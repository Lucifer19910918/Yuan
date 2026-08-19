#include "lu_decomposition.h"
#include "newton_raphson.h"
#include "power_system.h"
#include "sparse_matrix.h"
#include "thread_pool.h"
#include "timer.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace powerflow;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Test 1: SparseLU correctness on a small tri-diagonal system.
// ---------------------------------------------------------------------------
static bool testSparseLU() {
    SparseMatrix A(4);
    A.add(0, 0, 4.0); A.add(0, 1, 1.0);
    A.add(1, 0, 1.0); A.add(1, 1, 5.0); A.add(1, 2, 1.0);
    A.add(2, 1, 1.0); A.add(2, 2, 6.0); A.add(2, 3, 1.0);
    A.add(3, 2, 1.0); A.add(3, 3, 7.0);
    A.buildCSR();

    SparseLU lu;
    lu.factorizeAll(A);
    if (lu.singular()) {
        std::printf("[test] SparseLU: SINGULAR\n");
        return false;
    }
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4];
    lu.solve(b, x);

    double r[4];
    A.matvec(x, r);
    double maxr = 0.0;
    for (int i = 0; i < 4; ++i) maxr = std::max(maxr, std::fabs(r[i] - b[i]));
    std::printf("[test] SparseLU residual = %.3e  (nnzL=%zu nnzU=%zu)\n",
                maxr, lu.nnzL(), lu.nnzU());
    return maxr < 1e-10;
}

// ---------------------------------------------------------------------------
// Test 2: small 5-bus power flow, verify convergence + power balance.
// ---------------------------------------------------------------------------
static bool testSmallPowerFlow() {
    PowerSystem sys;
    sys.baseMVA = 100.0;
    sys.buses.resize(5);
    for (int i = 0; i < 5; ++i) {
        sys.buses[i].id = i;
        sys.buses[i].index = i;
    }
    sys.buses[0].type = BusType::SLACK; sys.buses[0].v_spec = 1.0; sys.buses[0].v = 1.0;
    sys.buses[1].type = BusType::PV;    sys.buses[1].v_spec = 1.0; sys.buses[1].p_gen = 40.0;
    sys.buses[2].type = BusType::PQ;   sys.buses[2].p_load = 20.0; sys.buses[2].q_load = 10.0;
    sys.buses[3].type = BusType::PQ;   sys.buses[3].p_load = 30.0; sys.buses[3].q_load = 15.0;
    sys.buses[4].type = BusType::PQ;   sys.buses[4].p_load = 20.0; sys.buses[4].q_load = 10.0;

    auto addBr = [&](int f, int t, double r, double x) {
        Branch b; b.from = f; b.to = t; b.r = r; b.x = x; b.b = 0.0; b.tap = 1.0; b.phi = 0.0;
        sys.branches.push_back(b);
    };
    addBr(0, 1, 0.01, 0.03);
    addBr(0, 2, 0.02, 0.05);
    addBr(1, 3, 0.03, 0.08);
    addBr(2, 4, 0.02, 0.06);
    addBr(3, 4, 0.01, 0.04);

    NewtonRaphson nr(sys);
    nr.setup();
    NewtonRaphson::Options opt;
    opt.verbose = true;
    opt.tol = 1e-8;
    opt.maxIter = 50;
    auto res = nr.solve(opt);

    // Verify KCL: sum_i P_calc[i] (total injection == total losses) must equal
    // the total branch active losses computed independently from the solved
    // voltages. This cross-checks the Y assembly and the Jacobian formulas.
    double sumCalc = 0.0;
    for (const auto& b : sys.buses) sumCalc += nr.Pcalc()[b.index];

    auto cplx = [](double mag, double ang) {
        return std::complex<double>(mag * std::cos(ang), mag * std::sin(ang));
    };
    double branchLoss = 0.0;
    for (const auto& br : sys.branches) {
        const double r = br.r, x = br.x;
        const double z2 = r * r + x * x;
        const double g_ser =  r / z2;
        const double b_ser = -x / z2;
        const double b_half = 0.5 * br.b;
        const double a = br.tap, phi = br.phi;
        const std::complex<double> Vf = cplx(sys.buses[br.from].v, sys.buses[br.from].theta);
        const std::complex<double> Vt = cplx(sys.buses[br.to].v,   sys.buses[br.to].theta);
        const std::complex<double> ys(g_ser, b_ser);
        const std::complex<double> tap(a * std::cos(phi), a * std::sin(phi));
        // Consistent with PowerSystem::buildAdmittance:
        //   Y_ff = ys/a^2 + jb/2,  Y_ft = -ys/conj(tap),
        //   Y_tf = -ys/tap,       Y_tt = ys + jb/2.
        const std::complex<double> If = (ys / (a * a)) * Vf
                                      - (ys / std::conj(tap)) * Vt
                                      + std::complex<double>(0.0, b_half) * Vf;
        const std::complex<double> It = ys * Vt
                                      - (ys / tap) * Vf
                                      + std::complex<double>(0.0, b_half) * Vt;
        const double Pf = std::real(Vf * std::conj(If));
        const double Pt = std::real(Vt * std::conj(It));
        branchLoss += (Pf + Pt);
    }
    std::printf("[test] losses check: sumPcalc=%.6f  branchLoss=%.6f  diff=%.3e\n",
                sumCalc, branchLoss, std::fabs(sumCalc - branchLoss));
    std::printf("[test] bus results:\n");
    for (const auto& b : sys.buses) {
        const char* ts = (b.type == BusType::SLACK) ? "SLACK"
                       : (b.type == BusType::PV)    ? "PV"
                                                    : "PQ";
        std::printf("  bus %d %-5s  V=%.6f  theta=%.4f deg  P=%.4f Q=%.4f\n",
                    b.id, ts, b.v, b.theta * 180.0 / M_PI,
                    nr.Pcalc()[b.index], nr.Qcalc()[b.index]);
    }
    return res.converged && std::fabs(sumCalc - branchLoss) < 1e-6;
}

// ---------------------------------------------------------------------------
// Test 3: large random grid benchmark.
// ---------------------------------------------------------------------------
static void benchmarkLarge(int N) {
    std::printf("\n=== Benchmark: %d buses ===\n", N);
    PowerSystem sys;
    sys.baseMVA = 100.0;
    sys.buses.resize(N);
    std::mt19937 rng(20260819);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const int nPV = std::max(1, N / 10);
    for (int i = 0; i < N; ++i) {
        auto& b = sys.buses[i];
        b.id = i;
        b.index = i;
        if (i == 0) {
            b.type = BusType::SLACK;
            b.v_spec = 1.0;
            b.v = 1.0;
        } else if (i <= nPV) {
            b.type = BusType::PV;
            b.v_spec = 1.0 + 0.02 * (u01(rng) - 0.5);
        } else {
            b.type = BusType::PQ;
            b.p_load = 1.0 + 4.0 * u01(rng);   // 1..5 MW
            b.q_load = 0.3 + 1.5 * u01(rng);    // 0.3..1.8 MVAr
        }
    }

    // Balance generation to ~105% of load (slack absorbs the slack + losses).
    double totalLoad = 0.0;
    for (const auto& b : sys.buses) totalLoad += b.p_load;
    const double genPerPV = (totalLoad * 1.05) / std::max(1, nPV);
    for (int i = 1; i <= nPV; ++i) sys.buses[i].p_gen = genPerPV;

    auto addBr = [&](int f, int t, double r, double x) {
        Branch b; b.from = f; b.to = t; b.r = r; b.x = x; b.b = 0.0;
        b.tap = 1.0; b.phi = 0.0;
        sys.branches.push_back(b);
    };
    // Spanning tree: each node i>0 connects to a random earlier node.
    for (int i = 1; i < N; ++i) {
        int parent = static_cast<int>(u01(rng) * i);
        double r = 0.004 + 0.012 * u01(rng);
        double x = 0.025 + 0.080 * u01(rng);
        addBr(i, parent, r, x);
    }
    // Extra meshing edges (rings) to exercise fill-in.
    const int extra = N / 5;
    for (int k = 0; k < extra; ++k) {
        int a = 1 + static_cast<int>(u01(rng) * (N - 1));
        int b = 1 + static_cast<int>(u01(rng) * (N - 1));
        if (a == b) continue;
        double r = 0.006 + 0.020 * u01(rng);
        double x = 0.04 + 0.10 * u01(rng);
        addBr(a, b, r, x);
    }

    std::printf("buses=%zu  branches=%zu  PV=%d  PQ=%zu\n",
                sys.buses.size(), sys.branches.size(), nPV,
                sys.buses.size() - nPV - 1);

    Timer setupT;
    NewtonRaphson nr(sys);
    nr.setup();
    double setupMs = setupT.elapsed_ms();

#ifdef _OPENMP
    std::printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    std::printf("Jacobian nnz = %zu,  factor L+U nnz = %zu (fill factor x%.2f)\n",
                nr.jacobianNnz(), nr.factorNnz(),
                nr.jacobianNnz() ? (double)nr.factorNnz() / nr.jacobianNnz() : 0.0);
    std::printf("setup (Y assembly + symbolic LU): %.2f ms\n", setupMs);

    NewtonRaphson::Options opt;
    opt.verbose = true;
    opt.tol = 1e-7;
    opt.maxIter = 50;
    nr.solve(opt);
}

int main(int argc, char** argv) {
    std::printf("============ Power Flow C++ Solver ============\n");
    bool okLU = testSparseLU();
    bool okPF = testSmallPowerFlow();
    std::printf("[summary] SparseLU %s, small power flow %s\n",
                okLU ? "PASS" : "FAIL", okPF ? "PASS" : "FAIL");

    int N = 1000;
    if (argc > 1) N = std::atoi(argv[1]);
    if (N < 5) N = 5;
    benchmarkLarge(N);
    return (okLU && okPF) ? 0 : 1;
}
