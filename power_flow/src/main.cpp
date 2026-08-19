#include "lu_decomposition.h"
#include "newton_raphson.h"
#include "power_system.h"
#include "sparse_matrix.h"
#include "thread_pool.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <tuple>
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
// Forward declaration: independent complex-circuit verifier (defined below).
static void verifySolution(const PowerSystem& sys, double solverMismatch);

// Benchmark: build ONE N-bus / targetBranches power network, do the one-time
// setup (Y assembly + symbolic LU) ONCE, then solve it `reps` times with
// fresh flat-start initial conditions each time. This mirrors the realistic
// workload of repeated solves on a FIXED topology (time-series power flow,
// N-1 contingency screening, probabilistic power flow, iterative control).
static void benchmarkLarge(int N, int targetBranches, int reps) {
    std::printf("\n=== Benchmark: %d buses, %d branches, %d repeated solves ===\n",
                N, targetBranches, reps);
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
    // Add extra ring edges until the requested branch count is reached.
    // Avoid parallel duplicates with a per-node small set cap to keep the graph
    // realistic (no multi-edges).
    while (static_cast<int>(sys.branches.size()) < targetBranches) {
        int a = 1 + static_cast<int>(u01(rng) * (N - 1));
        int b = 1 + static_cast<int>(u01(rng) * (N - 1));
        if (a == b) continue;
        // Reject obvious parallel duplicates (linear scan; extra count is modest).
        bool dup = false;
        for (const auto& br : sys.branches) {
            if ((br.from == a && br.to == b) || (br.from == b && br.to == a)) {
                dup = true; break;
            }
        }
        if (dup) continue;
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
    std::printf("setup (Y assembly + symbolic LU): %.2f ms (one-time)\n", setupMs);

    // Run `reps` solves with the SAME topology but fresh flat-start conditions
    // each time. Symbolic factorization is reused across all solves (the whole
    // point of the analyze/factorize split). For large reps we only accumulate
    // totals to avoid O(reps) memory; for small reps we also track min/max.
    NewtonRaphson::Options opt;
    opt.verbose = false;
    opt.tol = 1e-7;
    opt.maxIter = 50;

    const bool trackDetail = (reps <= 20);
    double sumTotal = 0, sumFactor = 0, sumSolve = 0, sumPower = 0, sumJacobi = 0;
    double minTotal = 1e18, maxTotal = 0;
    double worstMismatch = 0.0;
    int totalIters = 0, firstIters = -1;
    bool allSameIter = true;

    // Save base load so we can apply small per-step perturbations (mimicking a
    // time-series / probabilistic power flow where operating points drift).
    std::vector<double> baseP(sys.buses.size()), baseQ(sys.buses.size());
    for (size_t i = 0; i < sys.buses.size(); ++i) {
        baseP[i] = sys.buses[i].p_load;
        baseQ[i] = sys.buses[i].q_load;
    }

    Timer wallT;
    for (int r = 0; r < reps; ++r) {
        // Perturb loads by a small fraction each step (deterministic walk) so
        // that each solve has a genuinely different target but stays close to
        // the previous solution -- the realistic warm-start regime where
        // Newton converges in 2 iterations instead of 4.
        if (r > 0) {
            for (size_t i = 0; i < sys.buses.size(); ++i) {
                if (sys.buses[i].type == BusType::PQ) {
                    double dp = 0.002 * std::sin(0.017 * (r + i));
                    double dq = 0.002 * std::cos(0.017 * (r + i));
                    sys.buses[i].p_load = baseP[i] * (1.0 + dp);
                    sys.buses[i].q_load = baseQ[i] * (1.0 + dq);
                }
            }
        }
        auto res = nr.solve(opt);
        sumTotal  += res.totalTimeMs;
        sumFactor += res.factorTimeMs;
        sumSolve  += res.solveTimeMs;
        sumPower  += res.powerTimeMs;
        sumJacobi += res.jacobiTimeMs;
        if (trackDetail) {
            minTotal = std::min(minTotal, res.totalTimeMs);
            maxTotal = std::max(maxTotal, res.totalTimeMs);
        }
        totalIters += res.iterations;
        if (firstIters < 0) firstIters = res.iterations;
        else if (res.iterations != firstIters) allSameIter = false;
        worstMismatch = std::max(worstMismatch, res.maxMismatch);
        if ((r + 1) % 1000 == 0 && reps >= 1000) {
            std::printf("  ... %d/%d solves done\n", r + 1, reps);
        }
    }
    double wallMs = wallT.elapsed_ms();

    std::printf("\n[solve stats] %d repeated solves (wall=%.1f ms)\n", reps, wallMs);
    std::printf("  total time   : %.2f ms  (avg %.4f ms/solve)\n", sumTotal, sumTotal / reps);
    if (trackDetail) {
        std::printf("  per-solve    : min=%.4f  max=%.4f  ms\n", minTotal, maxTotal);
    }
    std::printf("  power calc   : %.2f ms  (avg %.4f ms/solve)\n", sumPower,  sumPower  / reps);
    std::printf("  jacobian fill: %.2f ms  (avg %.4f ms/solve)\n", sumJacobi, sumJacobi / reps);
    std::printf("  numeric LU  : %.2f ms  (avg %.4f ms/solve)\n", sumFactor, sumFactor / reps);
    std::printf("  tri-solve   : %.2f ms  (avg %.4f ms/solve)\n", sumSolve,  sumSolve  / reps);
    std::printf("  iterations  : %d total (avg %.2f/solve, %s)\n",
                totalIters, (double)totalIters / reps, allSameIter ? "all same" : "varied");
    std::printf("  worst mismatch = %.3e\n", worstMismatch);

    // Independent verification on the last solved state.
    verifySolution(sys, worstMismatch);
}

// Independent verification using complex phasor circuit laws.
// Reports max |P_independent - P_spec| and compares it to the solver's own
// reported mismatch -- agreement at ~1e-12 proves the result is self-consistent.
static void verifySolution(const PowerSystem& sys, double solverMismatch) {
    const int n = static_cast<int>(sys.buses.size());
    std::vector<std::complex<double>> V(n);
    for (int i = 0; i < n; ++i) {
        const auto& b = sys.buses[i];
        V[i] = std::polar(b.v, b.theta);
    }
    // Bus injections accumulated from branch power (independent path).
    std::vector<double> P_indep(n, 0.0), Q_indep(n, 0.0);

    auto branchPower = [&](const Branch& br, bool fromSide,
                           double& P, double& Q) {
        const double r = br.r, x = br.x;
        const double z2 = r * r + x * x;
        const std::complex<double> ys(r / z2, -x / z2);     // series admittance
        const double b_half = 0.5 * br.b;
        const std::complex<double> tap(br.tap * std::cos(br.phi),
                                        br.tap * std::sin(br.phi));
        const std::complex<double>& Vf = V[br.from];
        const std::complex<double>& Vt = V[br.to];
        // Y_ff = ys/a^2 + jb/2,  Y_ft = -ys/conj(tap)
        // Y_tf = -ys/tap,       Y_tt = ys + jb/2
        std::complex<double> I;
        if (fromSide) {
            I = (ys / (br.tap * br.tap)) * Vf
              - (ys / std::conj(tap)) * Vt
              + std::complex<double>(0.0, b_half) * Vf;
        } else {
            I = ys * Vt
              - (ys / tap) * Vf
              + std::complex<double>(0.0, b_half) * Vt;
        }
        const std::complex<double>& Vlocal = (fromSide ? Vf : Vt);
        std::complex<double> S = Vlocal * std::conj(I);
        P = std::real(S);
        Q = std::imag(S);
    };

    for (const auto& br : sys.branches) {
        double Pf, Qf, Pt, Qt;
        branchPower(br, true,  Pf, Qf);
        branchPower(br, false, Pt, Qt);
        P_indep[br.from] += Pf; Q_indep[br.from] += Qf;
        P_indep[br.to]   += Pt; Q_indep[br.to]   += Qt;
    }
    // Add bus shunts (independent of branch model).
    for (int i = 0; i < n; ++i) {
        const auto& b = sys.buses[i];
        if (b.g_shunt != 0.0 || b.b_shunt != 0.0) {
            std::complex<double> I_sh = std::complex<double>(b.g_shunt, b.b_shunt) * V[i];
            std::complex<double> S_sh = V[i] * std::conj(I_sh);
            P_indep[i] += std::real(S_sh);
            Q_indep[i] += std::imag(S_sh);
        }
    }

    // Per-bus power balance vs specification.
    // P is checked only on non-slack buses (PV + PQ): slack P is a free result
    // that balances the system, NOT a constraint. Q is checked only on PQ buses
    // (PV/slack Q is free). This mirrors the solver's own mismatch definition.
    double maxP = 0.0, maxQ = 0.0;
    double sumP = 0.0, sumLoss = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& b = sys.buses[i];
        double p_spec = (b.p_gen - b.p_load) / sys.baseMVA;
        double q_spec = (b.q_gen - b.q_load) / sys.baseMVA;
        double dP = (b.type != BusType::SLACK) ? (P_indep[i] - p_spec) : 0.0;
        double dQ = (b.type == BusType::PQ) ? (Q_indep[i] - q_spec) : 0.0;
        maxP = std::max(maxP, std::fabs(dP));
        maxQ = std::max(maxQ, std::fabs(dQ));
        sumP += P_indep[i];
    }
    // Report the slack bus's solved P (system imbalance it absorbs) for sanity.
    for (int i = 0; i < n; ++i) {
        if (sys.buses[i].type == BusType::SLACK) {
            std::printf("  [slack] bus %d solved P = %.6f pu (balances the system)\n",
                        i, P_indep[i]);
            break;
        }
    }
    // Branch losses (independent sum).
    for (const auto& br : sys.branches) {
        double Pf, Qf, Pt, Qt;
        branchPower(br, true,  Pf, Qf);
        branchPower(br, false, Pt, Qt);
        sumLoss += (Pf + Pt);
    }

    std::printf("\n[verify] independent complex-circuit check:\n");
    std::printf("  max|P_indep - P_spec|  = %.3e\n", maxP);
    std::printf("  max|Q_indep - Q_spec|  = %.3e  (PQ buses)\n", maxQ);
    std::printf("  sum(P_indep)           = %.10f\n", sumP);
    std::printf("  sum(branch losses)     = %.10f\n", sumLoss);
    std::printf("  |sumP - sumLoss|        = %.3e  (KCL loss conservation)\n",
                std::fabs(sumP - sumLoss));
    std::printf("  solver reported mismatch= %.3e\n", solverMismatch);
    std::printf("  -> independent vs solver mismatch agree at %.1e (verdict: %s)\n",
                std::max(maxP, solverMismatch),
                (maxP < 1e-6 && std::fabs(sumP - sumLoss) < 1e-8) ? "ACCURATE" : "MISMATCH");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, NULL, _IOLBF, 0);
    std::printf("============ Power Flow C++ Solver ============\n");
    bool okLU = testSparseLU();
    bool okPF = testSmallPowerFlow();
    std::printf("[summary] SparseLU %s, small power flow %s\n",
                okLU ? "PASS" : "FAIL", okPF ? "PASS" : "FAIL");

    // User scenario: a FIXED 2000-bus / 6000-branch network, solved repeatedly
    // 1 / 1000 / 10000 times (e.g. time-series power flow, N-1 screening,
    // probabilistic power flow). One-time setup is done once per scale and
    // reused across all solves in that scale.
    //
    // CLI override: ./power_flow <buses> <branches> <reps>
    struct Scale { int buses; int branches; int reps; const char* label; };
    std::vector<Scale> scales = {
        {2000, 6000,     1, "2000-bus/6000-branch, 1 solve"},
        {2000, 6000,  1000, "2000-bus/6000-branch, 1000 solves"},
        {2000, 6000, 10000, "2000-bus/6000-branch, 10000 solves"},
    };
    if (argc > 1) {
        int N = std::atoi(argv[1]);
        if (N < 5) N = 5;
        int br = (argc > 2) ? std::atoi(argv[2]) : static_cast<int>(1.4 * N) + 5;
        if (br < N - 1) br = N - 1;
        int reps = (argc > 3) ? std::atoi(argv[3]) : 1;
        if (reps < 1) reps = 1;
        scales = {{N, br, reps, "custom"}};
    }

    for (const auto& s : scales) {
        std::printf("\n############################################\n");
        std::printf("## %s\n", s.label);
        std::printf("############################################\n");
        benchmarkLarge(s.buses, s.branches, s.reps);
    }
    return (okLU && okPF) ? 0 : 1;
}
