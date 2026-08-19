// Benchmark: 100x100 pentadiagonal (5-diagonal) sparse linear system.
//
// Matrix construction:
//   * Symmetric, strictly diagonally dominant, 5 non-zeros per row on the
//     main diagonal plus ±1, ±2 off-diagonals (pentadiagonal band).
//   * Pattern mimics a 2D FDM Poisson-like stencil or a meshed distribution
//     network -- realistic sparsity, and bandedness makes the fill-in profile
//     interesting for MD ordering comparisons.
//
// Ground truth: x_true[i] = i+1, b = A * x_true (computed via SpMV so b is
// exact to machine precision for the assembled A).
//
// Three independent solvers are compared:
//   1. Our SparseLU   (CSR + MD ordering + symbolic/numeric split LU) --
//      the SAME code used for the 2000-bus power flow.
//   2. Dense LU with partial pivoting  (std O(n^3) Gaussian elimination) --
//      used as a gold-standard correctness reference; also a naive speed
//      baseline.
//   3. Eigen::SparseLU  (Eigen 3.4, MD ordering, Supernodal-ish sparse LU) --
//      a well-known open-source production sparse solver.
//
// Correctness is asserted three ways:
//   * residual ‖A·x - b‖∞
//   * solution error ‖x - x_true‖∞
//   * cross-solver agreement: max_ij |x_i[solver_a] - x_i[solver_b]|.
//
// Speed is measured for (a) one-shot (build A + factorize + solve) and
// (b) constant-matrix repeated solves (factorize once, solve N times -- the
// FDLF / repeated-solve regime used by the power flow).
#include "lu_decomposition.h"
#include "sparse_matrix.h"
#include "timer.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec = std::vector<double>;

// Build an n x n PENTADIAGONAL strictly-diagonally-dominant sparse matrix.
//   A(i,i)   = 10 + rand(0,1)         [diagonal, large enough to dominate]
//   A(i,i±1) = -2 + rand(-0.1,0.1)    [±1 band]
//   A(i,i±2) = -1 + rand(-0.05,0.05)  [±2 band]
// Returns SparseMatrix in CSR form, and also fills the dense matrix
// `denseA` (row-major, n*n doubles) for the dense-LU reference solver.
powerflow::SparseMatrix buildMatrix(int n, Vec& denseA, int seed) {
    using powerflow::SparseMatrix;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d01(0.0, 1.0);
    std::uniform_real_distribution<double> dp1(-0.1, 0.1);
    std::uniform_real_distribution<double> dp2(-0.05, 0.05);

    SparseMatrix A(n);
    denseA.assign(n * n, 0.0);
    auto setDense = [&](int i, int j, double v) { denseA[i*n + j] = v; };

    for (int i = 0; i < n; ++i) {
        // Diagonal
        double d = 10.0 + d01(rng);
        A.add(i, i, d);
        setDense(i, i, d);

        auto addIfIn = [&](int i_, int j_, double v) {
            if (j_ >= 0 && j_ < n) { A.add(i_, j_, v); setDense(i_, j_, v); }
        };
        // Off-diagonals: store the matrix as NON-symmetric intentionally
        // (just to test general non-symmetric sparse LU). The diagonal still
        // dominates, so no pivoting issues.
        addIfIn(i, i-2, -1.0 + dp2(rng));
        addIfIn(i, i-1, -2.0 + dp1(rng));
        addIfIn(i, i+1, -2.0 + dp1(rng));
        addIfIn(i, i+2, -1.0 + dp2(rng));
    }
    A.buildCSR();
    return A;
}

// Compute y = A_dense * x  (dense matvec for verification).
void denseMatvec(const Vec& A, const Vec& x, Vec& y, int n) {
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double* row = A.data() + i*n;
        for (int j = 0; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
}

// ================ Solver 2: Dense LU with partial pivoting ================
// Returns true on success (non-singular). Performs in-place factorization.
struct DenseLU {
    int n = 0;
    Vec LU;            // packed L+U: L unit lower (no diag stored)
    std::vector<int> ipiv;  // row permutation from partial pivoting
    bool factorized = false;

    void factorize(const Vec& denseA, int n_) {
        n = n_;
        LU = denseA;                 // copy
        ipiv.assign(n, 0);
        for (int k = 0; k < n; ++k) {
            // Partial pivoting: find max-abs row in column k below diagonal.
            int maxRow = k;
            double maxVal = std::fabs(LU[k*n + k]);
            for (int i = k + 1; i < n; ++i) {
                double v = std::fabs(LU[i*n + k]);
                if (v > maxVal) { maxVal = v; maxRow = i; }
            }
            ipiv[k] = maxRow;
            if (maxRow != k) {
                for (int j = 0; j < n; ++j) std::swap(LU[k*n + j], LU[maxRow*n + j]);
            }
            double piv = LU[k*n + k];
            if (std::fabs(piv) < 1e-18) { factorized = false; return; }
            for (int i = k + 1; i < n; ++i) {
                double lik = LU[i*n + k] / piv;
                LU[i*n + k] = lik;
                for (int j = k + 1; j < n; ++j) {
                    LU[i*n + j] -= lik * LU[k*n + j];
                }
            }
        }
        factorized = true;
    }

    // Solve A x = b using the factored form.
    void solve(const Vec& b, Vec& x) const {
        if (!factorized) { x.clear(); return; }
        x = b;
        // Apply row permutation to rhs: y = P b
        // (ipiv records swap(k, ipiv[k]) applied during elimination; apply
        //  the same sequence to b to get the permuted rhs.)
        Vec y(n);
        y = b;
        for (int k = 0; k < n; ++k) std::swap(y[k], y[ipiv[k]]);
        // Forward substitution L z = y  (L unit diag)
        for (int i = 0; i < n; ++i) {
            double s = y[i];
            for (int j = 0; j < i; ++j) s -= LU[i*n + j] * y[j];
            y[i] = s;
        }
        // Back substitution U x = y
        for (int i = n - 1; i >= 0; --i) {
            double s = y[i];
            for (int j = i + 1; j < n; ++j) s -= LU[i*n + j] * x[j];
            x[i] = s / LU[i*n + i];
        }
    }
};

// Utility: norm helpers.
double infNormDiff(const Vec& a, const Vec& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
double infNorm(const Vec& a) {
    double m = 0.0;
    for (double v : a) m = std::max(m, std::fabs(v));
    return m;
}

std::string fmtMs(double ms) {
    std::ostringstream os;
    if (ms < 0.001)    os << std::fixed << std::setprecision(3) << ms*1000.0 << " us";
    else if (ms < 1.0) os << std::fixed << std::setprecision(3) << ms << " ms";
    else               os << std::fixed << std::setprecision(2) << ms << " ms";
    return os.str();
}

} // namespace


int main(int argc, char** argv) {
    using namespace powerflow;

    const int n = (argc > 1) ? std::atoi(argv[1]) : 100;
    const int seed = 20260819;
    const int warm_reps = 1000;  // repeated-solve regime (constant matrix)
    std::printf("=========================================================\n");
    std::printf(" Benchmark: %dx%d pentadiagonal sparse linear system\n", n, n);
    std::printf("=========================================================\n");

    // -------- 1. Build problem -----------------------------------------
    Timer buildT;
    Vec denseA;
    SparseMatrix A = buildMatrix(n, denseA, seed);
    Vec x_true(n);
    for (int i = 0; i < n; ++i) x_true[i] = static_cast<double>(i + 1);
    Vec b(n), r_check(n);
    denseMatvec(denseA, x_true, b, n);   // b = A * x_true (from dense copy,
                                         // independent of the CSR copy below)
    double buildMs = buildT.elapsed_ms();

    // Print matrix info.
    int nnz = static_cast<int>(A.nnz());
    std::printf("Matrix: %dx%d, nnz=%d (density=%.4f%%), avg %.1f nonzeros/row\n",
                n, n, nnz, 100.0 * nnz / double(n*n),
                double(nnz) / n);
    std::printf("Pattern: pentadiagonal (diag ±1, ±2), seeded random values\n");
    std::printf("Ground truth: x_true[i]=i+1  (‖x_true‖∞=%.1f)\n", infNorm(x_true));
    std::printf("Build time: %s\n", fmtMs(buildMs).c_str());

    // Cross-check: the CSR copy used by our sparse solver must agree with
    // the dense copy used by the dense solver (this catches buildCSR bugs).
    Vec b_csr(n, 0.0);
    A.matvec(x_true.data(), b_csr.data());
    double csr_dense_b_diff = 0.0;
    for (int i = 0; i < n; ++i) csr_dense_b_diff =
        std::max(csr_dense_b_diff, std::fabs(b_csr[i] - b[i]));
    std::printf("CSR vs dense matvec: max |b_CSR - b_dense| = %.2e  %s\n",
                csr_dense_b_diff, (csr_dense_b_diff < 1e-12) ? "[OK]" : "[MISMATCH!]");

    // Solver results.
    Vec x_ours(n), x_dense(n), x_eigen(n);
    double time_ours = 0.0, time_dense = 0.0, time_eigen = 0.0;
    double time_ours_analyze = 0.0, time_ours_factor = 0.0, time_ours_solve = 0.0;
    double factorNnzL = 0, factorNnzU = 0;
    bool our_ok = true, dense_ok = true, eigen_ok = true;

    // -------- 2. Our SparseLU -----------------------------------------
    std::printf("\n---- [1] Our SparseLU (CSR + MD ordering) ----\n");
    {
        SparseLU lu;
        Timer aT;  lu.analyze(A);           double aMs = aT.elapsed_ms();
        Timer fT;  lu.factorize(A);         double fMs = fT.elapsed_ms();
        Timer sT;  lu.solve(b.data(), x_ours.data());  double sMs = sT.elapsed_ms();
        time_ours = aMs + fMs + sMs;
        time_ours_analyze = aMs; time_ours_factor = fMs; time_ours_solve = sMs;
        factorNnzL = static_cast<double>(lu.nnzL());
        factorNnzU = static_cast<double>(lu.nnzU());
        our_ok = !lu.singular();
        std::printf("  analyze (MD+symbolic): %s\n", fmtMs(aMs).c_str());
        std::printf("  factorize (numeric):   %s\n", fmtMs(fMs).c_str());
        std::printf("  solve (tri-solve):     %s\n", fmtMs(sMs).c_str());
        std::printf("  total (analyze+fact+solve): %s\n", fmtMs(time_ours).c_str());
        std::printf("  L nnz=%zu  U nnz=%zu  total=%zu  (fill factor vs A: %.2fx)\n",
                    lu.nnzL(), lu.nnzU(), lu.nnzLU(),
                    nnz ? double(lu.nnzLU()) / nnz : 0.0);
        if (!our_ok) std::printf("  !! SINGULAR !!\n");
    }

    // -------- 3. Dense LU (partial pivoting) --------------------------
    std::printf("\n---- [2] Dense LU (O(n^3), partial pivoting, gold ref) ----\n");
    {
        DenseLU dlu;
        Timer fT;  dlu.factorize(denseA, n);  double fMs = fT.elapsed_ms();
        Timer sT;  dlu.solve(b, x_dense);      double sMs = sT.elapsed_ms();
        time_dense = fMs + sMs;
        dense_ok = dlu.factorized;
        std::printf("  factorize:  %s  (O(n^3) = %.0f flops)\n",
                    fmtMs(fMs).c_str(), (2.0/3.0)*n*n*n);
        std::printf("  solve:      %s\n", fmtMs(sMs).c_str());
        std::printf("  total:      %s\n", fmtMs(time_dense).c_str());
        if (!dense_ok) std::printf("  !! SINGULAR !!\n");
    }

    // -------- 4. Eigen::SparseLU (open-source ref) --------------------
    std::printf("\n---- [3] Eigen::SparseLU (Eigen 3.4, open source) ----\n");
    {
        // Convert our CSR to Eigen's compressed format.
        using SpMat = Eigen::SparseMatrix<double>;
        using Tri = Eigen::Triplet<double>;
        std::vector<Tri> trips;
        trips.reserve(nnz);
        const auto& Ap = A.rowPtr();
        const auto& Aj = A.colIdx();
        const auto& Av = A.values();
        for (int i = 0; i < n; ++i) {
            for (int p = Ap[i]; p < Ap[i+1]; ++p) trips.emplace_back(i, Aj[p], Av[p]);
        }
        SpMat Ae(n, n);
        Ae.setFromTriplets(trips.begin(), trips.end());
        Ae.makeCompressed();

        Eigen::VectorXd be(n), xe(n);
        for (int i = 0; i < n; ++i) be(i) = b[i];

        Eigen::SparseLU<SpMat> solver;
        Timer aT;  solver.analyzePattern(Ae);      double aMs = aT.elapsed_ms();
        Timer fT;  solver.factorize(Ae);           double fMs = fT.elapsed_ms();
        Timer sT;  xe = solver.solve(be);          double sMs = sT.elapsed_ms();
        time_eigen = aMs + fMs + sMs;
        eigen_ok = (solver.info() == Eigen::Success);
        for (int i = 0; i < n; ++i) x_eigen[i] = xe(i);
        std::printf("  analyzePattern: %s\n", fmtMs(aMs).c_str());
        std::printf("  factorize:      %s\n", fmtMs(fMs).c_str());
        std::printf("  solve:          %s\n", fmtMs(sMs).c_str());
        std::printf("  total:          %s\n", fmtMs(time_eigen).c_str());
        if (!eigen_ok) std::printf("  !! FAILED (info=%d) !!\n", int(solver.info()));
    }

    // -------- 5. Correctness verification -----------------------------
    std::printf("\n=========================================================\n");
    std::printf(" Correctness verification (3 independent checks)\n");
    std::printf("=========================================================\n");

    struct Check { std::string label; double value; };
    std::vector<Check> checks;

    // a) Residual ‖A x - b‖∞ (computed via dense matvec so all three solvers
    //    share the SAME reference matrix, not their copies).
    auto residual = [&](const Vec& x) {
        Vec r(n);
        denseMatvec(denseA, x, r, n);
        double m = 0.0;
        for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(r[i] - b[i]));
        return m;
    };
    checks.push_back({"Our SparseLU  ‖Ax-b‖∞",      our_ok   ? residual(x_ours)  : -1.0});
    checks.push_back({"Dense LU      ‖Ax-b‖∞",      dense_ok ? residual(x_dense) : -1.0});
    checks.push_back({"Eigen::SLU    ‖Ax-b‖∞",      eigen_ok ? residual(x_eigen) : -1.0});

    // b) Solution error vs exact ground truth x_true.
    checks.push_back({"Our SparseLU  ‖x - x_true‖∞", our_ok   ? infNormDiff(x_ours,  x_true) : -1.0});
    checks.push_back({"Dense LU      ‖x - x_true‖∞", dense_ok ? infNormDiff(x_dense, x_true) : -1.0});
    checks.push_back({"Eigen::SLU    ‖x - x_true‖∞", eigen_ok ? infNormDiff(x_eigen, x_true) : -1.0});

    // c) Cross-solver agreement (pairwise max difference).
    double ours_vs_dense = (our_ok && dense_ok) ? infNormDiff(x_ours, x_dense) : -1.0;
    double ours_vs_eigen = (our_ok && eigen_ok) ? infNormDiff(x_ours, x_eigen) : -1.0;
    double dense_vs_eigen = (dense_ok && eigen_ok) ? infNormDiff(x_dense, x_eigen) : -1.0;
    checks.push_back({"|ours - dense|∞",  ours_vs_dense});
    checks.push_back({"|ours - eigen|∞",  ours_vs_eigen});
    checks.push_back({"|dense - eigen|∞", dense_vs_eigen});

    for (const auto& c : checks) {
        const char* verdict =
            (c.value < 0)                ? "N/A" :
            (c.value < 1e-10)            ? "[EXCELLENT]" :
            (c.value < 1e-6)             ? "[GOOD]" :
            (c.value < 1e-2)             ? "[FAIR]" :
                                             "[POOR]";
        if (c.value < 0)
            std::printf("  %-35s : N/A\n", c.label.c_str());
        else
            std::printf("  %-35s : %.3e  %s\n", c.label.c_str(), c.value, verdict);
    }

    // Overall pass/fail.
    const double acceptTol = 1e-9;
    bool allPass =
        our_ok && dense_ok && eigen_ok &&
        checks[0].value < acceptTol &&   // ours residual
        checks[1].value < acceptTol &&   // dense residual
        checks[2].value < acceptTol &&   // eigen residual
        ours_vs_dense < acceptTol &&
        ours_vs_eigen < acceptTol &&
        dense_vs_eigen < acceptTol;
    std::printf("\n  -> Overall correctness verdict: %s\n",
                allPass ? "ALL PASS ✓ (所有求解器结果一致，残差机器精度级)"
                        : "FAILED ✗");

    // -------- 6. Efficiency comparison --------------------------------
    std::printf("\n=========================================================\n");
    std::printf(" Efficiency comparison (one-shot + repeated solves)\n");
    std::printf("=========================================================\n");
    std::printf("\n [One-shot total time: analyze + factorize + solve]\n");
    std::printf("   Our SparseLU     : %s  (%.2fx vs dense,  %.2fx vs Eigen)\n",
                fmtMs(time_ours).c_str(),
                time_dense > 0 ? time_dense / time_ours : 0,
                time_eigen > 0 ? time_eigen / time_ours : 0);
    std::printf("   Dense LU (ref)   : %s  -- baseline\n", fmtMs(time_dense).c_str());
    std::printf("   Eigen::SparseLU  : %s  -- open-source baseline\n", fmtMs(time_eigen).c_str());

    // Repeated solves: factorize once, then solve `warm_reps` times.
    // This is the FDLF regime: constant matrix, only triangular solves.
    std::printf("\n [Constant-matrix repeated solves (%d reps, one-shot factor)]\n", warm_reps);
    double ours_warm = 0, eigen_warm = 0, dense_warm = 0;

    // Our SparseLU warm loop.
    {
        SparseLU lu;
        lu.analyze(A);
        lu.factorize(A);
        SparseLU::WorkSpace ws;
        std::vector<double> tmp(n);
        Timer t;
        for (int k = 0; k < warm_reps; ++k) {
            // Slightly wobble b each rep (force real work, not a cache no-op).
            b[0] += 1e-15;
            lu.solve(b.data(), tmp.data(), ws);
        }
        ours_warm = t.elapsed_ms();
    }

    // Dense LU warm loop.
    {
        DenseLU dlu;
        dlu.factorize(denseA, n);
        Vec tmp(n);
        Timer t;
        for (int k = 0; k < warm_reps; ++k) {
            b[0] += 1e-15;
            dlu.solve(b, tmp);
        }
        dense_warm = t.elapsed_ms();
    }

    // Eigen warm loop.
    {
        using SpMat = Eigen::SparseMatrix<double>;
        using Tri = Eigen::Triplet<double>;
        std::vector<Tri> trips; trips.reserve(nnz);
        const auto& Ap = A.rowPtr();
        const auto& Aj = A.colIdx();
        const auto& Av = A.values();
        for (int i = 0; i < n; ++i)
            for (int p = Ap[i]; p < Ap[i+1]; ++p) trips.emplace_back(i, Aj[p], Av[p]);
        SpMat Ae(n, n); Ae.setFromTriplets(trips.begin(), trips.end()); Ae.makeCompressed();
        Eigen::VectorXd be(n), xe(n);
        for (int i = 0; i < n; ++i) be(i) = b[i];
        Eigen::SparseLU<SpMat> solver;
        solver.analyzePattern(Ae);
        solver.factorize(Ae);
        Timer t;
        for (int k = 0; k < warm_reps; ++k) {
            be(0) += 1e-15;
            xe = solver.solve(be);
        }
        eigen_warm = t.elapsed_ms();
    }

    double ours_avg   = ours_warm   / warm_reps;
    double dense_avg  = dense_warm  / warm_reps;
    double eigen_avg  = eigen_warm  / warm_reps;
    std::printf("   Our SparseLU    : %s total  (avg %s/solve,  %.2fx vs dense,  %.2fx vs Eigen)\n",
                fmtMs(ours_warm).c_str(),  fmtMs(ours_avg).c_str(),
                dense_avg > 0 ? dense_avg / ours_avg : 0,
                eigen_avg > 0 ? eigen_avg / ours_avg : 0);
    std::printf("   Dense LU (ref)  : %s total  (avg %s/solve) -- baseline\n",
                fmtMs(dense_warm).c_str(), fmtMs(dense_avg).c_str());
    std::printf("   Eigen::SparseLU : %s total  (avg %s/solve) -- open-source baseline\n",
                fmtMs(eigen_warm).c_str(), fmtMs(eigen_avg).c_str());

    std::printf("\n Sparsity facts for %dx%d matrix:\n", n, n);
    std::printf("   A nnz           = %d  (%.1f / row)\n", nnz, double(nnz)/n);
    std::printf("   L+U nnz (ours)  = %.0f  (fill %.2fx)\n",
                factorNnzL + factorNnzU,
                nnz ? (factorNnzL + factorNnzU) / nnz : 0);
    std::printf("   Dense matrix    = %d entries  (%.1fx the sparse A)\n",
                n*n, double(n*n)/nnz);

    return allPass ? 0 : 1;
}
