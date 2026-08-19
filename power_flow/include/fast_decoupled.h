#ifndef POWER_FLOW_FAST_DECOUPLED_H
#define POWER_FLOW_FAST_DECOUPLED_H

#include <string>
#include <vector>
#include "lu_decomposition.h"
#include "power_system.h"
#include "timer.h"

namespace powerflow {

// Fast Decoupled Load Flow (FDLF / XB scheme).
//
// Key performance advantage over Newton-Raphson:
//   * B' (P-theta) and B'' (Q-V) are CONSTANT matrices derived from the
//     network topology. They are factorized ONCE in setup().
//   * Every iteration and every repeated solve only performs triangular
//     solves (forward/back substitution) -- NO numeric LU factorization.
//   * This makes each solve O(nnz_factor) instead of O(nnz_factor * flops),
//     typically 10-50x faster per iteration than Newton-Raphson.
//
// Trade-off: linear convergence (vs Newton's quadratic), so more iterations
// are needed, but each iteration is so cheap that total time is much lower.
//
// XB scheme:
//   B'  = -Im(Y)  with slack row/col removed         (for dTheta)
//   B'' = -Im(Y)  with slack AND PV row/col removed  (for dV)
class FastDecoupled {
public:
    struct Options {
        int    maxIter  = 100;
        double tol      = 1e-7;    // max |mismatch| in pu
        bool   flatStart = true;
        bool   verbose   = true;
        // When true and flatStart==false (warm start), perform at most one
        // P-theta + one Q-V sweep per solve. This trades a small amount of
        // accuracy for a large speedup in repeated-solve workloads where each
        // step is close to the previous solution.
        bool   singleSweep = false;
        // When true, skip the final power re-evaluation at the end of solve.
        // The final pass only refreshes Pcalc()/Qcalc() for external readers;
        // in a tight repeated-solve loop nobody reads them between solves, so
        // skipping it saves one full O(nnz) power pass per solve.
        bool   skipFinalPower = false;
    };

    struct Result {
        bool   converged   = false;
        int    iterations  = 0;
        double maxMismatch = 0.0;
        double totalTimeMs  = 0.0;
        double powerTimeMs  = 0.0;   // cumulative power/mismatch eval
        double solveTimeMs  = 0.0;   // cumulative triangular solve time
        std::string message;
    };

    explicit FastDecoupled(PowerSystem& sys);

    // Build B' and B'', factorize them once. This is the one-time setup cost.
    void setup();

    // Run the FDLF iteration. Reuses the pre-factorized B' and B''.
    Result solve(const Options& opt);

    // Access results.
    const std::vector<double>& Pcalc() const { return p_calc_; }
    const std::vector<double>& Qcalc() const { return q_calc_; }
    size_t factorBpNnz()  const { return luBp_.nnzLU(); }
    size_t factorBppNnz() const { return luBpp_.nnzLU(); }

private:
    PowerSystem& sys_;
    SparseMatrix Yg_, Yb_;     // admittance G, B (shared pattern)

    // Index maps for the two sub-problems.
    // P-theta: all non-slack buses.
    // Q-V:     all PQ buses.
    std::vector<int> p_eq_;     // bus -> P equation index (-1 if slack)
    std::vector<int> q_eq_;     // bus -> Q equation index (-1 if not PQ)
    int np_ = 0;               // dimension of B'  (non-slack buses)
    int nq_ = 0;               // dimension of B'' (PQ buses)

    SparseMatrix Bp_, Bpp_;     // constant coefficient matrices
    SparseLU luBp_, luBpp_;     // factorizations (done once in setup)
    mutable SparseLU::WorkSpace wsBp_, wsBpp_;  // persistent solve workspace

    // Workspace.
    std::vector<double> p_calc_, q_calc_;
    std::vector<double> dP_, dQ_;    // mismatches (rhs vectors)
    std::vector<double> dTheta_, dV_; // corrections
};

} // namespace powerflow

#endif // POWER_FLOW_FAST_DECOUPLED_H
