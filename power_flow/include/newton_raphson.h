#ifndef POWER_FLOW_NEWTON_RAPHSON_H
#define POWER_FLOW_NEWTON_RAPHSON_H

#include <string>
#include <vector>
#include "lu_decomposition.h"
#include "power_system.h"
#include "thread_pool.h"
#include "timer.h"

namespace powerflow {

// Polar Newton-Raphson power-flow solver.
//
// Performance design:
//   * Admittance Y = G + jB assembled once into CSR.
//   * Jacobian *pattern* is derived once from the topology / bus types
//     (JacEntry table). Every iteration only re-evaluates the numerical
//     values in place -- no triplet re-assembly, no re-sort.
//   * SparseLU symbolic factorization runs once; subsequent iterations reuse
//     the pattern and only perform numeric factorization + solve.
//   * Power / mismatch computation and Jacobian value fill are parallelized
//     with OpenMP; a ThreadPool is also available for task-level dispatch.
//
// State vector x = [theta (non-slack), V (PQ buses)].
class NewtonRaphson {
public:
    struct Options {
        int    maxIter       = 50;
        double tol           = 1e-8;   // max |mismatch| in pu
        bool   flatStart     = true;
        bool   verbose       = true;
    };

    struct Result {
        bool   converged   = false;
        int    iterations  = 0;
        double maxMismatch = 0.0;
        double solveTimeMs = 0.0;
        double factorTimeMs = 0.0;
        double jacobiTimeMs = 0.0;
        double powerTimeMs  = 0.0;
        std::string message;
    };

    explicit NewtonRaphson(PowerSystem& sys);

    // Optional: attach an external thread pool for task-level parallelism.
    void setThreadPool(ThreadPool* pool) { pool_ = pool; }

    // Precompute Y, index maps and the static Jacobian pattern + LU symbolic
    // factorization. Must be called before solve().
    void setup();

    // Run the Newton iteration.
    Result solve(const Options& opt);

    // Access results / state.
    const std::vector<double>& Pcalc() const { return p_calc_; }
    const std::vector<double>& Qcalc() const { return q_calc_; }
    size_t jacobianNnz() const { return J_.nnz(); }
    size_t factorNnz()   const { return lu_.nnzLU(); }

private:
    // Metadata for one Jacobian nonzero entry, enough to recompute its value
    // from the current state without touching the topology again.
    struct JacEntry {
        int    row;
        int    col;
        int    i;     // equation bus index
        int    j;     // neighbor bus index (== i for diagonal)
        double G;     // G_ij (constant)
        double B;     // B_ij (constant)
        int    kind;  // 0..7 (see computeValue)
    };

    double computeValue(const JacEntry& e,
                        const std::vector<double>& V,
                        const std::vector<double>& th,
                        const std::vector<double>& Pcalc,
                        const std::vector<double>& Qcalc) const;

    PowerSystem& sys_;
    ThreadPool*  pool_ = nullptr;

    // Admittance matrices (shared sparsity pattern).
    SparseMatrix Yg_, Yb_;

    // Index maps.
    std::vector<int> p_eq_;      // P-equation row per bus (-1 if none)
    std::vector<int> q_eq_;      // Q-equation row per bus (-1 if none)
    std::vector<int> theta_col_;  // theta unknown column per bus (-1 if slack)
    std::vector<int> v_col_;      // V unknown column per bus (-1 if not PQ)
    int np_eq_ = 0;              // number of P-equations (non-slack buses)
    int nq_eq_ = 0;              // number of Q-equations (PQ buses)
    int m_     = 0;              // total Jacobian dimension

    // Jacobian as a fixed-pattern SparseMatrix, plus the entry metadata.
    SparseMatrix J_;
    std::vector<JacEntry> jac_;
    bool pattern_built_ = false;

    // Solver.
    SparseLU lu_;
    bool lu_analyzed_ = false;

    // Workspace.
    std::vector<double> p_calc_, q_calc_;
    std::vector<double> mismatch_;   // RHS (size m_)
    std::vector<double> dx_;         // solution (size m_)
};

} // namespace powerflow

#endif // POWER_FLOW_NEWTON_RAPHSON_H
