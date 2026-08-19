#ifndef POWER_FLOW_LU_DECOMPOSITION_H
#define POWER_FLOW_LU_DECOMPOSITION_H

#include <vector>
#include "sparse_matrix.h"

namespace powerflow {

// Sparse LU factorization with separated symbolic and numeric phases and a
// minimum-degree (MD) fill-reducing ordering.
//
//   analyze(A)      -> (1) compute an MD ordering P over A's symmetric pattern,
//                      (2) build the permuted structure P A P^T,
//                      (3) symbolic factorization (reachability) -> L/U patterns.
//   factorize(A)    -> numeric factorization reusing the pattern. Re-callable
//                      for matrices with the SAME structure but different values
//                      (the Newton-Raphson hot path: analyze once, factorize
//                      every iteration).
//   solve(b, x)      -> forward + back substitution with permutation: x = A^{-1} b.
//
// L is unit lower triangular (implicit unit diagonal), U includes its diagonal.
// Both are stored row-wise (CSR) in the permuted index space.
class SparseLU {
public:
    void analyze(const SparseMatrix& A);
    void factorize(const SparseMatrix& A);
    void factorizeAll(const SparseMatrix& A);

    void solve(const double* b, double* x) const;
    void solve(const std::vector<double>& b, std::vector<double>& x) const;

    int    n()          const { return n_; }
    bool   analyzed()   const { return analyzed_; }
    bool   factorized() const { return factorized_; }
    bool   singular()   const { return singular_; }
    size_t nnzL() const { return L_col_.size(); }
    size_t nnzU() const { return U_col_.size(); }
    size_t nnzLU() const { return nnzL() + nnzU(); }

    // Elimination ordering: perm_[new_index] = old_index.
    const std::vector<int>& permutation() const { return perm_; }

private:
    int  n_ = 0;
    bool analyzed_   = false;
    bool factorized_ = false;
    bool singular_   = false;

    std::vector<int> perm_;     // new -> old
    std::vector<int> invperm_;  // old -> new

    // Permuted input matrix P A P^T: structure (constant) + value source map.
    std::vector<int>    Ap_ptr_, Ap_col_;
    std::vector<int>    a_src_;    // Ap nonzero k comes from A.values[a_src_[k]]
    std::vector<double> Ap_val_;   // refreshed every factorize()

    // L (unit lower) and U (upper, diag stored separately) in permuted order.
    std::vector<int>    L_ptr_, L_col_;
    std::vector<double> L_val_;
    std::vector<int>    U_ptr_, U_col_;
    std::vector<double> U_val_;
    std::vector<double> U_diag_;

    // Persistent workspace.
    std::vector<double> w_;
    std::vector<int>    marker_;
};

} // namespace powerflow

#endif // POWER_FLOW_LU_DECOMPOSITION_H
