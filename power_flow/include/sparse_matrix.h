#ifndef POWER_FLOW_SPARSE_MATRIX_H
#define POWER_FLOW_SPARSE_MATRIX_H

#include <vector>
#include <cstddef>
#include <string>
#include "thread_pool.h"

namespace powerflow {

// Compressed Sparse Row (CSR) sparse matrix of real values.
//
// Assembly model:
//   1. add(row, col, value) accumulates triplets (COO).
//   2. buildCSR() sorts rows by column index and merges duplicate entries.
//   3. The matrix is then ready for SpMV and LU extraction.
//
// Designed for power-flow Jacobian / admittance matrices which are highly
// sparse (a few off-diagonals per row), so CSR keeps memory traffic low.
class SparseMatrix {
public:
    SparseMatrix();
    explicit SparseMatrix(size_t n);          // square n x n
    SparseMatrix(size_t rows, size_t cols);

    void resize(size_t rows, size_t cols);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz()   const { return values_.size(); }
    bool   built() const { return built_; }

    // ---- Triplet (COO) assembly ---------------------------------------
    void reserve(size_t n);
    void add(size_t row, size_t col, double value);
    // Convert triplets to sorted CSR (duplicates merged by summation).
    void buildCSR();

    // ---- Sparse matrix-vector multiply -------------------------------
    // y = A * x  (x length == cols_, y length == rows_)
    void matvec(const double* x, double* y) const;
    void matvec(const std::vector<double>& x, std::vector<double>& y) const;

    // OpenMP-parallel SpMV (falls back to serial if OpenMP unavailable).
    void matvecParallel(const double* x, double* y) const;

    // ThreadPool-driven SpMV (task-level parallelism over row chunks).
    void matvecThreadPool(const double* x, double* y,
                          ThreadPool& pool, size_t chunk = 64) const;

    // ---- CSR accessors ----------------------------------------------
    const std::vector<int>&    rowPtr() const { return row_ptr_; }
    const std::vector<int>&    colIdx() const { return col_idx_; }
    const std::vector<double>& values() const { return values_; }

    // Mutable CSR access (used by SparseLU to populate factors).
    std::vector<int>&    rowPtrMut() { return row_ptr_; }
    std::vector<int>&    colIdxMut() { return col_idx_; }
    std::vector<double>& valuesMut() { return values_; }

    // Dense look-up of A(i,j); returns 0.0 if not present.
    double at(size_t i, size_t j) const;

    void clear();

private:
    size_t rows_ = 0, cols_ = 0;
    std::vector<int>    coo_row_, coo_col_;
    std::vector<double> coo_val_;
    bool built_ = false;
    std::vector<int>    row_ptr_;   // length rows_+1
    std::vector<int>    col_idx_;   // length nnz
    std::vector<double> values_;    // length nnz
};

} // namespace powerflow

#endif // POWER_FLOW_SPARSE_MATRIX_H
