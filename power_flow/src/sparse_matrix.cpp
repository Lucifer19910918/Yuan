#include "sparse_matrix.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace powerflow {

SparseMatrix::SparseMatrix() = default;

SparseMatrix::SparseMatrix(size_t n)
    : rows_(n), cols_(n), row_ptr_(n + 1, 0) {}

SparseMatrix::SparseMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

void SparseMatrix::resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    coo_row_.clear();
    coo_col_.clear();
    coo_val_.clear();
    row_ptr_.assign(rows + 1, 0);
    col_idx_.clear();
    values_.clear();
    built_ = false;
}

void SparseMatrix::reserve(size_t n) {
    coo_row_.reserve(n);
    coo_col_.reserve(n);
    coo_val_.reserve(n);
}

void SparseMatrix::add(size_t row, size_t col, double value) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("SparseMatrix::add index out of range");
    }
    coo_row_.push_back(static_cast<int>(row));
    coo_col_.push_back(static_cast<int>(col));
    coo_val_.push_back(value);
    built_ = false;
}

void SparseMatrix::buildCSR() {
    const size_t ntrip = coo_row_.size();
    row_ptr_.assign(rows_ + 1, 0);

    // 1. Count triplets per row.
    for (size_t k = 0; k < ntrip; ++k) {
        row_ptr_[coo_row_[k] + 1]++;
    }
    // 2. Prefix sum -> temporary row offsets (pre-merge).
    for (size_t i = 0; i < rows_; ++i) {
        row_ptr_[i + 1] += row_ptr_[i];
    }

    // 3. Scatter triplets into a (row-sorted, col-unsorted) buffer.
    col_idx_.resize(ntrip);
    values_.resize(ntrip);
    std::vector<int> cursor(rows_, 0);
    for (size_t k = 0; k < ntrip; ++k) {
        int r = coo_row_[k];
        int pos = row_ptr_[r] + cursor[r]++;
        col_idx_[pos] = coo_col_[k];
        values_[pos] = coo_val_[k];
    }

    // 4. For each row: sort by column index and merge duplicates by sum.
    //    First pass: compute the compressed (merged) length of every row.
    std::vector<int> new_ptr(rows_ + 1, 0);
    std::vector<std::pair<int, double>> buf; // reused per row (thread-local)

    #pragma omp parallel
    {
        std::vector<std::pair<int, double>> local_buf;
        #pragma omp for schedule(dynamic, 64)
        for (int r = 0; r < static_cast<int>(rows_); ++r) {
            int begin = row_ptr_[r];
            int end = row_ptr_[r + 1];
            local_buf.clear();
            local_buf.reserve(end - begin);
            for (int p = begin; p < end; ++p) {
                local_buf.emplace_back(col_idx_[p], values_[p]);
            }
            std::sort(local_buf.begin(), local_buf.end());
            int unique = 0;
            int prev = -1;
            for (auto& pr : local_buf) {
                if (pr.first != prev) { ++unique; prev = pr.first; }
            }
            new_ptr[r + 1] = unique;
        }
    }

    // 5. Prefix sum of merged lengths.
    for (size_t i = 0; i < rows_; ++i) {
        new_ptr[i + 1] += new_ptr[i];
    }
    const int total = new_ptr[rows_];
    std::vector<int>    final_col(total);
    std::vector<double> final_val(total);

    // 6. Second pass: emit merged entries into the compact buffers.
    #pragma omp parallel
    {
        std::vector<std::pair<int, double>> local_buf;
        #pragma omp for schedule(dynamic, 64)
        for (int r = 0; r < static_cast<int>(rows_); ++r) {
            int begin = row_ptr_[r];
            int end = row_ptr_[r + 1];
            local_buf.clear();
            local_buf.reserve(end - begin);
            for (int p = begin; p < end; ++p) {
                local_buf.emplace_back(col_idx_[p], values_[p]);
            }
            std::sort(local_buf.begin(), local_buf.end());

            int out = new_ptr[r];
            if (!local_buf.empty()) {
                final_col[out] = local_buf[0].first;
                final_val[out] = local_buf[0].second;
                int w = out + 1;
                for (size_t k = 1; k < local_buf.size(); ++k) {
                    if (local_buf[k].first == final_col[w - 1]) {
                        final_val[w - 1] += local_buf[k].second;
                    } else {
                        final_col[w] = local_buf[k].first;
                        final_val[w] = local_buf[k].second;
                        ++w;
                    }
                }
            }
        }
    }

    col_idx_ = std::move(final_col);
    values_ = std::move(final_val);
    row_ptr_ = std::move(new_ptr);

    // Free COO buffers.
    coo_row_.clear();   coo_row_.shrink_to_fit();
    coo_col_.clear();   coo_col_.shrink_to_fit();
    coo_val_.clear();   coo_val_.shrink_to_fit();

    built_ = true;
}

void SparseMatrix::matvec(const double* x, double* y) const {
    for (size_t i = 0; i < rows_; ++i) {
        double s = 0.0;
        const int end = row_ptr_[i + 1];
        for (int p = row_ptr_[i]; p < end; ++p) {
            s += values_[p] * x[col_idx_[p]];
        }
        y[i] = s;
    }
}

void SparseMatrix::matvec(const std::vector<double>& x,
                          std::vector<double>& y) const {
    if (x.size() != cols_) {
        throw std::invalid_argument("SparseMatrix::matvec dimension mismatch");
    }
    if (y.size() != rows_) y.resize(rows_);
    matvec(x.data(), y.data());
}

void SparseMatrix::matvecParallel(const double* x, double* y) const {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(rows_); ++i) {
        double s = 0.0;
        const int end = row_ptr_[i + 1];
        for (int p = row_ptr_[i]; p < end; ++p) {
            s += values_[p] * x[col_idx_[p]];
        }
        y[i] = s;
    }
#else
    matvec(x, y);
#endif
}

void SparseMatrix::matvecThreadPool(const double* x, double* y,
                                    ThreadPool& pool, size_t chunk) const {
    parallel_for(pool, rows_, chunk, [&](size_t i) {
        double s = 0.0;
        const int end = row_ptr_[i + 1];
        for (int p = row_ptr_[i]; p < end; ++p) {
            s += values_[p] * x[col_idx_[p]];
        }
        y[i] = s;
    });
}

double SparseMatrix::at(size_t i, size_t j) const {
    if (i >= rows_) return 0.0;
    const int end = row_ptr_[i + 1];
    for (int p = row_ptr_[i]; p < end; ++p) {
        if (col_idx_[p] == static_cast<int>(j)) return values_[p];
        if (col_idx_[p] > static_cast<int>(j)) break; // row is sorted
    }
    return 0.0;
}

void SparseMatrix::clear() {
    coo_row_.clear();
    coo_col_.clear();
    coo_val_.clear();
    row_ptr_.assign(rows_ + 1, 0);
    col_idx_.clear();
    values_.clear();
    built_ = false;
}

} // namespace powerflow
