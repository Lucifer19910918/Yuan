#include "lu_decomposition.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace powerflow {

namespace {

// Minimum-degree ordering over the symmetric pattern of A (A + A^T), using a
// bucket-queue priority structure (O(1) amortized min-degree selection) and
// vector-based adjacency with a marker array (avoids hash-set overhead).
// Returns perm[new] = old and invperm[old] = new.
void minimumDegreeOrdering(const SparseMatrix& A,
                           std::vector<int>& perm,
                           std::vector<int>& invperm) {
    const int n = static_cast<int>(A.rows());
    if (n == 0) { perm.clear(); invperm.clear(); return; }

    // Adjacency as vectors. Membership during fill-in is checked via a marker
    // array tagged with the current outer index, so no per-edge hashing.
    std::vector<std::vector<int>> adj(n);
    const auto& Ap = A.rowPtr();
    const auto& Aj = A.colIdx();
    for (int i = 0; i < n; ++i) {
        for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
            int j = Aj[p];
            if (j != i) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    // De-duplicate neighbors (symmetric input would otherwise add each edge
    // twice). Sort + unique keeps adjacency compact and bucket state consistent.
    int maxDeg = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        maxDeg = std::max(maxDeg, static_cast<int>(adj[i].size()));
    }

    std::vector<int> degree(n);
    for (int i = 0; i < n; ++i) degree[i] = static_cast<int>(adj[i].size());

    // Bucket queue: buckets[d] holds live nodes whose current degree is d.
    std::vector<std::vector<int>> buckets(maxDeg + 1);
    std::vector<int> bucketPos(n, -1);
    auto place = [&](int v, int d) {
        if (d >= static_cast<int>(buckets.size())) buckets.resize(d + 1);
        bucketPos[v] = static_cast<int>(buckets[d].size());
        buckets[d].push_back(v);
    };
    auto remove = [&](int v) {
        int d = degree[v];
        int pos = bucketPos[v];
        auto& bucket = buckets[d];
        int last = bucket.back();
        bucket[pos] = last;
        bucketPos[last] = pos;
        bucket.pop_back();
        bucketPos[v] = -1;
    };
    auto moveDeg = [&](int v, int newD) {
        remove(v);
        degree[v] = newD;
        place(v, newD);
    };
    for (int i = 0; i < n; ++i) place(i, degree[i]);

    // mark[w] == token  means w is currently adjacent to the node whose fill-in
    // scan is in progress. token is a monotonically increasing global counter,
    // so stale marks from previous scans never match -- no reset needed.
    std::vector<int> fillMark(n, -1);
    int token = 0;

    perm.assign(n, -1);
    invperm.assign(n, -1);
    std::vector<char> eliminated(n, 0);
    int minDeg = 0;

    for (int step = 0; step < n; ++step) {
        while (minDeg < static_cast<int>(buckets.size()) && buckets[minDeg].empty()) ++minDeg;
        if (minDeg >= static_cast<int>(buckets.size())) break;
        int best = buckets[minDeg].back();
        remove(best);
        eliminated[best] = 1;
        perm[step] = best;
        invperm[best] = step;

        // Take a snapshot of best's neighbors (adj[best] will be cleared below).
        std::vector<int> nbrs = adj[best];
        // 1. Detach best from each neighbor's adjacency (swap-remove), drop degree.
        for (int a : nbrs) {
            auto& la = adj[a];
            for (size_t i = 0; i < la.size(); ++i) {
                if (la[i] == best) {
                    la[i] = la.back();
                    la.pop_back();
                    break;
                }
            }
            int nd = degree[a] - 1;
            moveDeg(a, nd);
            if (nd < minDeg) minDeg = nd;
        }
        // 2. Form a clique among the neighbors (fill edges), using fillMark to
        //    avoid duplicates and existing edges.
        for (size_t a = 0; a < nbrs.size(); ++a) {
            int u = nbrs[a];
            // Tag u's current neighbors so we can test membership in O(1).
            ++token;
            for (int w : adj[u]) fillMark[w] = token;
            for (size_t b = a + 1; b < nbrs.size(); ++b) {
                int v = nbrs[b];
                if (fillMark[v] != token) {
                    // New fill edge u-v.
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                    moveDeg(u, degree[u] + 1);
                    moveDeg(v, degree[v] + 1);
                    fillMark[v] = token; // now adjacent to u
                }
            }
        }
        adj[best].clear();
    }
}

} // namespace

void SparseLU::analyze(const SparseMatrix& A) {
    n_ = static_cast<int>(A.rows());
    if (static_cast<int>(A.cols()) != n_) {
        throw std::invalid_argument("SparseLU::analyze requires a square matrix");
    }

    // 1. Minimum-degree ordering over the symmetric pattern.
    minimumDegreeOrdering(A, perm_, invperm_);

    // 2. Build permuted structure P A P^T with a value-source map.
    //    trips: (new_row, new_col, source_position_in_A)
    const auto& Ap = A.rowPtr();
    const auto& Aj = A.colIdx();
    std::vector<std::tuple<int, int, int>> trips;
    trips.reserve(A.nnz());
    for (int i = 0; i < n_; ++i) {
        for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
            int j = Aj[p];
            trips.emplace_back(invperm_[i], invperm_[j], p);
        }
    }
    std::sort(trips.begin(), trips.end(),
              [](const std::tuple<int,int,int>& a,
                 const std::tuple<int,int,int>& b) {
                  if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
              });

    Ap_ptr_.assign(n_ + 1, 0);
    for (const auto& t : trips) Ap_ptr_[std::get<0>(t) + 1]++;
    for (int r = 0; r < n_; ++r) Ap_ptr_[r + 1] += Ap_ptr_[r];

    const int nnz = static_cast<int>(trips.size());
    Ap_col_.resize(nnz);
    a_src_.resize(nnz);
    Ap_val_.assign(nnz, 0.0);
    for (int k = 0; k < nnz; ++k) {
        Ap_col_[k] = std::get<1>(trips[k]);
        a_src_[k]  = std::get<2>(trips[k]);
    }

    // 3. Symbolic factorization (reachability) over the permuted graph.
    std::vector<std::vector<int>> L_pat(n_), U_pat(n_);
    marker_.assign(n_, -1);
    std::vector<int> pattern;
    pattern.reserve(64);

    for (int k = 0; k < n_; ++k) {
        pattern.clear();
        for (int p = Ap_ptr_[k]; p < Ap_ptr_[k + 1]; ++p) {
            int j = Ap_col_[p];
            if (marker_[j] != k) { marker_[j] = k; pattern.push_back(j); }
        }
        size_t idx = 0;
        while (idx < pattern.size()) {
            int j = pattern[idx++];
            if (j < k) {
                const auto& uj = U_pat[j];
                for (int c : uj) {
                    if (c > j && marker_[c] != k) {
                        marker_[c] = k;
                        pattern.push_back(c);
                    }
                }
            }
        }
        std::sort(pattern.begin(), pattern.end());
        pattern.erase(std::unique(pattern.begin(), pattern.end()), pattern.end());
        for (int j : pattern) {
            if (j < k) L_pat[k].push_back(j);
            else       U_pat[k].push_back(j);
        }
        if (U_pat[k].empty() || U_pat[k].front() != k) {
            U_pat[k].insert(U_pat[k].begin(), k);
        }
    }

    // Compact patterns to CSR.
    L_ptr_.assign(n_ + 1, 0);
    U_ptr_.assign(n_ + 1, 0);
    for (int k = 0; k < n_; ++k) {
        L_ptr_[k + 1] = L_ptr_[k] + static_cast<int>(L_pat[k].size());
        U_ptr_[k + 1] = U_ptr_[k] + static_cast<int>(U_pat[k].size());
    }
    L_col_.resize(L_ptr_[n_]);
    U_col_.resize(U_ptr_[n_]);
    L_val_.assign(L_col_.size(), 0.0);
    U_val_.assign(U_col_.size(), 0.0);
    U_diag_.assign(n_, 0.0);
    for (int k = 0; k < n_; ++k) {
        for (size_t t = 0; t < L_pat[k].size(); ++t)
            L_col_[L_ptr_[k] + t] = L_pat[k][t];
        for (size_t t = 0; t < U_pat[k].size(); ++t)
            U_col_[U_ptr_[k] + t] = U_pat[k][t];
    }

    w_.assign(n_, 0.0);
    analyzed_   = true;
    factorized_ = false;
    singular_   = false;
}

void SparseLU::factorize(const SparseMatrix& A) {
    if (!analyzed_) {
        throw std::runtime_error("SparseLU::factorize called before analyze");
    }
    if (static_cast<int>(A.rows()) != n_) {
        throw std::invalid_argument("SparseLU::factorize structure mismatch");
    }
    const auto& Av = A.values();

    // Refresh permuted matrix values from A via the source map.
    const int nap = static_cast<int>(Ap_val_.size());
    for (int k = 0; k < nap; ++k) {
        Ap_val_[k] = Av[a_src_[k]];
    }

    singular_ = false;
    std::fill(L_val_.begin(), L_val_.end(), 0.0);
    std::fill(U_val_.begin(), U_val_.end(), 0.0);

    const double kPivotFloor = 1e-18;

    for (int k = 0; k < n_; ++k) {
        // Load permuted row k into the dense workspace.
        for (int p = Ap_ptr_[k]; p < Ap_ptr_[k + 1]; ++p) {
            w_[Ap_col_[p]] = Ap_val_[p];
        }

        const int lb = L_ptr_[k];
        const int le = L_ptr_[k + 1];
        for (int lp = lb; lp < le; ++lp) {
            const int j = L_col_[lp];
            const double l_kj = w_[j] / U_diag_[j];
            L_val_[lp] = l_kj;
            const int ub = U_ptr_[j] + 1; // skip diagonal (first entry of U row j)
            const int ue = U_ptr_[j + 1];
            for (int up = ub; up < ue; ++up) {
                w_[U_col_[up]] -= l_kj * U_val_[up];
            }
        }

        const int u0 = U_ptr_[k];
        double diag = w_[k];
        if (std::fabs(diag) < kPivotFloor) {
            singular_ = true;
            diag = (diag < 0.0 ? -kPivotFloor : kPivotFloor);
        }
        U_diag_[k] = diag;
        U_val_[u0] = diag;
        for (int up = u0 + 1; up < U_ptr_[k + 1]; ++up) {
            U_val_[up] = w_[U_col_[up]];
        }

        // Reset touched workspace.
        for (int lp = lb; lp < le; ++lp) w_[L_col_[lp]] = 0.0;
        for (int up = u0; up < U_ptr_[k + 1]; ++up) w_[U_col_[up]] = 0.0;
    }

    factorized_ = true;
}

void SparseLU::factorizeAll(const SparseMatrix& A) {
    analyze(A);
    factorize(A);
}

void SparseLU::solve(const double* b, double* x) const {
    if (!factorized_) {
        throw std::runtime_error("SparseLU::solve called before factorize");
    }
    // y = P b  (forward into permuted space).
    std::vector<double> y(n_);
    for (int i = 0; i < n_; ++i) y[i] = b[perm_[i]];

    // Forward substitution: L z = y (L has unit diagonal).
    std::vector<double> z(n_);
    for (int k = 0; k < n_; ++k) {
        double s = y[k];
        const int* col = L_col_.data();
        const double* val = L_val_.data();
        for (int lp = L_ptr_[k]; lp < L_ptr_[k + 1]; ++lp) {
            s -= val[lp] * z[col[lp]];
        }
        z[k] = s;
    }
    // Back substitution: U xp = z.
    std::vector<double> xp(n_);
    for (int k = n_ - 1; k >= 0; --k) {
        double s = z[k];
        const int* col = U_col_.data();
        const double* val = U_val_.data();
        for (int up = U_ptr_[k] + 1; up < U_ptr_[k + 1]; ++up) {
            s -= val[up] * xp[col[up]];
        }
        xp[k] = s / U_diag_[k];
    }
    // x = P^T xp.
    for (int i = 0; i < n_; ++i) x[perm_[i]] = xp[i];
}

void SparseLU::solve(const std::vector<double>& b, std::vector<double>& x) const {
    if (static_cast<int>(x.size()) != n_) x.resize(n_);
    solve(b.data(), x.data());
}

void SparseLU::solve(const double* b, double* x, WorkSpace& ws) const {
    if (!factorized_) {
        throw std::runtime_error("SparseLU::solve called before factorize");
    }
    if (static_cast<int>(ws.y.size()) != n_) ws.resize(n_);
    double* y  = ws.y.data();
    double* z  = ws.z.data();
    double* xp = ws.xp.data();

    // y = P b  (gather b via perm, sequential write to y).
    const int* perm = perm_.data();
    for (int i = 0; i < n_; ++i) y[i] = b[perm[i]];

    // Forward substitution: L z = y (L unit diagonal).
    const int* Lcol = L_col_.data();
    const double* Lval = L_val_.data();
    const int* Lptr = L_ptr_.data();
    for (int k = 0; k < n_; ++k) {
        double s = y[k];
        for (int lp = Lptr[k]; lp < Lptr[k + 1]; ++lp) {
            s -= Lval[lp] * z[Lcol[lp]];
        }
        z[k] = s;
    }
    // Back substitution: U xp = z.
    const int* Ucol = U_col_.data();
    const double* Uval = U_val_.data();
    const int* Uptr = U_ptr_.data();
    const double* Udiag = U_diag_.data();
    for (int k = n_ - 1; k >= 0; --k) {
        double s = z[k];
        for (int up = Uptr[k] + 1; up < Uptr[k + 1]; ++up) {
            s -= Uval[up] * xp[Ucol[up]];
        }
        xp[k] = s / Udiag[k];
    }
    // x = P^T xp  (gather xp via invperm, SEQUENTIAL write to x -- cache
    // friendlier than scattering x[perm[i]] = xp[i] which writes x out of
    // order and thrashes cache lines).
    const int* invperm = invperm_.data();
    for (int i = 0; i < n_; ++i) x[i] = xp[invperm[i]];
}

} // namespace powerflow
