#include "newton_raphson.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace powerflow {

NewtonRaphson::NewtonRaphson(PowerSystem& sys) : sys_(sys) {}

double NewtonRaphson::computeValue(const JacEntry& e,
                                   const std::vector<double>& V,
                                   const std::vector<double>& th,
                                   const std::vector<double>& Pcalc,
                                   const std::vector<double>& Qcalc) const {
    const double Vi = V[e.i];
    switch (e.kind) {
        // Diagonal forms: note theta_ii = theta_i - theta_i = 0 is constant, so
        // the j=i term does NOT contribute to theta derivatives. This yields the
        // extra -V_i^2*B_ii (H_ii) and -V_i^2*G_ii (M_ii) terms below, which are
        // essential -- without them the Jacobian is singular at flat start.
        case 0: return -Qcalc[e.i] - Vi * Vi * e.B;  // H_ii = -Q_i - V_i^2*B_ii
        case 1: return  Pcalc[e.i] / Vi + Vi * e.G;  // N_ii =  P_i/V_i + V_i*G_ii
        case 2: return  Pcalc[e.i] - Vi * Vi * e.G; // M_ii =  P_i - V_i^2*G_ii
        case 3: return  Qcalc[e.i] / Vi - Vi * e.B;  // L_ii =  Q_i/V_i - V_i*B_ii
        default: break;
    }
    const double Vj = V[e.j];
    const double dt = th[e.i] - th[e.j];
    const double c = std::cos(dt);
    const double s = std::sin(dt);
    const double G = e.G, B = e.B;
    switch (e.kind) {
        case 4: return  Vi * Vj * (G * s - B * c);   // H_ij
        case 5: return  Vi * (G * c + B * s);        // N_ij
        case 6: return -Vi * Vj * (G * c + B * s);  // M_ij
        case 7: return  Vi * (G * s - B * c);        // L_ij
        default: return 0.0;
    }
}

void NewtonRaphson::setup() {
    sys_.buildAdmittance(Yg_, Yb_);
    const int n = static_cast<int>(sys_.buses.size());

    // ---- Index maps ----------------------------------------------------
    p_eq_.assign(n, -1);
    q_eq_.assign(n, -1);
    theta_col_.assign(n, -1);
    v_col_.assign(n, -1);

    np_eq_ = 0;
    nq_eq_ = 0;
    for (int i = 0; i < n; ++i) {
        if (sys_.buses[i].type != BusType::SLACK) ++np_eq_;
        if (sys_.buses[i].type == BusType::PQ)     ++nq_eq_;
    }
    m_ = np_eq_ + nq_eq_;

    int p_idx = 0, q_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (sys_.buses[i].type != BusType::SLACK) {
            p_eq_[i] = p_idx;
            theta_col_[i] = p_idx;
            ++p_idx;
        }
        if (sys_.buses[i].type == BusType::PQ) {
            q_eq_[i] = q_idx;
            v_col_[i] = np_eq_ + q_idx;
            ++q_idx;
        }
    }

    // ---- Build static Jacobian pattern (JacEntry table) ----------------
    jac_.clear();
    jac_.reserve(static_cast<size_t>(n) * 8);

    const auto& Ap  = Yg_.rowPtr();
    const auto& Aj  = Yg_.colIdx();
    const auto& Ag  = Yg_.values();
    const auto& Ab  = Yb_.values();

    for (int i = 0; i < n; ++i) {
        // Diagonal Gii, Bii (used for diagonal Jacobian entries).
        double Gii = 0.0, Bii = 0.0;
        for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
            if (Aj[p] == i) { Gii = Ag[p]; Bii = Ab[p]; break; }
        }

        // P equation (every non-slack bus).
        if (p_eq_[i] >= 0) {
            const int row = p_eq_[i];
            jac_.push_back({row, theta_col_[i], i, i, Gii, Bii, 0}); // H_ii
            if (v_col_[i] >= 0) {
                jac_.push_back({row, v_col_[i], i, i, Gii, Bii, 1});  // N_ii
            }
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                int j = Aj[p];
                if (j == i) continue;
                const double Gij = Ag[p], Bij = Ab[p];
                if (theta_col_[j] >= 0)
                    jac_.push_back({row, theta_col_[j], i, j, Gij, Bij, 4}); // H_ij
                if (v_col_[j] >= 0)
                    jac_.push_back({row, v_col_[j], i, j, Gij, Bij, 5});     // N_ij
            }
        }

        // Q equation (PQ buses only).
        if (q_eq_[i] >= 0) {
            const int row = np_eq_ + q_eq_[i];
            jac_.push_back({row, theta_col_[i], i, i, Gii, Bii, 2}); // M_ii
            jac_.push_back({row, v_col_[i], i, i, Gii, Bii, 3});     // L_ii
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                int j = Aj[p];
                if (j == i) continue;
                const double Gij = Ag[p], Bij = Ab[p];
                if (theta_col_[j] >= 0)
                    jac_.push_back({row, theta_col_[j], i, j, Gij, Bij, 6}); // M_ij
                if (v_col_[j] >= 0)
                    jac_.push_back({row, v_col_[j], i, j, Gij, Bij, 7});     // L_ij
            }
        }
    }

    // Sort entries in (row, col) order so they coincide with CSR layout.
    std::sort(jac_.begin(), jac_.end(),
              [](const JacEntry& a, const JacEntry& b) {
                  if (a.row != b.row) return a.row < b.row;
                  return a.col < b.col;
              });

    // ---- Build J_ CSR directly from the sorted table -------------------
    J_.resize(m_, m_);
    auto& Jptr = J_.rowPtrMut();
    auto& Jcol = J_.colIdxMut();
    auto& Jval = J_.valuesMut();
    Jptr.assign(m_ + 1, 0);
    for (const auto& e : jac_) Jptr[e.row + 1]++;
    for (int r = 0; r < m_; ++r) Jptr[r + 1] += Jptr[r];
    const int nnz = static_cast<int>(jac_.size());
    Jcol.resize(nnz);
    Jval.assign(nnz, 0.0);
    for (int k = 0; k < nnz; ++k) {
        Jcol[k] = jac_[k].col;
    }
    // (jac_[k] is now aligned 1:1 with CSR slot k because both are row-major.)

    // ---- Symbolic LU factorization (once) ------------------------------
    lu_.analyze(J_);
    lu_analyzed_ = true;
    pattern_built_ = true;

    p_calc_.assign(n, 0.0);
    q_calc_.assign(n, 0.0);
    mismatch_.assign(m_, 0.0);
    dx_.assign(m_, 0.0);
}

NewtonRaphson::Result NewtonRaphson::solve(const Options& opt) {
    Result res;
    if (!pattern_built_) {
        throw std::runtime_error("NewtonRaphson::solve called before setup");
    }
    const int n = static_cast<int>(sys_.buses.size());
    const double baseMVA = sys_.baseMVA;

    std::vector<double> V(n), th(n);
    std::vector<double> p_spec(n, 0.0), q_spec(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const auto& b = sys_.buses[i];
        if (opt.flatStart) {
            V[i]  = (b.type == BusType::PQ) ? 1.0 : b.v_spec;
            th[i] = 0.0;
        } else {
            V[i]  = b.v;
            th[i] = b.theta;
        }
        p_spec[i] = (b.p_gen - b.p_load) / baseMVA;
        q_spec[i] = (b.q_gen - b.q_load) / baseMVA;
    }

    Timer total;
    bool broke = false;

    for (int iter = 0; iter < opt.maxIter; ++iter) {
        // 1. Compute P, Q at every bus (parallel).
        Timer pt;
        p_calc_.assign(n, 0.0);
        q_calc_.assign(n, 0.0);
        const auto& Ap = Yg_.rowPtr();
        const auto& Aj = Yg_.colIdx();
        const auto& Ag = Yg_.values();
        const auto& Ab = Yb_.values();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; ++i) {
            double Pi = 0.0, Qi = 0.0;
            const double Vi = V[i], ti = th[i];
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                const int j = Aj[p];
                const double Vj = V[j];
                const double dt = ti - th[j];
                const double c = std::cos(dt);
                const double s = std::sin(dt);
                const double G = Ag[p], B = Ab[p];
                Pi += Vj * (G * c + B * s);
                Qi += Vj * (G * s - B * c);
            }
            p_calc_[i] = Vi * Pi;
            q_calc_[i] = Vi * Qi;
        }
        res.powerTimeMs += pt.elapsed_ms();

        // 2. Build mismatch and check convergence.
        double maxm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (p_eq_[i] >= 0) {
                const double dP = p_spec[i] - p_calc_[i];
                mismatch_[p_eq_[i]] = dP;
                const double a = std::fabs(dP);
                if (a > maxm) maxm = a;
            }
            if (q_eq_[i] >= 0) {
                const double dQ = q_spec[i] - q_calc_[i];
                mismatch_[np_eq_ + q_eq_[i]] = dQ;
                const double a = std::fabs(dQ);
                if (a > maxm) maxm = a;
            }
        }
        res.maxMismatch = maxm;
        res.iterations = iter + 1;
        if (maxm < opt.tol) {
            res.converged = true;
            broke = true;
            break;
        }
        if (iter == opt.maxIter - 1) {
            broke = true; // ran out of iterations; do not solve again
            break;
        }

        // 3. Fill Jacobian values (parallel, in place).
        Timer jt;
        auto& Jval = J_.valuesMut();
        const int nnz = static_cast<int>(jac_.size());
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nnz; ++k) {
            Jval[k] = computeValue(jac_[k], V, th, p_calc_, q_calc_);
        }
        res.jacobiTimeMs += jt.elapsed_ms();

        // 4. Numeric LU factorization (pattern reused).
        Timer ft;
        lu_.factorize(J_);
        res.factorTimeMs += ft.elapsed_ms();
        if (lu_.singular()) {
            res.message = "singular Jacobian encountered";
            broke = true;
            break;
        }

        // 5. Solve J dx = mismatch.
        Timer st;
        lu_.solve(mismatch_, dx_);
        res.solveTimeMs += st.elapsed_ms();

        // 6. Update state.
        for (int i = 0; i < n; ++i) {
            if (theta_col_[i] >= 0) th[i] += dx_[theta_col_[i]];
            if (v_col_[i]     >= 0) V[i]  += dx_[v_col_[i]];
        }
    }

    // Final power evaluation so p_calc_/q_calc_ reflect the final state.
    {
        const auto& Ap = Yg_.rowPtr();
        const auto& Aj = Yg_.colIdx();
        const auto& Ag = Yg_.values();
        const auto& Ab = Yb_.values();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; ++i) {
            double Pi = 0.0, Qi = 0.0;
            const double Vi = V[i], ti = th[i];
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                const int j = Aj[p];
                const double Vj = V[j];
                const double dt = ti - th[j];
                const double c = std::cos(dt);
                const double s = std::sin(dt);
                const double G = Ag[p], B = Ab[p];
                Pi += Vj * (G * c + B * s);
                Qi += Vj * (G * s - B * c);
            }
            p_calc_[i] = Vi * Pi;
            q_calc_[i] = Vi * Qi;
        }
        double maxm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (p_eq_[i] >= 0) {
                const double a = std::fabs(p_spec[i] - p_calc_[i]);
                if (a > maxm) maxm = a;
            }
            if (q_eq_[i] >= 0) {
                const double a = std::fabs(q_spec[i] - q_calc_[i]);
                if (a > maxm) maxm = a;
            }
        }
        res.maxMismatch = maxm;
    }

    // Write state back into the system.
    for (int i = 0; i < n; ++i) {
        sys_.buses[i].v     = V[i];
        sys_.buses[i].theta = th[i];
    }

    res.totalTimeMs = total.elapsed_ms();
    (void)broke;

    if (opt.verbose) {
        std::printf("Newton-Raphson: %s in %d iter, maxMismatch=%.3e\n",
                    res.converged ? "CONVERGED" : "NOT CONVERGED",
                    res.iterations, res.maxMismatch);
        std::printf("  total=%.2f ms | power=%.2f | jacobi=%.2f | factor=%.2f | tri-solve=%.2f\n",
                    res.totalTimeMs, res.powerTimeMs, res.jacobiTimeMs,
                    res.factorTimeMs, res.solveTimeMs);
    }
    return res;
}

} // namespace powerflow
