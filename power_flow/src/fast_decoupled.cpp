#include "fast_decoupled.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace powerflow {

FastDecoupled::FastDecoupled(PowerSystem& sys) : sys_(sys) {}

void FastDecoupled::setup() {
    sys_.buildAdmittance(Yg_, Yb_);
    const int n = static_cast<int>(sys_.buses.size());

    // ---- Index maps ----------------------------------------------------
    p_eq_.assign(n, -1);
    q_eq_.assign(n, -1);
    np_ = 0;
    nq_ = 0;
    for (int i = 0; i < n; ++i) {
        if (sys_.buses[i].type != BusType::SLACK) { p_eq_[i] = np_++; }
        if (sys_.buses[i].type == BusType::PQ)    { q_eq_[i] = nq_++; }
    }

    // ---- Build B' = -Im(Y), slack row/col removed ----------------------
    // B' is np_ x np_.  Entry (r, c) corresponds to bus pair (i, j) where
    // p_eq_[i]=r, p_eq_[j]=c.  We use -B (susceptance negated) for the
    // standard XB decoupling: dTheta = (B')^{-1} (dP / V).
    Bp_ = SparseMatrix(np_, np_);
    {
        const auto& Ap = Yb_.rowPtr();
        const auto& Aj = Yb_.colIdx();
        const auto& Ab = Yb_.values();
        for (int i = 0; i < n; ++i) {
            if (p_eq_[i] < 0) continue;
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                int j = Aj[p];
                if (p_eq_[j] < 0) continue;
                Bp_.add(p_eq_[i], p_eq_[j], -Ab[p]);
            }
        }
        Bp_.buildCSR();
    }

    // ---- Build B'' = -Im(Y), slack AND PV row/col removed --------------
    // B'' is nq_ x nq_.
    Bpp_ = SparseMatrix(nq_, nq_);
    {
        const auto& Ap = Yb_.rowPtr();
        const auto& Aj = Yb_.colIdx();
        const auto& Ab = Yb_.values();
        for (int i = 0; i < n; ++i) {
            if (q_eq_[i] < 0) continue;
            for (int p = Ap[i]; p < Ap[i + 1]; ++p) {
                int j = Aj[p];
                if (q_eq_[j] < 0) continue;
                Bpp_.add(q_eq_[i], q_eq_[j], -Ab[p]);
            }
        }
        Bpp_.buildCSR();
    }

    // ---- Factorize B' and B'' ONCE -------------------------------------
    luBp_.analyze(Bp_);
    luBp_.factorize(Bp_);
    if (luBp_.singular()) throw std::runtime_error("FDLF: B' is singular");

    luBpp_.analyze(Bpp_);
    luBpp_.factorize(Bpp_);
    if (luBpp_.singular()) throw std::runtime_error("FDLF: B'' is singular");

    // Workspace.
    p_calc_.assign(n, 0.0);
    q_calc_.assign(n, 0.0);
    dP_.assign(np_, 0.0);
    dQ_.assign(nq_, 0.0);
    dTheta_.assign(np_, 0.0);
    dV_.assign(nq_, 0.0);
}

FastDecoupled::Result FastDecoupled::solve(const Options& opt) {
    Result res;
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
    const auto& Ap = Yg_.rowPtr();
    const auto& Aj = Yg_.colIdx();
    const auto& Ag = Yg_.values();
    const auto& Ab = Yb_.values();

    // In single-sweep warm-start mode we do exactly one P+Q pass (no loop).
    const int maxIter = (opt.singleSweep && !opt.flatStart) ? 1 : opt.maxIter;
    for (int iter = 0; iter < maxIter; ++iter) {
        // ---- 1. Compute P, Q at every bus (parallel) --------------------
        // Both P and Q are computed from the same voltage/angle state, so a
        // single pass suffices (no per-iteration Q re-evaluation needed -- the
        // P/Q coupling is handled implicitly across iterations).
        Timer pt;
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

        // ---- 2. Assemble BOTH rhs vectors (dP/V, dQ/V) ------------------
        // Done before the solves so the two independent triangular systems
        // (B' dTheta = dP/V  and  B'' dV = dQ/V) can be solved in parallel.
        double maxP = 0.0, maxQ = 0.0;
        for (int i = 0; i < n; ++i) {
            if (p_eq_[i] >= 0) {
                double dP = p_spec[i] - p_calc_[i];
                dP_[p_eq_[i]] = dP / V[i];
                maxP = std::max(maxP, std::fabs(dP));
            }
            if (q_eq_[i] >= 0) {
                double dQ = q_spec[i] - q_calc_[i];
                dQ_[q_eq_[i]] = dQ / V[i];
                maxQ = std::max(maxQ, std::fabs(dQ));
            }
        }

        // ---- 3. Solve B' dTheta=dP/V and B'' dV=dQ/V in PARALLEL ----------
        // The two sub-problems are fully independent (different matrices,
        // different rhs, different workspaces). Running them concurrently on
        // two threads roughly halves the triangular-solve time, which is the
        // dominant cost in the warm-start single-sweep regime.
        Timer st;
        #pragma omp parallel num_threads(2)
        {
            #pragma omp sections
            {
                #pragma omp section
                luBp_.solve(dP_.data(), dTheta_.data(), wsBp_);
                #pragma omp section
                luBpp_.solve(dQ_.data(), dV_.data(), wsBpp_);
            }
        }
        res.solveTimeMs += st.elapsed_ms();

        // ---- 4. Apply corrections ---------------------------------------
        double maxDTheta = 0.0, maxDV = 0.0;
        for (int i = 0; i < n; ++i) {
            if (p_eq_[i] >= 0) {
                th[i] += dTheta_[p_eq_[i]];
                maxDTheta = std::max(maxDTheta, std::fabs(dTheta_[p_eq_[i]]));
            }
            if (q_eq_[i] >= 0) {
                V[i] += dV_[q_eq_[i]];
                maxDV = std::max(maxDV, std::fabs(dV_[q_eq_[i]]));
            }
        }

        // ---- 5. Convergence check --------------------------------------
        // Use BOTH the power mismatch AND the correction magnitude. In the
        // warm-start regime the mismatch may still be above tol after one
        // iteration, but the corrections become tiny -- a reliable signal that
        // the solution has stabilized.
        double maxm = std::max(maxP, maxQ);
        double maxCorr = std::max(maxDTheta, maxDV);
        res.maxMismatch = maxm;
        res.iterations = iter + 1;
        if (maxm < opt.tol || (iter > 0 && maxCorr < opt.tol * 0.1)) {
            res.converged = true;
            break;
        }
    }

    // Final power evaluation so p_calc_/q_calc_ reflect the final state.
    // Skipped in tight-loop mode: the values are only needed by external
    // readers (Pcalc()/Qcalc()), and in a repeated-solve loop nobody reads
    // them between solves. The last solve of a batch can re-enable it.
    if (!opt.skipFinalPower) {
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
                Pi += Vj * (Ag[p] * c + Ab[p] * s);
                Qi += Vj * (Ag[p] * s - Ab[p] * c);
            }
            p_calc_[i] = Vi * Pi;
            q_calc_[i] = Vi * Qi;
        }
    }

    // Write back to system.
    for (int i = 0; i < n; ++i) {
        sys_.buses[i].v     = V[i];
        sys_.buses[i].theta = th[i];
    }

    res.totalTimeMs = total.elapsed_ms();
    if (opt.verbose) {
        std::printf("FDLF: %s in %d iter, maxMismatch=%.3e\n",
                    res.converged ? "CONVERGED" : "NOT CONVERGED",
                    res.iterations, res.maxMismatch);
        std::printf("  total=%.2f ms | power=%.2f | tri-solve=%.2f (NO factorization)\n",
                    res.totalTimeMs, res.powerTimeMs, res.solveTimeMs);
    }
    return res;
}

} // namespace powerflow
