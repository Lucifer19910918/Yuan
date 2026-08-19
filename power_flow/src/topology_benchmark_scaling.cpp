// Benchmark: topology tracing speed
//   Scales: 10, 100, 1000, 10000 nodes (radial tree + small mesh links).
//   Solvers:
//     (1) Standard BFS queue (adjacency lists, node-level visited array)
//         -> the traditional DFS/BFS "push_back on list" family.
//     (2) Matrix method (CSR adjacency + repeated SpMV OR-expansion)
//         -> what we built in topology_matrix_trace.cpp.
//   For each size we also measure:
//     (a) ONE single-solve (build graph + trace from source)
//     (b) REPEATED solves (keep graph, only retrace) -- this is the
//         switch-scanning regime: same topology, retrace 1000 times.
//   Correctness is cross-checked: the two solvers must agree on the
//   energized set and shortest-path layer for every node.

#include "sparse_matrix.h"
#include "timer.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using powerflow::SparseMatrix;

// ---------------------------------------------------------------------------
// Radial distribution network generator.
//   Produces a tree with average degree ~2 and a few small meshes (~1% of
//   edges are loop-closing tie-switches) to mimic a realistic weakly-meshed
//   distribution network.
//   Source is always node 0.
// ---------------------------------------------------------------------------
struct Graph {
    int n;
    int nedges;                         // undirected edge count (for info)
    SparseMatrix A;                     // CSR adjacency (binary, symmetric)
    std::vector<std::vector<int>> adj;  // adjacency list for BFS

    Graph(int n_, SparseMatrix&& A_, int nedges_,
          std::vector<std::vector<int>>&& adj_)
        : n(n_), nedges(nedges_), A(std::move(A_)), adj(std::move(adj_)) {}
};

static Graph buildRadialGraph(int n, int seed) {
    std::mt19937 rng(seed);
    // Attach each node i>=1 to a random parent in [0, i-1].  This builds a
    // random tree (Prüfer-like) with average depth ~log(n), then sprinkle
    // ~1% mesh edges to break strict tree sparsity.
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> ui;

    std::vector<std::pair<int,int>> edges;
    edges.reserve(n + n/100);
    for (int i = 1; i < n; ++i) {
        // Bias parents toward more recent ones for shorter diameter.
        int minP = std::max(0, i - std::min(i, 50));
        std::uniform_int_distribution<int> up(minP, i - 1);
        int p = up(rng);
        edges.emplace_back(p, i);
    }
    // Mesh edges: ~0.5% of n (realistic weakly-meshed feeder rate).
    int mesh = std::max(0, n / 200);
    for (int k = 0; k < mesh; ++k) {
        // Pick two random non-adjacent nodes and link them.
        for (int attempt = 0; attempt < 20; ++attempt) {
            int a = ui(rng) % n;
            int b = ui(rng) % n;
            if (a == b) continue;
            if (a > b) std::swap(a, b);
            // Accept (approximate; we just avoid self loop & ensure a<b).
            edges.emplace_back(a, b);
            break;
        }
    }
    int nedges = (int)edges.size();

    // --- Build CSR adjacency matrix via our SparseMatrix (binary, 0/1) ---
    SparseMatrix A(n);
    for (const auto& e : edges) {
        A.add(e.first, e.second, 1.0);
        A.add(e.second, e.first, 1.0);
    }
    A.buildCSR();

    // --- Build adjacency list for BFS (same edge set) ---
    std::vector<std::vector<int>> adj(n);
    for (const auto& e : edges) {
        adj[e.first ].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    // Sort + unique to avoid multi-edge artifacts (the CSR build handles
    // duplicates internally via summation but the BFS list is cleanest deduped).
    for (auto& v : adj) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return Graph(n, std::move(A), nedges, std::move(adj));
}

// ---------------------------------------------------------------------------
// Solver 1: Standard BFS with a simple int queue (vector-as-deque push_back +
// head pointer). The textbook implementation; no recursion.
// ---------------------------------------------------------------------------
static int traceBFS(const Graph& g, std::vector<int>& layer,
                    std::vector<int>& energized) {
    const int n = g.n;
    layer.assign(n, -1);
    std::vector<int> q;
    q.reserve(n);
    // Multi-source support: in a realistic setting there could be >1 slack.
    // We use node 0 as source (consistent with matrix solver below).
    layer[0] = 0;
    q.push_back(0);
    int diameter = 0;
    for (size_t head = 0; head < q.size(); ++head) {
        int u = q[head];
        for (int v : g.adj[u]) {
            if (layer[v] < 0) {
                layer[v] = layer[u] + 1;
                diameter = std::max(diameter, layer[v]);
                q.push_back(v);
            }
        }
    }
    energized.clear();
    energized.reserve(q.size());
    for (int i = 0; i < n; ++i) if (layer[i] >= 0) energized.push_back(i);
    return diameter;
}

// ---------------------------------------------------------------------------
// Solver 2: Matrix method (repeated SpMV frontier expansion + OR update +
// layer assignment).  Exact algorithm from topology_matrix_trace.cpp.
// ---------------------------------------------------------------------------
static int traceMatrix(const Graph& g, std::vector<int>& layer,
                       std::vector<int>& energized, int& iterations) {
    const int n = g.n;
    const SparseMatrix& A = g.A;
    layer.assign(n, -1);
    std::vector<double> v(n, 0.0);
    std::vector<double> frontier(n, 0.0);
    layer[0] = 0;
    v[0] = 1.0;
    frontier[0] = 1.0;
    std::vector<double> next(n);
    iterations = 0;
    int diameter = 0;
    while (true) {
        A.matvec(frontier.data(), next.data());
        bool changed = false;
        const int k = iterations;
        for (int i = 0; i < n; ++i) {
            if (next[i] > 0.5 && layer[i] < 0) {
                layer[i] = k + 1;
                v[i] = 1.0;
                frontier[i] = 1.0;
                changed = true;
                diameter = std::max(diameter, k + 1);
            } else {
                frontier[i] = 0.0;
            }
        }
        if (!changed) break;
        ++iterations;
        if (iterations > n) break;
    }
    energized.clear();
    energized.reserve(n);
    for (int i = 0; i < n; ++i) if (v[i] > 0.5) energized.push_back(i);
    return diameter;
}

// ---------------------------------------------------------------------------
// Report helpers.
// ---------------------------------------------------------------------------
static std::string fmtUs(double us) {
    char buf[64];
    if (us < 1.0)       std::snprintf(buf, sizeof(buf), "%.1f ns", us * 1000.0);
    else if (us < 1000) std::snprintf(buf, sizeof(buf), "%.1f us", us);
    else                std::snprintf(buf, sizeof(buf), "%.2f ms", us / 1000.0);
    return buf;
}
static std::string fmt(double v, const std::string& unit) {
    char buf[64];
    if (std::fabs(v) >= 1000)      std::snprintf(buf, sizeof(buf), "%.0f", v);
    else if (std::fabs(v) >= 100)  std::snprintf(buf, sizeof(buf), "%.1f", v);
    else if (std::fabs(v) >= 10)   std::snprintf(buf, sizeof(buf), "%.2f", v);
    else                           std::snprintf(buf, sizeof(buf), "%.3f", v);
    return std::string(buf) + unit;
}

// Warmup rounds + timing loop: returns average microseconds per call.
template <typename F>
static double benchUs(F&& fn, int minRepeats, double minSec) {
    // warmup (1 or a small fixed amount to get caches hot)
    int warm = std::min(3, std::max(1, minRepeats / 100));
    for (int i = 0; i < warm; ++i) fn();
    int reps = std::max(minRepeats, 1);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double totalUs = 1.0e-3 *
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    // If total time was too short (noise), expand reps and retry once.
    double totalSec = totalUs / 1.0e6;
    if (totalSec < minSec && reps < 100000000) {
        int target = (int)std::ceil(reps * (minSec / std::max(totalSec, 1e-9)));
        reps = std::max(reps, std::min(target, 100000000));
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        t1 = std::chrono::high_resolution_clock::now();
        totalUs = 1.0e-3 *
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    }
    return totalUs / reps;
}

struct CaseResult {
    int n;
    int nedges;
    int diameter;
    int iter;           // matrix method iteration count (=#SpMV rounds)
    double bfs_us;
    double mat_us;
    bool   match;       // result correctness
};

static CaseResult runOneCase(int n, int seed) {
    CaseResult r; r.n = n;

    Graph g = buildRadialGraph(n, seed);
    r.nedges = g.nedges;

    std::vector<int> layerB, layerM, energB, energM;

    // ---- BFS timing (repeated solve: keep graph, redo visited) ----
    double bfs_us = benchUs([&]() { (void)traceBFS(g, layerB, energB); },
                            200, 0.30);
    r.bfs_us = bfs_us;
    int diamB = traceBFS(g, layerB, energB);

    // ---- Matrix timing (repeated solve) ----
    int iter = 0;
    double mat_us = benchUs([&]() { int it; (void)traceMatrix(g, layerM, energM, it); },
                            200, 0.30);
    r.mat_us = mat_us;
    int diamM = traceMatrix(g, layerM, energM, iter);

    // ---- Correctness check ----
    r.match = (energB == energM) && (layerB == layerM) && (diamB == diamM);
    r.diameter = diamB;
    r.iter = iter;
    return r;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, NULL, _IOLBF, 0);
    (void)argc; (void)argv;

    const int cases[] = { 10, 100, 1000, 10000,
                          20000, 50000, 100000 };
    const int numCases = sizeof(cases)/sizeof(cases[0]);
    const int seed = 20260819;

    std::printf("=================================================================\n");
    std::printf("  Topology trace benchmark: BFS (queue) vs Matrix (SpMV waves)\n");
    std::printf("  Graph family: random tree (radial feeder) + ~0.5%% mesh edges\n");
    std::printf("  Source = node 0; metric = wall-clock time per single trace\n");
    std::printf("  (both solvers return shortest-path layer per node)\n");
    std::printf("=================================================================\n\n");

    // Header row.
    std::printf("%-9s %-7s %-8s %-8s %-5s %14s %14s %10s %s\n",
                "nodes(n)", "edges", "deg_avg", "diam", "SpMV",
                "BFS_queue", "Matrix_SpMV", "BFS/Mat", "CHECK");
    std::printf("%-9s %-7s %-8s %-8s %-5s %14s %14s %10s %s\n",
                "--------", "-----", "-------", "----", "----",
                "--------------", "--------------", "---------", "-----");

    std::vector<CaseResult> results(numCases);
    for (int i = 0; i < numCases; ++i) {
        CaseResult r = runOneCase(cases[i], seed + i);
        results[i] = r;
        double degAvg = (2.0 * r.nedges) / std::max(1, r.n);
        double ratio = (r.mat_us > 0) ? (r.bfs_us / r.mat_us) : 0;
        std::printf("%-9d %-7d %-8.2f %-8d %-5d %14s %14s %10.2fx %s\n",
                    r.n, r.nedges, degAvg, r.diameter, r.iter,
                    fmtUs(r.bfs_us).c_str(), fmtUs(r.mat_us).c_str(),
                    ratio, r.match ? "OK" : "MISMATCH!");
    }

    // ---- Repeat mode (N-switch scanning regime): same graph, trace REPEAT
    // times in one block, measure THROUGHPUT.
    std::printf("\n=========================================================\n");
    std::printf("  REPEATED solve regime (like 1000 switch scans):\n");
    std::printf("  keep graph unchanged, retrace 1000x -> total time\n");
    std::printf("=========================================================\n");
    std::printf("%-9s %10s %12s %12s %8s\n",
                "n", "repeats", "BFS_total", "Mat_total", "BFS/Mat");
    for (int i = 0; i < numCases; ++i) {
        const int n = cases[i];
        // Pick repeat count based on graph size to keep total time sane.
        int reps = (n <= 100) ? 100000 : (n <= 1000) ? 10000 : (n <= 10000) ? 1000 : (n<=20000)?500:(n<=50000)?200:100;
        Graph g = buildRadialGraph(n, seed + i);
        std::vector<int> layer, energ;

        int iter = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < reps; ++k) traceBFS(g, layer, energ);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < reps; ++k) traceMatrix(g, layer, energ, iter);
        auto t2 = std::chrono::high_resolution_clock::now();
        double bfs_ms = 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        double mat_ms = 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        double ratio  = (mat_ms > 0) ? (bfs_ms / mat_ms) : 0.0;
        std::printf("%-9d %10d %10s ms %10s ms %7.2fx\n",
                    n, reps, fmt(bfs_ms,"").c_str(), fmt(mat_ms,"").c_str(), ratio);
    }

    // ---- Summary guidance. ----
    std::printf("\n==================== 结论 (Summary) ====================\n");
    for (const auto& r : results) {
        double ratio = r.bfs_us / std::max(1e-9, r.mat_us);
        const char* verdict =
            (ratio >= 1.0) ? "Matrix WINS  ✓" : "BFS WINS     ✓";
        std::printf("  n=%-7d  BFS=%10s  Mat=%10s  ratio=%6.2fx  %s\n",
                    r.n, fmtUs(r.bfs_us).c_str(), fmtUs(r.mat_us).c_str(),
                    ratio, verdict);
    }
    // Find breakeven: first n where Matrix catches up.
    for (int i = 0; i + 1 < numCases; ++i) {
        double r1 = results[i  ].bfs_us / std::max(1e-9, results[i  ].mat_us);
        double r2 = results[i+1].bfs_us / std::max(1e-9, results[i+1].mat_us);
        if (r1 < 1.0 && r2 >= 1.0) {
            std::printf("\n  交叉点（矩阵法追平/反超队列BFS）大约在 n = %d ~ %d 之间\n",
                        cases[i], cases[i+1]);
            break;
        }
    }
    std::printf("  备注：本benchmark = 单CPU核心, 纯标量循环。矩阵SpMV方法的优势\n");
    std::printf("  在多线程/GPU/批量开关扫描场景会进一步放大（SpMV天然SIMD化）。\n");

    // Mismatch summary.
    int mismatches = 0;
    for (const auto& r : results) if (!r.match) ++mismatches;
    std::printf("\n  正确性：%d/%zu cases PASS %s\n",
                (int)results.size() - mismatches, results.size(),
                mismatches == 0 ? "(完全一致, 无差异)" : "!! 有错 !!");
    return mismatches == 0 ? 0 : 1;
}
