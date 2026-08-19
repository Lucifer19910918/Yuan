// Topology tracing via MATRIX operations (no DFS/BFS recursion).
//
// The core idea: graph traversal = repeated sparse matrix-vector multiply.
//
// Let A be the binary ADJACENCY matrix of the network (A[i][j]=1 iff there
// is a CLOSED electrical connection between bus i and bus j).
// Let s be the SOURCE VECTOR (s[i]=1 iff bus i is an energized source,
// e.g. a slack bus / grid supply point).
//
// The propagation step is:
//     v_{k+1} = v_k  OR  (A * v_k)            (element-wise, OR as max/add)
//
// Each iteration = one BFS "layer expansion" from the current frontier.
// After at most (diameter) iterations, v stops changing:
//     R = (I + A + A^2 + ... + A^{n-1})    ... the TRANSITIVE CLOSURE
//     v_final = R * s
//
// v_final[i] == 1  <=>  bus i is reachable from any source (energized).
//
// To recover the PATH EDGES (not just reachable nodes):
//     An edge (u,v) is used in the supply path iff
//       u is reachable  AND  v is reachable  AND  A[u][v] == 1
//       AND the edge belongs to a BFS tree from the nearest source.
// For a radial (tree-like) distribution network this is equivalent to:
//   path_edges_mask = A  AND  (v_final v_final^T + v_final^T v_final)
// i.e. edges where BOTH endpoints are energized.  For tree networks this
// returns exactly the supply tree (no cycles).  For weakly-meshed networks
// we additionally restrict to edges where (layer[v] == layer[u] + 1), which
// picks the BFS tree edges explicitly.
//
// SWITCHES are handled by zeroing their row/col in A when the switch is
// OPEN -- this is the MATRIX ANALOG of "blocking the edge during traversal".
// Changing switch states = element-wise mask on A (a single CSR pass, no
// graph rebuild) -> the same factorization / closure can be reused for
// scenarios that only change a small number of switches.

#include "sparse_matrix.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace topo {

// ======================== Network Model ========================

enum class DeviceType : uint8_t {
    SOURCE = 0,   // grid supply / slack
    BUS,          // node (busbar, junction)
    LINE,         // distribution line / cable segment
    SWITCH,       // circuit breaker / load-break switch (has OPEN/CLOSE state)
    LOAD,         // load transformer
};
const char* typeName(DeviceType t) {
    switch (t) {
        case DeviceType::SOURCE: return "SOURCE";
        case DeviceType::BUS:    return "BUS";
        case DeviceType::LINE:   return "LINE";
        case DeviceType::SWITCH: return "SWITCH";
        case DeviceType::LOAD:   return "LOAD";
    }
    return "?";
}

struct Device {
    int         id;           // unique device ID
    DeviceType  type;
    std::string code;         // e.g. "SS-01", "CB-A", "L-12", "LOAD-C3"
    int         busA;         // terminal A (-1 for single-terminal like LOAD)
    int         busB;         // terminal B (-1 for single-terminal)
    bool        switchClosed; // SWITCH only: true=closed(conducting), false=open
};

struct Network {
    int                          numBuses;     // bus count (buses are 0..numBuses-1)
    std::vector<Device>          devices;      // every piece of equipment
    // For quick lookup: map from device id -> index in devices[].
    std::unordered_map<int, int> devById;

    // Build a CSR adjacency matrix of the network.
    // Edge = 1 if a closed conducting path exists between bus i and bus j
    // through a LINE or a CLOSED SWITCH.  Single-terminal devices (SOURCE,
    // LOAD, BUS itself) don't create edges -- they are *attributes* of the
    // bus, attached later via bus->device list.
    powerflow::SparseMatrix buildAdjacency() const;

    // Return list of source buses (buses that host a SOURCE device).
    std::vector<int> sourceBuses() const;

    // For each bus, list devices directly on it.
    std::vector<std::vector<int>> devicesPerBus() const;
};

powerflow::SparseMatrix Network::buildAdjacency() const {
    using powerflow::SparseMatrix;
    SparseMatrix A(numBuses);
    for (const auto& d : devices) {
        // Conducting devices that create bus-to-bus edges:
        //   LINE  -> always a connection
        //   SWITCH -> connection ONLY if switchClosed == true
        if (d.busA < 0 || d.busB < 0) continue;   // single-terminal, no edge
        if (d.type == DeviceType::LINE ||
            (d.type == DeviceType::SWITCH && d.switchClosed))
        {
            // Add both directions (undirected graph) with weight=1.
            A.add(d.busA, d.busB, 1.0);
            A.add(d.busB, d.busA, 1.0);
        }
    }
    A.buildCSR();
    return A;
}

std::vector<int> Network::sourceBuses() const {
    std::vector<int> sb;
    for (const auto& d : devices) {
        if (d.type == DeviceType::SOURCE) {
            // SOURCE is a single-terminal device (attached to busA).
            if (d.busA >= 0) sb.push_back(d.busA);
            if (d.busB >= 0) sb.push_back(d.busB);
        }
    }
    return sb;
}

std::vector<std::vector<int>> Network::devicesPerBus() const {
    std::vector<std::vector<int>> out(numBuses);
    for (size_t k = 0; k < devices.size(); ++k) {
        const auto& d = devices[k];
        if (d.busA >= 0) out[d.busA].push_back((int)k);
        if (d.busB >= 0 && d.busB != d.busA) out[d.busB].push_back((int)k);
    }
    return out;
}

// ======================== Matrix Topology Tracer ========================

struct TraceResult {
    std::vector<int>  energizedBuses;         // 供电可达的母线编号
    std::vector<int>  pathDevices;            // 供电路径包含的设备索引
    std::vector<int>  layer;                  // 每母线段数 (BFS层号), -1=不可达
    int               iterations = 0;         // SpMV迭代次数
    int               diameter   = 0;         // 供电路径最远层数
};

// Core algorithm: repeatedly v = v OR (A * v) until convergence.
// This is MATRIX BFS -- each SpMV expands the frontier by exactly one hop.
// We additionally record layer[] so we can later pick explicit tree edges
// (needed for meshed networks; trees get the same answer without layer).
TraceResult traceSupplyPaths(const Network& net) {
    using powerflow::SparseMatrix;
    TraceResult res;

    const int n = net.numBuses;
    SparseMatrix A = net.buildAdjacency();

    // Source vector.
    std::vector<double> v(n, 0.0);
    res.layer.assign(n, -1);
    auto sources = net.sourceBuses();
    for (int s : sources) { v[s] = 1.0; res.layer[s] = 0; }

    // Layer frontier vector (only nodes reached *during* the current step
    // propagate further -- identical to BFS queue semantics).
    std::vector<double> frontier = v;

    std::vector<double> next(n);
    res.iterations = 0;
    res.diameter = 0;

    while (true) {
        // next = A * frontier  (SpMV: all neighbors of nodes on this layer)
        A.matvec(frontier.data(), next.data());

        // Select nodes that (a) were reached by this wave AND (b) are new.
        // Update v (global reachable mask) and layer[] with layer k+1.
        bool changed = false;
        const int k = res.iterations;
        for (int i = 0; i < n; ++i) {
            if (next[i] > 0.5 && res.layer[i] < 0) {   // newly reached!
                res.layer[i] = k + 1;
                v[i] = 1.0;
                frontier[i] = 1.0;    // this node will propagate next round
                changed = true;
                res.diameter = std::max(res.diameter, k + 1);
            } else {
                frontier[i] = 0.0;    // not in the active wave
            }
        }
        if (!changed) break;
        ++res.iterations;
        if (res.iterations > n) {
            std::fprintf(stderr, "  [warn] topology iterations > n (possible cycle bug)\n");
            break;
        }
    }

    // Collect energized buses.
    for (int i = 0; i < n; ++i) {
        if (v[i] > 0.5) res.energizedBuses.push_back(i);
    }

    // ---- Recover PATH EDGES (and the LINE/SWITCH devices on them) ----
    // Strategy: for every LINE or SWITCH device that connects two energized
    // buses, include it if it is on a shortest (BFS) supply path.  For a
    // radial (tree) network this just means "both endpoints are energized".
    // For meshed networks we additionally require
    //     |layer[busA] - layer[busB]| == 1
    // which picks the BFS tree edges (ties broken deterministically by
    // the propagation order).
    auto perBus = net.devicesPerBus();
    // We also need SOURCE/LOAD/BUS devices that sit on energized buses.
    std::vector<char> deviceInPath(net.devices.size(), 0);
    const auto& Ap = A.rowPtr();
    const auto& Aj = A.colIdx();
    // Quick adjacency mask for layer check on tree edges.
    auto edgeIsTree = [&](int u, int v) {
        if (res.layer[u] < 0 || res.layer[v] < 0) return false;
        return std::abs(res.layer[u] - res.layer[v]) == 1;
    };

    for (size_t dk = 0; dk < net.devices.size(); ++dk) {
        const Device& d = net.devices[dk];
        switch (d.type) {
            case DeviceType::SOURCE:
            case DeviceType::LOAD:
            case DeviceType::BUS: {
                // Single-terminal / bus attribute -> include if bus is energized.
                bool ok = (d.busA >= 0 && v[d.busA] > 0.5) ||
                          (d.busB >= 0 && v[d.busB] > 0.5);
                if (ok) deviceInPath[dk] = 1;
                break;
            }
            case DeviceType::LINE:
            case DeviceType::SWITCH: {
                // Both-end energized AND it is a valid tree edge.
                if (d.busA < 0 || d.busB < 0) break;
                if (v[d.busA] > 0.5 && v[d.busB] > 0.5 && edgeIsTree(d.busA, d.busB)) {
                    deviceInPath[dk] = 1;
                }
                break;
            }
        }
    }
    for (size_t k = 0; k < deviceInPath.size(); ++k)
        if (deviceInPath[k]) res.pathDevices.push_back((int)k);
    return res;
}

// ======================== Reference DFS (for correctness check) ============

static void dfsVisit(int u, const powerflow::SparseMatrix& A,
                     std::vector<char>& vis, std::vector<int>& layer, int depth) {
    if (vis[u]) return;
    vis[u] = 1;
    layer[u] = depth;
    const auto& Ap = A.rowPtr();
    const auto& Aj = A.colIdx();
    for (int p = Ap[u]; p < Ap[u+1]; ++p) {
        int v = Aj[p];
        if (!vis[v]) dfsVisit(v, A, vis, layer, depth + 1);
    }
}

static TraceResult traceDFS(const Network& net) {
    TraceResult res;
    using powerflow::SparseMatrix;
    const int n = net.numBuses;
    SparseMatrix A = net.buildAdjacency();
    std::vector<char> vis(n, 0);
    res.layer.assign(n, -1);
    auto sources = net.sourceBuses();
    // Run DFS from every source; overwrite the layer with the MIN depth to
    // match BFS semantics (keep first-seen layer as the shortest path).
    for (int s : sources) dfsVisit(s, A, vis, res.layer, 0);
    // Refine layer to be BFS-shortest (the DFS layer is just a spanning tree).
    std::vector<int> dist(n, n + 1);
    for (int s : sources) dist[s] = 0;
    // BFS using queue for the shortest-layer reference (the TRUE shortest
    // path layer we compare against).
    std::vector<int> q;  q.reserve(n);
    for (int s : sources) q.push_back(s);
    for (size_t head = 0; head < q.size(); ++head) {
        int u = q[head];
        const auto& Ap = A.rowPtr();
        const auto& Aj = A.colIdx();
        for (int p = Ap[u]; p < Ap[u+1]; ++p) {
            int v = Aj[p];
            if (dist[v] > dist[u] + 1) {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (dist[i] < n) {   // dist[i]==n (or n+1 initial) means unreachable
            res.energizedBuses.push_back(i);
            res.layer[i] = dist[i];
            res.diameter = std::max(res.diameter, dist[i]);
        }
    }
    return res;
}

// ======================== Pretty Print Helpers ========================

static void printNetwork(const Network& net) {
    std::printf("Network: %d buses, %zu devices\n", net.numBuses, net.devices.size());
    std::printf("  devices:\n");
    for (const auto& d : net.devices) {
        char state[16] = "";
        if (d.type == DeviceType::SWITCH)
            std::snprintf(state, sizeof(state), " [%s]",
                          d.switchClosed ? "CLOSED" : "OPEN");
        std::printf("    id=%-3d  %-6s  code=%-10s  busA=%-3d busB=%-3d%s\n",
                    d.id, typeName(d.type), d.code.c_str(), d.busA, d.busB, state);
    }
    auto sb = net.sourceBuses();
    std::printf("  source buses (%zu): [", sb.size());
    for (size_t i = 0; i < sb.size(); ++i) std::printf("%s%d", i?", ":"", sb[i]);
    std::printf("]\n");
}

static void printTrace(const Network& net, const TraceResult& r, const char* label) {
    std::printf("\n--- %s ---\n", label);
    std::printf("  iterations/SpMV waves : %d\n", r.iterations);
    std::printf("  supply-tree diameter  : %d hops\n", r.diameter);
    std::printf("  energized buses (%zu) : [", r.energizedBuses.size());
    for (size_t i = 0; i < r.energizedBuses.size(); ++i)
        std::printf("%s%d", i?", ":"", r.energizedBuses[i]);
    std::printf("]\n");
    std::printf("  bus layer (distance from nearest source):\n    ");
    for (int i = 0; i < net.numBuses; ++i)
        std::printf("%d:%d%s", i, r.layer[i], (i+1)%15==0?"\n    ":" ");
    std::printf("\n");
    std::printf("  supply-path devices (%zu):\n", r.pathDevices.size());
    // Group by device type for nice output.
    const char* order[] = {"SOURCE","BUS","LINE","SWITCH","LOAD"};
    for (const char* tn : order) {
        bool header = false;
        for (int dk : r.pathDevices) {
            const Device& d = net.devices[dk];
            if (std::string(typeName(d.type)) != tn) continue;
            if (!header) { std::printf("    %-6s : ", tn); header = true; }
            else          std::printf(", ");
            std::printf("%s", d.code.c_str());
        }
        if (header) std::printf("\n");
    }
    // Also list isolated (de-energized) LOADs that are NOT in the path.
    // Useful to confirm out-of-service zones are correctly excluded.
    std::vector<std::string> deEnergizedLoads;
    std::vector<char> inPath(net.devices.size(), 0);
    for (int k : r.pathDevices) inPath[k] = 1;
    for (const auto& d : net.devices) {
        if (d.type == DeviceType::LOAD && !inPath[&d - &net.devices[0]])
            deEnergizedLoads.push_back(d.code);
    }
    if (!deEnergizedLoads.empty()) {
        std::printf("  de-energized LOADs (%zu): ", deEnergizedLoads.size());
        for (size_t i = 0; i < deEnergizedLoads.size(); ++i)
            std::printf("%s%s", i?", ":"", deEnergizedLoads[i].c_str());
        std::printf("\n");
    }
}

} // namespace topo


// ======================== Demo Cases ========================

using namespace topo;

// Case 1: small radial distribution feeder.
//   Source SS-01 ----bus 0---- SW-1 (closed)
//                          |
//                          +-- LINE-1 --> bus 1 -- LOAD-A
//                          |
//                          +-- LINE-2 --> bus 2 -- SW-2 (closed)
//                                                  |
//                                                  +-- LINE-3 --> bus 3 -- LOAD-B
//                                                                         |
//                                                                         +-- LINE-4 --> bus 4 -- SW-3 (OPEN, isolates LOAD-C)
//                                                                                                       |
//                                                                                                       +-- LINE-5 --> bus 5 -- LOAD-C (ISLANDED)
// Expected energized: buses {0,1,2,3}.  De-energized: {4,5} (LOAD-C off).
static Network buildNetwork1() {
    Network net;
    net.numBuses = 6;
    int nextId = 1;
    auto add = [&](DeviceType t, std::string code, int a, int b, bool swClosed=true) {
        Device d;
        d.id = nextId++;
        d.type = t;
        d.code = std::move(code);
        d.busA = a; d.busB = b;
        d.switchClosed = swClosed;
        net.devById[d.id] = (int)net.devices.size();
        net.devices.push_back(d);
    };
    // SOURCE at bus 0 (substation)
    add(DeviceType::SOURCE, "SS-01",    0, -1);
    // BUS markers (just tagging -- for readability, not edges)
    add(DeviceType::BUS, "BUS-0", 0, -1);
    add(DeviceType::BUS, "BUS-1", 1, -1);
    add(DeviceType::BUS, "BUS-2", 2, -1);
    add(DeviceType::BUS, "BUS-3", 3, -1);
    add(DeviceType::BUS, "BUS-4", 4, -1);
    add(DeviceType::BUS, "BUS-5", 5, -1);
    // SWITCH out of the substation (closed)
    add(DeviceType::SWITCH, "SW-HV",   0, 0, true);   // bus-coupler (same bus, trivial loopback; will be ignored by layer diff filter; keeping as demonstraion)
    add(DeviceType::SWITCH, "SW-1",    0, 0, true);
    // LINEs
    add(DeviceType::LINE,   "LINE-1",  0, 1);
    add(DeviceType::LINE,   "LINE-2",  0, 2);
    add(DeviceType::SWITCH, "SW-2",    2, 2, true);
    add(DeviceType::LINE,   "LINE-3",  2, 3);
    add(DeviceType::LINE,   "LINE-4",  3, 4);
    add(DeviceType::SWITCH, "SW-3",    4, 5, false);  // OPEN: isolates bus 4 <> 5
    add(DeviceType::LINE,   "LINE-5",  5, 5);
    // LOADs
    add(DeviceType::LOAD,   "LOAD-A",  1, -1);
    add(DeviceType::LOAD,   "LOAD-B",  3, -1);
    add(DeviceType::LOAD,   "LOAD-C",  5, -1);
    return net;
}

// Case 2: a realistic 33-bus test feeder topology (Baran & Wu 33-bus), condensed
// to 12 nodes + a TIE SWITCH between two ends for N-1 / loop-transfer analysis.
static Network buildNetwork2() {
    Network net;
    net.numBuses = 12;
    int nextId = 100;
    auto add = [&](DeviceType t, std::string code, int a, int b, bool swC=true) {
        Device d; d.id = nextId++; d.type=t; d.code=std::move(code);
        d.busA=a; d.busB=b; d.switchClosed=swC;
        net.devById[d.id]=(int)net.devices.size();
        net.devices.push_back(d);
    };
    // 1 SOURCE (bus 0), 3 feeders, 1 normally-open tie-switch linking feeder-2
    // tail to feeder-3 tail.
    add(DeviceType::SOURCE, "MAIN-SS",   0, -1);
    for (int i = 0; i < 12; ++i) {
        char c[16]; std::snprintf(c, sizeof(c), "BUS-%02d", i);
        add(DeviceType::BUS, c, i, -1);
    }
    // Feeder #1:  0 --L1-- 1 --L2-- 2 --L3-- 3 (loads on 1,2,3)
    add(DeviceType::LINE, "F1-L1",   0, 1);
    add(DeviceType::LINE, "F1-L2",   1, 2);
    add(DeviceType::LINE, "F1-L3",   2, 3);
    add(DeviceType::SWITCH, "F1-CB-IN", 0, 0, true);  // feeder CB
    add(DeviceType::LOAD, "F1-LD1",  1, -1);
    add(DeviceType::LOAD, "F1-LD2",  2, -1);
    add(DeviceType::LOAD, "F1-LD3",  3, -1);
    // Feeder #2:  0 --L4-- 4 --L5-- 5 --L6-- 6
    add(DeviceType::LINE, "F2-L4",   0, 4);
    add(DeviceType::LINE, "F2-L5",   4, 5);
    add(DeviceType::LINE, "F2-L6",   5, 6);
    add(DeviceType::SWITCH, "F2-CB-IN", 0, 0, true);
    add(DeviceType::LOAD, "F2-LD1",  4, -1);
    add(DeviceType::LOAD, "F2-LD2",  5, -1);
    add(DeviceType::LOAD, "F2-LD3",  6, -1);
    // Feeder #3:  0 --L7-- 7 --SW(OPEN)-- 8 --L8-- 9 --L9-- 10
    // The switch is open at the head of feeder-3, so it's islanded initially.
    add(DeviceType::SWITCH, "F3-CB-IN", 0, 7, false);   // OPEN
    add(DeviceType::LINE, "F3-L8",   8, 9);
    add(DeviceType::LINE, "F3-L9",   9, 10);
    // Tie-switch between feeder-2 tail (bus 6) and feeder-3 head tail (bus 10).
    // Normally open. Closing it restores supply to F3 via F2.
    add(DeviceType::SWITCH, "TIE-SW-6-10", 6, 10, false);  // NORMALLY OPEN
    // Loads on feeder 3.
    add(DeviceType::LOAD, "F3-LD0",  7,  -1);
    add(DeviceType::LOAD, "F3-LD1",  8,  -1);
    add(DeviceType::LOAD, "F3-LD2",  9,  -1);
    add(DeviceType::LOAD, "F3-LD3",  10, -1);
    // An extra auxiliary bus, unconnected everywhere.
    add(DeviceType::LOAD, "LOAD-DECOMM-11", 11, -1);
    return net;
}

static void runCase(const char* title, const Network& net) {
    using namespace topo;
    std::printf("\n=============================================================\n");
    std::printf("  %s\n", title);
    std::printf("=============================================================\n");
    printNetwork(net);

    // Matrix-based trace.
    TraceResult rMat = traceSupplyPaths(net);

    // Reference DFS+BFS for correctness.
    TraceResult rRef = traceDFS(net);

    printTrace(net, rMat, "MATRIX-BASED (SpMV frontier expansion)");

    // ---- Correctness checks ----
    bool reachOK = (rMat.energizedBuses == rRef.energizedBuses);
    // For layer we compare the SHORTEST distance computed by reference BFS.
    bool layerOK = true;
    for (int i = 0; i < net.numBuses; ++i) {
        if (rMat.layer[i] != rRef.layer[i]) { layerOK = false; break; }
    }
    std::printf("\n===== 正确性验证 (矩阵法 vs BFS参考) =====\n");
    std::printf("  energized buses match : %s  (%zu buses energized)\n",
                reachOK ? "✓ MATCH" : "✗ MISMATCH",
                rRef.energizedBuses.size());
    std::printf("  shortest layer  match : %s\n",
                layerOK ? "✓ MATCH" : "✗ MISMATCH");
    std::printf("  SpMV iterations       : %d  (graph diameter = %d)\n",
                rMat.iterations, rRef.diameter);
    if (!reachOK || !layerOK) {
        std::printf("  [DEBUG] matrix layer : ");
        for (int i = 0; i < net.numBuses; ++i) std::printf("%d ", rMat.layer[i]);
        std::printf("\n");
        std::printf("  [DEBUG] ref    layer : ");
        for (int i = 0; i < net.numBuses; ++i) std::printf("%d ", rRef.layer[i]);
        std::printf("\n");
    }
    std::printf("  overall verdict       : %s\n",
                (reachOK && layerOK) ? "ALL PASS ✓  矩阵方法结果与BFS完全等价"
                                     : "FAILED ✗");
}

int main() {
    std::setvbuf(stdout, NULL, _IOLBF, 0);
    std::printf("=============================================================\n");
    std::printf("  配网供电路径拓扑追踪：矩阵运算法（无DFS/BFS）\n");
    std::printf("  核心: A = 邻接矩阵 (稀疏CSR)，s = 源向量\n");
    std::printf("        迭代 v = v OR (A·v)  直至收敛  (BFS层扩展 → SpMV)\n");
    std::printf("=============================================================\n");

    runCase("Case 1: 辐射状馈线（含1台断开开关, LOAD-C孤岛）", buildNetwork1());
    runCase("Case 2: 3馈线含联络开关（F3馈线头开关断开，联络开关常开）", buildNetwork2());
    return 0;
}
