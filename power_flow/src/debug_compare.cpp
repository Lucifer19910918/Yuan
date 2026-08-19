#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>
#include <functional>

struct Net {
    int N, K, n, M, srcA, srcB;
    struct Sw { int a,b,kind; };
    std::vector<Sw> switches;
};
static Net build(int N, int K, int seed=7777) {
    Net g; g.N=N; g.K=K; g.n=2*(N+1); g.srcA=0; g.srcB=N+1; g.M=2*N+K;
    for (int k=0;k<N;++k) g.switches.push_back({k,k+1,0});
    for (int k=0;k<N;++k) { int a=g.srcB+k; g.switches.push_back({a,a+1,0}); }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pa(std::max(0,N/2),N);
    std::uniform_int_distribution<int> pb(std::max(0,N/2),N);
    std::unordered_set<uint64_t> used;
    for (int t=0;t<K;++t) for (int att=0;att<50;++att) {
        int A=pa(rng), B=pb(rng);
        uint64_t kk=(uint64_t(A)<<32)|uint64_t(B);
        if (used.count(kk)) continue;
        used.insert(kk);
        g.switches.push_back({A, g.srcB+B, 1});
        break;
    }
    return g;
}
static bool refValid(const Net& g, long long mask) {
    std::vector<std::vector<int>> adj(g.n);
    for (int s=0;s<g.M;++s) if (mask&(1LL<<s)) {
        int a=g.switches[s].a, b=g.switches[s].b;
        adj[a].push_back(b); adj[b].push_back(a);
    }
    std::vector<int> col(g.n, 0);
    std::queue<int> q;
    col[g.srcA]=1; q.push(g.srcA);
    col[g.srcB]=2; q.push(g.srcB);
    int Eclosed = __builtin_popcountll(mask);
    while (!q.empty()) {
        int u=q.front(); q.pop();
        for (int v:adj[u]) {
            if (col[v]==0) { col[v]=col[u]; q.push(v); }
            else if (col[v] != col[u]) return false;
        }
    }
    for (int i=0;i<g.n;++i) if (col[i]==0) return false;
    if (Eclosed != 2*g.N) return false;
    return true;
}

// Copy of TraditionalSearch validate logic.
static bool tradValidate(const Net& g, long long mask) {
    std::vector<char> m(g.M, 0);
    for (int s=0;s<g.M;++s) if (mask&(1LL<<s)) m[s]=1;
    int closedCount = 0; for (int s=0;s<g.M;++s) if (m[s]) ++closedCount;
    if (closedCount != 2*g.N) return false;
    const int n = g.n, M = g.M;
    std::vector<int> color(n, 0);
    std::queue<int> q;
    color[g.srcA]=1; q.push(g.srcA);
    color[g.srcB]=2; q.push(g.srcB);
    static int cachedN = -1;
    static std::vector<std::vector<std::pair<int,int>>> nb_sw;
    if (cachedN != n) {
        cachedN = n;
        nb_sw.assign(n, {});
        for (int s=0;s<M;++s) {
            int a=g.switches[s].a, b=g.switches[s].b;
            nb_sw[a].emplace_back(b,s);
            nb_sw[b].emplace_back(a,s);
        }
    }
    while (!q.empty()) {
        int u=q.front(); q.pop();
        for (auto& p : nb_sw[u]) {
            int v=p.first, s=p.second;
            if (!m[s]) continue;
            if (color[v]==0) { color[v]=color[u]; q.push(v); }
            else if (color[v]!=color[u]) return false;
        }
    }
    for (int i=0;i<n;++i) if (color[i]==0) return false;
    return true;
}

int main() {
    Net g = build(3,2,7777);
    int r=0,t=0,bd=0,both=0;
    for (long long m=0; m<(1LL<<g.M); ++m) {
        bool R=refValid(g,m), T=tradValidate(g,m);
        if (R) ++r;
        if (T) ++t;
        if (R && !T) { printf("ref valid, trad INVALID: mask=%04lldb (%lld) edges=%d\n", m, m, __builtin_popcountll(m)); ++bd; }
        if (!R && T) { printf("ref INVALID, trad valid: mask=%04lldb (%lld) edges=%d\n", m, m, __builtin_popcountll(m)); ++both; }
    }
    printf("ref=%d trad=%d  [ref+trad-]=%d  [ref-trad+]=%d\n", r, t, bd, both);
    return 0;
}
