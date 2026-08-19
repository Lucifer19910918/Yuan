// Mini debugger for operating-modes logic: brute-force enumerates every
// mode of the N=3/K=2 case with a 2-source BFS reference validator.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

struct Net {
    int N, K, n, M, srcA, srcB;
    struct Sw { int a,b,kind; };
    std::vector<Sw> switches;
};
static Net build(int N, int K, int seed=7777) {
    Net g;
    g.N=N; g.K=K; g.n=2*(N+1); g.srcA=0; g.srcB=N+1; g.M=2*N+K;
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

int main() {
    Net g = build(3,2,7777);
    printf("Case: N=3 K=2 M=%d n=%d srcA=%d srcB=%d\n",g.M,g.n,g.srcA,g.srcB);
    for (int s=0;s<g.M;++s) {
        printf("  sw %d: kind=%d (%s)  %d <-> %d\n",
               s, g.switches[s].kind,
               g.switches[s].kind? "tie" : "sec",
               g.switches[s].a, g.switches[s].b);
    }
    long long total=0;
    for (long long m=0; m < (1LL<<g.M); ++m) {
        if (refValid(g,m)) { ++total; printf("  VALID mask=%08lldb\n",m); }
    }
    printf("TOTAL valid = %lld\n", total);
    return 0;
}
