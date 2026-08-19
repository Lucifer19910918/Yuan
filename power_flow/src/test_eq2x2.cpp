// Standalone test: use the project's SparseLU (the same solver used by
// Newton-Raphson / FDLF power flow) to solve the 2x2 linear system
//     X + Y = 3
//    5X - Y = 3
// This proves the general-equation wrapper works for tiny textbook examples,
// not just the 2000-bus power-flow use case.
#include "lu_decomposition.h"
#include "sparse_matrix.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    using namespace powerflow;
    std::printf("=== C++ SparseLU 求解二元一次方程组 ===\n");
    std::printf("   X + Y = 3\n");
    std::printf("  5X - Y = 3\n\n");

    // Build 2x2 coefficient matrix A in CSR form (via COO triplet assembly,
    // exactly the same pattern used by the power-flow Jacobian / B' / B'').
    SparseMatrix A(2);
    A.add(0, 0,  1.0);   // a11
    A.add(0, 1,  1.0);   // a12
    A.add(1, 0,  5.0);   // a21
    A.add(1, 1, -1.0);   // a22
    A.buildCSR();
    std::printf("A (CSR): rows=%zu  cols=%zu  nnz=%zu\n",
                A.rows(), A.cols(), A.nnz());

    // RHS vector b = [3, 3]
    double b[2] = {3.0, 3.0};
    double x[2] = {0.0, 0.0};

    // Reuse the exact same factorization pipeline as the power flow:
    //   analyze()   -> symbolic + MD ordering + fill pattern
    //   factorize() -> numeric LU on the fixed pattern
    //   solve()     -> forward + back substitution with permutation
    SparseLU lu;
    lu.analyze(A);
    lu.factorize(A);
    if (lu.singular()) {
        std::printf("ERROR: matrix is singular\n");
        return 1;
    }
    std::printf("Factorization done.  nnz(L)=%zu  nnz(U)=%zu\n",
                lu.nnzL(), lu.nnzU());

    lu.solve(b, x);

    // Verify: compute residual r = A x - b via sparse matvec
    double r[2] = {0.0, 0.0};
    A.matvec(x, r);
    r[0] -= b[0];
    r[1] -= b[1];
    double maxr = std::max(std::fabs(r[0]), std::fabs(r[1]));

    std::printf("\n解:\n");
    std::printf("  X = %.15g\n", x[0]);
    std::printf("  Y = %.15g\n", x[1]);
    std::printf("\n验证:\n");
    std::printf("  X + Y  = %.15g  (应=3, 差 %.2e)\n", x[0]+x[1], std::fabs(x[0]+x[1]-3.0));
    std::printf("  5X - Y = %.15g  (应=3, 差 %.2e)\n",
                5.0*x[0]-x[1], std::fabs(5.0*x[0]-x[1]-3.0));
    std::printf("  SparseLU 残差 ||Ax-b||∞ = %.2e\n", maxr);

    const bool ok = (std::fabs(x[0] - 1.0) < 1e-12 &&
                     std::fabs(x[1] - 2.0) < 1e-12 &&
                     maxr < 1e-12);
    std::printf("\n判定: %s  (X=1, Y=2)\n", ok ? "完全正确 ✓" : "错误 ✗");
    return ok ? 0 : 2;
}
