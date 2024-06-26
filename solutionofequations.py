import numpy as np
from scipy.linalg import lu_factor, lu_solve

# Coefficients matrix
A = np.array([
    [2, 2, -1, 1],
    [4, 3, -1, 2],
    [8, 5, -3, 4],
    [3, 3, -2, 2]
])

# Constants vector
B = np.array([4, 6, 12, 6])

# Solve the system of equations
solution = np.linalg.solve(A, B)
print(solution)

# Gauss Elimination method
def gauss_elimination(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = len(B)
    
    for i in range(n):
        # Make the diagonal contain all 1's
        factor = A[i][i]
        A[i] = A[i] / factor
        B[i] = B[i] / factor
        
        # Make the other rows contain 0's
        for j in range(i+1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            B[j] = B[j] - factor * B[i]
    
    # Back substitution
    X = np.zeros_like(B)
    for i in range(n-1, -1, -1):
        X[i] = B[i] - np.dot(A[i, i+1:], X[i+1:])
    
    return X

def lu_decomposition(A, B):
    # Perform LU decomposition with pivoting
    lu, piv = lu_factor(A)
    # Solve the system
    X = lu_solve((lu, piv), B)
    return X
    
# Matrix Inversion method
def matrix_inversion(A, B):
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, B)
    return X

# Applying all methods
gauss_sol = gauss_elimination(A.copy(), B.copy())
lu_sol = lu_decomposition(A, B)
inv_sol = matrix_inversion(A, B)
print("Gauss elimination, LU Decomposition, Matrix Inversion", gauss_sol, lu_sol, inv_sol)