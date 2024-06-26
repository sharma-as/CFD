import numpy as np

def gauss_elimin(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    m, n = a.shape
    if m != n:
        raise ValueError('Matrix a must be square')

    # Augment the matrix a with the vector b
    Aug = np.hstack((a, b.reshape(-1, 1)))
    
    nb = n + 1

    # Forward elimination
    for i in range(n-1):
        for j in range(i+1, n):
            factor = Aug[j, i] / Aug[i, i]
            Aug[j, i:nb] = Aug[j, i:nb] - factor * Aug[i, i:nb]
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = Aug[n-1, nb-1] / Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (Aug[i, nb-1] - np.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]
    
    return x

a = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
b = np.array([1, 0, 1], dtype=float)
x = gauss_elimin(a, b)
print(x)