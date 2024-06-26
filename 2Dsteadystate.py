import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10e-2  # 10 cm
H = 10e-2  # 10 cm

# Mesh
nx = 10
ny = 10

x = np.linspace(0, L, nx)  # Values 0 to L with nx points
y = np.linspace(0, H, ny)  # Values 0 to H with ny points
dx = L / (nx - 1)
dy = H / (ny-1)

# Initialization (initial guess values)
T = np.zeros((nx, ny))

# Boundary conditions
T[:, 0] = 100
T[0, :] = 80
T[:, -1] = 50
T[-1, :] = 20
T_old = T.copy()

beta = (dx / dy) ** 2
err = 100
tol = 1e-3
k = 0
err_p = []

while err > tol:
    k += 1
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T[i, j] = (1 / (2 + (1 + beta))) * (T[i+1, j] + T[i-1, j] + (beta * (T[i, j+1] + T[i, j-1])))

    err = np.abs(np.max(T - T_old))
    err_p.append(err)
    T_old = T.copy()

    plt.figure(1)
    plt.plot(err_p)
    plt.xlabel('No of Iterations', fontsize=14)
    plt.ylabel('Error/Residual [-]', fontsize=14)
    plt.title('Convergence graph', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    plt.pause(0.01)
    plt.clf()

    plt.figure(2)
    plt.contourf(x, y, T, levels=20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Length [m]', fontsize=14)
    plt.ylabel('Length [m]', fontsize=14)
    plt.title('2D Steady State Heat Conduction', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    plt.pause(0.01)
    plt.clf()

plt.show()