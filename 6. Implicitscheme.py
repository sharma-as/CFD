import numpy as np
import matplotlib.pyplot as plt
xmin = 0
xmax = 0.2  # length of the rod
N = 10  # number of nodes
dx = (xmax - xmin) / (N - 1)  # grid size
x = np.linspace(xmin, xmax, N)
dt = 1e-3  # time step
tmax = 0.5  # max time to simulate
t = np.arange(0, tmax + dt, dt)

alpha = 0.05  # diffusion coefficient/material property
Tcurrent = np.ones(N) * 20  # Initial Conditions
Tb = 200  # Boundary condition, T Base
Ttip = 20  # Boundary condition, T Tip

r = alpha * dt / (2 * dx**2)
A = np.zeros((N, N))
for i in range(1, N-1):
    A[i, i-1] = -r
    A[i, i] = 1 + 2*r
    A[i, i+1] = -r

A[0, 0] = 1
A[-1, -1] = 1

B = np.zeros((N, N))
for i in range(1, N-1):
    B[i, i-1] = r
    B[i, i] = 1 - 2*r
    B[i, i+1] = r

B[0, 0] = 1
B[-1, -1] = 1
for j in range(1, len(t)):  # loop for time step
    T = Tcurrent.copy()
    T[0] = Tb
    T[-1] = Ttip

    b = B @ T
    b[0] = Tb
    b[-1] = Ttip

    Tcurrent = np.linalg.solve(A, b)
    time = j * dt
    plt.plot(x, Tcurrent)
    plt.xlabel('Length of the rod [m]', fontsize=14)
    plt.ylabel('Temperature [°C]', fontsize=14)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    str1 = f'value of r = {r:.5f}'
    str2 = f'Time value = {time:.3f} s'
    plt.text(0.12, 150, str1, fontsize=14)
    plt.text(0.12, 130, str2, fontsize=14)
    plt.pause(0.01)
    plt.clf()

plt.show()