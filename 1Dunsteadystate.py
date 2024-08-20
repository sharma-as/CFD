import numpy as np
import matplotlib.pyplot as plt

# Geometry
xmin = 0
xmax = 0.2  # length of the rod
N = 10  # number of nodes
dx = (xmax - xmin) / (N - 1)  # grid size
x = np.linspace(xmin, xmax, N)
dt = 1e-3  # time step
tmax = 0.5  # max time to simulate
t = np.arange(0, tmax + dt, dt)

alpha = 0.05  # diffusion coefficient/material property

# Initial and Boundary Conditions
Tcurrent = np.ones(N) * 30  # Initial Conditions
Tb = 300  # Boundary condition, T Base
Ttip = 50  # Boundary condition, T Tip

# Solution
d = alpha * dt / dx**2

for j in range(1, len(t)):  # loop for time step
    T = Tcurrent.copy()
    for i in range(N):  # Space stepping
        if i == 0:
            T[i] = Tb
        elif i == N - 1:
            T[i] = Ttip
        else:
            T[i] = T[i] + d * (T[i + 1] - 2 * T[i] + T[i - 1])
    Tcurrent = T.copy()
    time = j * dt

    plt.plot(x, Tcurrent)
    plt.xlabel('Length of the rod [m]', fontsize=14)
    plt.ylabel('Temperature [°C]', fontsize=14)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    str1 = f'value of d = {d:.5f}'
    str2 = f'Time value = {time:.3f} s'
    plt.text(0.12, 150, str1, fontsize=14)
    plt.text(0.12, 130, str2, fontsize=14)
    plt.pause(0.01)
    plt.clf()
plt.show()