import numpy as np
import matplotlib.pyplot as plt

# Geometry
L = 10 / 100
N = 10
dx = L / (N - 1)

# BC and IC
T = np.zeros(N)
Tb = 200
Ttip = 20

# Solution
k = 100  # Number of iterations
for j in range(k):
    T[0] = Tb
    for i in range(1, N-1):
        T[i] = (T[i+1] + T[i-1]) / 2
    T[N-1] = T[N-2]
    plt.plot(T)
    plt.xlabel('Node')
    plt.ylabel('Temperature')
    plt.title('Temperature Distribution')
    plt.grid(True)
    plt.pause(0.01)  # Pause to allow dynamic plotting

plt.show()