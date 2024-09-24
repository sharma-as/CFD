import numpy as np
import matplotlib.pyplot as plt

# Geometry
L = 0.5
N = 10
dx = L / (N - 1)

# Initial and Boundary Conditions
T = np.zeros(N)
Tb = 2000
Ttip = 100

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
    plt.pause(0.01)  # Pause to allow dynamic plotting
plt.show()