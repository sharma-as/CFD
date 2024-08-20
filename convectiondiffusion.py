import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0         # Length of domain
rho = 1.0       # Density
Gamma = 0.1     # Diffusion coefficient

# Define grid
n_cells = 5     # Number of cells
n_points = n_cells + 1  # Number of grid points
dx = L / n_cells  # Grid spacing
x = np.linspace(0, L, n_points)  # Grid points

# Boundary conditions
phi_0 = 1        # Boundary condition at x = 0
phi_L = 0        # Boundary condition at x = L

# Define cases
cases = [0.1, 2.5]  # velocities

# Analytical solution function
def analytical_solution(u, x, L, rho, Gamma, phi_0, phi_L):
    return phi_0 + (phi_L - phi_0) * (np.exp(rho * u * x / Gamma) - 1) / (np.exp(rho * u * L / Gamma) - 1)

# Function to solve the problem using upwind scheme
def solve_convection_diffusion_upwind(u, n_points, dx, rho, Gamma, phi_0, phi_L):
    A = np.zeros((n_points - 2, n_points - 2))
    B = np.zeros(n_points - 2)
    
    if u > 0:
        # For positive velocities (upwind from the left)
        for i in range(n_points - 2):
            A[i, i] = -2 * (Gamma / dx**2) - u / dx
            if i > 0:
                A[i, i-1] = Gamma / dx**2 + u / dx
            if i < (n_points - 3):
                A[i, i+1] = Gamma / dx**2
        B[0] = -phi_0 * (Gamma / dx**2 - u / dx)
        B[-1] = -phi_L * (Gamma / dx**2 + u / dx)
    else:
        # For negative velocities (upwind from the right)
        for i in range(n_points - 2):
            A[i, i] = -2 * (Gamma / dx**2) + u / dx
            if i > 0:
                A[i, i-1] = Gamma / dx**2
            if i < (n_points - 3):
                A[i, i+1] = Gamma / dx**2 - u / dx
        B[0] = -phi_0 * (Gamma / dx**2 - u / dx)
        B[-1] = -phi_L * (Gamma / dx**2 + u / dx)
    
    phi_interior = np.linalg.solve(A, B)
    phi = np.concatenate(([phi_0], phi_interior, [phi_L]))
    return phi

# Function to solve the problem using central differencing scheme
def solve_convection_diffusion_central(u, n_points, dx, rho, Gamma, phi_0, phi_L):
    A = np.zeros((n_points - 2, n_points - 2))
    B = np.zeros(n_points - 2)
    
    for i in range(n_points - 2):
        A[i, i] = -2 * (Gamma / dx**2) - u / (2 * dx)
        if i > 0:
            A[i, i-1] = Gamma / dx**2 + u / (2 * dx)
        if i < (n_points - 3):
            A[i, i+1] = Gamma / dx**2 - u / (2 * dx)
    
    B[0] = -phi_0 * (Gamma / dx**2 - u / (2 * dx))
    B[-1] = -phi_L * (Gamma / dx**2 + u / (2 * dx))
    
    phi_interior = np.linalg.solve(A, B)
    phi = np.concatenate(([phi_0], phi_interior, [phi_L]))
    return phi

# Loop through each case and calculate
for i, u in enumerate(cases):
    # Solve for 5 nodes using upwind scheme
    phi_5_nodes_upwind = solve_convection_diffusion_upwind(u, n_points, dx, rho, Gamma, phi_0, phi_L)

    # Analytical solution for 5 nodes
    phi_analytical_5_nodes = analytical_solution(u, x, L, rho, Gamma, phi_0, phi_L)

    # Solve for 5 nodes using central differencing scheme
    phi_5_nodes_central = solve_convection_diffusion_central(u, n_points, dx, rho, Gamma, phi_0, phi_L)

    # Display results
    print(f'Case {i + 1} (u = {u} m/s)')
    print('Numerical solution with 5 nodes (Upwind scheme):')
    print(phi_5_nodes_upwind)
    print('Numerical solution with 5 nodes (Central differencing scheme):')
    print(phi_5_nodes_central)
    print('Analytical solution with 5 nodes:')
    print(phi_analytical_5_nodes)

    # Plotting
    plt.figure()
    plt.plot(x, phi_5_nodes_upwind, 'ro-', label='Upwind Scheme')
    plt.plot(x, phi_5_nodes_central, 'g*-', label='Central Differencing')
    plt.plot(x, phi_analytical_5_nodes, 'b*-', label='Analytical')
    plt.title(f'Comparison for u = {u} m/s')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('φ')
    plt.grid(True)
    plt.show()

# Recalculate with 20 nodes for Case 2
n_cells_20 = 20
n_points_20 = n_cells_20 + 1
dx_20 = L / n_cells_20
x_20 = np.linspace(0, L, n_points_20)

# Solve for 20 nodes using upwind scheme
phi_20_nodes_upwind = solve_convection_diffusion_upwind(cases[1], n_points_20, dx_20, rho, Gamma, phi_0, phi_L)

# Analytical solution for 20 nodes
phi_analytical_20_nodes = analytical_solution(cases[1], x_20, L, rho, Gamma, phi_0, phi_L)

# Plotting for 20 nodes
plt.figure()
plt.plot(x_20, phi_20_nodes_upwind, 'ro-', label='Numerical (Upwind)')
plt.plot(x_20, phi_analytical_20_nodes, 'b*-', label='Analytical')
plt.title('Comparison for u = 2.5 m/s with 20 nodes')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('φ')
plt.grid(True)
plt.show()