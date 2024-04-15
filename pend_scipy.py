import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

D_omeg = 0.15
K = 0.3
T = 392
p = 0.6

initial_conditions = [0.2, 0.2, 0.01, 0.1, 0.2, 0.5]

# Function defining the system of ODEs
def odesystem(t, y, d_omeg, theta):
    E1_real, E1_imag, E2_real, E2_imag, N1, N2 = y
    E1 = E1_real + 1j * E1_imag
    E2 = E2_real + 1j * E2_imag
    
    dE1_dt = (1 + 1j * D_omeg) * N1 * E1 - 0.5 * E1 - 1j * d_omeg * E1 + K * np.exp(1j * theta) * E2
    dE2_dt = (1 + 1j * D_omeg) * N2 * E2 - 0.5 * E2 + 1j * d_omeg * E2 + K * np.exp(1j * theta) * E1
    dN1_dt = (1 / T) * (p - (N1 * (E1_real**2 + E1_imag**2)) - 2 * N1 * (E1_real**2 + E1_imag**2))
    dN2_dt = (1 / T) * (p - (N2 * (E2_real**2 + E2_imag**2)) - 2 * N2 * (E2_real**2 + E2_imag**2))
    
    return [dE1_dt.real, dE1_dt.imag, dE2_dt.real, dE2_dt.imag, dN1_dt, dN2_dt]

# Values of theta for which we want to plot abs(E1)
theta_values = [np.pi/2, 3*np.pi/2]

# Initialize arrays to store the solutions
solutions = []

# Time span for solving the ODEs
t_span = (0, 100)

# Time span for plotting
t_plot = np.linspace(0, 100, 1000)

# Iterate over the values of theta
for theta in theta_values:
    # Solve the ODEs for the current value of theta
    solution = solve_ivp(odesystem, t_span, initial_conditions, args=(D_omeg, theta), dense_output=True)
    # Evaluate the solution at desired time points
    y_eval = solution.sol(t_plot)
    # Store the solution
    solutions.append((t_plot, np.abs(y_eval[0])))

# Plot the absolute value of E1 for each value of theta
for i, theta in enumerate(theta_values):
    plt.plot(solutions[i][0], solutions[i][1], label=r'$\theta = {}$'.format(theta))
plt.xlabel(r'Time ($t$)')
plt.ylabel(r'$|E1|$')
plt.title('Absolute value of E1 over time for different values of theta')
plt.legend()
plt.grid(True)
plt.savefig('absolute_value_E1_vs_time.pdf')
plt.show()


