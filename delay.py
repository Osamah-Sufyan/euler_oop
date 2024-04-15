import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 20, 1000)
# Define the function to compute the derivatives with delay
def f(t, y, D_omeg, d_omeg, K, theta, p, T, tau):
    N1, N2, E1, E2 = y
    
   # Compute the delayed terms for E1 and E2
    delayed_E1 = np.interp(t - tau, t, E1, left=0, right=0)
    delayed_E2 = np.interp(t - tau, t, E2, left=0, right=0)
    
    # Compute the derivatives with delayed terms
    dE1_dt = (1 + 1j * D_omeg) * N1 * E1 - 0.5 * E1 - 1j * d_omeg * E1 + K * np.exp(1j * theta) * delayed_E2
    dE2_dt = (1 + 1j * D_omeg) * N2 * E2 - 0.5 * E2 + 1j * d_omeg * E2 + K * np.exp(1j * theta) * delayed_E1
    dN1_dt = (1 / T) * (p - (N1 * E1) - 2 * N1 * np.abs(E1))
    dN2_dt = (1 / T) * (p - (N2 * E2) - 2 * N2 * np.abs(E2))
    
    return np.array([dN1_dt, dN2_dt, dE1_dt, dE2_dt])

# Runge-Kutta 4th order method with delay
def RK4(f, y0, t, D_omeg, d_omeg, K, theta, p, T, tau):
    n = len(t)
    y = np.zeros((n, len(y0)), dtype=complex)
    y[0] = y0
    h = t[1] - t[0]
    for i in range(n - 1):
        k1 = h * f(t[i], y[i], D_omeg, d_omeg, K, theta, p, T, tau)
        k2 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k1, D_omeg, d_omeg, K, theta, p, T, tau)
        k3 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k2, D_omeg, d_omeg, K, theta, p, T, tau)
        k4 = h * f(t[i] + h, y[i] + k3, D_omeg, d_omeg, K, theta, p, T, tau)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

# Define the parameter ranges
d_omeg_range = np.linspace(-0.2, 0.2, 100)
theta_range = np.linspace(0, 2 * np.pi, 100)

# Initialize the color plot
maxima_counts = np.zeros((len(d_omeg_range), len(theta_range)))

# Define other constants
D_omeg = 3
K = 0.1
p = 2
T = 329
tau = 1  # Define the delay parameter tau

# Define initial conditions
y0 = np.array([2, 2, 3, 1])

# Define time array


# Scan parameters and count maxima
for i, d_omeg in enumerate(d_omeg_range):
    for j, theta in enumerate(theta_range):
        # Solve the system of differential equations using RK4 with delay
        solution = RK4(f, y0, t, D_omeg, d_omeg, K, theta, p, T, tau)
        # Extract E1 solution
        E1_solution = solution[:, 2]
        # Count the number of maxima
        maxima_counts[i, j] = np.sum((E1_solution[1:-1] > E1_solution[:-2]) & (E1_solution[1:-1] > E1_solution[2:]))

# Plot the color plot
plt.imshow(maxima_counts, extent=[0, 2 * np.pi, -0.2, 0.2], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Number of Maxima')
plt.xlabel('Theta')
plt.ylabel('d_omeg')
plt.title('Number of Maxima of |E1| with respect to Theta and d_omeg')
plt.show()
