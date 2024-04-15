import numpy as np
import matplotlib.pyplot as plt

# Define the function to compute the derivatives

def f( y, D_omeg, d_omeg, K, theta, p, T):
    E1, E2, N1, N2 = y
    
    # Define your system of equations here
    dE1_dt = (1 + 1j * D_omeg) * N1 * E1 - 0.5 * E1 - 1j * d_omeg * E1 + K * np.exp(1j * theta) * E2
    dE2_dt = (1 + 1j * D_omeg) * N2 * E2 - 0.5 * E2 + 1j * d_omeg * E2 + K * np.exp(1j * theta) * E1
    dN1_dt = (1 / T) * (p - (N1 * E1) - 2 * N1 * np.abs(E1))
    dN2_dt = (1 / T) * (p - (N2 * E2) - 2 * N2 * np.abs(E2))
    
    return np.array([dN1_dt, dN2_dt, dE1_dt, dE2_dt])


# Runge-Kutta 4th order method
def RK4(f, y0, t, d_omeg, D_omeg, K, theta, p, T):
    n = len(t)
    y = np.zeros((n, len(y0)), dtype=complex)
    y[0] = y0
    h = t[1] - t[0]
    for i in range(n - 1):
        k1 = h * f(y[i], d_omeg, D_omeg, K, theta, p, T)
        k2 = h * f(y[i] + 0.5 * k1, d_omeg, D_omeg, K, theta, p, T)
        k3 = h * f(y[i] + 0.5 * k2, d_omeg, D_omeg, K, theta, p, T)
        k4 = h * f(y[i] + k3, d_omeg, D_omeg, K, theta, p, T)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y
# Example usage
# Define constants
d_omeg = 0.05
D_omeg = 3
K = 0.1
theta = 3*np.pi/2
p = 2
T = 392

# Define initial conditions
y0 = np.array([0.2, 0.2, 0.3, 0.1])

# Define time array
t = np.linspace(0, 800, 10000)

# Solve the system of differential equations using RK4
solution = RK4(f, y0, t, d_omeg, D_omeg,K,theta, p, T)

# Extracting N1 and E2 solutions
N1_solution = solution[:, 2]
E1_solution = solution[:, 0]
print(len(E1_solution))
print(t)
print(len(t))
# Plotting N1 and E2 absolute values with time
theta = round(theta,3)
# plt.plot(t, np.abs(E1_solution), label='|E1|')
# plt.xlabel('Time')
# plt.ylabel('|E1|')
# plt.title(f'Absolute value of E1 over time, ($\Delta\omega$= {d_omeg}, $\delta\omega$= {D_omeg},$p$= {p},\n $K$= {K}, $\\theta=\\frac{{3}}{{2}}\\pi$)' )
# plt.legend()
# plt.show()
# print(solution)
