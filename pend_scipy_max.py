import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

D_omeg = 0.15
K = 0.3
T = 392
p = 0.6

initial_conditions = [0.5, 0.01, 0.01, 0.1, 0.1, 0.02]

def odesystem(t, y, d_omeg, theta):
    E1_real, E1_imag, E2_real, E2_imag, N1, N2 = y
    E1 = E1_real + 1j * E1_imag
    E2 = E2_real + 1j * E2_imag
    
    dE1_dt = (1 + 1j * D_omeg) * N1 * E1 - 0.5 * E1 - 1j * d_omeg * E1 + K * np.exp(1j * theta) * E2
    dE2_dt = (1 + 1j * D_omeg) * N2 * E2 - 0.5 * E2 + 1j * d_omeg * E2 + K * np.exp(1j * theta) * E1
    dN1_dt = (1 / T) * (p - (N1 * (E1_real**2 + E1_imag**2)) - 2 * N1 * (E1_real**2 + E1_imag**2))
    dN2_dt = (1 / T) * (p - (N2 * (E2_real**2 + E2_imag**2)) - 2 * N2 * (E2_real**2 + E2_imag**2))
    
    return [dE1_dt.real, dE1_dt.imag, dE2_dt.real, dE2_dt.imag, dN1_dt, dN2_dt]

def count_maxima(t, y):
    maxima_indices = (y[:-2] > y[1:-1]) & (y[1:-1] > y[2:])
    return np.sum(maxima_indices)

d_omeg_range = np.linspace(-0.2, 0.2, 100)
theta_range = np.linspace(0, 2*np.pi, 100)

num_maxima = np.zeros((len(d_omeg_range), len(theta_range)))

t_span = (0, 100)

for i, d_omeg in enumerate(d_omeg_range):
    for j, theta in enumerate(theta_range):
        
        solution = solve_ivp(odesystem, t_span, initial_conditions, args=(d_omeg, theta), dense_output=True)
        
        t_eval = solution.t
        y_eval = solution.y
        
        num_maxima[i, j] = count_maxima(t_eval, y_eval[0])

plt.imshow(num_maxima, extent=[0, 2*np.pi, -0.2, 0.2], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Number of maxima')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\delta\omega$')
plt.title('Number of maxima in E1 time series')
plt.savefig('maixma_bifurc_diagram.pdf')
plt.show()
