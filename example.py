import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# FitzHugh-Nagumo model
def fitzhugh_nagumo(t, y, a, b, I):
    v, w = y
    dvdt = v - (v**3 / 3) - w + I
    dwdt = epsilon * (v + a - b*w)
    return [dvdt, dwdt]

# Parameters
epsilon = 0.08  # Small parameter
I = 0.5         # External current
a_values = np.linspace(-0.5, 1.5, 50)  # Range of a values
b_values = np.linspace(0, 2, 50)       # Range of b values
final_v = np.zeros((len(a_values), len(b_values)))  # To store the final v values

# Solve the system for each pair of (a, b)
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        sol = solve_ivp(fitzhugh_nagumo, [0, 1000], [0.5, 0.5], args=(a, b, I), rtol=1e-5, atol=1e-8)
        final_v[i, j] = sol.y[0, -1]  # Take the last value of v

# Plotting
plt.figure(figsize=(10, 8))
plt.contourf(b_values, a_values, final_v, levels=50, cmap='viridis')
plt.colorbar(label='Final v value')
plt.xlabel('b')
plt.ylabel('a')
plt.title('Exploratory Bifurcation Diagram of FitzHugh-Nagumo Model')
plt.show()
