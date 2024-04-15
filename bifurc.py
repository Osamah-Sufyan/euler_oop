import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the parameters
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4

# Define the system of ODEs
def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Initial conditions
z0 = [10, 5]

# Time span
t = np.linspace(0, 15, 400)

# Solve the system of ODEs
sol = solve_ivp(lotka_volterra, [0, 15], z0, t_eval=t)

# Plotting
plt.plot(sol.t, sol.y[0], label='Prey (x)')
plt.plot(sol.t, sol.y[1], label='Predators (y)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Lotka-Volterra Model')
plt.show()