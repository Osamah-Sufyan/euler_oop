import numpy as np
# Set up simulation parameters
p = 1
a = 1  # Delay parameter
s_max = 20  # End time for simulation
dt = 0.0001  # Time step size
steps = int(s_max / dt) + 1  # Number of time steps
initial_population = 0.5  # Initial condition for x(s) for s <= 0

# Initialize arrays to store time steps and x values
time_steps = np.linspace(0, s_max, steps)
x_values = np.full(steps, initial_population)

# Populate the x_values array considering the delay
for i in range(1, steps):
    t = i * dt  # Current time step
    delay_index = int(max(0, t - a) / dt)  

    # Calculate the rate of change dx/ds
    dx_dt = p*x_values[i-1] * (1 - x_values[delay_index])
    
    # Update the current value of x
    x_values[i] = x_values[i-1] + dx_dt * dt

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_steps, x_values, label='x(s)')
plt.title('Solution of Delay Logistic Differential Equation')
plt.xlabel('Dimensionless time, s')
plt.ylabel('Normalized population, x(s)')
plt.legend()
plt.grid(True)
plt.show()
