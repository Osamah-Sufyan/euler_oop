import numpy as np
import matplotlib.pyplot as plt

# Constants
D_omeg = 3
d_omeg = 0.2
K = 0.1
theta = 3*np.pi / 2
T = 329
p = 2


def solve(D_omeg,d_omeg, K, theta, T, p):

# Initial conditions
        E1_0 = 1 + 0j
        E2_0 = 1 + 0j
        N1_0 = 1.0
        N2_0 = 1.0


        # Time step and simulation time
        dt = 0.1
        time = np.arange(0, 4000, dt)

        # Initialize arrays to store the values
        E1 = np.zeros(len(time), dtype=complex)
        E2 = np.zeros(len(time), dtype=complex)
        N1 = np.zeros(len(time))
        N2 = np.zeros(len(time))
        E1_abs = np.zeros(len(time))  # Absolute value of E1

        # Set initial values
        E1[0] = E1_0
        E2[0] = E2_0
        N1[0] = N1_0
        N2[0] = N2_0

        # Euler method to update values
        for i in range(1, len(time)):
            dE1_dt = (1 + 1j * D_omeg) * N1[i-1] * E1[i-1] - 0.5 * E1[i-1] - 1j * d_omeg * E1[i-1] + K * np.exp(1j * theta) * E2[i-1]
            dE2_dt = (1 + 1j * D_omeg) * N2[i-1] * E2[i-1] - 0.5 * E2[i-1] + 1j * d_omeg * E2[i-1] + K * np.exp(1j * theta) * E1[i-1]
            dN1_dt = (1 / T) * (p - (N1[i-1] * np.abs(E1[i-1])) - 2 * N1[i-1] * np.abs(E1[i-1]))
            dN2_dt = (1 / T) * (p - (N2[i-1] * np.abs(E2[i-1])) - 2 * N2[i-1] * np.abs(E2[i-1]))

            E1[i] = E1[i-1] + dE1_dt * dt
            E2[i] = E2[i-1] + dE2_dt * dt
            N1[i] = N1[i-1] + dN1_dt * dt
            N2[i] = N2[i-1] + dN2_dt * dt

            E1_abs[i] = np.abs(E1[i])

        return E1_abs, time


def count_unique_maxima(x, tolerance=1e-4):
    # Find indices of local maxima
    max_indices = np.where((np.roll(x, 1) < x) & (np.roll(x, -1) < x))[0] + 1
    
    # Get unique maxima values
    unique_maxima = np.unique(np.round(x[max_indices], int(np.ceil(-np.log10(tolerance)))))
    
    return unique_maxima, len(unique_maxima)-1

# Example usage:
# D_omeg = 3
# d_omeg = 0.2
# K = 0.1
# theta = 3*np.pi / 2
# T = 329
# p = 2

time = solve(3, 0.2,0.1, 2*np.pi, 392, 2)[1]
E1_abs= solve(3, 0.2,0.1, 2*np.pi, 392, 2)[0]
x = E1_abs
num_unique_maxima = count_unique_maxima(x)[1]
print("Number of unique maxima:", num_unique_maxima)
plt.plot(time[12000:], E1_abs[12000:], label='|E1| over time')
plt.xlabel('Time[ps]')
plt.ylabel('|E1|')
plt.title(f'Absolute value of E1 over time, ($\Delta\omega$= {d_omeg}, $\delta\omega$= {D_omeg},$p$= {p},\n $K$= {K}, $\\theta=\\frac{{3}}{{2}}\\pi$)' )
plt.legend()
plt.grid(True)




d_omeg_values = np.linspace(-0.2, 0.2, 100)
theta_values = np.linspace(0, 2*np.pi, 100)
num_maxima = np.zeros((len(theta_values), len(d_omeg_values)))


# Plotting
for i, d_omeg in enumerate(d_omeg_values):
    for j, theta in enumerate(theta_values):
        E1_abs = solve(3, i, 0.1, j, 392, 2)[0][20000:30000]
        
        num_maxima[i, j] = count_unique_maxima(E1_abs[20000:30000])


# Define colors for different ranges of unique maxima counts
colors = ['lightblue', 'lightgreen', 'orange', 'red']
bounds = [1, 2, 8, np.inf]
cmap = plt.cm.colors.ListedColormap(colors)
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(num_maxima.T, extent=[0, 2*np.pi, -0.2, 0.2], origin='lower', cmap=cmap, norm=norm)
plt.colorbar(label='Number of unique maxima')
plt.xlabel('Theta')
plt.ylabel('d_omega')
plt.title('Number of unique maxima heatmap')
plt.grid(True)
plt.show()


