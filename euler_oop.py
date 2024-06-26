import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors




class LaserDESolver:
    def __init__(self, D_omeg, d_omeg, K, theta, T, p):
        self.D_omeg = D_omeg
        self.d_omeg = d_omeg
        self.K = K
        self.theta = theta
        self.T = T
        self.p = p

    def solve(self):
        dt = 0.05
        
        time = np.arange(0, 10000, dt)

        E1 = np.zeros(len(time), dtype=complex)
        E2 = np.zeros(len(time), dtype=complex)
        N1 = np.zeros(len(time))
        N2 = np.zeros(len(time))
        E1_abs = np.zeros(len(time)) 
        # Initial conditions
        E1_0 = 5*np.random.rand() + 5*np.random.rand() * 1j
        E2_0 = 5*np.random.rand() + 5*np.random.rand() * 1j
        N1_0 = 5*np.random.rand()
        N2_0 = 5*np.random.rand()

        # Time step and simulation time
        

        # Initialize arrays to store the values
         # Absolute value of E1

        # Set initial values
        E1[0] = E1_0
        E2[0] = E2_0
        N1[0] = N1_0
        N2[0] = N2_0

        # Euler method to update values
        for i in range(1, len(time)):
            dE1_dt = (1 + 1j * self.D_omeg) * N1[i-1] * E1[i-1] - 0.5 * E1[i-1] - 1j * self.d_omeg * E1[i-1] + self.K * np.exp(1j * self.theta) * E2[i-1]
            dE2_dt = (1 + 1j * self.D_omeg) * N2[i-1] * E2[i-1] - 0.5 * E2[i-1] + 1j * self.d_omeg * E2[i-1] + self.K * np.exp(1j * self.theta) * E1[i-1]
            dN1_dt = (1 / self.T) * (self.p - N1[i-1]  - 2 * N1[i-1] * (np.abs(E1[i-1]))**2)
            dN2_dt = (1 / self.T) * (self.p - N2[i-1]  - 2 * N2[i-1] * (np.abs(E2[i-1]))**2)

            E1[i] = E1[i-1] + dE1_dt * dt
            E2[i] = E2[i-1] + dE2_dt * dt
            N1[i] = N1[i-1] + dN1_dt * dt
            N2[i] = N2[i-1] + dN2_dt * dt

            E1_abs[i] = (np.abs(E1[i]))**2

        return E1_abs, time

    @staticmethod
    def count_unique_maxima(x,tolerance= 1e-1):
    # Find indices of local maxima
        
        unique_maxima = np.array([])
        if np.all(np.isclose(x, x[0], atol=tolerance)):
            num_maxima = 0
        else:
            max_indices = np.where((np.roll(x, 1) < x) & (np.roll(x, -1) < x))[0] 
            unique_maxima = np.unique(np.round(x[max_indices], int(np.ceil(-np.log10(tolerance)))))
            
        num_maxima = len(unique_maxima)     
        return unique_maxima, num_maxima

 # Example usage:
# solver = LaserDESolver(D_omeg=3, d_omeg=0.1, K=0.1, theta=np.pi, T=329, p=2)
# dt = 0.05
# start_count = round(7000/dt)
# E1_abs, time = solver.solve()
# x = E1_abs[start_count:]
# unique_maxima, num_maxima = LaserDESolver.count_unique_maxima(x)
# print(f"Number of unique maxima: {num_maxima}")
# print(E1_abs)

# plt.plot(time, E1_abs, label='|E1| over time')

# plt.xlabel('Time[ps]')
# plt.ylabel('|E1|')
# plt.title(f'Absolute value of E1 over time' )
# plt.legend()
# plt.grid(True)
# plt.savefig('time_series_|E1|_zoomed.pdf')
# plt.show()
# num_maxima =  LaserDESolver.count_unique_maxima(x)[1]
# print(f"Number of unique maxima: {num_maxima}")
    



tolerance= 1e-0
d_omeg_values = np.linspace(-0.2, 0.2, 10)
theta_values = np.linspace(0, 2*np.pi, 10)
d_omeg_grid, theta_grid = np.meshgrid(d_omeg_values, theta_values)# Initialize an array to store the counts of unique maxima
umax = [[[] for _ in range(d_omeg_grid.shape[1])] for _ in range(d_omeg_grid.shape[0])]
maxima_counts = np.zeros_like(d_omeg_grid)


# Loop over each pair of d_omeg and theta
for i in range(d_omeg_grid.shape[0]):
    for j in range(d_omeg_grid.shape[1]):
        dt = 0.05
        start_count = round(7000/dt)
        # Solve the equations and count the unique maxima
        solver = LaserDESolver(D_omeg=3, d_omeg=d_omeg_grid[i, j], K=0.1, theta=theta_grid[i, j], T=392, p=2)
        E1_abs, time = solver.solve()
        x = E1_abs[start_count:]
        
        unique_maxima, num_maxima = LaserDESolver.count_unique_maxima(x)
        maxima_counts[i, j] = num_maxima
        umax[i][j]  = unique_maxima.tolist()
        
        






# Define the colormap

# for i in range(d_omeg_grid.shape[0]):
#     for j in range(d_omeg_grid.shape[1]):
#         print(f"umax[{i}][{j}] = {umax[i][j]}")

# solver = LaserDESolver(D_omeg=3, d_omeg=d_omeg_grid[2,2], K=0.1, theta=theta_grid[2,2], T=392, p=2)
# dt = 0.05
# start_count = round(7000/dt)
# E1_abs, time = solver.solve()
# x = E1_abs[start_count:]

# if np.all(np.isclose(x, x[0], atol=tolerance)):
#     num_maxima = 0
# else:
#     max_indices = np.where((np.roll(x, 1) < x) & (np.roll(x, -1) < x))[0] 
#     unique_maxima = np.unique(np.round(x[max_indices], int(np.ceil(-np.log10(tolerance)))))
#     num_maxima = len(unique_maxima)

# print(f"Number of unique maxima: {num_maxima}")


# plt.plot(time, E1_abs, label='|E1| over time')
# plt.xlabel('Time[ps]')
# plt.ylabel('|E1|')
# plt.title(f'Absolute value of E1 over time)' )
# plt.legend()
# plt.grid(True)
# plt.show()
        



colors = ['yellow', 'lightblue', 'blue', 'darkblue', 'black']
bounds = [-0.5, 0.5, 1.5, 8, 200]    # Boundaries for the colors
norm = mcolors.BoundaryNorm(bounds, len(colors))  # Create a BoundaryNorm object
cmap = mcolors.ListedColormap(colors)  # Create a colormap from the list of colors

# Create a heatmap of the counts
plt.imshow(maxima_counts, origin='lower', extent=[0, 2*np.pi, -0.2, 0.2], aspect='auto', cmap=cmap, norm=norm)
plt.colorbar(label='Number of unique maxima')
plt.xlabel('theta')
plt.ylabel('d_omeg')
plt.title('Number of unique maxima for different d_omeg and theta_0')
plt.savefig('unique_maxima_heatmap_corrected_1.pdf')
plt.show()