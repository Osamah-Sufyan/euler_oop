import numpy as np
import matplotlib.pyplot as plt

# Constants



def solve(D_omeg,d_omeg, K, theta, T, p):

# Initial conditions
        
        dt = 0.1
        
        time = np.arange(0, 10000, dt)
        
        
        E1 = np.zeros(len(time), dtype=complex)
        E2 = np.zeros(len(time), dtype=complex)
        N1 = np.zeros(len(time))
        N2 = np.zeros(len(time))
        E1_abs = np.zeros(len(time))
        
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
            dE1_dt = (1 + 1j * D_omeg) * N1[i-1] * E1[i-1] - 0.5 * E1[i-1] - 1j * d_omeg * E1[i-1] + K * np.exp(1j * theta) * E2[i-1]
            dE2_dt = (1 + 1j * D_omeg) * N2[i-1] * E2[i-1] - 0.5 * E2[i-1] + 1j * d_omeg * E2[i-1] + K * np.exp(1j * theta) * E1[i-1]
            dN1_dt = (1 / T) * (p - (N1[i-1] * np.abs(E1[i-1])) - 2 * N1[i-1] * np.abs(E1[i-1]))
            dN2_dt = (1 / T) * (p - (N2[i-1] * np.abs(E2[i-1])) - 2 * N2[i-1] * np.abs(E2[i-1]))

            E1[i] = E1[i-1] + dE1_dt * dt
            E2[i] = E2[i-1] + dE2_dt * dt
            N1[i] = N1[i-1] + dN1_dt * dt
            N2[i] = N2[i-1] + dN2_dt * dt

            E1_abs[i] = np.abs(E1[i])

        return E1_abs**2 , time


def count_unique_maxima(x, tolerance=2):
    # Find indices of local maxima
    max_indices = np.where((np.roll(x, 1)[1:-1] < x[1:-1]) & (np.roll(x, -1)[1:-1] < x[1:-1]))[0] + 1
    
    
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
D_omeg = 3
d_omeg = 0.2
K = 0.1
theta = 3*np.pi / 2
T = 329
p = 2



theta_values = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
d_omeg_values = [-0.2, 0.1, 0.0, 0.1, 0.2]
plt.figure(figsize=( 4 * len(theta_values),4))
dt = 0.1
start_count = round(8000/dt)
time = solve(3, 0 ,0.1, np.pi, 392, 2)[1]
E1_abs= solve(3, 0 ,0.1, np.pi, 392, 2)[0]
x = E1_abs[start_count:]
print(x)
num_unique_maxima = count_unique_maxima(x)[1]
umax = count_unique_maxima(x)[0]
print(umax)
print("Number of unique maxima:", num_unique_maxima)

# Add a subplot for each iteration


plt.plot(time[start_count:], E1_abs[start_count:], label='|E1| over time')
plt.xlabel('Time[ps]')
plt.ylabel('|E1|')
plt.title(f'Absolute value of E1 over time, ($\Delta\omega$= {d_omeg}, $\delta\omega$= {D_omeg},$p$= {p},\n $K$= {K}, $\\theta=\\frac{{3}}{{2}}\\pi$)' )
plt.legend()
plt.grid(True)

# for i, value in enumerate(theta_values):
#     time = solve(3, 0.2,0.1, value, 392, 2)[1]
#     E1_abs= solve(3, 0.2,0.1, value, 392, 2)[0]
#     x = E1_abs
#     num_unique_maxima = count_unique_maxima(x)[1]
#     print("Number of unique maxima:", num_unique_maxima)

#     # Add a subplot for each iteration
#     plt.subplot(len(theta_values), 1, i+1)

#     plt.plot(time, E1_abs, label='|E1| over time')
#     plt.xlabel('Time[ps]')
#     plt.ylabel('|E1|')
#     plt.title(f'Absolute value of E1 over time, ($\Delta\omega$= {d_omeg}, $\delta\omega$= {D_omeg},$p$= {p},\n $K$= {K}, $\\theta=\\frac{{3}}{{2}}\\pi$)' )
#     plt.legend()
#     plt.grid(True)

# Save the figure after the loop
plt.savefig('time_series_|E1|_zoomed.pdf')
plt.show()


