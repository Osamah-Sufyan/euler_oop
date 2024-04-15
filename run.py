from multiprocessing import Pool
import numpy as np
from euler_oop_parallel import LaserDESolver

tolerance= 1e-0

def simulate(params):
    d_omeg, theta = params
    solver = LaserDESolver(D_omeg=3, d_omeg=d_omeg, K=0.1, theta=theta, T=392, p=2)
    E1_abs, time = solver.solve()
    x = E1_abs[round(7000/0.05):]
    unique_maxima, num_maxima = LaserDESolver.count_unique_maxima(x)
    return num_maxima

if __name__ == '__main__':
    # Create parameter combinations
    params = [(d_omeg, theta) for d_omeg in np.linspace(-0.2, 0.2, 10) for theta in np.linspace(0, 2*np.pi, 10)]

    # Parallelize simulations
    with Pool() as p:
        maxima_counts = p.map(simulate, params)

    # Reshape maxima_counts to fit your grid if necessary
    maxima_counts = np.array(maxima_counts).reshape((10, 10))