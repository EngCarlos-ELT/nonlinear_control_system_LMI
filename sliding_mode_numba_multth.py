import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from numba import njit
import multiprocessing

# Parameters for the inverted pendulum
m = 0.2  # Mass of the pendulum (kg)
M = 0.5  # Mass of the cart (kg)
L = 0.3  # Length of the pendulum (m)
g = 9.81  # Gravitational acceleration (m/s^2)

# Sliding Mode Controller parameters
k_s = 10.0  # Sliding surface gain
lambda_s = 1.0  # Sliding surface coefficient

# Define the system dynamics (state space model)
#@njit
def pendulum_dynamics(t, u):
    θ, θ̇, x, ẋ = u

    # Equations of motion for the inverted pendulum
    sinθ = np.sin(θ)
    cosθ = np.cos(θ)

    # Angular acceleration (θ̈)
    θ̈ = (g * sinθ + cosθ * (-k_s * θ̇ - lambda_s * θ)) / (L * (4 / 3 - (m * cosθ ** 2) / (M + m)))

    # Cart acceleration (ẍ)
    ẍ = (-m * L * θ̈ * cosθ + m * L * sinθ * θ̇ ** 2) / (M + m)

    # Sliding mode control law
    s = θ̇ + k_s * θ  # Sliding surface
    u_ctrl = -k_s * np.sign(s)  # Control input based on the sliding surface

    # State derivatives
    return [θ̇, θ̈ + u_ctrl, ẋ, ẍ]


# Function to compute the ODE solution for a given time span
def solve_ode_for_chunk(t_chunk):
    u0 = [0.1, 0.0, 0.0, 0.0]
    tspan = (t_chunk[0], t_chunk[1])
    t_eval = np.arange(t_chunk[0], t_chunk[1], 0.05)
    sol = solve_ivp(pendulum_dynamics, tspan, u0, t_eval=t_eval)
    return sol


def main():
    start = time.time()

    # Define the time span for the full simulation
    t_total = 10.0
    num_chunks = 4  # Number of chunks to divide the time span into
    chunk_size = t_total / num_chunks

    # Create chunks of time spans to run in parallel
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]

    # Use multiprocessing to compute the ODE solution for each time chunk in parallel num_chunks
    with multiprocessing.Pool(processes=num_chunks) as pool:
        results = pool.map(solve_ode_for_chunk, chunks)

    # Combine the results from all chunks
    t_combined = np.concatenate([res.t for res in results])
    y_combined = np.concatenate([res.y for res in results], axis=1)

    print("Plotting...", time.time() - start)

    # Plot the results
    fig, axs = plt.subplots(3, figsize=(10, 10))

    # Plot all variables on the same graph
    axs[0].plot(t_combined, y_combined[0], label="Angle (θ)", linestyle='-', color='b')
    axs[0].plot(t_combined, y_combined[1], label="Angular Velocity (θ̇)", linestyle='--', color='r')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("State Variables")
    axs[0].set_title("Inverted Pendulum Dynamics with Sliding Mode Control")
    axs[0].legend()

    # Phase plane plot
    axs[1].plot(y_combined[0], y_combined[1], label="Phase Plane", color='g')
    axs[1].set_xlabel("Angle (θ)")
    axs[1].set_ylabel("Angular Velocity (θ̇)")
    axs[1].set_title("Phase Plane of the Pendulum")
    axs[1].legend()

    # Controller plot
    axs[2].plot(t_combined, y_combined[1] + k_s * y_combined[0], label="Sliding Surface", color='m')
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Sliding Surface")
    axs[2].set_title("Controller")
    axs[2].legend()

    # Layout so plots do not overlap
    fig.tight_layout()

    end_ = time.time()
    print(f"Total time: {end_ - start:.2f} seconds")

    plt.show()


if __name__ == "__main__":
    main()
