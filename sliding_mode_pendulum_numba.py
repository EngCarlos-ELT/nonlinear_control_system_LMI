import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from numba import njit

start=time.time()
print("start", time.time() - start)
# Parameters for the inverted pendulum
m = 0.2   # Mass of the pendulum (kg)
M = 0.5   # Mass of the cart (kg)
L = 0.3   # Length of the pendulum (m)
g = 9.81  # Gravitational acceleration (m/s^2)

# Sliding Mode Controller parameters
k_s = 10.0  # Sliding surface gain
lambda_s = 1.0  # Sliding surface coefficient

# Define the system dynamics (state space model)
@njit
def pendulum_dynamics(t, u):
    θ, θ̇, x, ẋ = u
    
    # Equations of motion for the inverted pendulum
    sinθ = np.sin(θ)
    cosθ = np.cos(θ)
    
    # Angular acceleration (θ̈)
    θ̈ = (g * sinθ + cosθ * (-k_s * θ̇ - lambda_s * θ)) / (L * (4/3 - (m * cosθ**2) / (M + m)))
    
    # Cart acceleration (ẍ)
    ẍ = (-m * L * θ̈ * cosθ + m * L * sinθ * θ̇**2) / (M + m)
    
    # Sliding mode control law
    s = θ̇ + k_s * θ  # Sliding surface
    u_ctrl = -k_s * np.sign(s)  # Control input based on the sliding surface
    
    # State derivatives
    return [θ̇, θ̈ + u_ctrl, ẋ, ẍ]

# Initial conditions [θ, θ̇, x, ẋ]
u0 = [0.1, 0.0, 0.0, 0.0]

# Time span for simulation
tspan = (0.0, 10.0)

# Time points to save
t_eval = np.arange(0.0, 10.0, 0.05)


print("Solve", time.time() - start)
# Solve the differential equations using the ODE solver
sol = solve_ivp(pendulum_dynamics, tspan, u0, t_eval=t_eval)

print("Plot", time.time() - start)
# Plot the results
plt.figure(figsize=(10, 6))

# Plot all variables on the same graph
plt.plot(sol.t, sol.y[0], label="Angle (θ)", linestyle='-', color='b')
plt.plot(sol.t, sol.y[1], label="Angular Velocity (θ̇)", linestyle='--', color='r')
plt.plot(sol.t, sol.y[2], label="Cart Position (x)", linestyle='-.', color='g')
plt.plot(sol.t, sol.y[3], label="Cart Velocity (ẋ)", linestyle=':', color='m')

# Labels and legend
plt.xlabel("Time (s)")
plt.ylabel("State Variables")
plt.title("Inverted Pendulum Dynamics with Sliding Mode Control")
plt.legend()

# Show the plot
plt.tight_layout()
end_=time.time()
print(end_ - start)

plt.show()

