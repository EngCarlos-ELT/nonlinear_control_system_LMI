import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the state-space matrices of the inverted pendulum
A = np.array([[0, 1, 0, 0],
              [0, 0, -9.8, 0],
              [0, 0, 0, 1],
              [0, 0, 24.5, 0]])
B = np.array([[0],
              [1],
              [0],
              [-1]])

# Define the size of the state and input
n, m = B.shape

# Variables for LMI
P = cp.Variable((n, n), symmetric=True)
Y = cp.Variable((m, n))

# Define the LMI constraints
lmi = cp.bmat([
    [P, A @ P + B @ Y],
    [(A @ P + B @ Y).T, P]
])
constraints = [P >> 0, lmi >> 0]

# Define the objective (minimize trace of P for simplicity)
objective = cp.Minimize(cp.trace(P))

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract the feedback gain K
if problem.status == cp.OPTIMAL:
    P_value = P.value
    Y_value = Y.value
    K = Y_value @ np.linalg.inv(P_value)
    print("Optimal feedback gain K:")
    print(K)
else:
    print("LMI problem did not converge to an optimal solution.")
    exit()

# Closed-loop simulation
def inverted_pendulum_dynamics(t, x):
    u = K @ x  # State-feedback control law
    return (A + B @ K) @ x  # Closed-loop dynamics

# Simulation parameters
x0 = np.array([0.1, 0, 0.1, 0])  # Initial state [angle, angular velocity, position, velocity]
t_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Time points for evaluation

# Solve the differential equation
solution = solve_ivp(inverted_pendulum_dynamics, t_span, x0, t_eval=t_eval)

# Extract time and state trajectories
time = solution.t
states = solution.y

# Plot the results
plt.figure(figsize=(12, 8))

# Plot each state
state_labels = ["Angle (rad)", "Angular Velocity (rad/s)", "Position (m)", "Velocity (m/s)"]
for i in range(n):
    plt.subplot(2, 2, i + 1)
    plt.plot(time, states[i, :], label=state_labels[i])
    plt.xlabel("Time (s)")
    plt.ylabel(state_labels[i])
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
