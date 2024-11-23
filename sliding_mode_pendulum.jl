using DifferentialEquations
using Plots

# Start timing
start = time()
println("start: ", time() - start)

# Parameters for the inverted pendulum
m = 0.2  # Mass of the pendulum (kg)
M = 0.5  # Mass of the cart (kg)
L = 0.3  # Length of the pendulum (m)
g = 9.81  # Gravitational acceleration (m/s^2)

# Sliding Mode Controller parameters
k_s = 10.0  # Sliding surface gain
lambda_s = 1.0  # Sliding surface coefficient

# Define the system dynamics (state space model)
function pendulum_dynamics(du, u, p, t)
    θ, θ̇, x, ẋ = u

    # Equations of motion for the inverted pendulum
    sinθ = sin(θ)
    cosθ = cos(θ)

    # Angular acceleration (θ̈)
    θ̈ = (g * sinθ + cosθ * (-k_s * θ̇ - lambda_s * θ)) / (L * (4 / 3 - (m * cosθ^2) / (M + m)))

    # Cart acceleration (ẍ)
    ẍ = (-m * L * θ̈ * cosθ + m * L * sinθ * θ̇^2) / (M + m)

    # Sliding mode control law
    s = θ̇ + k_s * θ  # Sliding surface
    u_ctrl = -k_s * sign(s)  # Control input based on the sliding surface

    # State derivatives
    du[1] = θ̇
    du[2] = θ̈ + u_ctrl
    du[3] = ẋ
    du[4] = ẍ

    # Debugging output (optional)
    # println("t=$t, θ=$θ, θ̇=$θ̇, θ̈=$θ̈, x=$x, ẋ=$ẋ, ẍ=$ẍ, u_ctrl=$u_ctrl")
end

# Initial conditions [θ, θ̇, x, ẋ]
u0 = [0.1, 0.0, 0.0, 0.0]

# Time span for simulation
tspan = (0.0, 10.0)

println("Solve: ", time() - start)

# Solve the differential equations using a stiff solver (e.g., Rodas5)
prob = ODEProblem(pendulum_dynamics, u0, tspan)
sol = solve(prob, Rodas5(), saveat=0.05, maxiters=1e7, abstol=1e-6, reltol=1e-6)

println("Plot: ", time() - start)

# Plot the results
plot_layout = @layout [a; b; c]
p = plot(layout=plot_layout, size=(1000, 1000))

# Plot all variables on the same graph
plot!(sol.t, sol[1, :], label="Angle (θ)", color=:blue, linestyle=:solid, subplot=1)
plot!(sol.t, sol[2, :], label="Angular Velocity (θ̇)", color=:red, linestyle=:dash, subplot=1)
plot!(xlabel="Time (s)", ylabel="State Variables", title="Inverted Pendulum Dynamics with Sliding Mode Control", subplot=1)

# Phase plane plot
plot!(sol[1, :], sol[2, :], label="Phase Plane", color=:green, xlabel="Angle (θ)", ylabel="Angular Velocity (θ̇)", title="Phase Plane of the Pendulum", subplot=2)

# Controller plot
sliding_surface = sol[2, :] + k_s * sol[1, :]
plot!(sol.t, sliding_surface, label="Sliding Surface", color=:magenta, xlabel="Time (s)", ylabel="Sliding Surface", title="Controller", subplot=3)

# Display the plots
display(p)

println(time() - start)
