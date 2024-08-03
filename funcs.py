# funcs.py

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
d = 2
alpha = 1
sigma = 1
h = 0.001  # Step size
epsilon = 1e-6  # 2.1 line 6 epsilon


# GENERAL SOLVING
def f(r, y):
    u, u_prime = y
    if r == 0:
        return np.array([u_prime, (u - alpha * abs(u) ** (2 * sigma) * u) / d])
    return np.array([u_prime, -((d - 1) / r) * u_prime + u - alpha * abs(u) ** (2 * sigma) * u])


def rk4(f, y0, r0, rf, h):
    n = int((rf - r0) / h)
    r = np.linspace(r0, rf, n + 1)
    y = np.zeros((n + 1, 2))
    y[0] = y0

    for i in range(n):
        k1 = h * f(r[i], y[i])
        k2 = h * f(r[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(r[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(r[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return r, y


def shoot(beta, R):
    r, y = rk4(f, np.array([beta, 0]), 0, R, h)
    return r, y[:, 0]


# CALCULATIONS
def find_ground_state(R, max_iterations):
    a, b = 0, 4
    a_values = [a]
    b_values = [b]
    beta_values = []
    solutions = []

    for i in range(max_iterations):
        beta = (a + b) / 2
        beta_values.append(beta)
        r, u = shoot(beta, R)
        solutions.append((r, u))

        if np.any(u < 0):
            b = beta
        elif np.all(u > epsilon):
            a = beta
        else:
            break

        a_values.append(a)
        b_values.append(b)

    return beta, r, u, a_values, b_values, beta_values, solutions


def find_excited_state(k, R=20, max_iterations=10, a0=0, b0=5):
    a, b = a0, b0
    a_values = [a]
    b_values = [b]
    beta_values = []
    solutions = []

    for i in range(max_iterations):
        beta = (a + b) / 2
        beta_values.append(beta)
        r, u = shoot(beta, R)
        solutions.append((r, u))
        zero_crossings = np.sum(np.diff(np.sign(u)) != 0)
        print(f"Iteration {i + 1}: beta = {beta:.6f}, zero_crossings = {zero_crossings}")
        if zero_crossings > k:
            b = beta
        elif zero_crossings <= k:
            a = beta
        else:
            if np.abs(u[-1]) < epsilon or (b - a) < epsilon:
                return beta, r, u, a_values, b_values, beta_values, solutions
            else:
                b = beta
        a_values.append(a)
        b_values.append(b)

    print(f"Max iterations reached. Last beta = {beta:.6f}")
    return beta, r, u, a_values, b_values, beta_values, solutions


# PLOTTING
def plot_figure_1(R, max_iterations):
    # Find the ground state
    beta_star, r, u, a_values, b_values, beta_values, solutions = find_ground_state(R, max_iterations)

    # Print the coefficients for each iteration
    print("Iteration | a | b | beta")
    print("-" * 50)
    for i, (a, b, beta) in enumerate(zip(a_values, b_values, beta_values)):
        print(f"{i:9d} | {a:.10f} | {b:.10f} | {beta:.10f}")

    # Plot the results for specific iterations
    #plt.figure(figsize=(12, 8))
    beta = 4
    r_0, u_0 = shoot(beta, 10)
    # Plot 0th iteration
    plt.plot(r_0, u_0, label=f'0th iteration, β ≈ {beta_values[0]:.10f}', )

    # Plot 1st iteration
    r_1, u_1 = solutions[0]  # 9th iteration (index 0)
    plt.plot(r_1, u_1, label=f'1st iteration, β ≈ {beta_values[1]:.10f}')

    # Plot final solution
    plt.plot(r, u, label=f'Final, β* ≈ {beta_star:.10f}', linewidth=2, linestyle='--')

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Ground State of NLS Equation (d=2, α=σ=1)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nFinal approximate β* = {beta_star:.10f}")

    # Print a, b, and beta values
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])


def plot_figure_2(R, max_iterations):
    # Find the ground state
    beta_star, r, u, a_values, b_values, beta_values, solutions = find_ground_state(R, max_iterations)

    # Print the coefficients for each iteration
    print("Iteration | a | b | beta")
    print("-" * 50)
    for i, (a, b, beta) in enumerate(zip(a_values, b_values, beta_values)):
        print(f"{i:9d} | {a:.10f} | {b:.10f} | {beta:.10f}")

    # Plot the results for specific iterations
    #plt.figure(figsize=(12, 8))

    # Plot 8th iteration
    r_8, u_8 = solutions[7]  # 8th iteration (index 7)
    plt.plot(r_8, u_8, label=f'8th iteration, β ≈ {beta_values[7]:.10f}', linestyle='--')

    # Plot 9th iteration
    r_9, u_9 = solutions[8]  # 9th iteration (index 8)
    plt.plot(r_9, u_9, label=f'9th iteration, β ≈ {beta_values[8]:.10f}', linestyle=':')

    # Plot final solution
    plt.plot(r, u, label=f'Final, β* ≈ {beta_star:.10f}', linewidth=2)

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Ground State of NLS Equation (d=2, α=σ=1)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nFinal approximate β* = {beta_star:.10f}")

    # Print a, b, and beta values
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])


def plot_figure_3(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)

    # Plot the solutions for the last three iterations
    plt.plot(r, solutions[-3][1], label=f"β = (a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    plt.plot(r, solutions[-2][1], label=f"β = a9 ≈ {a_values[-1]:.6f}")
    plt.plot(r, solutions[-1][1], label=f"β = b9 ≈ {b_values[-1]:.6f}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()

    print(f"a9 ≈ {a_values[-1]:.6f}")
    print(f"b9 ≈ {b_values[-1]:.6f}")
    print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")


def plot_figure_4(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)

    for i, (a, b, beta) in enumerate(zip(a_values, b_values, beta_values)):
        print(f"{i:9d} | {a:.10f} | {b:.10f} | {beta:.10f}")
    # Plot the solutions for the last three iterations
    plt.plot(r, solutions[-1][1], label=f"β = (a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    plt.plot(r, solutions[-2][1], label=f"β = a9 ≈ {a_values[-1]:.6f}")
    plt.plot(r, solutions[-3][1], label=f"β = b9 ≈ {b_values[-1]:.6f}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])
