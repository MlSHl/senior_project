# funcs.py
import math

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
d = 2
alpha = 1
sigma = 1
h = 0.001  # Step size
epsilon = 1e-8  # 2.1 line 6 epsilon


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

    beta = b0
    for i in range(max_iterations):
        beta_values.append(beta)
        r, u = shoot(beta, R)
        solutions.append((r, u))
        zero_crossings = np.sum(np.diff(np.sign(u)) != 0)
        # if i in (max_iterations - 3, max_iterations - 2):
        #     plt.plot(r, u, label=f"{i} β ≈ {beta_values[i]:.10f}")
        # elif i == (max_iterations - 1):
        #     plt.plot(r, u, label=f"{i} β ≈ {beta_values[i]:.10f}", linestyle='--')
        #plt.plot(r, u, label=f"{i}: β = {beta_values[i]:.10f}, a={a_values[i]}, b={b_values[i]}, int{zero_crossings}")
        #print(f"Iteration {i + 1}: beta = {beta:.6f}, zero_crossings = {zero_crossings}")
        if zero_crossings > k:
            b = beta
        elif zero_crossings <= k:
            a = beta
        a_values.append(a)
        b_values.append(b)

        beta = (a + b) / 2

        print(f"{i:9d} | {a_values[i]:.10f} | {b_values[i]:.10f} | {beta_values[i]:.10f} | {zero_crossings}")
    #print(f"Max iterations reached. Last beta = {beta:.6f}")
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
    plt.plot(r_8, u_8, label=f'8th iteration, β ≈ {beta_values[7]:.10f}')

    # Plot 9th iteration
    r_9, u_9 = solutions[8]  # 9th iteration (index 8)
    plt.plot(r_9, u_9, label=f'9th iteration, β ≈ {beta_values[8]:.10f}')

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


def plot_figure_3(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)
    r, u = shoot(5, R)
    # Plot the solutions for the last three iterations
    # plt.plot(r, u, label=f"β = (a1+b1)/2 ≈ {beta_values[-2]:.6f}")
    # plt.plot(r, solutions[-2][1], label=f"β = b1 ≈ {5:.6f}")
    # plt.plot(r, solutions[-3][1], label=f"β = a1 ≈ {beta_values[-3]:.6f}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 2nd iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()

    print(f"a1 ≈ {a_values[-1]:.6f}")
    print(f"b1 ≈ {b_values[-1]:.6f}")
    print(f"(a1+b1)/2 ≈ {beta_values[-1]:.6f}")
    print(beta_values)
    print(a_values)
    print(b_values)


def plot_figure_4(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)

    # Plot the solutions for the last three iterations
    # plt.plot(r, solutions[-1][1], label=f"β = (a9+b9)/2 ≈ {beta_values[-1]:.6f}, {len(solutions)-1}")
    # plt.plot(r, solutions[-2][1], label=f"β = a9 ≈ {beta_values[-2]:.6f}, {len(solutions)-2}")
    # plt.plot(r, solutions[-3][1], label=f"β = b9 ≈ {beta_values[-3]:.6f}, {len(solutions)-3}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])


def test():
    b = 4.150390625
    a = 4.140625
    beta = 4.1601562500
    r, u = shoot(4.1601562500 , 20)
    plt.plot(r, u, label=f"beta = {4.1601562500 }")

    beta = (4.1601562500 + 4.1406250000)/2
    r, u = shoot(beta, 20)
    plt.plot(r, u, label=f"beta {beta} 2")

    r, u = shoot((a + b) / 2, 20)
    # plt.plot(r, u, label=f"beta = (a+b)/2 = {(a + b) / 2}")

    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()


def plot_excited(R, max_iterations, k):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(k, R, max_iterations)

    # Plot the solutions for the last three iterations
    plt.plot(r, solutions[-1][1], label=f"k = 0, {len(solutions)-1}")
    plt.plot(r, solutions[-2][1], label=f"k = 0, {len(solutions)-2}")
    plt.plot(r, solutions[-3][1], label=f"k = 0, {len(solutions)-3}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, R)
    plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])

def plot_figure_5(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta0, r0, u0, a_values0, b_values0, beta_values0, solutions0 = find_ground_state(R, max_iterations)
    beta1, r1, u1, a_values1, b_values1, beta_values1, solutions1 = find_excited_state(1, R, max_iterations)
    beta2, r2, u2, a_values2, b_values2, beta_values2, solutions2 = find_excited_state(2, R, max_iterations)

    # Plot the solutions for the last three iterations
    plt.plot(r0, np.log(solutions0[-1][1]),  label=f"k = 0")
    plt.plot(r1, np.log(solutions1[-1][1]),  label=f"k = 1")
    plt.plot(r2, np.log(solutions2[-1][1]),  label=f"k = 2")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Graphs for k = 0, 1 and 2')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, R)
    #plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    # print("\na values:", [f"{a:.10f}" for a in a_values])
    # print("\nb values:", [f"{b:.10f}" for b in b_values])
    # print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])
# funcs.py
import math

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

    beta = b0
    for i in range(max_iterations):
        beta_values.append(beta)
        r, u = shoot(beta, R)
        solutions.append((r, u))
        zero_crossings = np.sum(np.diff(np.sign(u)) != 0)
        # if i in (max_iterations - 3, max_iterations - 2):
        #     plt.plot(r, u, label=f"{i} β ≈ {beta_values[i]:.10f}")
        # elif i == (max_iterations - 1):
        #     plt.plot(r, u, label=f"{i} β ≈ {beta_values[i]:.10f}", linestyle='--')
        #plt.plot(r, u, label=f"{i}: β = {beta_values[i]:.10f}, a={a_values[i]}, b={b_values[i]}, int{zero_crossings}")
        #print(f"Iteration {i + 1}: beta = {beta:.6f}, zero_crossings = {zero_crossings}")
        if zero_crossings > k:
            b = beta
        elif zero_crossings <= k:
            a = beta
        a_values.append(a)
        b_values.append(b)

        beta = (a + b) / 2

        print(f"{i:9d} | {a_values[i]:.10f} | {b_values[i]:.10f} | {beta_values[i]:.10f} | {zero_crossings}")
    #print(f"Max iterations reached. Last beta = {beta:.6f}")
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
    plt.plot(r_8, u_8, label=f'8th iteration, β ≈ {beta_values[7]:.10f}')

    # Plot 9th iteration
    r_9, u_9 = solutions[8]  # 9th iteration (index 8)
    plt.plot(r_9, u_9, label=f'9th iteration, β ≈ {beta_values[8]:.10f}')

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


def plot_figure_3(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)
    r, u = shoot(5, R)
    # Plot the solutions for the last three iterations
    # plt.plot(r, u, label=f"β = (a1+b1)/2 ≈ {beta_values[-2]:.6f}")
    # plt.plot(r, solutions[-2][1], label=f"β = b1 ≈ {5:.6f}")
    # plt.plot(r, solutions[-3][1], label=f"β = a1 ≈ {beta_values[-3]:.6f}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 2nd iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()

    print(f"a1 ≈ {a_values[-1]:.6f}")
    print(f"b1 ≈ {b_values[-1]:.6f}")
    print(f"(a1+b1)/2 ≈ {beta_values[-1]:.6f}")
    print(beta_values)
    print(a_values)
    print(b_values)


def plot_figure_4(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(2, R, max_iterations)

    # Plot the solutions for the last three iterations
    # plt.plot(r, solutions[-1][1], label=f"β = (a9+b9)/2 ≈ {beta_values[-1]:.6f}, {len(solutions)-1}")
    # plt.plot(r, solutions[-2][1], label=f"β = a9 ≈ {beta_values[-2]:.6f}, {len(solutions)-2}")
    # plt.plot(r, solutions[-3][1], label=f"β = b9 ≈ {beta_values[-3]:.6f}, {len(solutions)-3}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])


def test():
    b = 4.150390625
    a = 4.140625
    beta = 4.1601562500
    r, u = shoot(4.1601562500 , 20)
    plt.plot(r, u, label=f"beta = {4.1601562500 }")

    beta = (4.1601562500 + 4.1406250000)/2
    r, u = shoot(beta, 20)
    plt.plot(r, u, label=f"beta {beta} 2")

    r, u = shoot((a + b) / 2, 20)
    # plt.plot(r, u, label=f"beta = (a+b)/2 = {(a + b) / 2}")

    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(-3, 5)
    plt.show()


def plot_excited(R, max_iterations, k):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta, r, u, a_values, b_values, beta_values, solutions = find_excited_state(k, R, max_iterations)

    # Plot the solutions for the last three iterations
    plt.plot(r, solutions[-1][1], label=f"k = 0, {len(solutions)-1}")
    plt.plot(r, solutions[-2][1], label=f"k = 0, {len(solutions)-2}")
    plt.plot(r, solutions[-3][1], label=f"k = 0, {len(solutions)-3}")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Determination of the 2nd excited state: 10th iteration')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, R)
    plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    print("\na values:", [f"{a:.10f}" for a in a_values])
    print("\nb values:", [f"{b:.10f}" for b in b_values])
    print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])

def plot_figure_5b(R, max_iterations):
    #plt.figure(figsize=(12, 8))

    # Find the 2nd excited state (k=2)
    beta0, r0, u0, a_values0, b_values0, beta_values0, solutions0 = find_ground_state(R, max_iterations)
    beta1, r1, u1, a_values1, b_values1, beta_values1, solutions1 = find_excited_state(1, R, max_iterations)
    beta2, r2, u2, a_values2, b_values2, beta_values2, solutions2 = find_excited_state(2, R, max_iterations)

    # Plot the solutions for the last three iterations
    plt.plot(r0, np.log(solutions0[-1][1]),  label=f"k = 0")
    plt.plot(r1, np.log(solutions1[-1][1]),  label=f"k = 1")
    plt.plot(r2, np.log(solutions2[-1][1]),  label=f"k = 2")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Graphs for k = 0, 1 and 2')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, R)
    #plt.ylim(-2, 5)
    plt.show()

    # print(f"a9 ≈ {a_values[-1]:.6f}")
    # print(f"b9 ≈ {b_values[-1]:.6f}")
    # print(f"(a9+b9)/2 ≈ {beta_values[-1]:.6f}")
    # print("\na values:", [f"{a:.10f}" for a in a_values])
    # print("\nb values:", [f"{b:.10f}" for b in b_values])
    # print("\nbeta values:", [f"{beta:.10f}" for beta in beta_values])
def plot_figure_5c(R, max_iterations):
    # Find the ground state and first two excited states
    beta0, r0, u0, _, _, _, _ = find_ground_state(R, max_iterations)
    beta1, r1, u1, _, _, _, _ = find_excited_state(1, R, max_iterations)
    beta2, r2, u2, _, _, _, _ = find_excited_state(2, R, max_iterations)

    # Plot the solutions
    plt.figure(figsize=(10, 8))
    plt.plot(r0, np.log(np.abs(u0)), label="k = 0")
    plt.plot(r1, np.log(np.abs(u1)), label="k = 1")
    plt.plot(r2, np.log(np.abs(u2)), label="k = 2")

    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Graphs for k = 0, 1 and 2')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 14.7)
    plt.ylim(-15, 5)
    plt.show()

    # Print the values at the origin for each state
    print(f"Ground state (k = 0) value at origin: {u0[0]:.6f}")
    print(f"First excited state (k = 1) value at origin: {u1[0]:.6f}")
    print(f"Second excited state (k = 2) value at origin: {u2[0]:.6f}")