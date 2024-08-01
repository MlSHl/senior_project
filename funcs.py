# funcs.py

import numpy as np

# Global parameters
d = 2
alpha = 1
sigma = 1
R = 20
h = 0.001  # Smaller step size for better accuracy
epsilon = 1e-6  # Small value to check if solution remains positive


def f(r, y):
    u, u_prime = y
    if r == 0:
        return np.array([u_prime, d * (u - alpha * abs(u) ** (2 * sigma) * u)])
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


def shoot(beta):
    r, y = rk4(f, np.array([beta, 0]), 0, R, h)
    return r, y[:, 0]


def find_ground_state():
    a, b = 0, 4
    a_values = [a]
    b_values = [b]
    beta_values = []
    solutions = []
    max_iterations = 10

    for i in range(max_iterations):
        beta = (a + b) / 2
        beta_values.append(beta)
        r, u = shoot(beta)
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


def find_excited_state(k):
    """
    Find the kth excited state (k=0 for ground state, k=1 for 1st excited, k=2 for 2nd excited, etc.)
    """
    a, b = 0, 10  # Increased upper bound to accommodate higher states
    a_values = [a]
    b_values = [b]
    beta_values = []
    solutions = []
    max_iterations = 10  # Increased max iterations

    for i in range(max_iterations):
        beta = (a + b) / 2
        beta_values.append(beta)
        r, u = shoot(beta)
        solutions.append((r, u))

        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(u)) != 0)

        if zero_crossings > k:
            b = beta
        elif zero_crossings < k:
            a = beta
        else:
            # Check if the solution decays to zero
            if np.abs(u[-1]) < epsilon:
                break
            else:
                b = beta

        a_values.append(a)
        b_values.append(b)

    return beta, r, u, a_values, b_values, beta_values, solutions