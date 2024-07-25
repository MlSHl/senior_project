import numpy as np

d = 2
alpha = 1
sigma = 1


# Solving the original (2.1) equation from the paper
# Parameters, as in their example: α = σ = 1 an d = 2, set above
# First splitting into a system of first order equations
def f(r, y):
    y1, y2 = y
    dy1dr = y2
    dy2dr = -y2 * (d - 1) / r + y1 - alpha * abs(y1) ** (2 * sigma) * y1
    return np.array([dy1dr, dy2dr])


# Implement the RK4 method
# Here f is a function defined above ^
# Return 2 vectors or r and y vector
def rk4(f, y0, r0, rf, h):
    n = int((rf - r0) / h)
    r = r0
    y = y0
    r_vector = [r]
    y_vector = [y]

    for _ in range(n):
        k1 = h * f(r, y)
        k2 = h * f(r + h / 2, y + k1 / 2)
        k3 = h * f(r + h / 2, y + k2 / 2)
        k4 = h * f(r + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        r = r + h

        r_vector.append(r)
        y_vector.append(y)

    return np.array(r_vector), np.array(y_vector)


def count_sign_changes(vector):
    # Ensure the vector is a numpy array
    vector = np.array(vector)

    # Calculate the signs of the elements in the vector
    signs = np.sign(vector)

    # Count the number of sign changes
    sign_changes = np.diff(signs)

    # The number of sign changes is the number of non-zero elements in sign_changes
    return np.sum(sign_changes != 0)
