import numpy as np
import matplotlib.pyplot as plt

d = 2
alpha = 1
sigma = 1


def f(r, y):
    y1, y2 = y
    dy1dr = y2
    dy2dr = -y2 * (d - 1) / r + y1 - alpha * abs(y1) ** (2 * sigma) * y1
    return np.array([dy1dr, dy2dr])


def rk4(f, y, r0, rf, h):
    n = int((rf - r0) / h)
    r = r0
    # y = y0
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
    vector = np.array(vector)
    signs = np.sign(vector)
    sign_changes = np.diff(signs)
    return np.sum(sign_changes != 0)


def plot(r_vector, u_vector, color, label):
    plt.plot(r_vector, u_vector, color=color, label=label)
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('Solution of u\'\'(r) + (d-1) u\'(r)/r - u(r) +α|u(r)|^(2σ) u(r) = 0 using RK4')
    plt.grid(True)
