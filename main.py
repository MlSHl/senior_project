import matplotlib.pyplot as plt
import numpy as np

from funcs import rk4, f, count_sign_changes

# an and bn variables, 2.1 line 4 of the paper
# CHANGES: make a and b vectors
a = []
b = []
a.append(0)
b.append(4)

d = 2
alpha = 1
sigma = 1

# Arbitrary number of iteration after which we stop
iter_count = 10
k = 0

# Interval and step size
r0 = 0.01
R = 10
h = 0.01

# Epsilone in section 2.1 line 6
epsilone = 0.01

# Initial conditions
beta = (a[0] + b[0]) / 2
du0 = 0

y0 = np.array([beta, du0])

u_at_R_last_time = 1

for i in range(iter_count):
    beta = (a[len(a) - 1] + b[len(b) - 1]) / 2
    y0 = np.array([beta, du0])
    r_vector, y_vector = rk4(f, y0, r0, R, h)

    u_vector = y_vector[:, 0]
    u_len = len(u_vector) - 1
    u_at_R = u_vector[u_len]
    u_at_R_last_time = u_at_R

    sign_change_count = count_sign_changes(u_vector)
    print(f" TEST  + {sign_change_count}")
    # If u at R has the same sign as last time
    if sign_change_count > k:
        b.append(beta)
        a.append(a[len(a) - 1])
    else:
        a.append(beta)
        b.append(b[len(b) - 1])

# Logging
print(a)
print(b)
print(beta)

# Plot the solution
plt.plot(r_vector, u_vector)
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Solution of u\'\'(r) + (d-1) u\'(r)/r - u(r) +α|u(r)|^(2σ) u(r) = 0 using RK4')
plt.grid(True)
plt.show()
