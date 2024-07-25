import numpy as np
from funcs import rk4, f, count_sign_changes, plot
import matplotlib.pyplot as plt

a = []
b = []
a.append(0)
b.append(4)

iter_count = 11
k = 0

r0 = 0.01
R = 10
h = 0.01

epsilone = 0.01

du0 = 0

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

    if sign_change_count > k:
        b.append(beta)
        a.append(a[len(a) - 1])
    else:
        a.append(beta)
        b.append(b[len(b) - 1])

    if i > 8:
        if beta == a[9]:
            plot(r_vector, u_vector, color='r', label='Iteration 8')
        elif beta == b[9]:
            plot(r_vector, u_vector, color='g', label='Iteration 9')
        elif beta == (a[9] + b[9]) / 2:
            plot(r_vector, u_vector, color='b', label='Iteration 10')

print(a)
print(b)
print(beta)

plot(r_vector, u_vector, color='k', label='Final Iteration')
plt.legend()
plt.show()
