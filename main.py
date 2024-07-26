import numpy as np
from funcs import rk4, f, count_sign_changes, plot
import matplotlib.pyplot as plt

a= {0: 0}
b= {0: 4}

iter_count = 11
k = 0

r0 = 0.01
R = 10
h = 0.01

epsilone = 0.01

du0 = 0

beta = {0: 4}

u_at_R_last_time = 1
r_vector, u_vector = 0, 0
# funky count
for i in range(iter_count):
    #print(i)
    y0 = np.array([beta.get(i), du0])
    r_vector, y_vector = rk4(f, y0, r0, R, h)

    u_vector = y_vector[:, 0]
    u_len = len(u_vector) - 1
    u_at_R = u_vector[u_len]
    u_at_R_last_time = u_at_R

    sign_change_count = count_sign_changes(u_vector)
    print(f" Sign changes: {sign_change_count}")

    if sign_change_count > k:
        b[i] = beta[i]
        a[i] = a[len(a) - 1]
    else:
        a[i] = beta[i]
        b[i] = b[len(b) - 1]

    beta[i+1] = (a[len(a) - 1] + b[len(b) - 1]) / 2

    if i == 7:
        plot(r_vector, u_vector, 'r', '7')

    if i == 8:
        plot(r_vector, u_vector, 'g', '8')

    # if i == 9:
    #     plot(r_vector, u_vector, 'b', '8')

    if i == 10:
        plot(r_vector, u_vector, 'y', '9')

print(f"a:    {a}")
print(f"b:    {b}")
print(f"beta: {beta}")

plot(r_vector, u_vector, color='k', label='Final Iteration')
plt.legend()
plt.show()
