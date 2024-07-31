# main.py

import numpy as np
import matplotlib.pyplot as plt
from funcs import find_ground_state

# Find the ground state
beta_star, r, u, a_values, b_values, beta_values, solutions = find_ground_state()

# Print the coefficients for each iteration
print("Iteration | a | b | beta")
print("-" * 50)
for i, (a, b, beta) in enumerate(zip(a_values, b_values, beta_values)):
    print(f"{i:9d} | {a:.10f} | {b:.10f} | {beta:.10f}")

# Plot the results for specific iterations
plt.figure(figsize=(12, 8))

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