import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    1.0, 1.5, 2.0, 2.5, 3.0,
    3.5, 4.0, 4.5, 5.0, 5.5,
    6.0, 6.5, 7.0, 7.5, 8.0
])

y = np.array([
    2.20, 3.10, 14.50, 4.90, 5.80,
    -3.50, 7.90, 8.80, 19.20, 10.90,
    11.80, 1.00, 14.70, 22.50, 16.60
])

# Linear model
X_linear = np.column_stack([x, np.ones_like(x)])
w_linear = np.linalg.inv(X_linear.T @ X_linear) @ X_linear.T @ y

# Quadratic model
X_quad = np.column_stack([x**2, x, np.ones_like(x)])
w_quad = np.linalg.inv(X_quad.T @ X_quad) @ X_quad.T @ y

print("Linear coefficients:", w_linear)
print("Quadratic coefficients:", w_quad)

# Predictions at x = 12
linear_12 = w_linear[0] * 12 + w_linear[1]
quad_12 = w_quad[0] * 12**2 + w_quad[1] * 12 + w_quad[2]

print("\nPrediction at x = 12")
print("Linear:", linear_12)
print("Quadratic:", quad_12)

# Training errors
linear_fit = X_linear @ w_linear
quad_fit = X_quad @ w_quad

mse_linear = np.mean((y - linear_fit)**2)
mse_quad = np.mean((y - quad_fit)**2)

print("\nLinear MSE:", mse_linear)
print("Quadratic MSE:", mse_quad)

# Plot both models
x_plot = np.linspace(0, 13, 500)
linear_plot = w_linear[0] * x_plot + w_linear[1]
quad_plot = w_quad[0] * x_plot**2 + w_quad[1] * x_plot + w_quad[2]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="black", label="Data", zorder=3)
plt.plot(x_plot, linear_plot, color="blue", label="Linear model")
plt.plot(x_plot, quad_plot, color="red", label="Quadratic model")

plt.scatter(12, linear_12, color="blue", marker="*", s=200,
            label="Linear prediction", zorder=4)
plt.scatter(12, quad_12, color="red", marker="*", s=200,
            label="Quadratic prediction", zorder=4)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear and quadratic regression")
plt.xlim(0, 13)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()