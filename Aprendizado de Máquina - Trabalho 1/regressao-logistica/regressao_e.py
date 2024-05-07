import numpy as np
import matplotlib.pyplot as plt

# Conjunto de dados {(x,y)}
mean0, std0 = -0.4, 0.5
mean1, std1 = 0.9, 0.3
m = 200

x1s = np.random.randn(m // 2) * std1 + mean1
x0s = np.random.randn(m // 2) * std0 + mean0
xs = np.hstack((x1s, x0s))

ys = np.hstack((np.ones(m // 2), np.zeros(m // 2)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, theta):
    return sigmoid(np.dot(x, theta))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def J(theta, xs, ys):
    return cost(h(xs, theta), ys)

# Criar uma grade de valores de theta
theta0_values = np.linspace(-5, 5, 100)
theta1_values = np.linspace(-5, 5, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_values, theta1_values)
cost_grid = np.zeros_like(theta0_grid)

# Calcular o custo para cada combinação de theta
for i in range(len(theta0_values)):
    for j in range(len(theta1_values)):
        theta = np.array([theta0_values[i], theta1_values[j]])
        cost_grid[j, i] = J(theta, np.c_[np.ones(xs.shape), xs], ys)

# Plotar a função de custo em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_grid, theta1_grid, cost_grid, cmap='viridis')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Custo')
ax.set_title('Função de custo J(Theta)')
plt.show()

