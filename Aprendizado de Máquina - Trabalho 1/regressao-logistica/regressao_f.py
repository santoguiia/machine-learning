import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Conjunto de dados {(x,y)}
mean0, std0 = -0.4, 0.5
mean1, std1 = 0.9, 0.3
m = 200

x1s = np.random.randn(m // 2) * std1 + mean1
x2s = np.random.randn(m // 2) * std1 + mean1
x1s_0 = np.random.randn(m // 2) * std0 + mean0
x2s_0 = np.random.randn(m // 2) * std0 + mean0

xs = np.vstack((np.hstack((x1s, x1s_0)), np.hstack((x2s, x2s_0))))
ys = np.hstack((np.ones(m // 2), np.zeros(m // 2)))

# Plotar os dados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs[0, :], xs[1, :], ys)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, theta):
    return sigmoid(np.dot(theta.T, x))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def J(theta, xs, ys):
    return cost(h(xs, theta), ys)

def gradient(theta, xs, ys):
    return np.dot(xs, (h(xs, theta) - ys).T) / ys.size

alpha = 0.01
epochs = 2000
theta = np.random.randn(3)  # Inicialização aleatória dos parâmetros (incluindo o termo de bias)

cost_history = []

# Adicionar um termo de bias (1) às características
xs = np.vstack((np.ones(xs.shape[1]), xs))

for k in range(epochs):
    # Aplicar descida de gradiente
    theta -= alpha * gradient(theta, xs, ys)

    # Registrar custo
    cost_history.append(J(theta, xs, ys))

# Plotar função de custo ao longo das épocas
plt.plot(cost_history)
plt.title('Função de custo ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()

# Imprimir informações do modelo final
print('Parâmetros do Modelo:', theta)

# Plotar a fronteira de decisão
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs[1, :], xs[2, :], ys)

x1_vals = np.linspace(-3, 3, 100)
x2_vals = np.linspace(-3, 3, 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
h_grid = sigmoid(theta[0] + theta[1] * x1_grid + theta[2] * x2_grid)

ax.plot_surface(x1_grid, x2_grid, h_grid, alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()
