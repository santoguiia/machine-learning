import numpy as np
import matplotlib.pyplot as plt

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

# Imprimir informações do modelo final
print('Parâmetros do Modelo:', theta)

# Plotar a fronteira de decisão e os dados
plt.scatter(xs[1, :], xs[2, :], c=ys, cmap='bwr', edgecolors='k', label='Dados')
x1_min, x1_max = xs[1, :].min(), xs[1, :].max()
x2_min, x2_max = xs[2, :].min(), xs[2, :].max()
x1_vals = np.linspace(x1_min, x1_max, 100)
x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]  # Calcula os valores de x2 para a fronteira de decisão
plt.plot(x1_vals, x2_vals, '-r', label='Fronteira de Decisão')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Fronteira de Decisão e Dados')
plt.legend()
plt.show()
