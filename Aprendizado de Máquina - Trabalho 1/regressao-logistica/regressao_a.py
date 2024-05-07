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

plt.plot(xs[:m // 2], ys[:m // 2], '.')
plt.plot(xs[m // 2:], ys[m // 2:], '.')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, theta):
    return sigmoid(np.dot(x, theta))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def J(theta, xs, ys):
    return cost(h(xs, theta), ys)

def gradient(theta, xs, ys):
    return np.dot(xs.T, (h(xs, theta) - ys)) / ys.size

def plot_fronteira(theta):
    plt.plot(xs[:m // 2], ys[:m // 2], '.')
    plt.plot(xs[m // 2:], ys[m // 2:], '.')
    plt.plot(xs, sigmoid(np.dot(np.c_[np.ones(xs.shape), xs], theta)), '-')
    plt.show()

def print_modelo(theta, xs, ys):
    predictions = (h(xs, theta) >= 0.5).astype(int)
    print('Parâmetros do Modelo:', theta)
    print('Acurácia:', accuracy(ys, predictions))

def accuracy(ys, predictions):
    num = sum(ys == predictions)
    return num / len(ys)

alpha = 0.01
epochs = 600
theta = np.random.randn(2)  # Inicialização aleatória dos parâmetros

for k in range(epochs):
    # Aplicar descida de gradiente
    theta -= alpha * gradient(theta, np.c_[np.ones(xs.shape), xs], ys)

    # Mostrar desempenho de classificação
    if k % 100 == 0:
        print('Iteração:', k, 'Custo:', J(theta, np.c_[np.ones(xs.shape), xs], ys))

print_modelo(theta, np.c_[np.ones(xs.shape), xs], ys)
plot_fronteira(theta)
