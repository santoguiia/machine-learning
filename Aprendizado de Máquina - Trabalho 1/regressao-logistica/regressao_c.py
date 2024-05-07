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
epochs = 2000
theta = np.random.randn(2)  # Inicialização aleatória dos parâmetros

cost_history = []
accuracy_history = []
boundary_history = []

for k in range(epochs):
    # Aplicar descida de gradiente
    theta -= alpha * gradient(theta, np.c_[np.ones(xs.shape), xs], ys)

    # Registrar custo
    cost_history.append(J(theta, np.c_[np.ones(xs.shape), xs], ys))

    # Calcular acurácia
    predictions = (h(np.c_[np.ones(xs.shape), xs], theta) >= 0.5).astype(int)
    accuracy_history.append(accuracy(ys, predictions))

    # Calcular a localização da fronteira
    boundary_history.append(-theta[0] / theta[1])

# Plotar função de custo ao longo das épocas
plt.plot(cost_history)
plt.title('Função de custo ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()

# Plotar acurácia ao longo das épocas
plt.plot(accuracy_history)
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.show()

# Plotar localização da fronteira ao longo das épocas
plt.plot(boundary_history)
plt.title('Localização da fronteira ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Localização da fronteira')
plt.show()

# Imprimir informações do modelo final
print_modelo(theta, np.c_[np.ones(xs.shape), xs], ys)
plot_fronteira(theta)
