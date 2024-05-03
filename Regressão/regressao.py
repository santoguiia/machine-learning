print('Guilherme dos Santos')

import numpy as np
import matplotlib.pyplot as plt

def f_true(x):
    return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = f_true(xs) + np.random.randn(100) * 0.5

# hipotese
def h(x, theta):
    theta = theta[0] + theta[1] * x
    return theta

# funcao de cost
def J(theta, xs, ys):
    media = np.mean((h(xs, theta) - ys) ** 2) / 2
    return media

# derivada parcial
def gradient(i, theta, xs, ys):
    if i == 0:
        return np.mean(h(xs, theta) - ys)
    elif i == 1:
        return np.mean((h(xs, theta) - ys) * xs)

# plot do modelo
def print_modelo(theta, xs, ys):
    plt.scatter(xs, ys, label='ruido')
    plt.plot(xs, f_true(xs), label='real function')
    plt.plot(xs, h(xs, theta), label='hipotese')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('model x real function')
    plt.show()

# parametros
theta = (np.random.randn(2))
lr = 0.0001 # taxa de aprendizado
epochs = 5000
cost = []

# gradient descent
for i in range(epochs):
    theta -= lr * np.array([gradient(j, theta, xs, ys) for j in range(2)])
    cost.append(J(theta, xs, ys))

print('theta:', theta)
print('cost:', cost[-1])
print_modelo(theta, xs, ys)
plt.plot(cost, '.-')
plt.show()