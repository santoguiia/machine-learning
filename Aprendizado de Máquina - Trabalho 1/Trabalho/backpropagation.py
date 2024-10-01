import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10
MAX_ITER = 100
LEARNING_RATE = 0.2
INPUT_SIZE = 2
HIDDEN_SIZES = [2]
OUTPUT_SIZE = 1
FILE_PATH = 'classification2.txt'

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, seed=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.seed = seed

        self.weights = []
        self.biases = []
        self.initialize_weights()
    
    def initialize_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = []
        self.biases = []
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]))
            self.biases.append(np.random.randn(1, sizes[i + 1]))
    
    def train(self, X, y, epochs, max_iter, learning_rate):
        best_loss = float('inf')
        best_weights = None
        best_biases = None
        for epoch in range(epochs):
            for _ in range(max_iter):
                activations = self.forward(X)
                grads_w, grads_b = self.backward(X, y, activations)
                self.update(grads_w, grads_b, learning_rate)
            loss = np.mean(np.square(activations[-1] - y))
            print(f"Epoch {epoch}: Loss {loss}")
            if loss < best_loss:
                best_loss = loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
        self.weights = best_weights
        self.biases = best_biases
        print(f"Best Loss: {best_loss}")
        print(f"Best Weights: {best_weights}")

    def forward(self, X):
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = np.dot(X, w) + b
            X = np.tanh(X)
            activations.append(X)
        activations[-1] = 1 / (1 + np.exp(-activations[-1]))
        return activations
    
    def backward(self, X, y, activations):
        deltas = [None] * len(self.weights)
        deltas[-1] = activations[-1] - y
        for i in range(len(deltas) - 2, -1, -1):
            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * (1 - np.power(activations[i + 1], 2))
        grads_w = [np.dot(activations[i].T, deltas[i]) / len(X) for i in range(len(self.weights))]
        grads_b = [np.mean(deltas[i], axis=0, keepdims=True) for i in range(len(self.biases))]
        return grads_w, grads_b
    
    def update(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]
    
    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    return X, y

def plot_decision_boundary(nn, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(grid_input)
    Z = np.where(Z > 0.5, 1, 0)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.title('Decision Boundary')
    plt.show()

def numerical_gradient(nn, X, y, epsilon=1e-5):
    num_grads_w = []
    num_grads_b = []

    for i in range(len(nn.weights)):
        grad_w = np.zeros_like(nn.weights[i])
        for j in range(grad_w.size):
            original_value = nn.weights[i].flat[j]
            nn.weights[i].flat[j] = original_value + epsilon # Perturba o peso positivo
            loss_plus = np.mean(np.square(nn.forward(X)[-1] - y))
            nn.weights[i].flat[j] = original_value - epsilon # Perturba o peso negativo
            loss_minus = np.mean(np.square(nn.forward(X)[-1] - y))
            grad_w.flat[j] = (loss_plus - loss_minus) / (2 * epsilon) # Derivada numérica
            nn.weights[i].flat[j] = original_value
        num_grads_w.append(grad_w)

    for i in range(len(nn.biases)):
        grad_b = np.zeros_like(nn.biases[i])
        for j in range(grad_b.size):
            original_value = nn.biases[i].flat[j]
            nn.biases[i].flat[j] = original_value + epsilon
            loss_plus = np.mean(np.square(nn.forward(X)[-1] - y))
            nn.biases[i].flat[j] = original_value - epsilon
            loss_minus = np.mean(np.square(nn.forward(X)[-1] - y))
            grad_b.flat[j] = (loss_plus - loss_minus) / (2 * epsilon)
            nn.biases[i].flat[j] = original_value
        num_grads_b.append(grad_b)
    return num_grads_w, num_grads_b


X, y = load_data(FILE_PATH)
nn = NeuralNetwork(input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE, seed=42)
nn.train(X, y, epochs=EPOCHS, max_iter=MAX_ITER, learning_rate=LEARNING_RATE)

grads_w, grads_b = nn.backward(X, y, nn.forward(X))
num_grads_w, num_grads_b = numerical_gradient(nn, X, y)

for i in range(len(grads_w)):
    print(f"Layer {i} - Weights Gradient Error: {np.max(np.abs(grads_w[i] - num_grads_w[i]))}")

for i in range(len(grads_b)):
    print(f"Layer {i} - Biases Gradient Error: {np.max(np.abs(grads_b[i] - num_grads_b[i]))}")


plot_decision_boundary(nn, X, y)