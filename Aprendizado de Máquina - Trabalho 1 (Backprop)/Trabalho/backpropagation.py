import numpy as np
import matplotlib.pyplot as plt

# Hiperparâmetros
EPOCHS = 1000
LEARNING_RATE = 0.4
INPUT_SIZE = 2
HIDDEN_SIZES = [4, 4]
OUTPUT_SIZE = 1
FILE_PATH = 'classification2.txt'
TEST_SIZE = 0.2  # 20% para teste
SEED = 42  # Semente para reprodutibilidade

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
            # Inicialização com distribuição normal padrão e escala reduzida
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, sizes[i + 1])))
    
    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate):
        best_loss = float('inf')
        best_weights = None
        best_biases = None
        costs = []  # Lista para armazenar os valores de perda

        for epoch in range(epochs):
            activations, zs = self.forward(X_train)
            grads_w, grads_b = self.backward(X_train, y_train, activations, zs)
            self.update(grads_w, grads_b, learning_rate)
            # Calcula a perda no conjunto de treino
            train_loss = np.mean(np.square(activations[-1] - y_train))
            # Calcula a perda no conjunto de teste
            test_activations, _ = self.forward(X_test)
            test_loss = np.mean(np.square(test_activations[-1] - y_test))

            costs.append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
            if train_loss < best_loss:
                best_loss = train_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
        # Restaura os melhores pesos e vieses
        self.weights = best_weights
        self.biases = best_biases

        print(f"Best Train Loss: {best_loss:.4f}")
        plot_decision_boundary(self, X_train, y_train, X_test, y_test)
        plot_cost_function(costs)  # Plota o gráfico da função de custo

    def forward(self, X):
        activations = [X]
        zs = []  # Armazena os valores pré-ativação
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(X, w) + b  # Pré-ativação
            zs.append(z)
            if i < len(self.weights) - 1:
                # Camadas ocultas com ativação tanh
                a = np.tanh(z) # Ativação tanh
                # a = np.maximum(0, z) # Ativação ReLU
            else:
                a = 1 / (1 + np.exp(-z))  # Última camada com ativação sigmoid
            activations.append(a)
            X = a  # Atualiza a entrada para a próxima camada
        return activations, zs

    def backward(self, X, y, activations, zs):
        deltas = [None] * len(self.weights)
        # Erro na camada de saída
        deltas[-1] = activations[-1] - y  # Derivada da MSE com sigmoid

        # Propagação do erro para as camadas ocultas
        for i in range(len(deltas) - 2, -1, -1):
            delta = np.dot(deltas[i + 1], self.weights[i + 1].T) * (1 - np.power(activations[i + 1], 2)) # Derivada da tanh
            # delta = np.dot(deltas[i + 1], self.weights[i + 1].T) * np.where(activations[i + 1] > 0, 1, 0) # Derivada da ReLU
            deltas[i] = delta

        # Gradientes para pesos e vieses
        grads_w = [np.dot(activations[i].T, deltas[i]) / len(X) for i in range(len(self.weights))]
        grads_b = [np.mean(deltas[i], axis=0, keepdims=True) for i in range(len(self.biases))]
        return grads_w, grads_b

    def update(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

def load_data(file_path, test_size=0.2, seed=None):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    
    # Normalização z-score
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Embaralhar os dados
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Dividir em treino e teste
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test

def plot_decision_boundary(nn, X_train, y_train, X_test, y_test):
    # Definir os limites do gráfico com base em todos os dados
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.1, X_all[:, 0].max() + 0.1
    y_min, y_max = X_all[:, 1].min() - 0.1, X_all[:, 1].max() + 0.1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(grid_input)
    Z = np.where(Z > 0.5, 1, 0)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    
    # Plotar os dados de treino
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap=plt.cm.Spectral, marker='o', edgecolors='k', label='Train')
    
    # Plotar os dados de teste
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap=plt.cm.Spectral, marker='x', edgecolors='k', label='Test')
    
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_cost_function(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Função de Custo (MSE)')
    plt.xlabel('Épocas')
    plt.ylabel('Custo')
    plt.title('Evolução da Função de Custo ao Longo das Épocas')
    plt.legend()
    plt.grid(True)
    plt.show()

def numerical_gradient(nn, X, y, epsilon=1e-5):
    num_grads_w = []
    num_grads_b = []

    for i in range(len(nn.weights)):
        grad_w = np.zeros_like(nn.weights[i])
        for j in range(grad_w.size):
            original_value = nn.weights[i].flat[j]
            nn.weights[i].flat[j] = original_value + epsilon
            loss_plus = np.mean(np.square(nn.predict(X) - y))
            nn.weights[i].flat[j] = original_value - epsilon
            loss_minus = np.mean(np.square(nn.predict(X) - y))
            grad_w.flat[j] = (loss_plus - loss_minus) / (2 * epsilon)
            nn.weights[i].flat[j] = original_value
        num_grads_w.append(grad_w)

    for i in range(len(nn.biases)):
        grad_b = np.zeros_like(nn.biases[i])
        for j in range(grad_b.size):
            original_value = nn.biases[i].flat[j]
            nn.biases[i].flat[j] = original_value + epsilon
            loss_plus = np.mean(np.square(nn.predict(X) - y))
            nn.biases[i].flat[j] = original_value - epsilon
            loss_minus = np.mean(np.square(nn.predict(X) - y))
            grad_b.flat[j] = (loss_plus - loss_minus) / (2 * epsilon)
            nn.biases[i].flat[j] = original_value
        num_grads_b.append(grad_b)
    return num_grads_w, num_grads_b

def __main__():
    # Carrega os dados
    X_train, y_train, X_test, y_test = load_data(FILE_PATH, test_size=TEST_SIZE, seed=SEED)

    # Inicializa a rede neural
    nn = NeuralNetwork(input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE, seed=SEED)

    # Treina a rede neural
    nn.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # Calcula os gradientes analíticos
    activations, zs = nn.forward(X_train)
    grads_w, grads_b = nn.backward(X_train, y_train, activations, zs)

    # Calcula os gradientes numéricos
    num_grads_w, num_grads_b = numerical_gradient(nn, X_train, y_train)

    # Verifica a diferença entre gradientes analíticos e numéricos
    for i in range(len(grads_w)):
        error = np.max(np.abs(grads_w[i] - num_grads_w[i]))
        print(f"Layer {i} - Weights Gradient Error: {error:.8f}")

    for i in range(len(grads_b)):
        error = np.max(np.abs(grads_b[i] - num_grads_b[i]))
        print(f"Layer {i} - Biases Gradient Error: {error:.8f}")

if __name__ == '__main__':
    __main__()