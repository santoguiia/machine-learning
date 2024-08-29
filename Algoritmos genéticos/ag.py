import random
import math
import matplotlib.pyplot as plt

# Parâmetros do algoritmo genético
POPULATION_SIZE = 100
GENERATIONS = 100

# Operadores do algoritmo genético
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
SELECTION_RATE = 0.5

# Função de fitness
def fitness_function(y):
    """ Função para maximizar """
    return y + abs(math.sin(32 * y))

# Geração da população
def generate_population(POPULATION_SIZE):
    """ Gera uma população de tamanho POPULATION_SIZE com cromossomos representando números float """
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = random.uniform(0, math.pi)  # Gera números float aleatórios entre -10 e 10
        population.append(chromosome)
    return population

# Operador de crossover
def crossover(parent1, parent2):
    """ Realiza crossover de dois pais para gerar dois filhos """
    return (parent1 + parent2) / 2, (parent1 + parent2) / 2

# Operador de mutação
def mutate(chromosome):
    """ Realiza mutação em um cromossomo, aplicando uma pequena variação """
    if random.random() < MUTATION_RATE:
        chromosome += random.uniform(-1.0, 1.0)  # Aplica uma pequena variação ao número
    return chromosome

# Operador de seleção
def select(population, fitness_values):
    """ Seleciona uma parte da população com base nos valores de fitness """
    selected_population = []
    for _ in range(int(SELECTION_RATE * POPULATION_SIZE)):
        idx = random.choices(range(POPULATION_SIZE), weights=fitness_values)[0]
        selected_population.append(population[idx])
    return selected_population



# Função para plotar a evolução da função de fitness
def plot_evolution(best_fitness_values):
    plt.plot(best_fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of the best fitness value')
    plt.show()

def plot_fitness_function():
    x = [i / 100 for i in range(0, int(math.pi * 100))]
    y = [fitness_function(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Fitness')
    plt.title('Fitness function')
    plt.show()



# Algoritmo genético
def genetic_algorithm():
    population = generate_population(POPULATION_SIZE)
    best_fitness_values = []
    for generation in range(GENERATIONS):
        fitness_values = [fitness_function(chromosome) for chromosome in population]
        
        # Ajusta os valores de fitness para garantir que sejam todos positivos
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]  # Desloca para cima
        
        best_fitness = max(fitness_values)
        best_fitness_values.append(best_fitness)
        print(f'Generation {generation}: best fitness = {best_fitness}')
        
        # Seleção
        population = select(population, fitness_values)
        
        # Crossover e mutação para gerar a próxima geração
        next_generation = []
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population, 2)
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            next_generation.extend([mutate(child1), mutate(child2)])
        population = next_generation[:POPULATION_SIZE]
    
    return best_fitness_values

# Running the genetic algorithm
best_fitness_values = genetic_algorithm()
plot_evolution(best_fitness_values)
plot_fitness_function()
