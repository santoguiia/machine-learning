# Guilherme dos Santos
# Utilizei o template fornecido pelo professor para implementar o algoritmo genético

import random
import math
import struct
import matplotlib.pyplot as plt

# Constants
L = 4 * 8  # size of chromosome in bits

POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 100

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

def bitsToFloat(b):
    s = struct.pack('>L', b)
    return struct.unpack('>f', s)[0]

# Exemplo:  1.23 -> '00010111100'
def get_bits(x):
    x = floatToBits(x)
    N = 4 * 8
    bits = ''
    for bit in range(N):
        b = x & (2**bit)
        bits += '1' if b > 0 else '0'
    return bits

# Exemplo:  '00010111100' ->  1.23
def get_float(bits):
    x = 0
    assert(len(bits) == L)
    for i, bit in enumerate(bits):
        bit = int(bit)  # 0 or 1
        x += bit * (2**i)
    return bitsToFloat(x)

# Function to maximize
def fitness_function(y):
    return y + abs(math.sin(32 * y))

def generate_population(size):
    return [float(random.uniform(0, math.pi)) for _ in range(size)]

# Crossover operator
def crossover(parent1, parent2):
    point = random.randint(0, L - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation operator
def mutate(chromosome):
    mutated_chromosome = list(chromosome)
    for i in range(L):
        if random.random() < MUTATION_RATE:
            mutated_chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(mutated_chromosome)

# Genetic algorithm
def genetic_algorithm():
    population = generate_population(POPULATION_SIZE)
    best_fitness_values = []  # List to store the best fitness value in each generation
    for generation in range(GENERATIONS):
        next_generation = []
        # Elitism: keep the best individual
        best_individual = max(population, key=fitness_function)
        best_fitness_values.append(fitness_function(best_individual))
        next_generation.append(best_individual)
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(get_bits(parent1), get_bits(parent2))
                child1 = mutate(child1)
                child2 = mutate(child2)
                next_generation.extend([get_float(child1), get_float(child2)])
            else:
                next_generation.extend([parent1, parent2])
        population = next_generation
    return best_fitness_values  # Return the list of best fitness values

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

# Running the genetic algorithm
best_fitness_values = genetic_algorithm()
plot_evolution(best_fitness_values)
plot_fitness_function()
