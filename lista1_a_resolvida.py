import random
import math
import struct

# Constants
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 1000

# Chromosome size
L = 4 * 8

# Function to maximize
def fitness_function(y):
    return y + abs(math.sin(32 * y))

# Helper functions for converting float to bits and vice versa
def float_to_bits(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

def bits_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]

# Generate initial population
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
    for generation in range(GENERATIONS):
        next_generation = []
        # Elitism: keep the best individual
        best_individual = max(population, key=fitness_function)
        next_generation.append(best_individual)
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(float_to_bits(parent1), float_to_bits(parent2))
                child1 = mutate(child1)
                child2 = mutate(child2)
                next_generation.extend([bits_to_float(child1), bits_to_float(child2)])
            else:
                next_generation.extend([parent1, parent2])
        population = next_generation
    return max(population, key=fitness_function)

# Running the genetic algorithm
best_solution = genetic_algorithm()
print("Best solution:", best_solution)
print("Fitness:", fitness_function(best_solution))
