# Guilherme dos Santos

import random
import math
import struct
import matplotlib.pyplot as plt

L = 4 * 8  # size of chromosome in bits

POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 100

def floatToBits(f):
    """ Convert a float to a 32-bit integer """
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

def bitsToFloat(b):
    """ Convert a 32-bit integer to a float """
    s = struct.pack('>L', b)
    return struct.unpack('>f', s)[0]

def get_bits(x):
    """ Convert a float to a string of bits """
    x = floatToBits(x)
    N = 4 * 8
    bits = ''
    for bit in range(N):
        b = x & (2**bit)
        bits += '1' if b > 0 else '0'
    return bits

def get_float(bits):
    """ Convert a string of bits to a float """
    x = 0
    assert(len(bits) == L)
    for i, bit in enumerate(bits):
        bit = int(bit)  # 0 or 1
        x += bit * (2**i)
    return bitsToFloat(x)

def fitness_function(y):
    """ Function to maximize """
    return y + abs(math.sin(32 * y))

def generate_population(size):
    """ Generate a population of size size """
    return [float(random.uniform(0, math.pi)) for _ in range(size)]

# Crossover operator
# def crossover()

# Mutation operator
# def mutate()

# Genetic algorithm
def genetic_algorithm():
    population = generate_population(POPULATION_SIZE)
    print(population)


# Running the genetic algorithm
best_fitness_values = genetic_algorithm()
# plot_evolution(best_fitness_values)
# plot_fitness_function()