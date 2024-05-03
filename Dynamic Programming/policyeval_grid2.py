import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(height, width):
    return np.zeros((height, width))

def evaluate_policy(grid, terminals):
    height, width = grid.shape
    new_grid = np.zeros_like(grid)
    delta = 0
    for i in range(height):
        for j in range(width):
            if (i, j) in terminals:
                continue
            old_val = grid[i, j]
            east_val = -1 + grid[min(i + 1, height - 1), j]
            west_val = -1 + grid[max(i - 1, 0), j]
            north_val = -1 + grid[i, min(j + 1, width - 1)]
            south_val = -1 + grid[i, max(j - 1, 0)]
            new_val = (east_val + west_val + north_val + south_val) / 4
            new_grid[i, j] = new_val
            delta = max(delta, abs(new_val - old_val))
    return new_grid, delta

def display_grid(grid):
    for row in grid:
        for value in row:
            print("%.3f\t" % value, end="")
        print()

height, width = 4, 4
grid = initialize_grid(height, width)

terminals = {(0, 0), (3, 3)}

delta_threshold = 0.01
delta = float('inf')
iterations = 0

while delta > delta_threshold:
    grid, delta = evaluate_policy(grid, terminals)
    iterations += 1

print(f"Converged after {iterations} iterations.")
print("Estimated values of the states (v_pi):")
display_grid(grid)
