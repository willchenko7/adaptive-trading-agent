import numpy as np
from forward import forward, sigmoid

def initialize_population(pop_size, layer_sizes):
    population = []
    for _ in range(pop_size):
        w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size) for i, size in enumerate(layer_sizes)]
        b = [np.random.rand(size) for size in layer_sizes]
        population.append((w, b))
    return population

def compute_fitness(solution, x):
    w, b = solution
    output = forward(x, w, b, n_layers)
    return output  # Since the goal is to minimize the output

def select_parents(population, fitnesses, num_parents):
    parents = list(np.argsort(fitnesses)[:num_parents])
    return [population[p] for p in parents]

def crossover(parent1, parent2):
    child_w = [(w1 + w2) / 2 for w1, w2 in zip(parent1[0], parent2[0])]
    child_b = [(b1 + b2) / 2 for b1, b2 in zip(parent1[1], parent2[1])]
    return child_w, child_b

def mutate(solution, mutation_rate):
    w, b = solution
    mutated_w = [w_layer + np.random.randn(*w_layer.shape) * mutation_rate for w_layer in w]
    mutated_b = [b_layer + np.random.randn(*b_layer.shape) * mutation_rate for b_layer in b]
    return mutated_w, mutated_b

def ea(n_layers,input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,x):
    # Initialize population
    population = initialize_population(pop_size, layer_sizes)

    # Evolution
    for generation in range(num_generations):
        # Compute fitness for each solution
        fitnesses = [compute_fitness(sol, x) for sol in population]

        # Select parents
        parents = select_parents(population, fitnesses, num_parents)

        # Generate next generation
        next_generation = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child = crossover(parents[i], parents[i + 1])
                child = mutate(child, mutation_rate)
                next_generation.append(child)
        
        # Replace worst solutions with new ones
        worst_indices = np.argsort(fitnesses)[-len(next_generation):]
        for idx, new_sol in zip(worst_indices, next_generation):
            population[idx] = new_sol

        print(f"Generation {generation}: Best Fitness = {np.min(fitnesses)}")

    # Best solution
    best_index = np.argmin([compute_fitness(sol, x) for sol in population])
    best_solution = population[best_index]
    print("Best Solution:", best_solution)
    return

if __name__ == "__main__":
    # Parameters
    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    pop_size = 50  # Population size
    num_generations = 1000  # Number of generations
    num_parents = 10  # Number of parents for crossover
    mutation_rate = 0.01  # Mutation rate
    x = np.random.rand(2, 1000)  # Input data
    #normalizing input data
    x = (x - np.mean(x)) / np.std(x)
    ea(n_layers,input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,x)