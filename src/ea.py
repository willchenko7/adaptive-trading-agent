import numpy as np
from sim import sim
import pickle
import os

'''
goal: use evolutionary algorithm to find optimal weights to maximize fitness (fitness is based on how much money is made)

input:
    n_layers - number of layers in model
    input_size - size of input data
    layer_sizes - list of layer sizes
    pop_size - population size
    num_generations - number of generations
    num_parents - number of parents for crossover
    mutation_rate - mutation rate
    start_times - list of start times for simulation
    model_name - name of model (used for saving best solution)

output:
    saves solution as pickle file in models folder (models/best_solution_{model_name}.pkl)
'''

def mutate_initial_pop(solution, mutation_rate=0.01):
    w, b, attn_weights, attn_query, attn_keys, attn_values = solution
    mutated_w = [w_layer + np.random.randn(*w_layer.shape) * mutation_rate for w_layer in w]
    mutated_b = [b_layer + np.random.randn(*b_layer.shape) * mutation_rate for b_layer in b]
    mutated_attn_weights = attn_weights + np.random.randn(*attn_weights.shape) * mutation_rate
    mutated_attn_query = attn_query + np.random.randn(*attn_query.shape) * mutation_rate
    mutated_attn_keys = attn_keys + np.random.randn(*attn_keys.shape) * mutation_rate
    mutated_attn_values = attn_values + np.random.randn(*attn_values.shape) * mutation_rate
    return mutated_w, mutated_b, mutated_attn_weights, mutated_attn_query, mutated_attn_keys, mutated_attn_values

def initialize_population(pop_size, layer_sizes,from_file=False):
    '''
    goal: initialize population of size pop_size with random weights and biases for each layer
    -optional: initialize population based on best solution
    '''
    if from_file:
        #if we want to initialize population based on best solution
        population = []
        best_solution = pickle.load(open(os.path.join('models','best_solution.pkl'),'rb'))
        for _ in range(pop_size):
            mutated_solution = mutate_initial_pop(best_solution)
            population.append(mutated_solution)
        return population
    population = []
    buffer = 1
    for _ in range(pop_size):
        w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size)*buffer for i, size in enumerate(layer_sizes)]
        b = [np.random.rand(size)*buffer for size in layer_sizes]
        attention_layer_index = 0 
        layer_output_dim = layer_sizes[attention_layer_index]
        attn_dim = layer_output_dim 
        attn_query = np.random.rand(attn_dim).astype(np.float64)
        attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
        attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
        attn_weights = np.random.rand(attn_dim).astype(np.float64)
        population.append((w, b, attn_weights, attn_query, attn_keys, attn_values))
    return population

def compute_fitness(solution, start_times,input_size,layer_sizes):
    '''
    goal: compute fitness of solution
    -fitness is based on how much money is made in simulation
    -fitness is the average of the fitnesses of each simulation (each simulation is based on a different start time)
    -fitness is negative because we want to minimize the output
    '''
    w, b, attn_weights, attn_query, attn_keys, attn_values = solution
    n_layers = len(layer_sizes)
    fitnesses = []
    for start_time in start_times:
        fitness, next_thoughts = sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values, start_time)
        fitnesses.append(fitness)
    fitness = np.mean(fitnesses)
    print(f"Final Fitness: {fitness}")
    return fitness*-1  # Since the goal is to minimize the output

def select_parents(population, fitnesses, num_parents):
    '''
    goal: select parents for crossover
    -parents are selected based on fitness
    -the best num_parents solutions are selected as parents
    '''
    parents = list(np.argsort(fitnesses)[:num_parents])
    return [population[p] for p in parents]

def crossover(parent1, parent2):
    '''
    goal: perform crossover on parents
    -crossover is performed by taking the average of the weights and biases of the parents
    '''
    child_w = [(w1 + w2) / 2 for w1, w2 in zip(parent1[0], parent2[0])]
    child_b = [(b1 + b2) / 2 for b1, b2 in zip(parent1[1], parent2[1])]
    child_attn_weights = (parent1[2] + parent2[2]) / 2
    child_attn_query = (parent1[3] + parent2[3]) / 2
    child_attn_keys = (parent1[4] + parent2[4]) / 2
    child_attn_values = (parent1[5] + parent2[5]) / 2
    return child_w, child_b, child_attn_weights, child_attn_query, child_attn_keys, child_attn_values

def mutate(solution, mutation_rate):
    '''
    goal: mutate solution
    -mutation is performed by adding a random number to each weight and bias
    -adjust mutation rate to control how much mutation occurs
    '''
    w, b, attn_weights, attn_query, attn_keys, attn_values = solution
    mutated_w = [w_layer + np.random.randn(*w_layer.shape) * mutation_rate for w_layer in w]
    mutated_b = [b_layer + np.random.randn(*b_layer.shape) * mutation_rate for b_layer in b]
    mutated_attn_weights = attn_weights + np.random.randn(*attn_weights.shape) * mutation_rate
    mutated_attn_query = attn_query + np.random.randn(*attn_query.shape) * mutation_rate
    mutated_attn_keys = attn_keys + np.random.randn(*attn_keys.shape) * mutation_rate
    mutated_attn_values = attn_values + np.random.randn(*attn_values.shape) * mutation_rate
    return mutated_w, mutated_b, mutated_attn_weights, mutated_attn_query, mutated_attn_keys, mutated_attn_values

def ea(input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,start_times,model_name):
    '''
    goal: perform evolutionary algorithm
    '''
    # Initialize population
    population = initialize_population(pop_size, layer_sizes)
    # Evolution
    for generation in range(num_generations):
        # Compute fitness for each solution
        fitnesses = [compute_fitness(sol, start_times,input_size,layer_sizes) for sol in population]
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
    best_index = np.argmin(fitnesses)
    best_solution = population[best_index]
    print("Best Solution:", best_solution)
    print("Best Fitness:", fitnesses[best_index])
    #save best solution as pickle
    pickle.dump(best_solution, open(os.path.join('models',f'best_solution_{model_name}.pkl'),'wb'))
    return

if __name__ == "__main__":
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    pop_size = 10
    num_generations = 10
    num_parents = 10
    mutation_rate = 3.0
    start_times = ["2023-11-27 11:46:00","2023-12-01 11:46:00","2023-12-05 11:46:00","2023-12-09 11:46:00"]
    model_name = "1127-1209"
    ea(input_size,layer_sizes,pop_size,num_generations,num_parents,mutation_rate,start_times,model_name)