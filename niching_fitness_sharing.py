import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sin(10 * np.pi * x) * x + np.cos(2 * np.pi * x) * x


def create_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)


# TODO: Implement fitness sharing
def fitness_sharing(population, fitness, sigma_share, alpha):
    '''
    Parameters:
    population (np.array): Array of population values
    fitness (np.array): Array of fitness values
    sigma_share (float): Sharing distance
    alpha (float): Sharing exponent

    Return:
    shared_fitness
    '''
    niche_count = np.zeros(len(population))

    for i in range(len(population)):
        for j in range(len(population)):
            distance = np.abs(population[i] - population[j])
            if distance < sigma_share:
                niche_count[i] += 1 - (distance / sigma_share) ** alpha

    shared_fitness = fitness / niche_count
    return shared_fitness


# TODO: Implement your own parent selection operator
def tournament_selection(population, fitness, tournament_size):
    indices = np.random.choice(len(population), tournament_size)
    best_index = indices[np.argmax(fitness[indices])]
    return population[best_index]


# TODO: Implement your own crossover operator
def crossover(parent1, parent2):
    return (parent1 + parent2) / 2


# TODO: Implement your own mutation operator
def mutation(offspring, mutation_rate, lower_bound, upper_bound):
    if np.random.random() < mutation_rate:
        offspring = np.random.uniform(lower_bound, upper_bound)
    return offspring


# TODO: Implement main genetic algorithm process
def run_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                          mutation_rate, tournament_size, sigma_share, alpha, plot_generation=10):
    # Initialize population
    population = create_population(population_size, lower_bound, upper_bound)

    # Run GA
    for generation in range(generations):
        fitness = objective_function(population)
        shared_fitness = fitness_sharing(population, fitness, sigma_share, alpha)

        new_population = []
        for _ in range(population_size):
            # TODO: Implement evolutionary process
            parent1 = tournament_selection(population, shared_fitness, tournament_size)
            parent2 = tournament_selection(population, shared_fitness, tournament_size)

            offspring = crossover(parent1, parent2)
            offspring = mutation(offspring, mutation_rate, lower_bound, upper_bound)

            new_population.append(offspring)

        population = np.array(new_population)

        if generation % plot_generation == 0:
            plot_population(population, generation)

    return population


def plot_population(population, generation):
    x = np.linspace(lower_bound, upper_bound, 1000)
    y = objective_function(x)
    plt.plot(x, y, label="Objective function")

    plt.scatter(population, objective_function(population), color="red", label=f"Population (Gen {generation})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# GA parameters
population_size = 100
generations = 100
lower_bound = 0
upper_bound = 1
mutation_rate = 0.1
tournament_size = 5
sigma_share = 0.1
alpha = 1

population = run_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                                   mutation_rate, tournament_size, sigma_share, alpha, plot_generation=10)

# Plot results
x = np.linspace(lower_bound, upper_bound, 1000)
y = objective_function(x)
plt.plot(x, y, label="Objective function")

plt.scatter(population, objective_function(population), color="red", label="Final population")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
