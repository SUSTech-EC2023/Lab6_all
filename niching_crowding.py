import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sin(10 * np.pi * x) * x + np.cos(2 * np.pi * x) * x


def create_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)


# TODO: Implement your selection operator
def tournament_selection(population, fitness, tournament_size):
    indices = np.random.choice(len(population), tournament_size)
    best_index = indices[np.argmax(fitness[indices])]
    return population[best_index]


# TODO: Implement your crossover operator
def crossover(parent1, parent2):
    return (parent1 + parent2) / 2


# TODO: Implement your mutation operator
def mutation(offspring, mutation_rate, lower_bound, upper_bound):
    if np.random.random() < mutation_rate:
        offspring = np.random.uniform(lower_bound, upper_bound)
    return offspring


# TODO: Implement main genetic algorithm process with crowding
def deterministic_crowding(population, fitness, mutation_rate, lower_bound, upper_bound):
    """
    Parameters:
    population (np.array): Array of population values
    fitness (np.array): Array of fitness values
    mutation_rate (float): Mutation rate
    lower_bound (float): Lower bound of search space
    upper_bound (float): Upper bound of search space
    """
    new_population = np.copy(population)

    for i in range(len(population) // 2):
        parent1 = tournament_selection(population, fitness, 2)
        parent2 = tournament_selection(population, fitness, 2)

        offspring1 = crossover(parent1, parent2)
        offspring1 = mutation(offspring1, mutation_rate, lower_bound, upper_bound)

        offspring2 = crossover(parent2, parent1)
        offspring2 = mutation(offspring2, mutation_rate, lower_bound, upper_bound)

        d1 = np.abs(parent1 - offspring1) + np.abs(parent2 - offspring2)
        d2 = np.abs(parent1 - offspring2) + np.abs(parent2 - offspring1)

        offspring1_fitness = objective_function(offspring1)
        offspring2_fitness = objective_function(offspring2)

        if d1 < d2:
            if offspring1_fitness > fitness[np.where(population == parent1)[0][0]]:
                new_population[np.where(population == parent1)[0][0]] = offspring1
            if offspring2_fitness > fitness[np.where(population == parent2)[0][0]]:
                new_population[np.where(population == parent2)[0][0]] = offspring2
        else:
            if offspring1_fitness > fitness[np.where(population == parent2)[0][0]]:
                new_population[np.where(population == parent2)[0][0]] = offspring1
            if offspring2_fitness > fitness[np.where(population == parent1)[0][0]]:
                new_population[np.where(population == parent1)[0][0]] = offspring2

    return new_population


def run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate, plot_generation=10):
    population = create_population(population_size, lower_bound, upper_bound)

    for generation in range(generations):
        fitness = objective_function(population)

        population = deterministic_crowding(population, fitness, mutation_rate, lower_bound, upper_bound)

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

population = run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate,
                                   plot_generation=10)

# Plot results
x = np.linspace(lower_bound, upper_bound, 1000)
y = objective_function(x)
plt.plot(x, y, label="Objective function")

plt.scatter(population, objective_function(population), color="red", label="Final population")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
