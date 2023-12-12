import numpy as np


class GeneticAlgorithm:
  def __init__(
    self, fitness_function, initial_population_generator,
        selection_function, crossover_function, mutation_function,
        population_size=100, num_generations=1000,
        crossover_probability=0.99,
        mutation_probability=None
):
    self.fitness_function = fitness_function
    self.initial_population_generator = initial_population_generator

    self.selection = selection_function
    self.crossover = crossover_function
    self.mutation = mutation_function

    self.population_size = population_size
    self.num_generations = num_generations

    self.crossover_probability = crossover_probability
    self.mutation_probability = mutation_probability if mutation_probability is not None else 1 / self.population_size

    self.population = None
    self.best_solution = None
    self.best_score = float('inf')

  def fit(self):
    # def dup(x):
    #   return list(filter(lambda items: items[1] > 1, Counter(x).items()))

    self.population = self.initial_population_generator(population_size=self.population_size)
    for generation in range(self.num_generations):
      fitness_scores = [self.fitness_function(solution=individual) for individual in self.population]
      best_idx = np.argmin(fitness_scores)

      if fitness_scores[best_idx] < self.best_score:
        self.best_score = fitness_scores[best_idx]
        self.best_solution = self.population[best_idx]
        #print('best_score', self.best_score)
        if generation == self.num_generations - 1: break

      new_population = []
      for _ in range(self.population_size):
        # Selection
        parents_indices = self.selection(fitness_scores, 2)
        #print('parents_indices', parents_indices)
        parent1, parent2 = self.population[parents_indices[0]], self.population[parents_indices[1]]
        #print('parent1, parent2', type(parent1), type(parent2))
        # Crossover
        if np.random.rand() < self.crossover_probability:
          child = self.crossover(parent1, parent2)
        else:
          child = parent1.copy()
        #print('child', type(child), dup(child))
        # Mutation
        mutated_child = self.mutation(child, self.mutation_probability)
        #print('mutated_child', type(mutated_child), dup(mutated_child))
        new_population.append(mutated_child)

      self.population = new_population

    return self.best_solution

  def predict(self):
      return self.best_solution

  def score(self):
      return self.best_score