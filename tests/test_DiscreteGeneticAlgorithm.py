from functools import partial
import numpy as np
import random
import unittest
from pnuopt.alg import GeneticAlgorithm
from collections import Counter


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.population_size = 200
        self.cities = [
            (8, 31), (54, 97), (50, 50), (65, 16), (70, 47), (25, 100), (55, 74), (77, 87),
            (6, 46), (70, 78), (13, 38), (100, 32), (26, 35), (55, 16), (26, 77), (17, 67),
            (40, 36), (38, 27), (33, 2), (48, 9), (62, 20), (17, 92), (30, 2), (80, 75),
            (32, 36), (43, 79), (57, 49), (18, 24), (96, 76), (81, 39)
        ]

        fitness_func = partial(self.fitness_func, cities=self.cities)
        initial_population_generator = partial(self.initial_population_generator, num_cities=len(self.cities))

        self.ga = GeneticAlgorithm(
            fitness_function=fitness_func,
            initial_population_generator=initial_population_generator,
            selection_function=self.selection_function,
            crossover_function=self.crossover_function,
            mutation_function=self.mutation_function,
            population_size=self.population_size
        )

    def fitness_func(self, solution, cities):
        total_distance = 0
        for i in range(len(solution)):
            start_city = cities[solution[i]]
            end_city = cities[solution[(i + 1) % len(solution)]]
            total_distance += np.sqrt((start_city[0] - end_city[0]) ** 2 + (start_city[1] - end_city[1]) ** 2)
        return total_distance

    def initial_population_generator(self, population_size, num_cities):
        return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

    def selection_function(self, fitness_scores, num_parents):
        total_fitness = sum(fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]
        return np.random.choice(range(len(fitness_scores)), size=num_parents, p=selection_probs)

    def crossover_function(self, parent1, parent2):
        size = len(parent1)
        start, stop = sorted(np.random.choice(range(size), 2, replace=False))
        child = [None] * size
        if stop < start:
            child[stop:start] = parent1[stop:start]
        else:
            child[start:stop] = parent1[start:stop]
        p2_elements = [item for item in parent2 if item not in parent1[start:stop]]
        p2_pointer = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_elements[p2_pointer]
                p2_pointer += 1
        return child

    def mutation_function(self, individual, mutation_probability):
        if random.random() < mutation_probability:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def test_evolution_process(self):
        best_solution = self.ga.fit()
        self.assertEqual(len(self.dup(best_solution)), 0)
        best_score = self.ga.score()
        self.assertLess(best_score, 1000)

    def dup(self, x):
        return list(filter(lambda items: items[1] > 1, Counter(x).items()))


if __name__ == '__main__':
    unittest.main()