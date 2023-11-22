import unittest
from functools import partial

import numpy as np

from pnuopt.alg import FirstChoiceHillClimber, SteepestAscentHillClimber, \
    GradientDescentHillClimber, SimulatedAnnealingHillClimber


class TestContinuousHillClimber(unittest.TestCase):
    def setUp(self):
        def convex_function(x):
            return (x[0] - 2) ** 2 + 5 * (x[1] - 5) ** 2 + 8 * (x[2] + 8) ** 2 + 3 * (x[3] + 1) ** 2 + 6 * (
                        x[4] - 7) ** 2

        def continuous_generate_neighbor(solution, step_size, num_neighbors):
            neighbors = []
            for _ in range(num_neighbors):
                neighbors.append(solution + np.random.uniform(-step_size, step_size, len(solution)))
            return neighbors

        def initial_solution_generator(n=5):
            return [np.random.uniform(-30, 30) for _ in range(n)]

        self.goal_score = 800
        self.objective_function = convex_function
        self.generate_neighbor = continuous_generate_neighbor
        self.initial_solution_generator = initial_solution_generator

    def test_first_choice(self):
        hill_climber = FirstChoiceHillClimber(
            generate_neighbor=self.generate_neighbor,
            objective_function=self.objective_function,
            initial_solution_generator=self.initial_solution_generator,
        )
        hill_climber.fit()
        best_solution = hill_climber.predict()
        best_score = hill_climber.score()
        print('best_solution', best_solution)
        print('best_score', best_score)
        #self.assertLess(best_score, self.goal_score)

    def test_steepest(self):
        hill_climber = SteepestAscentHillClimber(
            generate_neighbor=self.generate_neighbor,
            objective_function=self.objective_function,
            initial_solution_generator=self.initial_solution_generator,
        )
        hill_climber.fit()
        best_solution = hill_climber.predict()
        best_score = hill_climber.score()
        print('best_solution', best_solution)
        print('best_score', best_score)
        self.assertLess(best_score, self.goal_score)

    def test_gradient(self):
        def numerical_gradient(f, solution, step_size=0.1, num_neighbors=1, learning_rate=0.01, h=1e-5):
            neighbors = []
            for _ in range(num_neighbors):
                grad = np.zeros_like(solution)
                np_solution = np.array(solution)
                for i in range(len(np_solution)):
                    xi = np_solution[i]
                    np_solution[i] = xi - h
                    fxh1 = f(np_solution)  # f(x-h)
                    np_solution[i] = xi + h
                    fxh2 = f(np_solution)  # f(x+h)
                    grad[i] = (fxh2 - fxh1) / (2*h)
                    solution[i] = xi
                np_solution = np_solution - learning_rate * grad
                neighbors.append(np_solution)
            return neighbors

        objective_function = partial(numerical_gradient, f=self.objective_function)

        hill_climber = GradientDescentHillClimber(
            gradient_function=objective_function,
            objective_function=self.objective_function,
            initial_solution_generator=self.initial_solution_generator,
        )
        hill_climber.fit()
        best_solution = hill_climber.predict()
        best_score = hill_climber.score()
        print('best_solution', best_solution)
        print('best_score', best_score)
        self.assertLess(best_score, self.goal_score)


    def test_simulated_annealing(self):
        hill_climber = SimulatedAnnealingHillClimber(
            generate_neighbor=self.generate_neighbor,
            objective_function=self.objective_function,
            initial_solution_generator=self.initial_solution_generator,
        )
        hill_climber.fit()
        best_solution = hill_climber.predict()
        best_score = hill_climber.score()
        print('best_solution', best_solution)
        print('best_score', best_score)
        self.assertLess(best_score, self.goal_score)


if __name__ == '__main__':
    unittest.main()
