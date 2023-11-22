import numpy as np


class HillClimber:
  def __init__(self, generate_neighbor,
                objective_function,
                initial_solution_generator,
                step_size=0.1,
                num_neighbors=1,
                max_iter=1000,
                mode='min',
                num_restarts=10,
                ):
    self.generate_neighbor = generate_neighbor
    self.objective_function = objective_function
    self.initial_solution_generator = initial_solution_generator
    self.step_size = step_size
    self.num_neighbors = num_neighbors
    self.max_iter = max_iter
    self.best_solution = None
    self.best_score = None
    self._setup_mode(mode)
    self.num_restarts = num_restarts

  def _setup_mode(self, mode):
    if mode == 'max':
      self.best_score = float('-inf')
      self.is_better = lambda new, best: new > best
      self.best_score_idx = lambda scores: np.argmax(scores)
    else:
      self.best_score = float('inf')
      self.is_better = lambda new, best: new < best
      self.best_score_idx = lambda scores: np.argmin(scores)

  def fit(self):
    self.current_solution = self.initial_solution_generator()
    for _ in range(self.max_iter):
      neighbors = self.generate_neighbor(solution=self.current_solution, step_size=self.step_size, num_neighbors=self.num_neighbors)
      neighbors_scores = [self.objective_function(neighbor) for neighbor in neighbors]
      best_neighbor_index = np.argmin(neighbors_scores)
      best_neighbor = neighbors[best_neighbor_index]

      if neighbors_scores[best_neighbor_index] < self.best_score:
        self.best_solution = best_neighbor
        self.best_score = neighbors_scores[best_neighbor_index]
        self.current_solution = best_neighbor

    return self.best_solution

  def predict(self):
    return self.best_solution

  def score(self):
    return self.best_score

  def random_restart(self):
    for _ in range(self.num_restarts):
      self.fit()
      new_solution = self.predict()
      new_score = self.score()

      if self.is_better(new_score, self.best_score):
        self.best_solution = new_solution
        self.best_score = new_score


class FirstChoiceHillClimber(HillClimber):
    pass


class SteepestAscentHillClimber(HillClimber):
    def _setup_mode(self, mode):
        super()._setup_mode(mode)
        if mode == 'max':
            self.best_score_idx = lambda scores: np.argmax(scores)
        else:
            self.best_score_idx = lambda scores: np.argmin(scores)

    def fit(self):
        self.current_solution = self.initial_solution_generator()
        for _ in range(self.max_iter):
            neighbors = self.generate_neighbor(self.current_solution, self.step_size, 10)
            neighbors_scores = [self.objective_function(neighbor) for neighbor in neighbors]
            best_neighbor_index = self.best_score_idx(neighbors_scores)
            best_neighbor = neighbors[best_neighbor_index]

            if self.is_better(neighbors_scores[best_neighbor_index], self.best_score):
                self.best_solution = best_neighbor
                self.best_score = neighbors_scores[best_neighbor_index]
                self.current_solution = best_neighbor

        return self.best_solution


class StochasticHillClimber(HillClimber):
    def fit(self):
        for _ in range(self.max_iter):
            neighbors = [self.generate_neighbor(self.current_solution, self.step_size) for _ in range(100)]
            better_neighbors = [(neighbor, self.objective_function(neighbor))
                                for neighbor in neighbors if self.objective_function(neighbor) > self.best_score]

            if better_neighbors:
                chosen_neighbor = better_neighbors[np.random.randint(len(better_neighbors))]
                self.best_solution = chosen_neighbor[0]
                self.best_score = chosen_neighbor[1]
                self.current_solution = chosen_neighbor[0]

        return self.best_solution


class GradientDescentHillClimber(HillClimber):
    def __init__(self, gradient_function, objective_function, initial_solution_generator, step_size=0.1, num_neighbors=1, learning_rate=0.01, max_iter=1000, mode='min', num_restarts=10):
        super().__init__(gradient_function, objective_function, initial_solution_generator, step_size, num_neighbors, max_iter, mode, num_restarts)
        self.learning_rate = learning_rate


class SimulatedAnnealingHillClimber(HillClimber):
    def __init__(self, generate_neighbor, objective_function, initial_solution_generator, step_size=0.1, num_neighbors=1, max_iter=10000, mode='min', num_restarts=10, temperature=1.0, alpha=1.0):
        super().__init__(generate_neighbor, objective_function, initial_solution_generator, step_size, num_neighbors, max_iter, mode, num_restarts)
        self.temperature = temperature
        self.alpha = alpha
        # self.args = {
        #     'step_size' : step_size,
        #     'num_neighbors' : num_neighbors
        # }

    def _acceptance_probability(self, cur_cost, new_cost):
        delta = new_cost - cur_cost
        if delta < 0:
          pass
          #print('delta', delta)
          #print('self.temperature', self.temperature)
          #print('p', self.alpha*np.exp(-delta / (self.temperature+1e-2)))
        return self.alpha*np.exp(-delta / (self.temperature+1e-2))

    def _cooling_schedule(self, k):
        self.temperature *= (1.0-((k+1.0)/self.max_iter))
        #print('self.temperature', self.temperature)
        return self.temperature

    def fit(self):
        self.current_solution = self.initial_solution_generator()
        best_score = self.objective_function(self.current_solution)
        for k in range(self.max_iter):
            self._cooling_schedule(k)
            neighbors = self.generate_neighbor(self.current_solution, self.step_size, self.num_neighbors)
            neighbors_scores = [self.objective_function(neighbor) for neighbor in neighbors]
            best_neighbor_index = np.argmin(neighbors_scores)
            best_neighbor = neighbors[best_neighbor_index]
            best_score = neighbors_scores[best_neighbor_index]
            cur_score = self.objective_function(self.current_solution)
            if self._acceptance_probability(cur_score, best_score) >= np.random.uniform(0, 1):
                #print('k', k)
                #print('acceptance_probability', self._acceptance_probability(cur_score, best_score))
                #print('cur_score', cur_score)
                #print('best_score', best_score)
                #print('self.best_score', self.best_score)
                self.best_solution = best_neighbor
                self.best_score = best_score
                self.current_solution = best_neighbor

        return self.best_solution