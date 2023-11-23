# pnuopt

`pnuopt` is a Python package for advanced optimization techniques, tailored for tackling a wide range of optimization problems, particularly in the field of machine learning and artificial intelligence. It provides a set of tools for hill climbing strategies, including variants like steepest ascent, first-choice, gradient descent, and simulated annealing.

## Features

- **Variety of Optimization Algorithms**: Includes First-Choice Hill Climber, Steepest Ascent Hill Climber, Gradient Descent Hill Climber, and Simulated Annealing Hill Climber.
- **Customizable Parameters**: Offers flexibility to tweak various parameters like step size, number of neighbors, learning rate, and number of restarts.
- **Extensible Framework**: Designed to be easily extendable for additional optimization methods and custom problem definitions.

## Installation

Install `pnuopt` using pip:

```bash
pip install git+https://github.com/kaper-edward/pnuopt.git
```

## Usage
Here is a quick example of how to use the Steepest Ascent Hill Climber from the pnuopt package:

```python
from pnuopt.alg import SteepestAscentHillClimber

def my_objective_function(solution):
    # Define your objective function here
    pass

def my_neighbor_generator(solution):
    # Define how neighbors are generated from a given solution
    pass

def my_initial_solution_generator():
    # Define how the initial solution is generated
    pass

hill_climber = SteepestAscentHillClimber(
    generate_neighbor=my_neighbor_generator,
    objective_function=my_objective_function,
    initial_solution_generator=my_initial_solution_generator
)

hill_climber.fit()
best_solution = hill_climber.predict()
print("Best Solution:", best_solution)
```

## Contributing
Contributions are welcome! Please submit pull requests, report issues, and suggest improvements.

## License
This project is licensed under the GPL 3.0 License.

## Acknowledgments
A special thanks to the contributors and users of pnuopt for their support and suggestions.
