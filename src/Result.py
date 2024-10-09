import numpy as np


class Position:

    def __init__(self, value: float = np.inf, position: tuple[float, float] = (0, 0)):
        self.position = position
        self.value = value


class Iteration:

    def __init__(self):
        self.history = []
        self.best = Position()

    def add_position(self, position: Position) -> 'Iteration':
        self.history.append(position)
        return self

    def set_best(self, best: Position) -> 'Iteration':
        self.best = best
        return self

    def get_best(self) -> (float, float, float):
        return self.best.value, self.best.position[0], self.best.position[1]


class Result:

    def __init__(self):
        self.iterations = []
        self.best = np.inf
        self.best_position = Position()

    def add_iteration(self, iteration: Iteration) -> 'Result':
        self.iterations.append(iteration)
        if iteration.best.value < self.best:
            self.best = iteration.best.value
            self.best_position = iteration.best
        return self

    def get_best(self) -> (float, float, float):
        return self.best, self.best_position.position[0], self.best_position.position[1]
