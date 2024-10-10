import numpy as np
from numpy import ndarray


class Position:

    def __init__(self, value: float, position: ndarray):
        self.position = position
        self.value = value


class Iteration:

    def __init__(self):
        self.history = []
        self.best: Position = Position(np.inf, np.array([0]))

    def add_position(self, position: Position) -> 'Iteration':
        """
        Add position to the iteration
        if the position is better than the current best, update the best
        :param position:
        :return:
        """
        self.history.append(position)
        if position.value < self.best.value:
            self.best = position
        return self

    def set_best(self, best: Position) -> 'Iteration':
        self.best = best
        return self

    def get_best(self) -> (float, float, float):
        return self.best.value, self.best.position[0], self.best.position[1]


class Result:

    def __init__(self):
        self.iterations = []

        self.best = Position(np.inf, np.array([0]))

    def add_iteration(self, iteration: Iteration) -> 'Result':
        """
        Add iteration to the result
        if the best position from iteration is better than the current best, update the best
        :param iteration:
        :return:
        """
        self.iterations.append(iteration)
        if iteration.best.value < self.best.value:
            self.best = iteration.best
        return self

    def get_best(self) -> (float, float, float):
        return self.best, self.best.position[0], self.best.position[1]
