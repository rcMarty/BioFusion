import numpy as np
from numpy import ndarray


class Point:
    def __init__(self, x: float, y: float, name: str):
        self.x: float = x
        self.y: float = y
        self.name: str = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Individual:
    def __init__(self):
        self.points: list[Point] = []
        self.fitness: float = np.inf
        self.feromones: ndarray = np.zeros((len(self.points), len(self.points)))

    def add(self, point: Point) -> Point:
        self.points.append(point)
        return point

    def calculate_cost(self):
        # for i in range(len(self.points) - 1):
        #     self.fitness += np.sqrt( (self.points[i].x - self.points[i + 1].x) ** 2 + (self.points[i].y - self.points[i + 1].y) ** 2)
        # self.fitness += np.sqrt( (self.points[0].x - self.points[-1].x) ** 2 + (self.points[0].y - self.points[-1].y) ** 2)

        coords = np.array([(point.x, point.y) for point in self.points])
        distances = np.sqrt(np.sum(np.diff(coords, axis=0, append=coords[:1]) ** 2, axis=1))
        self.fitness = np.sum(distances)

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __lt__(self, other):
        return self.fitness < other

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __gt__(self, other):
        return self.fitness > other

    def __eq__(self, other):
        return self.fitness == other.fitness


class Generation:
    def __init__(self):
        self.individuals: list[Individual] = []
        self.best_ind: Individual | None = None

    def add(self, individual: Individual) -> Individual:
        self.individuals.append(individual)
        if self.best_ind is None:
            self.best_ind = individual
        if individual.fitness < self.best_ind.fitness:
            self.best_ind = individual
        return individual

    def get_best(self) -> Individual:
        return self.best_ind

    def __lt__(self, other):
        return self.best_ind < other

    def __le__(self, other):
        return self.best_ind <= other.best_gen

    def __gt__(self, other):
        return self.best_ind > other

    def __ge__(self, other):
        return self.best_ind >= other.best_gen

    def __eq__(self, other):
        return self.best_ind == other.best_gen


class Genetic:
    def __init__(self):
        self.generations: list[Generation] = []
        self.best_gen: Generation | None = None

    def add(self, generation: Generation) -> Generation:
        self.generations.append(generation)
        if self.best_gen is None:
            self.best_gen = generation
        if generation < self.best_gen:
            self.best_gen = generation
        return generation

    def get_best(self) -> Individual:
        return self.best_gen.get_best()
