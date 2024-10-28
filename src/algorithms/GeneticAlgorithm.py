from copy import copy

import numpy as np

from src.render.Render2D import Render2D
from src.utils.Genetic import Genetic, Point, Generation, Individual


def generate_points(num_points: int,
                    x_range: tuple[float, float] = (0, 10),
                    y_range: tuple[float, float] = (0, 10)) -> list[Point]:
    points = []
    for i in range(num_points):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        points.append(Point(x, y, chr(65 + i)))
    return points


class GeneticAlgorithm:

    def __init__(self, cities: int = 10, population: int = 100, generations: int = 100, mutation_rate: float = 0.5):
        self.points: list[Point] = generate_points(cities)  # D number of "cities"
        self.no_of_cities: int = cities
        self.population_size: int = population  # NP size of population
        self.generations: int = generations  # G
        self.mutation_rate: float = mutation_rate
        self.result: Genetic = Genetic()

    def generate_generation(self) -> Generation:
        generation = Generation()
        for _ in range(self.no_of_cities):
            individual = Individual()
            for point in self.points:
                individual.add(point)
            individual.calculate_cost()
            generation.add(individual)

        return generation

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual()
        start, end = sorted(np.random.choice(range(self.no_of_cities), 2, replace=False))
        child.points = parent1.points[start:end]
        for point in parent2.points:
            if point not in child.points:
                child.add(point)
        child.calculate_cost()
        return child

    def mutate(self, individual: Individual):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(range(self.no_of_cities), 2, replace=False)
            individual.points[idx1], individual.points[idx2] = individual.points[idx2], individual.points[idx1]
            individual.calculate_cost()

    def run(self):
        last_gen: Generation = self.result.add(self.generate_generation())

        for _ in range(self.generations):
            new_gen = Generation()
            for _ in range(self.population_size):
                parent1 = copy(last_gen.best_ind)
                parent2 = np.random.choice(last_gen.individuals, replace=False)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_gen.add(child)
            last_gen = self.result.add(new_gen)

        best_individual = self.result.best_gen.get_best()
        print(f"Best individual fitness: {best_individual.fitness}")
        for point in best_individual.points:
            print(f"Point {point.name}: ({point.x}, {point.y})")

    def render(self):
        render = Render2D()
        render.plot_generation(self.result.generations)
