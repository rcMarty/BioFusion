import pickle
import sys
from copy import deepcopy

import numpy as np

from src.render.Render2D import Render2D
from src.utils.Genetic import Point, Generation, Individual, Genetic


def generate_points(num_points: int,
                    x_range: tuple[float, float] = (0, 10),
                    y_range: tuple[float, float] = (0, 10)) -> list[Point]:
    points = []
    for i in range(num_points):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        points.append(Point(x, y, chr(65 + i)))
    return points


class AntColonyOptimization:

    def __init__(self, cities: int = 50, ants: int = 100, generations: int = 100, alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.5):
        self.points: list[Point] = generate_points(cities)
        self.no_of_cities: int = cities
        self.ants: int = ants
        self.generations: int = generations
        self.alpha: float = alpha
        self.beta: float = beta
        self.evaporation_rate: float = evaporation_rate
        self.pheromone: np.ndarray = np.ones((cities, cities))
        self.result: Genetic = Genetic()

    def distance(self, point1: Point, point2: Point) -> float:
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def initialize_pheromone(self):
        self.pheromone = np.ones((self.no_of_cities, self.no_of_cities))

    def run(self) -> Genetic:
        self.initialize_pheromone()
        for _ in range(self.generations):
            generation = Generation()
            best_individual = self.result.get_best() if self.result.generations else None
            for _ in range(self.ants):
                individual = self.construct_solution()
                individual.calculate_cost()
                generation.add(individual)
            if best_individual:
                generation.add(best_individual)
            self.update_pheromone(generation)
            self.result.add(generation)
            self.print_progress()
        best_individual = self.result.get_best()
        print(f"\n\n\nBest individual fitness: {best_individual.fitness}")
        return self.result

    def construct_solution(self) -> Individual:
        individual = Individual()
        unvisited = set(self.points)
        current_point = np.random.choice(self.points)
        individual.add(current_point)
        unvisited.remove(current_point)
        while unvisited:
            next_point = self.select_next_point(current_point, unvisited)
            individual.add(next_point)
            unvisited.remove(next_point)
            current_point = next_point

        # add to individual feromones for visualization ()
        feromones_copy = deepcopy(individual.feromones)
        # filter paths under treshold 1.0-e2
        feromones_copy[feromones_copy < 1.0e-2] = 0
        # for each 

        return individual

    def select_next_point(self, current_point: Point, unvisited: set) -> Point:
        current_index = self.points.index(current_point)
        unvisited_indices = [self.points.index(point) for point in unvisited]

        tau = self.pheromone[current_index, unvisited_indices] ** self.alpha
        distances = np.array([self.distance(current_point, self.points[i]) for i in unvisited_indices])
        eta = (1 / distances) ** self.beta

        probabilities = tau * eta
        probabilities /= probabilities.sum()

        return np.random.choice(list(unvisited), p=probabilities)

    def update_pheromone(self, generation: Generation):
        self.pheromone *= (1 - self.evaporation_rate)
        for individual in generation.individuals:
            indices = [self.points.index(point) for point in individual.points]
            for i in range(len(indices) - 1):
                self.pheromone[indices[i], indices[i + 1]] += 1 / individual.fitness

    def print_matrix(self, matrix: np.ndarray) -> list[str]:
        with np.printoptions(precision=2, suppress=False, threshold=np.inf, linewidth=250):
            matrix_str = np.array2string(matrix, separator=', ')
        return matrix_str.split('\n')

    def print_multiline_string(self, strings: list[str]):
        for string in strings:
            sys.stdout.write(string + '\n')

    def print_progress(self):
        progress = len(self.result.generations) / self.generations
        bar_length = 100
        filled_length = int(bar_length * progress)
        bar = '#' * filled_length + '.' * (bar_length - filled_length)
        percent = "{:.2f}".format(progress * 100)
        line = f'[{bar}]  {percent.rjust(5, " ")}'
        sys.stdout.write('\r')
        sys.stdout.write(line)
        # lines = [line, f'Generation: {len(self.result.generations)} / {self.generations}']
        # matrix_lines = self.print_matrix(self.pheromone)
        # os.system('clear')
        # self.print_multiline_string(lines + matrix_lines)
        sys.stdout.flush()

    def render(self):
        render = Render2D()
        render.plot_generation(self.result.generations)

    def save(self, path: str, file_name: str):
        with open(path + file_name + ".pkl", "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str, file_name: str) -> 'AntColonyOptimization':
        with open(path + file_name + ".pkl", "rb") as f:
            return pickle.load(f)
