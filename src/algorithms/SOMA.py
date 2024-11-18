import math
import random
from copy import deepcopy

from src.Functions import Function
from src.render.Render3D import *
from src.utils.Result import *


def evaluate_population(population: list[ndarray], function: callable) -> Position:
    best_value = np.inf
    best_position = None
    for pos in population:
        value = function(pos)
        if value < best_value:
            best_value = value
            best_position = pos
    return Position(best_value, best_position)


class SelfOrganizingMigrationAlgorithm:
    def __init__(self, functions: Function, NP: int = 30, prt: float = 0.4, NStep: int = 3, Migrations: int = 40,
                 MinDist: float = 1e-6):
        self.NP = NP
        self.prt = prt
        self.NStep = NStep
        self.Migrations = Migrations
        self.MinDist = MinDist
        self.functions: Function = functions
        self.result: dict[callable, Result] = {}

    def generate_population(self, fn: callable, dimension: int):
        return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]

    def cost_function_sort(self, pop, function):
        pop = np.array(pop)
        costs = [function(ind) for ind in pop]
        sorted_indices = np.argsort(costs)
        return pop[sorted_indices]

    def create_prt_vector(self, prt, pop_size, num_of_x):
        prt_vector = np.zeros((num_of_x, pop_size))
        for j in range(pop_size):
            for i in range(num_of_x):
                if random.random() < prt:
                    prt_vector[i][j] = 1
        return prt_vector

    def movement(self, pop, prt, NStep, function):
        pop_size = len(pop)
        num_of_x = len(pop[0])
        new_pop = deepcopy(pop)
        PRT = self.create_prt_vector(prt, pop_size, num_of_x)

        for i in range(pop_size):
            vector_of_steps = np.zeros((num_of_x, 4 * NStep))
            for s in range(4 * NStep):
                vector_of_steps[:, s] = new_pop[i] + ((new_pop[0] - new_pop[i]) / (2 * NStep)) * PRT[:, i] * s
            new_pop[i] = self.cost_function_sort(vector_of_steps.T, function)[0]

        for i in range(pop_size, pop_size * 2):
            vector_of_steps = np.zeros((num_of_x, 2 * NStep))
            for s in range(2 * NStep):
                vector_of_steps[:, s] = new_pop[i % pop_size] + ((new_pop[1] - new_pop[i % pop_size]) / NStep) * PRT[:,
                                                                                                                 i % pop_size] * s
            new_pop[i % pop_size] = self.cost_function_sort(vector_of_steps.T, function)[0]

        for i in range(pop_size * 2, pop_size * 3):
            vector_of_steps = np.zeros((num_of_x, NStep))
            for s in range(NStep):
                vector_of_steps[:, s] = new_pop[i % pop_size] + (
                        (new_pop[2] - new_pop[i % pop_size]) / (NStep / 2)) * PRT[:, i % pop_size] * s
            new_pop[i % pop_size] = self.cost_function_sort(vector_of_steps.T, function)[0]

        return new_pop

    def elaboration(self, pop, NStep, prt, function):
        num_of_x = len(pop[0])
        NStep *= 10
        PRT = self.create_prt_vector(prt, 3, num_of_x)
        vector_of_steps = np.zeros((num_of_x, NStep))
        ans = np.zeros((num_of_x, 3))
        for i in range(3):
            for s in range(NStep):
                vector_of_steps[:, s] = pop[i] + ((pop[0] - pop[i]) / (NStep / 2)) * PRT[:, i] * s
            ans[:, i] = self.cost_function_sort(vector_of_steps.T, function)[0]
        ans = self.cost_function_sort(ans.T, function)
        return ans[0]

    def run_function(self, function: callable) -> Result:
        dimension = len(function.range)
        pop = self.generate_population(function, dimension)
        pop = self.cost_function_sort(pop, function)
        new_pop = np.concatenate((pop, pop, pop), axis=0)

        MCount = 0
        result = Result()

        while MCount < self.Migrations and math.sqrt((1 / 2) * ((function(new_pop[1]) - function(new_pop[0])) ** 2 + (
                function(new_pop[2]) - function(new_pop[0])) ** 2)) >= self.MinDist:
            new_pop = self.movement(new_pop, self.prt, self.NStep, function)
            new_pop = self.cost_function_sort(new_pop, function)
            new_pop = np.concatenate(
                (new_pop[:round(self.NP * (2 / 3))], self.generate_population(function, dimension)), axis=0)
            new_pop = self.cost_function_sort(new_pop, function)
            new_pop = np.concatenate((new_pop, new_pop, new_pop), axis=0)

            MCount += 1

            iteration = Iteration()
            for pos in new_pop[:self.NP]:
                iteration.add_position(Position(function(pos), pos))
            result.add_iteration(iteration)

        last_pop = new_pop[:3]
        ans = self.elaboration(last_pop, self.NStep, self.prt, function)

        self.result[function] = result
        return self.result[function]

    def run_all(self) -> dict[callable, Result]:
        for function in self.functions.get_all():
            self.run_function(function)
        return self.result

    def render(self, function: callable, is_2d: bool = False):
        render = Render3D()

        if function in self.result:
            render.render3d(self.result[function], function)
        else:
            self.run_function(function)
            render.render3d(self.result[function], function)

    def render_all(self):
        render = Render3D()
        for function in self.functions.get_all():
            render.render3d(self.result[function], function)

# def evaluate_population(population: list[ndarray], function: callable) -> Position:
#     best_value = np.inf
#     best_position = None
#     for pos in population:
#         value = function(pos)
#         if value < best_value:
#             best_value = value
#             best_position = pos
#     return Position(best_value, best_position)
#
#
# class SelfOrganizingMigrationAlgorithm:
#     def __init__(self, functions: Function, NP: int = 20, path_length: float = 3.0, step: float = 0.11,
#                  g_maxim: int = 50, treshold: float = 1e-4):
#         self.treshold = treshold
#         self.NP = NP
#         self.path_length = path_length
#         self.step = step
#         self.g_maxim = g_maxim
#         self.functions: Function = functions
#         self.result: dict[callable, Result] = {}
#
#     def generate_population(self, fn: callable, dimension: int):
#         return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]
#
#     def run_function(self, function: callable) -> Result:
#         dimension = len(function.range)
#         pop = self.generate_population(function, dimension)
#         g = 0
#         result = Result()
#
#         while g < self.g_maxim:
#             iteration = Iteration()
#             best_position = evaluate_population(pop, function)
#             leader = best_position.position
#
#             for i, x in enumerate(pop):
#                 if np.array_equal(x, leader):
#                     continue
#
#                 for t in np.arange(0, self.path_length, self.step):
#                     new_position = x + t * (leader - x)
#                     new_position = np.clip(new_position, function.range[0], function.range[1])
#                     f_new = function(new_position)
#
#                     if f_new < function(x):
#                         pop[i] = new_position
#                         iteration.add_position(Position(f_new, new_position))
#
#             best_position = evaluate_population(pop, function)
#             iteration.add_position(best_position)
#
#             g += 1
#
#             result.add_iteration(iteration)
#
#         self.result[function] = result
#         return self.result[function]
#
#     def run_all(self) -> dict[callable, Result]:
#         for function in self.functions.get_all():
#             self.run_function(function)
#         return self.result
#
#     def render(self, function: callable):
#         render = Render3D()
#         if function in self.result:
#             render.render3d(self.result[function], function)
#         else:
#             self.run_function(function)
#             render.render3d(self.result[function], function)
#
#     def render_all(self):
#         render = Render3D()
#
#         for function in self.functions.get_all():
#             render.render3d(self.result[function], function)
