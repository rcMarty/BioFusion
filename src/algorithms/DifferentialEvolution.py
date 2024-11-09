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


class DifferentialEvolution:
    def __init__(self, functions: Function, NP: int = 20, F: float = 0.5, CR: float = 0.5, g_maxim: int = 50,
                 treshold: float = 1e-4):
        self.treshold = treshold
        self.NP = NP
        self.F = F
        self.CR = CR
        self.g_maxim = g_maxim
        self.functions: Function = functions
        self.result: dict[callable, Result] = {}

    def generate_population(self, fn: callable, dimension: int):
        return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]

    def select_random_indices(self, exclude_index: int) -> tuple[int, int, int]:
        indices = list(range(self.NP))
        indices.remove(exclude_index)
        return np.random.choice(indices, 3, replace=False)

    def run_function(self, function: callable) -> Result:
        dimension = len(function.range)
        pop = self.generate_population(function, dimension)
        g = 0
        result = Result()

        while g < self.g_maxim:
            new_pop = deepcopy(pop)

            iteration = Iteration()

            for i, x in enumerate(pop):
                r1, r2, r3 = self.select_random_indices(i)
                v = (pop[r1] - pop[r2]) * self.F + pop[r3]

                v = np.maximum(v, function.range[0])
                v = np.minimum(v, function.range[1])

                u = np.zeros(dimension)
                j_rnd = np.random.randint(0, dimension)

                for j in range(dimension):
                    if np.random.uniform() < self.CR or j == j_rnd:
                        u[j] = v[j]
                    else:
                        u[j] = x[j]

                f_u = function(u)

                if f_u <= function(x):
                    new_pop[i] = u

                pop = new_pop
                iteration.add_position(Position(f_u, u))

                # if iteration.best.value < self.treshold:
                #    break

            best_position = evaluate_population(pop, function)
            iteration.add_position(best_position)

            g += 1

            # if iteration.best.value < self.treshold:
            #    break

            result.add_iteration(iteration)

        self.result[function] = result
        return self.result[function]

    def run_all(self) -> dict[callable, Result]:
        for function in self.functions.get_all():
            self.run_function(function)
        return self.result

    def render(self, function: callable, is_2d: bool = False):
        render = Render3D()
        if not is_2d:
            if function in self.result:
                render.render3d(self.result[function], function)
            else:
                self.run_function(function)
                render.render3d(self.result[function], function)
        else:
            if function in self.result:
                render.render2d(self.result[function], function)
            else:
                self.run_function(function)
                render.render2d(self.result[function], function)

    def render_all(self, is_2d: bool = False):
        render = Render3D()
        if not is_2d:
            for function in self.functions.get_all():
                render.render3d(self.result[function], function)
        else:
            for function in self.functions.get_all():
                render.render2d(self.result[function], function)
