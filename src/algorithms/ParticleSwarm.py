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


class ParticleSwarmOptimization:
    def __init__(self, functions: Function, NP: int = 20, w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
                 g_maxim: int = 50,
                 treshold: float = 1e-4):
        self.treshold = treshold
        self.NP = NP
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.g_maxim = g_maxim
        self.functions: Function = functions
        self.result: dict[callable, Result] = {}

    def generate_population(self, fn: callable, dimension: int):
        return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]

    def run_function(self, function: callable) -> Result:
        dimension = len(function.range)
        pop = self.generate_population(function, dimension)
        velocities = [np.random.uniform(-1, 1, dimension) for _ in range(self.NP)]
        personal_best_positions = deepcopy(pop)
        personal_best_values = [function(pos) for pos in pop]
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        g = 0
        result = Result()

        while g < self.g_maxim:
            iteration = Iteration()

            for i, x in enumerate(pop):
                r1, r2 = np.random.rand(dimension), np.random.rand(dimension)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - x) +
                                 self.c2 * r2 * (global_best_position - x))
                pop[i] = np.clip(x + velocities[i], function.range[0], function.range[1])
                f_x = function(pop[i])

                if f_x < personal_best_values[i]:
                    personal_best_positions[i] = pop[i]
                    personal_best_values[i] = f_x

                if f_x < function(global_best_position):
                    global_best_position = pop[i]

                iteration.add_position(Position(f_x, pop[i]))

            best_position = evaluate_population(pop, function)
            iteration.add_position(best_position)

            g += 1

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
