from src.Functions import Function
from src.render.Render3D import *
from src.utils.Result import *
from src.utils.Utils import functions_call_counter


def evaluate_population(population: list[ndarray], function: callable) -> Position:
    best_value = np.inf
    best_position = None
    for pos in population:
        value = function(pos)
        if value < best_value:
            best_value = value
            best_position = pos
    return Position(best_value, best_position)


class TeachingLearningBasedOptimization:
    def __init__(self, functions: Function, NP: int = 100, g_maxim: int = 30):
        self.NP = NP
        self.g_maxim = g_maxim
        self.functions: Function = functions
        self.result: dict[callable, Result] = {}

    def generate_population(self, fn: callable, dimension: int):
        return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]

    def run_function(self, function: callable) -> Result:
        dimension = function.dimension
        pop = self.generate_population(function, dimension)
        g = 0
        result = Result()

        while g < self.g_maxim and functions_call_counter.get_counts(function) < functions_call_counter.get_max_calls():
            iteration = Iteration()

            # Teaching Phase
            teacher = min(pop, key=function)
            mean = np.mean(pop, axis=0)
            for i in range(self.NP):
                TF = np.random.randint(1, 3)  # Teaching Factor
                new_position = pop[i] + np.random.rand(dimension) * (teacher - TF * mean)
                new_position = np.clip(new_position, function.range[0], function.range[1])
                if function(new_position) < function(pop[i]):
                    pop[i] = new_position

            # Learning Phase
            for i in range(self.NP):
                partner = pop[np.random.randint(self.NP)]
                if function(partner) < function(pop[i]):
                    new_position = pop[i] + np.random.rand(dimension) * (partner - pop[i])
                else:
                    new_position = pop[i] + np.random.rand(dimension) * (pop[i] - partner)
                new_position = np.clip(new_position, function.range[0], function.range[1])
                if function(new_position) < function(pop[i]):
                    pop[i] = new_position

                iteration.add_position(Position(function(pop[i]), pop[i]))

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

    def render(self, function: callable):
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
