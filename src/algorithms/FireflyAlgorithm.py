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


class FireflyAlgorithm:
    def __init__(self, functions: Function, NP: int = 100, alpha: float = 0.3, beta: float = 1.0, gamma: float = 1.0,
                 g_maxim: int = 30, ):
        """

        :param functions:
        :param NP:
        :param alpha:
        :param beta:
        :param gamma:
        :param g_maxim:
        """

        self.NP = NP
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.g_maxim = g_maxim
        self.functions: Function = functions
        self.result: dict[callable, Result] = {}

    def generate_population(self, fn: callable, dimension: int):
        return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]

    def attractiveness(self, distance: float) -> float:
        return self.beta * np.exp(-self.gamma * distance ** 2)

    def run_function(self, function: callable) -> Result:
        dimension = function.dimension
        pop = self.generate_population(function, dimension)
        brightness = [function(pos) for pos in pop]
        g = 0
        result = Result()

        while g < self.g_maxim and functions_call_counter.get_counts(function) < functions_call_counter.get_max_calls():
            iteration = Iteration()

            for i in range(self.NP):
                for j in range(self.NP):
                    if brightness[i] > brightness[j]:
                        distance = np.linalg.norm(pop[i] - pop[j])
                        beta = self.attractiveness(distance)

                        pop[i] = np.clip(pop[i] + beta * (pop[j] - pop[i]) + self.alpha * (np.random.rand(dimension) - 0.5),
                                         function.range[0],
                                         function.range[1])

                        brightness[i] = function(pop[i])

                iteration.add_position(Position(brightness[i], pop[i]))

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
