from src.Functions import Function
from src.render.Render3D import *
from src.utils.Result import *


class BlindSearch:

    def __init__(self, functions: Function, repeat_count: int = 10, population: int = 100):
        self.functions: Function = functions
        self.repeat_count: int = repeat_count  # Number of iterations
        self.population: int = population  # Number of particles
        self.best = np.inf
        self.result: dict[callable, Result] = {}

    def run_function(self, function: callable) -> Result:
        iteration = Iteration()
        local_best = np.inf
        for i in range(self.repeat_count):
            for j in range(self.population):

                xx = np.random.uniform(low=function.range[0], high=function.range[1], size=function.dimension)
                result = function(xx)
                iteration.add_position(Position(result, xx))

                if result < local_best:
                    local_best = result
                    iteration.set_best(Position(result, xx))

        return Result().add_iteration(iteration)

    def run_all(self) -> dict[callable, Result]:
        for function in self.functions.get_all():
            self.result[function] = self.run_function(function)
        return self.result

    def render(self, function: callable):
        render = Render3D()
        render.render3d(self.result[function], function)

    def render_all(self):
        render = Render3D()
        for function in self.functions.get_all():
            render.render3d(self.result[function], function)
