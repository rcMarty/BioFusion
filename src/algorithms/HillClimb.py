from src.Functions import Function
from src.render.Render import *
from src.utils.Result import *


class HillClimb:

    def __init__(self, functions: Function, population: int = 75, radius: float = 5):
        """

        :param functions:
        :param population: how many points to generate per iteration
        :param radius: radius of the searchspace in %
        """
        self.functions: Function = functions
        self.population: int = population  # Number of particles
        self.radius: float = radius
        self.result: dict[callable, Result] = {}

    def run_function(self, function: callable) -> Result:
        better = True
        radius_in_percent_for_scale = (function.range[1] - function.range[0]) * self.radius / 100
        location: ndarray = np.random.uniform(low=function.range[0], high=function.range[1],
                                              size=len(function.range))
        result = Result()
        while better:
            iteration = Iteration()
            for j in range(self.population):
                xx = np.random.normal(loc=location, scale=radius_in_percent_for_scale)
                z = function(xx)
                iteration.add_position(Position(z, xx))

            if iteration.best.value < result.best.value:
                result.best = iteration.best
            else:
                better = False

            location = iteration.best.position
            result.add_iteration(iteration)

        self.result[function] = result
        return self.result[function]

    def run_all(self) -> dict[callable, Result]:
        for function in self.functions.get_all():
            self.run_function(function)
        return self.result

    def render(self, function: callable):
        render = Render()
        if function in self.result:
            render.render(self.result[function], function)
        else:
            self.run_function(function)
            render.render(self.result[function], function)

    def render_all(self):
        render = Render()
        for function in self.functions.get_all():
            render.render(self.result[function], function)
