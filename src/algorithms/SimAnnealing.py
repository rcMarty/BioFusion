from src.Functions import Function
from src.render.Render3D import *
from src.utils.Result import *


class SimAnnealing:

    # simulated annealing algorithm class
    def __init__(self, functions: Function, initial_temp: float = 100, min_temp: float = 0.5, alpha: float = 0.95):
        """
        :param functions: Function object containing all functions to optimize
        :param initial_temp: Initial temperature for the annealing process
        :param min_temp: Minimum temperature for the annealing process
        :param alpha: Cooling rate
        """
        self.functions: Function = functions
        self.initial_temp: float = initial_temp
        self.min_temp: float = min_temp
        self.alpha: float = alpha
        self.result: dict[callable, Result] = {}

    def run_function(self, function: callable) -> Result:
        T = self.initial_temp
        location: ndarray = np.random.uniform(low=function.range[0], high=function.range[1], size=len(function.range))
        best_value = function(location)
        radius_in_percent_for_scale = (function.range[1] - function.range[0]) * 4 / 100
        result = Result()

        iteration = Iteration()
        # while T > T_min:
        while T > self.min_temp:

            # x_1 = generate neighbour of x in normal distribution
            new_location = np.random.normal(location, radius_in_percent_for_scale, size=len(function.range))
            # Evaluate x_1
            new_value = function(new_location)

            # if f(x_1) < f(x):
            if new_value < best_value:
                # x = x_1
                location = new_location
                best_value = new_value

            else:
                # r = random number in uniform distribution
                r = np.random.uniform()
                # if r < e^(-(f(x_1)-f(x))/T ):
                if r < np.exp((best_value - new_value) / T):
                    location = new_location

            iteration.add_position(Position(new_value, new_location))
            T *= self.alpha

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
            render.render(self.result[function], function)
        else:
            self.run_function(function)
            render.render(self.result[function], function)

    def render_all(self):
        render = Render3D()
        for function in self.functions.get_all():
            render.render(self.result[function], function)
