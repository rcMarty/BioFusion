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


class SelfOrganizingMigrationAlgorithm:

    def __init__(self, functions: Function, NP: int = 20, PRT: float = 0.4, path_length: float = 2.0, step: float = 0.11, M_max: int = 100, treshold: float = 1e-5):
        """
        :param functions:
        :param NP: number of individuals in the population
        :param PRT: perturbation rate
        :param path_length: the length of the path in the search space
        :param step: the step size for each movement
        :param M_max: maximum number of migrations
        :param treshold: the treshold for the algorithm to stop
        """
        self.functions: Function = functions
        self.pop_size: int = NP
        self.PRT: float = PRT
        self.path_length: float = path_length
        self.step: float = step
        self.M_max: int = M_max
        self.treshold: float = treshold
        self.result: dict[callable, Result] = {}

    def run_function(self, function: callable) -> Result:
        dimension = function.dimension
        # Initialize population
        population = np.random.uniform(low=function.range[0], high=function.range[1], size=(self.pop_size, dimension))
        fitness = np.apply_along_axis(function, 1, population)
        leader_index = np.argmin(fitness)
        leader = population[leader_index]

        result = Result()
        for _ in range(self.M_max):
            iteration = Iteration()
            i = 0
            while i < self.pop_size and functions_call_counter.get_counts(function) < functions_call_counter.get_max_calls():
                if i == leader_index:
                    i += 1
                    continue
                for t in np.arange(0, self.path_length, self.step):
                    PRT_vector = np.random.rand(dimension) < self.PRT
                    new_position = population[i] + t * (leader - population[i]) * PRT_vector
                    new_position = np.clip(new_position, function.range[0], function.range[1])
                    new_fitness = function(new_position)
                    iteration.add_position(Position(new_fitness, new_position))
                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness
                i += 1

            leader_index = np.argmin(fitness)
            leader = population[leader_index]
            iteration.add_position(Position(fitness[leader_index], leader))
            result.add_iteration(iteration)

            # if difference between population and leader is smaller than treshold, stop
            if np.linalg.norm(np.mean(population, axis=0) - leader) < self.treshold:
                break

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

    ###################################################################################################################
    # another versions of soma (sometimes reaaaaaly advanced (prof. Zelinka (https://ivanzelinka.eu/somaalgorithm/Codes.html) ))

    # def __init__(self, functions: Function, NP: int = 30, prt: float = 0.4, NStep: int = 3, Migrations: int = 40,
    #              MinDist: float = 1e-6):
    #     self.NP = NP
    #     self.prt = prt
    #     self.NStep = NStep
    #     self.Migrations = Migrations
    #     self.MinDist = MinDist
    #     self.functions: Function = functions
    #     self.result: dict[callable, Result] = {}
    #
    # def generate_population(self, fn: callable, dimension: int):
    #     return [np.random.uniform(fn.range[0], fn.range[1], dimension) for _ in range(self.NP)]
    #
    # def cost_function_sort(self, pop, function):
    #     pop = np.array(pop)
    #     costs = [function(ind) for ind in pop]
    #     sorted_indices = np.argsort(costs)
    #     return pop[sorted_indices]
    #
    # def create_prt_vector(self, prt, pop_size, num_of_x):
    #     prt_vector = np.zeros((num_of_x, pop_size))
    #     for j in range(pop_size):
    #         for i in range(num_of_x):
    #             if random.random() < prt:
    #                 prt_vector[i][j] = 1
    #     return prt_vector
    #
    # def movement(self, pop, prt, NStep, function):
    #     pop_size = len(pop)
    #     num_of_x = len(pop[0])
    #     new_pop = deepcopy(pop)
    #     PRT = self.create_prt_vector(prt, pop_size, num_of_x)
    #
    #     for i in range(pop_size):
    #         vector_of_steps = np.zeros((num_of_x, 4 * NStep))
    #         for s in range(4 * NStep):
    #             vector_of_steps[:, s] = new_pop[i] + ((new_pop[0] - new_pop[i]) / (2 * NStep)) * PRT[:, i] * s
    #         new_pop[i] = self.cost_function_sort(vector_of_steps.T, function)[0]
    #
    #     for i in range(pop_size, pop_size * 2):
    #         vector_of_steps = np.zeros((num_of_x, 2 * NStep))
    #         for s in range(2 * NStep):
    #             vector_of_steps[:, s] = new_pop[i % pop_size] + ((new_pop[1] - new_pop[i % pop_size]) / NStep) * PRT[:,
    #                                                                                                              i % pop_size] * s
    #         new_pop[i % pop_size] = self.cost_function_sort(vector_of_steps.T, function)[0]
    #
    #     for i in range(pop_size * 2, pop_size * 3):
    #         vector_of_steps = np.zeros((num_of_x, NStep))
    #         for s in range(NStep):
    #             vector_of_steps[:, s] = new_pop[i % pop_size] + (
    #                     (new_pop[2] - new_pop[i % pop_size]) / (NStep / 2)) * PRT[:, i % pop_size] * s
    #         new_pop[i % pop_size] = self.cost_function_sort(vector_of_steps.T, function)[0]
    #
    #     return new_pop
    #
    # def elaboration(self, pop, NStep, prt, function):
    #     num_of_x = len(pop[0])
    #     NStep *= 10
    #     PRT = self.create_prt_vector(prt, 3, num_of_x)
    #     vector_of_steps = np.zeros((num_of_x, NStep))
    #     ans = np.zeros((num_of_x, 3))
    #     for i in range(3):
    #         for s in range(NStep):
    #             vector_of_steps[:, s] = pop[i] + ((pop[0] - pop[i]) / (NStep / 2)) * PRT[:, i] * s
    #         ans[:, i] = self.cost_function_sort(vector_of_steps.T, function)[0]
    #     ans = self.cost_function_sort(ans.T, function)
    #     return ans[0]
    #
    # def run_function(self, function: callable) -> Result:
    #     dimension = len(function.range)
    #     pop = self.generate_population(function, dimension)
    #     pop = self.cost_function_sort(pop, function)
    #     new_pop = np.concatenate((pop, pop, pop), axis=0)
    #
    #     MCount = 0
    #     result = Result()
    #
    #     while MCount < self.Migrations and math.sqrt((1 / 2) * ((function(new_pop[1]) - function(new_pop[0])) ** 2 + (
    #             function(new_pop[2]) - function(new_pop[0])) ** 2)) >= self.MinDist:
    #         new_pop = self.movement(new_pop, self.prt, self.NStep, function)
    #         new_pop = self.cost_function_sort(new_pop, function)
    #         new_pop = np.concatenate(
    #             (new_pop[:round(self.NP * (2 / 3))], self.generate_population(function, dimension)), axis=0)
    #         new_pop = self.cost_function_sort(new_pop, function)
    #         new_pop = np.concatenate((new_pop, new_pop, new_pop), axis=0)
    #
    #         MCount += 1
    #
    #         iteration = Iteration()
    #         for pos in new_pop[:self.NP]:
    #             iteration.add_position(Position(function(pos), pos))
    #         result.add_iteration(iteration)
    #
    #     last_pop = new_pop[:3]
    #     ans = self.elaboration(last_pop, self.NStep, self.prt, function)
    #
    #     self.result[function] = result
    #     return self.result[function]
    #
    # def run_all(self) -> dict[callable, Result]:
    #     for function in self.functions.get_all():
    #         self.run_function(function)
    #     return self.result
    #
    # def render(self, function: callable, is_2d: bool = False):
    #     render = Render3D()
    #
    #     if function in self.result:
    #         render.render3d(self.result[function], function)
    #     else:
    #         self.run_function(function)
    #         render.render3d(self.result[function], function)
    #
    # def render_all(self):
    #     render = Render3D()
    #     for function in self.functions.get_all():
    #         render.render3d(self.result[function], function)
