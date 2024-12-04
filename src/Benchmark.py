from src.Functions import Function
from src.algorithms import DifferentialEvolution, FireflyAlgorithm, ParticleSwarmOptimization, TeachingLearningBasedOptimization, SelfOrganizingMigrationAlgorithm
from src.utils.Utils import functions_call_counter, FunctionCallCounter


class Benchmark:
    """
    class for benchmarking the performance of the algorithms which can be used on Functions

    metrics is

    In another words
    - BlindSearch
    - DifferentialEvolution
    - FireflyAlgorithm
    - HillClimbing
    - ParticleSwarmOptimization
    - SimAnnealing
    - SelfOrganizingMigratingAlgorithm
    - TeachingLearningBasedOptimization
    """

    def __init__(self, NP: int = 30, max_calls: int = 3000, dimensions: int = 20, number_of_tests: int = 30):
        self.max_calls = max_calls
        self.number_of_tests = number_of_tests
        self.functions = Function(dimensions)
        self.np = NP
        np = NP
        self.algorithms = [
            # BlindSearch(self.functions),
            # SimAnnealing(self.functions),
            # HillClimb(self.functions),
            DifferentialEvolution(self.functions, NP=np),
            FireflyAlgorithm(self.functions, NP=np),
            ParticleSwarmOptimization(self.functions, NP=np),
            SelfOrganizingMigrationAlgorithm(self.functions, NP=np),
            TeachingLearningBasedOptimization(self.functions, NP=np)
        ]

        self.all_results: list[dict] = []

    def run(self):
        FunctionCallCounter().set_max_calls(self.max_calls)
        assert functions_call_counter.get_counts() == {}

        print("Benchmark started")

        for _ in range(self.number_of_tests):
            print(f"Test {_ + 1}/{self.number_of_tests}")
            results = {}
            for algorithm in self.algorithms:
                algorithm.result = {}
                functions_call_counter.reset_counts()
                assert functions_call_counter.get_counts() == {}
                print(f"\tRunning {algorithm.__class__.__name__}", end=" ")
                results[algorithm] = algorithm.run_all()
                print("done")

            self.all_results.append(results)

        print("Benchmark finished")

    def save(self):
        """
        save to csv table the results of the benchmark
        structure is like this for each function:
        in the column is number of expoeriment and on the row is the algorihm

        [
            sphere:[
                1: {
                    DE: 0.001,
                    FA: 0.002,
                    PSO: 0.003,
                    SOMA: 0.004,
                    TLO: 0.005
                },
                2: {
                    DE: 0.001,
                    FA: 0.002,
                    PSO: 0.003,
                    SOMA: 0.004,
                    TLO: 0.005
                }, .....
            ],
            rosenbrock:[
                1: {
                    DE: 0.001,
                    FA: 0.002,
                    PSO: 0.003,
                    SOMA: 0.004,
                    TLO: 0.005
                },
                2: {
                    DE: 0.001,
                    FA: 0.002,
                    PSO: 0.003,
                    SOMA: 0.004,
                    TLO: 0.005
                }, .....
            ], ....
        ]

        :return: list of dictionaries where each dictionary is for one function
        """

        # save raw data to csv
        with open(f"../results/raw_data.csv", "w") as file:
            file.write("Algorithm,Function,Value\n")
            for test_index, test_results in enumerate(self.all_results, start=1):
                for algorithm, function_results in test_results.items():
                    for function, result in function_results.items():
                        file.write(f"{algorithm.__class__.__name__},{function.__name__},{result.get_best()[0].value}\n")
                file.write("\n")

        structured_data = {}

        for test_index, test_results in enumerate(self.all_results, start=1):
            for algorithm, function_results in test_results.items():
                for function, result in function_results.items():
                    function_name = function.__name__
                    if function_name not in structured_data:
                        structured_data[function_name] = []
                    if len(structured_data[function_name]) < test_index:
                        structured_data[function_name].append({})
                    structured_data[function_name][test_index - 1][algorithm.__class__.__name__] = result.get_best()[0].value

        # save this structure to each csv file for each function
        for function_name, data in structured_data.items():
            with open(f"../results/{function_name}.csv", "w") as file:
                file.write("Algorithm,")
                for i in range(1, self.number_of_tests + 1):
                    file.write(f"Test {i},")
                file.write("\n")
                for algorithm in self.algorithms:
                    algorithm_name = algorithm.__class__.__name__
                    file.write(f"{algorithm_name},")
                    for test in data:
                        file.write(f"{test[algorithm_name]},")
                    file.write("\n")

        return structured_data
