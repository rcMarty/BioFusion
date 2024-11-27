from src.algorithms import *
from src.render.Render2D import Render2D

if __name__ == '__main__':
    # algorithm = alg.BlindSearch(Function())
    # Render3D(100, 1, 0.0001, per_point=True)
    # algorithm = alg.SimAnnealing(Function())
    # algorithm.run_all()
    # algorithm.render_all()
    # algorithm.render(Function().ackley)
    # algorithm.render(Function().rosenbrock)

    # algorithm = GeneticAlgorithm(cities=50, population=200, generations=1000, mutation_rate=0.5)
    # algorithm.run()
    # algorithm.save("../results/", "data")
    # algorithm = GeneticAlgorithm.load("../results/", "data")
    # algorithm.render()

    # Render3D(per_generation_animation=True, only_best=False, as_surface=True, wait_iteration=0.5, new=False)

    # algorithm = DifferentialEvolution(Function())

    # algorithm = ParticleSwarmOptimization(Function())

    Render2D()

    algorithm = AntColonyOptimization(cities=60)

    algorithm.run()
    algorithm.render()
