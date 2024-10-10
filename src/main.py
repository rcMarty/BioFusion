import src.algorithms as alg
from src.Functions import Function

if __name__ == '__main__':
    # algorithm = alg.BlindSearch(Function())
    algorithm = alg.HillClimb(Function())
    algorithm.run_all()
    algorithm.render_all()
    # algorithm.render(Function().ackley)
    # algorithm.render(Function().rosenbrock)
