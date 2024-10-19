import src.algorithms as alg
from src.Functions import Function
from src.render.Render import Render

if __name__ == '__main__':
    # algorithm = alg.BlindSearch(Function())
    Render(100, 0.0001, 0.0001)
    algorithm = alg.SimAnnealing(Function())
    algorithm.run_all()
    algorithm.render_all()
    # algorithm.render(Function().ackley)
    # algorithm.render(Function().rosenbrock)
