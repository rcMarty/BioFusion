from src.Algorithms import BlindSearch
from src.Functions import Function

if __name__ == '__main__':
    algorithm = BlindSearch(Function(), 10, 10)
    algorithm.run_all()
    algorithm.render_all()
    # algorithm.render(Function().rosenbrock)
