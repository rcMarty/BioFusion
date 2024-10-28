from matplotlib import pyplot as plt

from src.utils.Genetic import Individual, Generation
from src.utils.Utils import singleton


@singleton
class Render2D:
    def __init__(self):
        plt.switch_backend('tkagg')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def plot_individual(self, individual: Individual):
        x_coords = [point.x for point in individual.points]
        y_coords = [point.y for point in individual.points]
        names = [point.name for point in individual.points]

        self.ax.clear()
        self.ax.scatter(x_coords, y_coords, color='blue')

        for i, name in enumerate(names):
            self.ax.text(x_coords[i], y_coords[i], name, fontsize=12, ha='right')

        self.ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]],
                     color='red')  # Connect the last point to the first

        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Points and Connections')
        self.ax.grid(True)

        # print individual fitness for points and lenght of path
        print("Points: ", individual.points)
        print("Individual fitness: ", individual.fitness)

        # plt.show()

    def plot_generation(self, generation: list[Generation]):
        self.ax.clear()
        for gen in generation:
            self.plot_individual(gen.best_ind)
            plt.pause(0.1)
            self.ax.clear()

        # plt.show()
