from matplotlib import pyplot as plt

from src.utils.Genetic import Individual, Generation
from src.utils.Utils import singleton


@singleton
class Render2D:
    def __init__(self):
        plt.switch_backend('tkagg')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.index = 0
        # self.ax.set_xlim(0, 10)
        # self.ax.set_ylim(0, 10)
        # self.ax.set_aspect('equal')

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
        self.ax.set_title(f'Points and Connections, generation {self.index} \n Fitness: ' + str(individual.fitness))
        self.ax.grid(True)

        # print individual fitness for points and lenght of path
        # print("Points: ", individual.points)
        # print("Individual fitness: ", individual.fitness)

        # plt.show()

    def plot_generation(self, generation: list[Generation], nth: int = 5):
        self.ax.clear()

        for (index, gen) in enumerate(generation):
            self.index = index
            # print(f"Generation {index + 1}")
            self.plot_individual(gen.best_ind)
            if index % int((len(generation) / nth)) == 0:
                plt.savefig(f'../results/generation_{index + 1}.png')  # Save the plot at each interval
            plt.pause(0.000001)
            self.ax.clear()

        # plt.show()
