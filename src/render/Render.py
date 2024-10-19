import random as pyrand

import matplotlib.pyplot as plt
import numpy as np

from src.utils.Result import Result, Iteration
from src.utils.Utils import singleton


@singleton
class Render:

    def __init__(self, resolution: int = 100, wait: float = 5, wait_iteration: float = 1):
        self.resolution = resolution
        plt.switch_backend('tkagg')
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.colors = ['red', 'blue', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'cyan', 'magenta']
        self.wait = wait
        self.wait_iteration = wait_iteration

    def render_graph(self, function: callable, clear: bool = False, time: int = 10):
        x = np.linspace(function.range[0], function.range[1], self.resolution)
        y = np.linspace(function.range[0], function.range[1], self.resolution)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
        self.ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='black', linewidth=0.01, alpha=0.1)
        if clear:
            plt.pause(time)
            self.ax.clear()

    def render_iteration(self, iteration: Iteration):

        pts = []
        best_pts = []

        color = pyrand.choice(self.colors)

        for point in iteration.history:
            pts.append(
                self.ax.scatter(point.position[0], point.position[1], point.value, color=color, alpha=1,
                                s=1))
            # plt.pause(0.001)

        best_pts.append(
            self.ax.scatter(iteration.best.position[0], iteration.best.position[1], iteration.best.value,
                            color='green'))

        plt.pause(self.wait_iteration)

    def render(self, result: Result, function: callable):
        if function is not None:
            self.render_graph(function)
        for iteration in result.iterations:
            print(
                f"Best position: {iteration.best.position} with value: {iteration.best.value} for function: {function.__name__}")
            self.render_iteration(iteration)

        plt.savefig(f"../results/{function.__name__}.png")
        plt.pause(self.wait)
        self.ax.clear()
