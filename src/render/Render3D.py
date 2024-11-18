import random as pyrand

import matplotlib.pyplot as plt
import numpy as np

from src.utils.Result import Result, Iteration
from src.utils.Utils import singleton


@singleton
class Render3D:

    def __init__(self, resolution: int = 100, wait: float = 5, wait_iteration: float = 1, new: bool = False,
                 per_point: bool = False, only_best: bool = False, per_generation_animation: bool = False,
                 as_surface: bool = False):
        self.as_surface = as_surface
        self.per_generation_animation = per_generation_animation
        self.new = new
        self.only_best = only_best
        self.per_point = per_point
        self.resolution = resolution
        plt.switch_backend('TkAgg')
        
        if not self.new:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        if self.as_surface:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot()
        self.colors = ['red', 'blue', 'purple', 'orange', 'black', 'pink', 'brown', 'magenta']
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

        if self.as_surface:
            contour = self.ax.contourf(X, Y, Z, cmap='viridis', alpha=0.75)
            self.fig.colorbar(contour)
        else:
            self.ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='black', linewidth=0.01, alpha=0.1)
        if clear:
            plt.pause(time)
            self.ax.clear()

    def render_iteration(self, iteration: Iteration):

        pts = []
        best_pts = []
        color = pyrand.choice(self.colors)

        if not self.only_best:
            for point in iteration.history:
                if self.as_surface:
                    pts.append(self.ax.scatter(point.position[0], point.position[1], color=color, marker='x', s=10))
                else:
                    pts.append(
                        self.ax.scatter(point.position[0], point.position[1], point.value, color=color, marker='x',
                                        s=10))
                if self.per_point:
                    plt.pause(self.wait_iteration / 10)

        if self.as_surface:
            best_pts.append(
                self.ax.scatter(iteration.best.position[0], iteration.best.position[1], color='green',
                                s=100))
        else:
            best_pts.append(
                self.ax.scatter(iteration.best.position[0], iteration.best.position[1], iteration.best.value,
                                color='green', s=100))

        plt.pause(self.wait_iteration)

        if self.per_generation_animation:
            for pt in pts:
                pt.remove()
            best_pts[0].remove()

    def render3d(self, result: Result, function: callable):

        if self.new:
            if self.as_surface:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot()
            else:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')

        best_position = result.get_best()
        plt.title(
            f"{function.__name__} \nBest Position: {best_position[1]:.2f}, {best_position[2]:.2f} \nwith Value: {best_position[0].value}")

        if function is not None:
            self.render_graph(function)
        for iteration in result.iterations:
            print(
                f"Best position: {iteration.best.position} with value: {iteration.best.value} for function: {function.__name__}")
            self.render_iteration(iteration)
        if self.as_surface:
            self.ax.scatter(best_position[1], best_position[2], color='blue', s=300)
            plt.savefig(f"../results/{function.__name__}_2d.png")
        else:
            self.ax.scatter(best_position[1], best_position[2], best_position[0].value, color='blue', s=300)
            plt.savefig(f"../results/{function.__name__}.png")

        if self.new:
            plt.show()
        else:
            plt.pause(self.wait)

        self.ax.clear()
