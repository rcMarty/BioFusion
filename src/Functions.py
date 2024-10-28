# new class with all test functions which can be parametrized
# Test functions - Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov, Ackley,

import math

import numpy as np

from src.utils.Utils import singleton, set_range


@singleton
class Function:

    def __init__(self):
        self.range = None

    def get_all(self):
        return [self.sphere, self.schwefel, self.rosenbrock, self.rastrigin, self.griewank, self.levy, self.michalewicz,
                self.zakharov, self.ackley]

    @set_range((-5.12, 5.12))
    def sphere(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-5.12, 5.12]

            The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal. The plot shows its two-dimensional form.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.

        :param xx: -> np.array
        :return: -> float
        """
        return np.sum(xx ** 2)

    @set_range((-500, 500))
    def schwefel(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-500, 500]

            The Schwefel function is complex, with many local minima. The plot shows the two-dimensional form of the function.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, d.

        Global Minimum:
            - f(x*) = 0 at x* = (420.9687, 420.9687)
        :param xx: -> np.array
        :return: -> float
        """
        d = len(xx)
        return 418.9829 * d - np.sum(xx * np.sin(np.sqrt(np.abs(xx))))

    @set_range((-10, 10))
    def rosenbrock(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-5, 10]

            The Rosenbrock function, also referred to as the Valley or Banana function, is a popular test problem for gradient-based optimization algorithms. It is shown in the plot above in its two-dimensional form.

            The function is unimodal, and the global minimum lies in a narrow, parabolic valley. However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d, although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, d.

        Global Minimum:
            - f(x*) = 0 at x* = (1, …, 1)

        :param xx: -> np.array
        :return: -> float
        """

        d = len(xx)
        xi = xx[0:d - 1]
        xnext = xx[1:d]
        sum = np.sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2)
        return sum

    @set_range((-5.12, 5.12))
    def rastrigin(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-5.12, 5.12]

            The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.

        Global Minimum:
            - f(x*) = 0 at x* = (0, …, 0)
        :param xx:
        :return:
        """
        d = len(xx)
        return 10 * d + np.sum(xx ** 2 - 10 * np.cos(2 * math.pi * xx))

    @set_range((-50, 50))
    def griewank(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-600, 600]

            The Griewank function has many widespread local minima, which are regularly distributed. The complexity is shown in the zoomed-in plots.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-600, 600], for all i = 1, …, d.

        Global Minimum:
            - f(x*) = 0 at x* = (0, …, 0)

        Source:
            https://www.sfu.ca/~ssurjano/griewank.html
        :param xx: -> np.array
        :return: -> float
        """
        ii = np.arange(1, len(xx) + 1)
        sum = np.sum(xx ** 2 / 4000)
        prod = np.prod(np.cos(xx / np.sqrt(ii)))
        return sum - prod + 1

    @set_range((-10, 10))
    def levy(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-10, 10]

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, d.

        Source:
            https://www.sfu.ca/~ssurjano/Code/levyr.html
        :param xx: -> np.array
        :return: -> float
        """

        d = len(xx)
        w = 1 + (xx - 1) / 4
        term1 = (np.sin(math.pi * w[0])) ** 2  # R <- (sin(pi*w[1]))^2
        term3 = (w[d - 1] - 1) ** 2 * (
                1 + (np.sin(2 * math.pi * w[d - 1])) ** 2)  # R <- (w[d]-1)^2*(1+(sin(2*pi*w[d]))^2)
        wi = w[0:d - 1]
        sum = np.sum(
            (wi - 1) ** 2 * (1 + 10 * (np.sin(math.pi * wi + 1)) ** 2))  # R <- sum((wi-1)^2*(1+10*(sin(pi*wi+1))^2))
        return term1 + sum + term3

    @set_range((0, math.pi))
    def michalewicz(self, xx: np.ndarray, m=10) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[0, pi]

            The Michalewicz function has d! local minima, and it is multimodal. The parameter m defines the steepness of they valleys and ridges; a larger m leads to a more difficult search. The recommended value of m is m = 10. The function's two-dimensional form is shown in the plot above.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [0, π], for all i = 1, …, d.

        Global Minimum:
            - d = 2, f(x*) = -1.8013 at x* = (2.20, 1.57)
            - d = 5, f(x*) = -4.6877
            - d = 10 f(x*) = -9.66015

        Source:
            https://www.sfu.ca/~ssurjano/Code/michal.html
        :param xx: -> np.array
        :param m: -> int constant (optional with default value of 10)
        :return: -> float
        """

        d = len(xx)
        i = np.arange(d) + 1
        return -np.sum(np.sin(xx) * np.sin(i * xx ** 2 / math.pi) ** (2 * m))

    @set_range((-10, 10))
    def zakharov(self, xx: np.ndarray) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-5, 10]

            The Zakharov function has no local minima except the global one. It is shown here in its two-dimensional form.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d.

        Source:
            https://www.sfu.ca/~ssurjano/Code/zakharov.html
        :param xx: -> np.array
        :return: -> float
        """

        d = len(xx)
        i = np.arange(d) + 1
        sum1 = np.sum(xx ** 2)
        sum2 = np.sum(0.5 * i * xx)
        return sum1 + sum2 ** 2 + sum2 ** 4

    @set_range((-32.768, 32.768))
    def ackley(self, xx: np.ndarray, a=20, b=0.2, c=(2 * math.pi)) -> float:
        """
        Description:
            - Dimensions: length of array xx
            - Search domain :[-32.768, 32.768]

            The Ackley function is widely used for testing optimization algorithms. In its two-dimensional form, as shown in the plot above, it is characterized by a nearly flat outer region, and a large hole at the centre. The function poses a risk for optimization algorithms, particularly hillclimbing algorithms, to be trapped in one of its many local minima.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all i = 1, …, d, although it may also be restricted to a smaller domain.

        Source:
            https://www.sfu.ca/~ssurjano/Code/ackley.html
        :param xx: -> np.array
        :param a: -> float (optional with default value of 20)
        :param b: -> float (optional with default value of 0.2)
        :param c: -> float (optional with default value of 2 * math.pi)
        :return: -> float
        """

        d = len(xx)
        sum1 = np.sum(xx ** 2)
        sum2 = np.sum(np.cos(c * xx))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + math.exp(1)
