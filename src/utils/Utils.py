import functools


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


# src/utils/Utils.py
def set_range(range_tuple):
    def decorator(func):
        func.range = range_tuple
        return func

    return decorator


def set_dimension(range):
    def decorator(func):
        func.dimension = range
        return func

    return decorator


@singleton
class FunctionCallCounter:
    def __init__(self, max_calls=3000):
        self.counts = {}
        self.max_calls = max_calls

    def set_max_calls(self, max_calls):
        self.max_calls = max_calls

    def get_max_calls(self):
        return self.max_calls

    def count_calls(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if func.__name__ not in self.counts:
                self.counts[func.__name__] = 0
            self.counts[func.__name__] += 1
            return func(*args, **kwargs)

        return wrapper

    def get_counts(self, func: callable = None) -> int:
        if func and func.__name__ not in self.counts:
            return 0
        return self.counts[func.__name__] if func else self.counts

    def reset_counts(self):
        self.counts = {}


functions_call_counter = FunctionCallCounter(int(1e100))
