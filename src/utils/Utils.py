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
