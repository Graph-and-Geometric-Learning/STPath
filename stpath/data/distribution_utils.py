import math
import random
import numpy as np
from functools import partial


def is_return_ratio(name: str):
    if "batch" in name:
        return False
    elif "uniform" in name and len(name.split("_")) > 0:
        return False
    else:
        return True


def get_distribution(name: str):
    if "constant" in name or "batch" in name:  # constant_0.2, batch_128
        ratio = float(name.split("_")[1])
        return partial(constant_distribution, ratio=ratio)
    elif "beta" in name:
        alpha, beta = [float(x) for x in name.split("_")[1:]]
        return partial(beta_distribution, alpha=alpha, beta=beta)
    elif name == "uniform":
        return uniform_distribution
    elif "uniform" in name:
        low, high = [int(x) for x in name.split("_")[1:]]
        return partial(uniform_distribution_range, low=low, high=high)
    elif name == "cosine":
        return cosine_distribution
    elif name == "sqrt":
        return square_root_distribution
    elif name == "square":
        return square_distribution
    else:
        raise ValueError(f"Unknown distribution: {name}")


def constant_distribution(ratio=0.2):
    return ratio


def uniform_distribution():
    return np.random.rand()


def uniform_distribution_range(low, high):
    return random.randint(low, high - 1)


def beta_distribution(alpha=3, beta=9):
    return np.random.beta(alpha, beta)


def cosine_distribution():
    return (1 - math.cos(np.random.rand() * math.pi * 0.5))


def square_root_distribution():
    return math.sqrt(np.random.rand())


def square_distribution():
    return np.random.rand() ** 2
