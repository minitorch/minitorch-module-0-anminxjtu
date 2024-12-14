"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    if x < y:
        return 1.0
    else:
        return 0.0

def eq(x: float, y: float) -> float:
    if -1e-3 <= x-y <= 1e-3:
        return 1.0
    else:
        return 0.0

def max(x: float, y: float) -> float:
    if x > y:
        return x
    else:
        return y

def is_close(x: float, y: float) ->float:
    if abs(x - y) < 1e-2:
        return 1.0
    else:
        return 0.0

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x)) 
    
    
def relu(x: float) -> float:
    if x >= 0:
        return x
    else:
        return 0

EPS = 1e-6

def log(x: float) -> float:
    # assert x > 0 
    return math.log(x + EPS)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    assert x != 0
    return 1/x

def log_back(x: float, d: float) -> float:
    assert x != 0
    return d / x

def inv_back(x: float, d: float) -> float:
    assert x != 0
    return -d /(x**2)

def relu_back(x: float, d: float) -> float:
    if x >= 0:
        return d
    else:
        return 0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:

    def apply_to_iterable(values: Iterable[float]) -> Iterable[float]:
        return [fn(value) for value in values]
    return apply_to_iterable

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:

    def apply_zip(values_1: Iterable[float], values_2: Iterable[float]) -> Iterable[float]:
        len1 = len(values_1)
        len2 = len(values_2)
        assert len1 == len2
        results = []
        for i in range(len1):
            results.append(fn(values_1[i], values_2[i]))
        return results
    return apply_zip

# def zipWith(
#     fn: Callable[[float, float], float]
# ) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:

#     def apply_zip(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
#         return [fn(x, y) for x, y in zip(ls1, ls2)]
#     return apply_zip

def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:

    def apply_reduce(ls: Iterable[float]) -> float:
        results = fn(ls[0], ls[1])
        for i in range(2, len(ls)):
            results = fn(results, ls[i])
        return results
    return apply_reduce

def negList(ls: Iterable[float]) -> Iterable[float]:
    neg_map = map(neg)
    return neg_map(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    add_zip = zipWith(add)
    return add_zip(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    if len(ls) == 0:
        return 0
    if len(ls) == 1:
        return ls[0]
    sum_reduce = reduce(add)
    return sum_reduce(ls)

def prod(ls: Iterable[float]) -> float:
    prod_reduce = reduce(mul)
    return prod_reduce(ls)

