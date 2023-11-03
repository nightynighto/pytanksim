# -*- coding: utf-8 -*-
"""Module for calculating derivatives numerically."""

__all__ = [
    "partial_derivative",
    "second_derivative",
    "mixed_second_derivative",
    "backward_partial_derivative",
    "forward_partial_derivative",
    "pardev",
    "backdev",
    "fordev",
    "secbackder",
    "secforder",
    "second_backward_derivative",
    "second_forward_derivative"
    ]

from typing import Callable, List


def pardev(func: Callable[[float], float],
           loc: float, stepsize: float) -> float:
    """Calculate the first derivative using centered finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    func : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    loc : float
        Location where the first derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation.

    Returns
    -------
    float
        The first derivative of  `func` evaluated at `loc`.

    """
    loc1 = (loc + stepsize)
    loc2 = (loc - stepsize)
    term1 = func(loc1)
    term2 = func(loc2)
    return (term1 - term2) / (loc1 - loc2)


def partial_derivative(func: Callable[..., float], var: int,
                       point: List, stepsize: float = 1e-3) -> float:
    """Calculate the first partial derivative using centered finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the partial derivative w.r.t. `x3`, then `var`
        should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-3.

    Returns
    -------
    float
        The first partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return pardev(wraps, point[var], stepsize)


def backdev(func: Callable[[float], float], loc: float,
            stepsize: float) -> float:
    """Calculate the first derivative using backwards finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    func : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    loc : float
        Location where the first derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation.

    Returns
    -------
    float
        The first derivative of `func` evaluated at `loc`.

    """
    loc2 = loc - stepsize
    loc3 = loc2 - stepsize
    term1 = func(loc)
    term2 = func(loc2)
    term3 = func(loc3)
    return (3*term1 - 4*term2 + term3) / (loc-loc3)


def backward_partial_derivative(func: Callable[..., float], var: int,
                                point: List, stepsize: float = 1e-3) -> float:
    """Find the first partial derivative using backwards finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the partial derivative w.r.t. `x3`, then `var`
        should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-3.

    Returns
    -------
    float
        The first partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return backdev(wraps, point[var], stepsize)


def fordev(func: Callable[[float], float], loc: float,
           stepsize: float) -> float:
    """Calculate the first derivative using forwards finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    func : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    loc : float
        Location where the first derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation.

    Returns
    -------
    float
        The first derivative of `func` evaluated at `loc`.

    """
    loc2 = loc + stepsize
    loc3 = loc2 + stepsize
    term1 = func(loc)
    term2 = func(loc2)
    term3 = func(loc3)
    return (- 3 * term1 + 4 * term2 - term3) / (loc3 - loc)


def forward_partial_derivative(func: Callable[..., float], var: int,
                               point: List, stepsize: float = 1e-3) -> float:
    """Find the first partial derivative using forwards finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the partial derivative w.r.t. `x3`, then `var`
        should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-3.

    Returns
    -------
    float
        The first partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return fordev(wraps, point[var], stepsize)


def secder(function: Callable[[float], float], location: float,
           stepsize: float = 1e-6) -> float:
    """Calculate the second derivative using centered finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    function : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    location : float
        Location where the second derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-6.

    Returns
    -------
    float
        The second derivative of `function` evaluated at `location`.

    """
    loc1 = (location - stepsize)
    loc2 = (location + stepsize)
    term1 = function(loc1)
    term2 = - 2 * function(location)
    term3 = function(loc2)
    stepsize = (loc2-loc1)/2
    return (term1 + term2 + term3) / (stepsize ** 2)


def second_derivative(func: Callable[..., float], var: int,
                      point: List, stepsize: float = 1e-6) -> float:
    """Find the second partial derivative using centered finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the second partial derivative w.r.t. `x3`, then
        `var` should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the second partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-6.

    Returns
    -------
    float
        The second partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return secder(wraps, point, stepsize)


def secforder(function: Callable[[float], float], location: float,
              stepsize: float = 1e-6) -> float:
    """Calculate the second derivative using forwards finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    function : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    location : float
        Location where the second derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-6.

    Returns
    -------
    float
        The second derivative of `function` evaluated at `location`.

    """
    loc1 = (location + stepsize)
    loc2 = (location + 2 * stepsize)
    term1 = function(location)
    term2 = -2 * function(loc1)
    term3 = function(loc2)
    stepsize = (loc2-location)/2
    return (term1 + term2 + term3) / (stepsize ** 2)


def second_forward_derivative(func: Callable[..., float], var: int,
                              point: List, stepsize: float = 1e-6) -> float:
    """Find the second partial derivative using forwards finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the second partial derivative w.r.t. `x3`, then
        `var` should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the second partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-6.

    Returns
    -------
    float
        The second partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return secforder(wraps, point, stepsize)


def secbackder(function: Callable[[float], float], location: float,
               stepsize: float = 1e-6) -> float:
    """Calculate the second derivative using backwards finite difference.

    This function in particular only works with functions which have only one
    argument.

    Parameters
    ----------
    function : Callable[[float], float]
        A function that takes in a floating point number and outputs a floating
        point number.

    location : float
        Location where the second derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation.

    Returns
    -------
    float
        The first derivative of `function` evaluated at `location`. The default
        is 1e-6.

    """
    loc1 = (location - stepsize)
    loc2 = (location - 2 * stepsize)
    term1 = function(location)
    term2 = - 2 * function(loc1)
    term3 = function(loc2)
    stepsize = (location-loc2)/2
    return (term1 + term2 + term3) / (stepsize ** 2)


def second_backward_derivative(func: Callable[..., float], var: int,
                               point: List, stepsize: float = 1e-6) -> float:
    """Find the second partial derivative using backwards finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (one or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least one
        argument which is a floating point number.

    var : int
        The integer showing the input order of the independent variable with
        respect to which the derivative of `func` is to be approximated. It
        uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
        ...). For example, if the function `func` has the following
        signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the second partial derivative w.r.t. `x3`, then
        `var` should be 2.

    point : List
        A list of input values for `func`. These input values indicate the
        location where the second partial derivative is to be evaluated.

    stepsize : float
        The stepsize for the finite difference approximation. The default is
        1e-6.

    Returns
    -------
    float
        The second partial derivative of `func` w.r.t. the variable specified
        by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)
    return secbackder(wraps, point, stepsize)


def mixsecder(function: Callable[[float, float], float], location: List[float],
              stepsize: List[float] = [1E3, 1E-4]) -> float:
    """Calculate a mixed variable second derivative using finite difference.

    This function in particular only works with functions which have only two
    arguments.

    Parameters
    ----------
    function : Callable[[float, float], float]
        A function that takes in two floating point numbers and outputs a
        floating point number.

    location : List[float]
        Location where the mixed second derivative is to be evaluated. It is a
        list of two floating point numbers.

    stepsize : List[float]
        The stepsizes for the finite difference approximation. It is a list
        of two floating point numbers. The default is [1E3, 1E-4].

    Returns
    -------
    float
        The mixed second derivative of `function` evaluated at `location`.

    """
    fpospos = function(location[0]+stepsize[0], location[1]+stepsize[1])
    fposmin = function(location[0]+stepsize[0], location[1]-stepsize[1])
    fminpos = function(location[0]-stepsize[0], location[1]+stepsize[1])
    fminmin = function(location[0]-stepsize[0], location[1]-stepsize[1])
    return (fpospos - fposmin - fminpos + fminmin) / (4 * (stepsize[0]
                                                           * stepsize[1]))


def mixed_second_derivative(func: Callable[..., float], var: List[int],
                            point: List,
                            stepsize: List[float] = [1E3, 1E-4]) -> float:
    """Find the mixed second derivative using finite difference.

    This version of the function works for functions which have an arbitrary
    number of arguments (two or more).

    Parameters
    ----------
    func : Callable[... , float]
        A function that outputs a floating point number, and has at least two
        arguments which are floating point numbers.

    var : List[int]
        A list of integers showing the input order of the two variables
        with respect to which the mixed second derivative of `func` is to be
        approximated. It uses python's convention for indexing (i.e., it counts
        from 0, 1, 2, 3, ...). For example, if the function `func` has the
        following signature::

            def some_function(x1, x2, x3):
                ....
                return y

        If we want to find the mixed second partial derivative w.r.t. `x1` and
        `x3`, then `var` should be [0, 2].

    point : List
        A list of input values for `func`. These input values indicate the
        location where the mixed second partial derivative is to be evaluated.

    stepsize : List[float]
        The stepsizes for the finite difference approximation. It is a list
        of two floating point numbers. The default is [1E3, 1E-4].

    Returns
    -------
    float
        The mixed second partial derivative of `func` w.r.t. the two variables
        specified by `var` evaluated at `point`.

    """
    args = point[:]

    def wraps(x):
        args[var[0]] = x[0]
        args[var[1]] = x[1]
        return func(*args)
    return mixsecder(wraps, [point[var[0]], point[var[1]]], stepsize)
