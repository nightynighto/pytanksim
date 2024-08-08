pytanksim.utils
===============

.. py:module:: pytanksim.utils

.. autoapi-nested-parse::

   Various aditional utilities used in pytanksim simulations.

   Includes logging functions and functions to calculate differentials using the
   finite difference method.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/pytanksim/utils/finitedifferences/index
   /autoapi/pytanksim/utils/logging/index


Functions
---------

.. autoapisummary::

   pytanksim.utils.partial_derivative
   pytanksim.utils.second_derivative
   pytanksim.utils.mixed_second_derivative
   pytanksim.utils.backward_partial_derivative
   pytanksim.utils.forward_partial_derivative
   pytanksim.utils.pardev
   pytanksim.utils.backdev
   pytanksim.utils.fordev
   pytanksim.utils.secbackder
   pytanksim.utils.secforder
   pytanksim.utils.second_backward_derivative
   pytanksim.utils.second_forward_derivative


Package Contents
----------------

.. py:function:: partial_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 0.001) -> float

   Calculate the first partial derivative using centered finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the partial derivative w.r.t. `x3`, then `var`
               should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-3.
   :type stepsize: float

   :returns: The first partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


.. py:function:: second_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 1e-06) -> float

   Find the second partial derivative using centered finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the second partial derivative w.r.t. `x3`, then
               `var` should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the second partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-6.
   :type stepsize: float

   :returns: The second partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


.. py:function:: mixed_second_derivative(func: Callable[Ellipsis, float], var: List[int], point: List, stepsize: List[float] = [1000.0, 0.0001]) -> float

   Find the mixed second derivative using finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (two or more).

   :param func: A function that outputs a floating point number, and has at least two
                arguments which are floating point numbers.
   :type func: Callable[... , float]
   :param var: A list of integers showing the input order of the two variables
               with respect to which the mixed second derivative of `func` is to be
               approximated. It uses python's convention for indexing (i.e., it counts
               from 0, 1, 2, 3, ...). For example, if the function `func` has the
               following signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the mixed second partial derivative w.r.t. `x1` and
               `x3`, then `var` should be [0, 2].
   :type var: List[int]
   :param point: A list of input values for `func`. These input values indicate the
                 location where the mixed second partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsizes for the finite difference approximation. It is a list
                    of two floating point numbers. The default is [1E3, 1E-4].
   :type stepsize: List[float]

   :returns: The mixed second partial derivative of `func` w.r.t. the two variables
             specified by `var` evaluated at `point`.
   :rtype: float


.. py:function:: backward_partial_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 0.001) -> float

   Find the first partial derivative using backwards finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the partial derivative w.r.t. `x3`, then `var`
               should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-3.
   :type stepsize: float

   :returns: The first partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


.. py:function:: forward_partial_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 0.001) -> float

   Find the first partial derivative using forwards finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the partial derivative w.r.t. `x3`, then `var`
               should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-3.
   :type stepsize: float

   :returns: The first partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


.. py:function:: pardev(func: Callable[[float], float], loc: float, stepsize: float) -> float

   Calculate the first derivative using centered finite difference.

   This function in particular only works with functions which have only one
   argument.

   :param func: A function that takes in a floating point number and outputs a floating
                point number.
   :type func: Callable[[float], float]
   :param loc: Location where the first derivative is to be evaluated.
   :type loc: float
   :param stepsize: The stepsize for the finite difference approximation.
   :type stepsize: float

   :returns: The first derivative of  `func` evaluated at `loc`.
   :rtype: float


.. py:function:: backdev(func: Callable[[float], float], loc: float, stepsize: float) -> float

   Calculate the first derivative using backwards finite difference.

   This function in particular only works with functions which have only one
   argument.

   :param func: A function that takes in a floating point number and outputs a floating
                point number.
   :type func: Callable[[float], float]
   :param loc: Location where the first derivative is to be evaluated.
   :type loc: float
   :param stepsize: The stepsize for the finite difference approximation.
   :type stepsize: float

   :returns: The first derivative of `func` evaluated at `loc`.
   :rtype: float


.. py:function:: fordev(func: Callable[[float], float], loc: float, stepsize: float) -> float

   Calculate the first derivative using forwards finite difference.

   This function in particular only works with functions which have only one
   argument.

   :param func: A function that takes in a floating point number and outputs a floating
                point number.
   :type func: Callable[[float], float]
   :param loc: Location where the first derivative is to be evaluated.
   :type loc: float
   :param stepsize: The stepsize for the finite difference approximation.
   :type stepsize: float

   :returns: The first derivative of `func` evaluated at `loc`.
   :rtype: float


.. py:function:: secbackder(function: Callable[[float], float], location: float, stepsize: float = 1e-06) -> float

   Calculate the second derivative using backwards finite difference.

   This function in particular only works with functions which have only one
   argument.

   :param function: A function that takes in a floating point number and outputs a floating
                    point number.
   :type function: Callable[[float], float]
   :param location: Location where the second derivative is to be evaluated.
   :type location: float
   :param stepsize: The stepsize for the finite difference approximation. The default
                    is 1e-6.
   :type stepsize: float

   :returns: The first derivative of `function` evaluated at `location`.
   :rtype: float


.. py:function:: secforder(function: Callable[[float], float], location: float, stepsize: float = 1e-06) -> float

   Calculate the second derivative using forwards finite difference.

   This function in particular only works with functions which have only one
   argument.

   :param function: A function that takes in a floating point number and outputs a floating
                    point number.
   :type function: Callable[[float], float]
   :param location: Location where the second derivative is to be evaluated.
   :type location: float
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-6.
   :type stepsize: float

   :returns: The second derivative of `function` evaluated at `location`.
   :rtype: float


.. py:function:: second_backward_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 1e-06) -> float

   Find the second partial derivative using backwards finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the second partial derivative w.r.t. `x3`, then
               `var` should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the second partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-6.
   :type stepsize: float

   :returns: The second partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


.. py:function:: second_forward_derivative(func: Callable[Ellipsis, float], var: int, point: List, stepsize: float = 1e-06) -> float

   Find the second partial derivative using forwards finite difference.

   This version of the function works for functions which have an arbitrary
   number of arguments (one or more).

   :param func: A function that outputs a floating point number, and has at least one
                argument which is a floating point number.
   :type func: Callable[... , float]
   :param var: The integer showing the input order of the independent variable with
               respect to which the derivative of `func` is to be approximated. It
               uses python's convention for indexing (i.e., it counts from 0, 1, 2, 3,
               ...). For example, if the function `func` has the following
               signature::

                   def some_function(x1, x2, x3):
                       ....
                       return y

               If we want to find the second partial derivative w.r.t. `x3`, then
               `var` should be 2.
   :type var: int
   :param point: A list of input values for `func`. These input values indicate the
                 location where the second partial derivative is to be evaluated.
   :type point: List
   :param stepsize: The stepsize for the finite difference approximation. The default is
                    1e-6.
   :type stepsize: float

   :returns: The second partial derivative of `func` w.r.t. the variable specified
             by `var` evaluated at `point`.
   :rtype: float


