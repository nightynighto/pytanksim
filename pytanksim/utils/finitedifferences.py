# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:12:26 2023

@author: nextf
"""

##This is a module for doing numerical finite differences

__all__ =[
    "partial_derivative",
    "second_derivative",
    "mixed_second_derivative"
    ]

def pardev(func, loc, stepsize):
    loc1 = (loc + stepsize)
    loc2 = (loc - stepsize)
    term1 = func(loc1)
    term2 = func(loc2)
    return (term1 - term2) / (2 * stepsize)

def partial_derivative(func, var=0, point=[], stepsize=1e-3):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return pardev(wraps, point[var], stepsize)

def secder(function, location, stepsize=1e-6):
    loc1 = (location - stepsize)
    loc2 = (location + stepsize)
    term1 = function(loc1)
    term2 = - 2 * function(location)
    term3 = function(loc2)
    #print(term1, term2/2, term3)
    stepsize = (loc2-loc1)/2
    return (term1 + term2 + term3)/(stepsize **2)

def second_derivative(func, var=0, point=[], stepsize=1e-6):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return secder(wraps, point[var], stepsize)

def mixsecder(function, location, stepsize = [1E3, 1E-4]):
    fpospos = function(location[0]+stepsize[0], location[1]+stepsize[1])
    fposmin = function(location[0]+stepsize[0], location[1]-stepsize[1])
    fminpos = function(location[0]-stepsize[0], location[1]+stepsize[1])
    fminmin = function(location[0]-stepsize[0], location[1]-stepsize[1])
    #print(fpospos, fposmin, fminpos, fminmin)
    return (fpospos - fposmin - fminpos + fminmin) / (4 * (stepsize[0] * stepsize[1]))

def mixed_second_derivative(func, var=[0,1], point=[], stepsize = [1E3, 1E-4]):
    args = point[:]
    def wraps(x,y):
        args[var[0]] = x
        args[var[1]] = y
        return func(*args)
    return mixsecder(wraps, [point[var[0]],point[var[1]]], stepsize)
