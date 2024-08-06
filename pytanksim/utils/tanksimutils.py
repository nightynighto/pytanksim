# -*- coding: utf-8 -*-
"""Module for additional utility functions for `pytanksim`.

Mainly used for the calculation of the combined Debye heat capacity."""
"""
Copyright 2024 Muhammad Irfan Maulana Kusdhany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = [
    "Cs_gen"
    ]

import scipy as sp
import numpy as np


def Cs_gen(mads, mcarbon, malum, msteel, Tads = 1500, MWads = 12.01E-3):
    R = sp.constants.R

    def Cdebye(T,theta):
        N = 50
        grid = np.linspace(0, theta/T, N)
        y = np.zeros_like(grid)

        def integrand(x):
            return(x**4) * np.exp(x) / ((np.exp(x)-1)**2)
        for i in range(1, N):
            y[i] = integrand(grid[i])
        return 9 * R * ((T/theta)**3) * sp.integrate.simps(y, grid)

    carbon_molar_mass = 12.01E-3
    alum_molar_mass = 26.98E-3
    iron_molar_mass = 55.845E-3

    def Cs(T):
        return (mads / MWads)*Cdebye(T, Tads) + \
            (mcarbon / carbon_molar_mass)*Cdebye(T, 1500) +\
            (malum / alum_molar_mass) * Cdebye(T, 389.4) +\
            (msteel / iron_molar_mass) * Cdebye(T, 500)

    return Cs