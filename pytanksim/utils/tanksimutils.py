# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:46:59 2023

@author: nextf
"""

__all__ =[
    "Cs_gen",
    "surface_potential_abs",
    "surface_potential_exc",
    "ads_energy_abs"
    ]

import CoolProp as CP
import scipy as sp
import numpy as np
import pytanksim.utils.finitedifferences as fd
from copy import deepcopy


#Create a function that gives the Cs as a function of T and structural component mass
def Cs_gen(mads, mcarbon, malum, msteel):
    R = sp.constants.R
    def Cdebye(T,theta):
        N=50
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
        return ((mads+mcarbon) / carbon_molar_mass)*Cdebye(T,1500) +\
            (malum/alum_molar_mass) * Cdebye(T, 389.4) +\
                (msteel /iron_molar_mass) * Cdebye(T, 500)
    return Cs

def tank_capacity_excess(nexcess, mads, rhoskel, tankvol, pempty, tempty, pfull, tfull, fluid):
    MW = 1.00784 * 2
    fluid.update(CP.PT_INPUTS, pempty, tempty)
    rho = fluid.rhomolar()
    nempty = rho * (tankvol - mads/rhoskel) + nexcess(pempty,tempty) * mads
    fluid.update(CP.PT_INPUTS, pfull, tfull)
    rho = fluid.rhomolar()
    nfull = rho * (tankvol - mads/rhoskel) + nexcess(pfull,tfull) * mads
    return (nfull - nempty) * MW /1000 

def tank_capacity_absolute(nabsolute, vadsorbed, mads, rhoskel, tankvol, pempty, tempty, pfull, tfull, fluid):
    MW = 1.00784 * 2
    fluid.update(CP.PT_INPUTS, pempty, tempty)
    rho = fluid.rhomolar()
    nempty = rho * (tankvol - mads/rhoskel - vadsorbed(tempty) * mads) + nabsolute(pempty,tempty) * mads
    fluid.update(CP.PT_INPUTS, pfull, tfull)
    rho = fluid.rhomolar()
    nfull = rho * (tankvol - mads/rhoskel - vadsorbed(tfull)*mads) + nabsolute(pfull,tfull) * mads
    return (nfull-nempty) * MW / 1000

def surface_potential_abs(nabs, p, T, vads, fluid):
    #Do an integral at constant temperature
    n= 100
    pres = np.linspace(0, p, n)
    func = np.zeros_like(pres)
    fug = np.zeros_like(func)
    for i in range(1,len(pres)):
        p = deepcopy(pres[i])
        fluid.update(CP.PT_INPUTS, p, T)
        fug[i] = fluid.fugacity(0)
        func[i] = nabs(p,T)/fug[i]
    I_trap = sp.integrate.simps(func, fug)
    return -sp.constants.R * T * I_trap + vads(p, T) * p
    
def surface_potential_exc(nexc, p, T, fluid):
    #Do an integral at constant temperature
    n = 100
    x = np.linspace(0, p, n)
    f = np.zeros_like(x)
    for i in range(1,len(x)):
        fluid.update(CP.PT_INPUTS, x[i], T)
        f[i] = nexc(x[i],T)/fluid.fugacity(0)
    I_trap = sp.integrate.simps(f, x)
    return -sp.constants.R * T * I_trap 

def ads_energy_abs(nabs, p, T, va, fluid):
    '''
    

    Parameters
    ----------
    nabs : FUNCTION
        Returns the absolute adsorption at a given P (Pa) and T (K).
    p : FLOAT
        Pressure (Pa).
    T : FLOAT
        Temperature (K).
    va : FUNCTION
        Returns the adsorbed phase volume in m^3/kg at a given temperature (K).

    Returns
    -------
    FLOAT
        The adsorbent mass-specific internal energy of the adsorbed phase (J/kg).

    '''
    fluid.update(CP.PT_INPUTS, p, T)
    hmolar = fluid.hmolar()
    def phi_over_T(p, T):
        return surface_potential_abs(nabs, p, T, va, fluid)/T
    return  nabs(p,T) * hmolar - (T**2) * fd.partial_derivative(phi_over_T, 1, [p, T],
                                                                T * 1E-5) 

def ads_energy_exc(nexcess, p, T, fluid):
    '''
    

    Parameters
    ----------
    nabs : FUNCTION
        Returns the absolute adsorption at a given P (Pa) and T (K).
    p : FLOAT
        Pressure (Pa).
    T : FLOAT
        Temperature (K).
    va : FUNCTION
        Returns the adsorbed phase volume in m^3/kg at a given temperature (K).

    Returns
    -------
    FLOAT
        The adsorbent mass-specific internal energy of the adsorbed phase (J/kg).

    '''
    fluid.update(CP.PT_INPUTS, p, T)
    hmolar = fluid.hmolar()
    def phi_over_T(p, T):
        return surface_potential_exc(nexcess, p, T)/T
    return nexcess(p, T) * hmolar - (T**2) * fd.partial_derivative(phi_over_T, 1, [p, T], phi_over_T(p,T) * 1E-5) 



