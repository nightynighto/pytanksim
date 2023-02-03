# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:39:58 2022

@author: nextf
"""

__all__ = ["N_ex", "N_abs", "fit_penalty", "mean_error"]


##This is a code with functions to calculate and fit the parameters
##of the potential theory of adsorption

import numpy as np
import scipy as sp
import CoolProp as CP


##First, use the Dubinin Raduskevich Potential for Micropores
def DRA(z, eps0, beta, lam, gam, T):
    z0 = lam + gam * T
    return eps0*np.log(z0/z)**(1.0/beta)

##Find maximum density at a given temperature that CoolProp can work with
def d_max(T, fluid):
    maxpres = fluid.pmax()/10
    fluid.update(CP.PT_INPUTS,maxpres,T)
    return fluid.rhomolar()

##Define a function to get chemical potential at a certain density and temperature
def chem(T, d, fluid):
    fluid.update(CP.DmolarT_INPUTS, d, T)
    return fluid.chemical_potential(0)

##Compute gas density in the adsorbed phase
def d_ads(z, eps0, beta, lam, gam, T, dB, d0, dvap, dliq, fluid):
    potential = chem(T, dB, fluid) + DRA(z, eps0, beta, lam, gam, T)
    ##f is the function to be minimized to get adsorbate density
    def f(d_z):
        y = potential - chem(T, d_z, fluid)
        return y
    ##The following two if conditions are to ensure stability
    ##This one ensures the optimizer does not try to get values outside of 
    ##Coolprop's range of validity for the equation of state
    if potential >= chem(T, d_max(T), fluid):
        return d_max(T, fluid)
    ##This one changes the default guess to be above liquid density if the 
    ##chemical potential is above the liquid chemical potential
    ##It is needed because there is a sharp change in density at the saturation
    ##which can disturb the stability of fsolve near saturation pressures.
    if potential > chem(T, dvap, fluid) and d0<dliq:
        d0 = dliq * 1.1
    return sp.optimize.fsolve(f,d0)[0]


def N_ex(eps0, beta, lam, gam, T, dB, fluid):
    d0 = dB
    z0 = lam + gam * T
    N = 50
    delta = z0/N
    integral=0
    if T < CP.CoolProp.PropsSI("Tcrit", "Hydrogen"):
        fluid.update(CP.QT_INPUTS, 1, T)
        dvap = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 0, T)
        dliq = fluid.rhomolar()
    else:
         dvap = d_max(T, fluid)
         dliq = dvap
    for i in range(0,N): 
        d_z = d_ads(z0 - i*delta - delta/2, eps0, beta, lam, gam, T, dB, d0, dvap, dliq, fluid)
        integral += d_z * delta
        d0 = d_z
    return integral - dB * z0

def N_abs(eps0, beta, lam, gam, T, dB, fluid):
    d0 = dB
    z0 = lam + gam * T
    N = 50
    delta = z0/N
    integral=0
    if T < CP.CoolProp.PropsSI("Tcrit", "Hydrogen"):
        fluid.update(CP.QT_INPUTS, 1, T)
        dvap = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 0, T)
        dliq = fluid.rhomolar()
    else:
         dvap = d_max(T, fluid)
         dliq = dvap
    for i in range(0,N): 
        d_z = d_ads(z0 - i*delta - delta/2, eps0, beta, lam, gam, T, dB, d0, dvap, dliq, fluid)
        integral += d_z * delta
        d0 = d_z
    return integral

def fit_penalty(params, dataD, dataAd, dataT, fluid):
    value = params.valuesdict()
    lam = value["lam"]
    gam = value["gam"]
    eps0 = value["eps0"]
    beta = value["beta"]
    difference = []
    for i in range(0, len(dataD)):
        difference.append(N_ex(eps0, beta, lam, gam, dataT[i], dataD[i], fluid) - dataAd[i])
    return difference

def mean_error(dataExp,dataModel):
    d=0
    for i in range(0,len(dataExp)):
        d += abs((dataExp[i]-dataModel[i])/dataExp[i])
    return 100*d/len(dataExp)

##Make a function to return another function which returns na as a function of P and T only
def n_exc_PT(eps0, beta, lam, gam, fluid, quality = 1):
    def na(p, T):
        if p == 0:
            return 0
        if p > fluid.pmax()/10:
            pres = fluid.pmax()/10
            fluid.update(CP.PT_INPUTS, pres, T)
            bulkdens = fluid.rhomolar()
            return N_ex(eps0, beta, lam, gam, T, bulkdens, fluid)
        ##Need to add an exception near saturation to ensure the density value used is same as the selected phase
        if T < fluid.T_critical():
            fluid.update(CP.QT_INPUTS, quality, T)
            psaturation = fluid.p()
            if np.abs(p-psaturation) > p * 1E-6:
                fluid.update(CP.PT_INPUTS, p, T)
            bulkdens = fluid.rhomolar()
        else:
            fluid.update(CP.PT_INPUTS, p, T)
            bulkdens = fluid.rhomolar()
        return N_ex(eps0, beta, lam, gam, T, bulkdens, fluid)
    return na

def n_abs_PT(eps0, beta, lam, gam, fluid, quality = 1):
    def na(p, T):
        if p == 0:
            return 0
        if p > fluid.pmax()/10:
            pres = fluid.pmax()/10
            fluid.update(CP.PT_INPUTS, pres, T)
            bulkdens = fluid.rhomolar()
            return N_abs(eps0, beta, lam, gam, T, bulkdens, fluid)
        ##Need to add an exception near saturation to ensure the density value used is same as the selected phase
        if T < fluid.T_critical():
            fluid.update(CP.QT_INPUTS, quality, T)
            psaturation = fluid.p()
            if np.abs(p-psaturation) > p * 1E-6:
                fluid.update(CP.PT_INPUTS, p, T)
            bulkdens = fluid.rhomolar()
        else:
            fluid.update(CP.PT_INPUTS, p, T)
            bulkdens = fluid.rhomolar()
        return N_abs(eps0, beta, lam, gam, T, bulkdens, fluid)
    return na
