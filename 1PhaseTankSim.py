# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:11:33 2023

@author: nextf
"""

import CoolProp as CP
import scipy as sp
import numpy as np
import FiniteDifferences as fd
import TankSimUtils as tsu
import time as time

##Molecular weight of hydrogen
MW = 1.00784 * 2


def dn_dp(p, T, nabs, vads, mads, tankvol, rhoskel, drho_dp):     
    term1 = drho_dp * (tankvol - mads/rhoskel - vads(T))
    term2 = fd.partial_derivative(nabs, 0, [p, T], p * 1E-5) * mads
    return term1 + term2

def dn_dT(p, T, nabs, vads, mads, tankvol, rhoskel, rhof, drho_dT):
    term = np.zeros(3)
    term[0] =  drho_dT * (tankvol - mads/rhoskel - vads(T))
    term[1] = - rhof * fd.pardev(vads, T, T*1E-5)
    term[2] = fd.partial_derivative(nabs, 1, [p,T], T*1E-5) * mads
    return sum(term)

def dU_dp(p, T, nabs, vads, mads, tankvol, rhoskel, drho_dp, dh_dp, rhof, hf):
    def phi_over_T(p, T):
        return tsu.surface_potential_abs(nabs, p, T, vads)/ T
    term = np.zeros(6)
    term[0] = -tankvol + (mads/rhoskel)
    term[1]= drho_dp * hf * (tankvol - mads/rhoskel - vads(T))
    term[2] = rhof * dh_dp * (tankvol - mads/rhoskel - vads(T))
    term[3] = mads * fd.partial_derivative(nabs, 0, [p, T], p * 1E-5) * hf
    term[4] = mads * nabs(p,T) * dh_dp
    term[5] = -mads * (T**2) * fd.mixed_second_derivative(phi_over_T, [0,1], [p, T], [p * 1E-5, T * 1E-5])
    return sum(term)

def dU_dT(p, T, nabs, vads, mads, tankvol, rhoskel, Cs, dh_dT, drho_dT, rhof, hf):
    def phi_over_T(p, T):
        return tsu.surface_potential_abs(nabs, p, T, vads)/ T
    term = np.zeros(8)
    term[0] = mads * fd.partial_derivative(nabs, 1, [p, T], T*1E-5) + hf
    term[1] = mads * nabs(p, T) * dh_dT
    term[2] = -mads * 2 * T * fd.partial_derivative(phi_over_T, 1, [p, T], T*1E-5)
    term[3] = -mads * (T**2) * fd.second_derivative(phi_over_T, 1, [p,T], T*1E-5)
    term[4] = drho_dT * hf * (tankvol - mads/rhoskel - vads(T))
    term[5] = rhof * dh_dT * (tankvol - mads/rhoskel - vads(T))
    term[6] = -rhof * hf * fd.pardev(vads, T, T * 1E-5)
    term[7] = Cs(T)
    return sum(term)

def dT_dt(p, T, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid):
    ##Convert kg/s to mol/s
    ndotin = mdotin * 1000 / MW
    ndotout = mdotout * 1000 / MW
    ##Get the input pressure at a condition
    Pinput = Pin(p)
    ##Get the molar enthalpy of the inlet fluid
    fluid.update(CP.PT_INPUTS, Pinput, Tin)
    hin = fluid.hmolar()
    ##Get the thermodynamic properties of the bulk fluid for later calculations
    fluid.update(CP.PT_INPUTS, p, T)
    hf = fluid.hmolar()
    drho_dp = fluid.first_partial_deriv(CP.iDmolar , CP.iP , CP.iT )
    drho_dT =  fluid.first_partial_deriv(CP.iDmolar , CP.iT , CP.iP )
    rhof = fluid.rhomolar()
    dh_dp = fluid.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT)
    dh_dT = fluid.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP)
    k1 = ndotin - ndotout
    k2 = ndotin * hin - ndotout * hf + Qin
    #print(hin, hgas)
    a = dn_dp(p, T, na, vads, mads, tankvol, rhoskel, drho_dp)
    b = dn_dT(p, T, na, vads, mads, tankvol, rhoskel, rhof, drho_dT)
    c = dU_dp(p, T, na, vads, mads, tankvol, rhoskel, drho_dp, dh_dp, rhof, hf)
    d = dU_dT(p, T, na, vads, mads, tankvol, rhoskel, Cs, dh_dT, drho_dT, rhof, hf)
    #Put in the right hand side of the mass and energy balance equations
    return (k2 * a - c * k1)/(d*a - b*c)

def dP_dt(p, T, na, vads, mads, tankvol, rhoskel, mdotin, mdotout, dTdt, fluid):
    fluid.update(CP.PT_INPUTS, p, T)
    drho_dp = fluid.first_partial_deriv(CP.iDmolar , CP.iP , CP.iT )
    rhof = fluid.rhomolar()
    drho_dT =  fluid.first_partial_deriv(CP.iDmolar , CP.iT , CP.iP )
    ndotin = mdotin * 1000 / MW
    ndotout = mdotout * 1000 / MW 
    k1 = ndotin - ndotout
    a = dn_dp(p, T, na, vads, mads, tankvol, rhoskel, drho_dp)
    b = dn_dT(p, T, na, vads, mads, tankvol, rhoskel, rhof, drho_dT)
    return (k1 - b * dTdt)/a

def init_condit(p, T, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid):
    nads = na(p,T) * mads
    fluid.update(CP.PT_INPUTS, p, T)
    rhofluid = fluid.rhomolar()
    Vg = tankvol - mads/rhoskel - vads(T)
    ng = Vg * rhofluid
    dTdt = dT_dt(p, T, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
    dPdt = dP_dt(p, T, na, vads, mads, tankvol, rhoskel, mdotin, mdotout, dTdt, fluid)
    return 0, p, T, nads, ng, dPdt, dTdt

def RK_4step(p, T, na, vads, mads,tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid, h=0.1):
    k_P = np.zeros(4)
    k_T = np.zeros(4)
    pres = p
    temp = T
    for i in range(4):
        dTdt = dT_dt(p, T, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
        dPdt = dP_dt(p, T, na, vads, mads, tankvol, rhoskel, mdotin, mdotout, dTdt, fluid)
        k_P[i] = dPdt
        k_T[i] = dTdt
        if i<2:
            pres = p + h/2 * dPdt
            temp = T + h/2 * dTdt
        else:
            pres = p + h * dPdt
            temp = T + h * dTdt
    presult = p + (h/6) * (k_P[0]+2*k_P[1]+2*k_P[2]+k_P[3])
    Tresult = T + (h/6) * (k_T[0]+2*k_T[1]+2*k_T[2]+k_T[3])
    nadsresult = na(presult,Tresult) * mads
    fluid.update(CP.PT_INPUTS, presult, Tresult)
    ngresult = fluid.rhomolar() * (tankvol - mads/rhoskel - vads(Tresult))
    dTdt = dT_dt(presult, Tresult, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
    dPdt = dP_dt(presult, Tresult, na, vads, mads, tankvol, rhoskel, mdotin, mdotout, dTdt, fluid)
    return presult, Tresult, nadsresult, ngresult, dPdt, dTdt


def ABM_4step(y, na, vads, mads,tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid, h=0.1):
    k_P = np.zeros(4)
    k_T = k_P
    ##Take the ABM coefficients from the array containing the previous 4 steps' results
    yflipped = np.flipud(y)
    k_T = yflipped[:,6]
    k_P = yflipped[:,5]
    ##Do Adams Bashforth 4 step prediction
    pres_predict =  y[3,1] + h * (55 * k_P[0] - 59 * k_P[1] + 37 * k_P[2] - 9 * k_P[3])/24
    temp_predict =  y[3,2] + h * (55 * k_T[0] - 59 * k_T[1] + 37 * k_T[2] - 9 * k_T[3])/24
    ##Calculate correction coefficient 
    corr_T = dT_dt(pres_predict, temp_predict, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
    corr_P = dP_dt(pres_predict, temp_predict,  na, vads, mads, tankvol, rhoskel, mdotin, mdotout, corr_T, fluid)
    ##Get final results for p and T
    presult = y[3,1] + h * (9 * corr_P + 19 * k_P[0] - 5 * k_P[1] + k_P[2])/24
    Tresult = y[3,2] + h * (9 * corr_T + 19 * k_T[0] - 5 * k_T[1] + k_T[2])/24
    ##Get other properties
    nadsresult = na(presult,Tresult) * mads
    fluid.update(CP.PT_INPUTS, presult, Tresult)
    ngresult = fluid.rhomolar() * (tankvol - mads/rhoskel - vads(Tresult))
    dTdt =dT_dt(presult, Tresult, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
    dPdt = dP_dt(presult, Tresult, na, vads, mads, tankvol, rhoskel, mdotin, mdotout, dTdt, fluid)
    return presult, Tresult, nadsresult, ngresult, dPdt, dTdt

def simulate_refueling(p0, T0, na, vads, mads,tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid, finaltime, timestep):
    rows = int(finaltime/timestep + 1)
    result_matrix = np.zeros((rows,7))
    ##The results matrix have columns in the order of: time, pressure, temperature, n excess, n gas
    ## as well as dP/dt and dT/dt
    #First get the initial condition
    print("Starting simulation", flush=True)
    start = time.time()
    result_matrix[0,:] = init_condit(p0, T0, na, vads, mads, tankvol, rhoskel, Cs, mdotin, mdotout, Tin, Pin, Qin, fluid)
    for i in range(rows):
        result_matrix[i,0] = i * timestep
    ##Then estimate the next 4 steps using Runge-Kutta
    for i in range(1,4):
        print("Simulated Time (s):", result_matrix[i,0], flush=True )
        current = time.time()
        print("Elapsed Comp. Time (s):", current-start, flush=True )
        result_matrix[i,1:] = RK_4step(result_matrix[i-1,1], result_matrix[i-1,2], \
                                       na, vads, mads,tankvol, rhoskel, Cs, \
                                       mdotin, mdotout, Tin, Pin, Qin, fluid, timestep)
    ##Continue until the final timestep using ABM
    for i in range(4,rows):
        print("Simulated Time (s):", result_matrix[i,0], flush=True )
        current = time.time()
        print("Elapsed Comp. Time (s):", current-start, flush=True )
        input_matrix = result_matrix[i-4:i, :]
        result_matrix[i,1:] = ABM_4step(input_matrix, na, vads, mads,tankvol, \
                                        rhoskel, Cs, mdotin, mdotout, Tin, Pin,\
                                            Qin, fluid, timestep)
    return result_matrix
 