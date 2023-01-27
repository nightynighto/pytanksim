# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:17:45 2023

@author: nextf
"""

import MPTA_ads as mpta
import numpy as np
import CoolProp as CP
import lmfit

class MPTAModel:
    def __init__(self, sorbentname, fluidname, EOS, eps0, beta, lam, gam):
        self.adsorbate = fluidname
        self.EOS = EOS
        self.backend = CP.AbstractState(fluidname, EOS)
        self.eps0 = eps0
        self.beta = beta
        self.lam = lam
        self.gam = gam
        self.sorbent = sorbentname
        
    @classmethod
    def from_ExcessIsotherms(cls, ExcessIsotherms, sorbentname = None, fluidname = None, EOS = None,
                           eps0guess = 2000, betaguess = 2, lamguess = 0.001, 
                           gamguess = -3E-6):
        
        if sorbentname == None:
            sorbentname = ExcessIsotherms[0].sorbent
        if fluidname == None:
            fluidname = ExcessIsotherms[0].adsorbate
        if EOS == None:
            EOS = ExcessIsotherms[0].EOS
        
        fluid = CP.AbstractState(fluidname, EOS)
        loading_combined = []
        temperature_combined = []
        density_combined = []
        for i in enumerate(ExcessIsotherms):
            pressure_data = ExcessIsotherms[i].pressure
            loading_data = ExcessIsotherms[i].loading
            temperature = ExcessIsotherms[i].temperature
            density = np.zeros_like(loading_data)
            for j in range(len(density)):
                fluid.update(CP.PT_INPUTS, pressure_data[j], temperature)
                density[j] = fluid.rhomolar()
            loading_combined.append(loading_data)
            temperature_combined.append(np.repeat(temperature,len(pressure_data)))
            density_combined.append(density)
            
        
        params = lmfit.Parameters()
        params.add("lam", lamguess, True, 0)
        params.add("gam", gamguess, True)
        params.add("eps0", eps0guess, True, 0 )
        params.add("beta", betaguess, True, 0, 10)
        fitting = lmfit.minimize(mpta.fit_penalty, params, args=(density_combined,
                                                                loading_combined, 
                                                                temperature_combined,
                                                                fluid))
        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()
        return cls(sorbentname, fluidname, EOS, paramsdict["eps0"], paramsdict["beta"],
                   paramsdict["lam"], paramsdict["gam"])
    
    def n_excess(self, p, T):
        fluid = self.backend
        fluid.update(CP.PT_INPUTS, p, T)
        bulk_density = fluid.rhomolar()
        value = mpta.N_ex(self.eps0, self.beta, self.lam, self.gam, T,
                          bulk_density, fluid)
        return value
    
    def n_absolute(self, p, T):
        fluid = self.backend
        fluid.update(CP.PT_INPUTS, p, T)
        bulk_density = fluid.rhomolar()
        value = mpta.N_abs(self.eps0, self.beta, self.lam, self.gam, T,
                          bulk_density, fluid)
        return value
        
    def v_ads(self, T):
        return self.lam + self.gam * T