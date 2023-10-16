# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:17:45 2023

@author: nextf
"""

__all__ = ["StoredFluid", "MPTAModel", "SorbentMaterial", "MDAModel", "DAModel"]

import pytanksim.mpta as mpta
import numpy as np
import CoolProp as CP
import lmfit
import pytanksim.utils.tanksimutils as tsu
from pytanksim.classes.excessisothermclass import ExcessIsotherm
from copy import deepcopy
import pytanksim.utils.finitedifferences as fd
from typing import List
import scipy as sp


class StoredFluid:
    def __init__(self,
                 fluid_name : str,
                 EOS : str):
        """
        

        Parameters
        ----------
        fluid_name : str, optional
            Name of the fluid.
        EOS : str, optional
            Name of the equation of state to be used for calculations. 
            
        Returns
        -------
        None.

        """
        
        self.fluid_name = fluid_name
        self.EOS = EOS
        self.backend = CP.AbstractState(EOS, fluid_name)
    
    def fluid_property_dict(self, p, T):
        """
        

        Parameters
        ----------
        p : float
            Pressure in Pa.
        T : float
            Temperature in K.

        Returns
        -------
        dict
            A dictionary containing the fluid properties at a given pressure and
            temperature.

        """
        
        backend = self.backend
        backend.update(CP.PT_INPUTS, p, T)
        return {
            "hf" : backend.hmolar(),
            "drho_dp" : backend.first_partial_deriv(CP.iDmolar , CP.iP , CP.iT ),
            "drho_dT" :  backend.first_partial_deriv(CP.iDmolar , CP.iT , CP.iP ),
            "rhof" : backend.rhomolar(),
            "dh_dp" : backend.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT),
            "dh_dT" : backend.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP),
            "uf" : backend.umolar(),
            "du_dp" : backend.first_partial_deriv(CP.iUmolar, CP.iP , CP.iT),
            "du_dT" : backend.first_partial_deriv(CP.iUmolar, CP.iT , CP.iP),
            "MW" : backend.molar_mass()
            }
    
    def saturation_property_dict(self,
                                 T:float,
                                 Q: int = 0):
        """
        

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        dict
            A dictionary containing the fluid properties at saturation
            at a given temperature.

        """
        
        backend = self.backend
        backend.update(CP.QT_INPUTS, Q, T)
        return {
            "psat" : backend.p(),
            "dps_dT" : backend.first_saturation_deriv(CP.iP, CP.iT),
            "hf" : backend.hmolar(),
            "drho_dp" : backend.first_partial_deriv(CP.iDmolar , CP.iP , CP.iT ),
            "drho_dT" :  backend.first_partial_deriv(CP.iDmolar , CP.iT , CP.iP ),
            "rhof" : backend.rhomolar(),
            "dh_dp" : backend.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT),
            "dh_dT" : backend.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP),
            "uf" : backend.umolar(),
            "du_dp" : backend.first_partial_deriv(CP.iUmolar, CP.iP , CP.iT),
            "du_dT" : backend.first_partial_deriv(CP.iUmolar, CP.iT , CP.iP),
            "MW" : backend.molar_mass()
            }
    
    def determine_phase(self,
                        p : float,
                        T : float):
        """
        Determines the phase of the bulk fluid inside the tank at
        a given pressure and temperature.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        str
            String that could either be "Supercritical", "Gas", "Liquid", 
            or "Saturated" depending on the bulk fluid phase.

        """
        
        fluid = self.backend
        Tcrit = fluid.T_critical()
        pcrit = fluid.p_critical()
        if T > Tcrit:
            if p > pcrit:   
                return "Supercritical"
            else: return "Gas"
        else:
            fluid.update(CP.QT_INPUTS, 0, T)
            psat = fluid.p()
            if np.abs(p-psat) <= (psat * 1E-5):
                return "Saturated"
            elif p < psat:
                return "Gas"
            elif p > psat and p > pcrit:
                return "Supercritical"
            elif p > psat and p < pcrit :
                return "Liquid"
            
            
            
        

class ModelIsotherm:
    def surface_potential(self, p : float, T : float) -> float:
        """
        

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Adsorption surface potential (J/kg).

        """
        
        return tsu.surface_potential_abs(nabs = self.n_absolute,
                                         p = p, T = T, vads = self.v_ads, fluid=self.stored_fluid.backend)
    
    
    def phi_over_T(self, p : float, T : float) -> float:
        """
        

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Surface potential divided by temperature (J/ (kg K)).

        """
        
        return (1/T) * self.surface_potential(p, T)
    
    def enthalpy_adsorbed_phase(self, p, T):
        return tsu.ads_energy_abs(self.n_absolute, p, T, self.v_ads, self.stored_fluid.backend)
 
    def pressure_from_absolute_adsorption(self, n_abs, T, p_max_guess = 20E6):
        p_max_guess = min(p_max_guess, self.stored_fluid.backend.pmax()/10)
        if n_abs == 0:
            return 0
        def optimum_pressure(p):
            return (self.n_absolute(p, T) - n_abs)**2
        root = sp.optimize.minimize_scalar( 
                                fun = optimum_pressure, 
                                bounds=(1,p_max_guess),
                                method = "bounded")
        return root.x
    
    def isosteric_enthalpy(self, p, T, q =1):
        nabs = self.n_absolute(p, T)
        fluid = self.stored_fluid.backend
        def diff_function(x):
            pres = self.pressure_from_absolute_adsorption(nabs, 1/x)
            phase = self.stored_fluid.determine_phase(pres, 1/x)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, 1/x)
            else:
                fluid.update(CP.QT_INPUTS, q, 1/x)
            return fluid.chemical_potential(0) * x
        
        phase = self.stored_fluid.determine_phase(p, T)
        x_loc = 1/T
        step = x_loc * 1E-2
        temp2 = 1/(x_loc + step)
        phase2 = self.stored_fluid.determine_phase(p, temp2)
        temp3 = 1/(x_loc - step)
        phase3 = self.stored_fluid.determine_phase(p, temp3)

        if phase == phase2 == phase3 != "Saturated":
            hadsorbed = fd.pardev(diff_function, x_loc, step)
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            if q == 1:
                hadsorbed = fd.backdev(diff_function, x_loc, step)
            else:
                hadsorbed = fd.fordev(diff_function, x_loc, step)
            if phase == "Saturated":
                fluid.update(CP.QT_INPUTS, q, T)
            else:
                fluid.update(CP.PT_INPUTS, p, T)
        hfluid = fluid.hmolar()
        return hfluid - hadsorbed 
    
    def isosteric_internal_energy(self, p, T, q = 1):
        nabs = self.n_absolute(p, T)
        vads = self.v_ads(p,T)
        fluid = self.stored_fluid.backend
        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)
        phase = self.stored_fluid.determine_phase(p, T)
        x_loc = T
        step = 1E-2
        temp2 = x_loc + step
        phase2 = self.stored_fluid.determine_phase(p, temp2)
        temp3 = x_loc - step
        phase3 = self.stored_fluid.determine_phase(p, temp3)
        if phase == phase2 == phase3 != "Saturated":
           hadsorbed = fd.pardev(diff_function, x_loc, step)
           fluid.update(CP.PT_INPUTS, p, T)
        else:
           if q == 0:
               hadsorbed = fd.backdev(diff_function, x_loc, step)
           else:
               hadsorbed = fd.fordev(diff_function, x_loc, step)
           if phase == "Saturated":
               fluid.update(CP.QT_INPUTS, q, T)
           else:
               fluid.update(CP.PT_INPUTS, p, T)
        chempot = fluid.chemical_potential(0)
        uadsorbed = chempot - T * hadsorbed
        ufluid = fluid.umolar()
        return ufluid - uadsorbed


    def _derivfunc(self, func, var, point, qinit, stepsize):
        pT = point[:2]
        def phase_func(x):
            pT[var] = x
            return self.stored_fluid.determine_phase(pT[0], pT[1])
        
        x0 = point[var]
        x1 = x0 + stepsize
        x2 = x0 - stepsize
        phase1 = phase_func(x0)
        phase2 = phase_func(x1)
        phase3 = phase_func(x2)
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.partial_derivative(func, var, point, stepsize)
        elif phase1 == "Saturated":
            if (qinit == 0 and var == 1) or (qinit == 1 and var == 0):
                return fd.backward_partial_derivative(func, var, point, stepsize)
            else:
                return fd.forward_partial_derivative(func, var, point, stepsize)
        else:
            if phase1 == phase3:
                return fd.backward_partial_derivative(func, var, point, stepsize)
            elif phase1 == phase2:
                return fd.forward_partial_derivative(func, var, point, stepsize)   
            
    def _derivfunc_second(self, func, point, qinit, stepsize):
        pT = point
        def phase_func(x):
            pT[1] = x
            return self.stored_fluid.determine_phase(pT[0], pT[1])
        
        x0 = point[1]
        x1 = x0 + stepsize
        x2 = x0 - stepsize
        phase1 = phase_func(x0)
        phase2 = phase_func(x1)
        phase3 = phase_func(x2)
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.secder(func, x0, stepsize)
        elif phase1 == "Saturated":
            if qinit == 0:
                return fd.secbackder(func, x0, stepsize)
            else:
                return fd.secforder(func, x0, stepsize)
        else:
            if phase1 == phase3:
                return fd.secbackder(func, x0, stepsize)
            elif phase1 == phase2:
                return fd.secforder(func, x0, stepsize)

    def isosteric_energy_temperature_deriv(self, p, T, q = 1, stepsize = 1E-3):
        nabs = self.n_absolute(p, T)
        vads = self.v_ads(p,T)
        fluid = self.stored_fluid.backend
        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        du_dT = fluid.first_partial_deriv(CP.iUmolar, CP.iT, CP.iP)
        dhads_dT = - T * self._derivfunc_second(diff_function, [p, T], q, stepsize)
        dnabs_dT = self._derivfunc(self.n_absolute, 1, [p, T], q, stepsize)
        dvads_dT = self._derivfunc(self.v_ads, 1, [p, T], q, stepsize)
        return du_dT - (dhads_dT - (p/(nabs **2))*(nabs * dvads_dT - dnabs_dT * vads))
    
    def differential_energy(self, p, T, q=1):
        nabs = self.n_absolute(p, T)
        fluid = self.stored_fluid.backend
        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)
        x_loc = T
        step = 1E-2
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            fluid.update(CP.QT_INPUTS, q, T)
        chempot = fluid.chemical_potential(0)
        hadsorbed = fd.pardev(diff_function, x_loc, step)
        uadsorbed = chempot - T * hadsorbed
        return uadsorbed
    
    def differential_heat(self, p, T):
        if p == 0:
            return 0
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, 1, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        u_molar = fluid.umolar()
        return u_molar - self.differential_energy(p,T)
    
    
    def internal_energy_adsorbed(self, p, T, q = 1):
        n_abs = self.n_absolute(p, T)
        # if n_abs > 1:
        #     n_grid = np.linspace(1, n_abs, 50)
        # else:
        n_grid = np.linspace(n_abs/50, n_abs, 50)
        p_grid = np.array([self.pressure_from_absolute_adsorption(n, T) if n!= 0 else 0 for n in n_grid ])
        heat_grid = np.array([self.differential_energy(pres, T, q) for pres in p_grid])
        return sp.integrate.simps(heat_grid, n_grid) / n_abs
    
    def areal_immersion_energy(self, T):
        fluid = self.stored_fluid.backend
        def sur_tension(T):
            fluid.update(CP.QT_INPUTS, 0, T)
            sur_ten = self.stored_fluid.backend.surface_tension()
            return sur_ten
        diff = fd.partial_derivative(sur_tension, 0, [T], 0.001) if T < fluid.T_critical() - 0.001 else \
            fd.backward_partial_derivative(sur_tension, 0, [T], 0.001)
        return T * diff - sur_tension(T)

    

class MPTAModel(ModelIsotherm):
    def __init__(self,
                 sorbent : str,
                 stored_fluid : StoredFluid,
                 eps0 : float,
                 beta : float,
                 lam : float,
                 gam : float):
        """
        Initializes the MPTAModel class.
        This class stores the parameters of the isotherm model based on
        the multicomponent potential theory of adsorption (MPTA).
        
        Parameters
        ----------
        sorbent : str
            Name of the sorbent sample.
        stored_fluid : StoredFluid, optional
            Object containing stored fluid properties and CoolProp backend.
        eps0 : float
            Characteristic energy of adsorption in J/mol.
        beta : float
            Pore distribution heterogeneity parameter.
        lam : float
            Parameter that represents the micropore volume of the material in m^3 / kg.
        gam : float
            Parameter that represents the change of micropore volume w.r.t. temperature.
            The unit is  m^3 / (kg K)
        backend : CP.AbstractState, optional
            Backend used to get fluid properties for calculations. The default is None.
        """
        
        self.stored_fluid = stored_fluid
        self.eps0 = eps0
        self.beta = beta
        self.lam = lam
        self.gam = gam
        self.sorbent = sorbent
        
    @classmethod
    def from_ExcessIsotherms(cls, 
                             ExcessIsotherms : List[ExcessIsotherm],
                             stored_fluid : StoredFluid = None,
                             sorbent: str = None,
                             eps0guess: float = 2000,
                             betaguess: float = 2,
                             lamguess: float = 0.001, 
                             gamguess: float = -3E-6,):
        """
        This function takes a list of excessisotherms object,
        fits an MPTA model, and uses the fitted parameters to
        instantiate an MPTAModel object.

        Parameters
        ----------
        ExcessIsotherms : list[ExcessIsotherm]
            A list of excess isotherms object describing experimental
            adsorption measurement of the same material at different 
            temperatures.
        stored_fluid : StoredFluid, optional
            Object containing stored fluid properties and CoolProp backend.
            The default is None.
        sorbent : str, optional
            Name of the sorbent material. The default is None.
        eps0guess : float, optional
            Initial guess for characteristic adsorption energy (J/mol).
            The default is 2000.
        betaguess : float, optional
            Initial guess for the pore size heterogeneity parameter.
            The default is 2, but this value should only be between 0-10.
        lamguess : float, optional
            Initial guess for micropore volume (m^3/kg).
            The default is 0.001.
        gamguess : float, optional
            Initial guess for change in micropore volume w.r.t. temperature.
            (m^3/(kg K))
            The default is -3E-6.

        Returns
        -------
        MPTAModel
            Class that contains MPTA model parameters as well as 
            methods to get the adsorbed amount at a given pressure
            and temperature.

        """
        
        excess_isotherms = deepcopy(ExcessIsotherms)
        
        #Take values from excess isotherm if not supplied in argument
        if sorbent == None:
            sorbent = excess_isotherms[0].sorbent
        if stored_fluid == None:
            stored_fluid = StoredFluid(
                fluid_name=excess_isotherms[0].adsorbate, EOS="HEOS")
        
        backend = stored_fluid.backend
        loading_combined = []
        temperature_combined = []
        density_combined = []
        
        
        for i in enumerate(excess_isotherms):
            pressure_data = excess_isotherms[i].pressure
            loading_data = excess_isotherms[i].loading
            temperature = excess_isotherms[i].temperature
            density = np.zeros_like(loading_data)
            for j in range(len(density)):
                backend.update(CP.PT_INPUTS, pressure_data[j], temperature)
                density[j] = backend.rhomolar()
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
                                                                backend))
        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()
        return cls(sorbent = sorbent,
                   stored_fluid = stored_fluid,
                   eps0 = paramsdict["eps0"],
                   beta = paramsdict["beta"],
                   lam = paramsdict["lam"],
                   gam = paramsdict["gam"])
    
    def n_excess(self, p : float, T : float,
                 quality: int = 1) -> float:
        """
        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).
        quality : int
            Quality of the fluid to be used at saturation point

        Returns
        -------
        float
            Excess adsorbed amount (mol/kg).

        """
        
        fluid = self.stored_fluid.backend
        pmax = fluid.pmax()
        phase = self.stored_fluid.determine_phase(p, T)
        if p == 0:
            return 0
        if p >= pmax:
            pres = pmax/10
        else:
            pres = p
        
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, quality, T)
        else:
            fluid.update(CP.PT_INPUTS, pres, T)
        bulk_density = fluid.rhomolar()
        value = mpta.N_ex(self.eps0, self.beta, self.lam, self.gam, T,
                      bulk_density, fluid)
        return value
    
    def n_absolute(self, p : float, T : float,
                   quality: int = 1) -> float:
        """
        

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).
        quality : int
            Quality of the fluid to be used at saturation point

        Returns
        -------
        float
            Absolute adsorbed amount (mol/kg).

        """
        
        fluid = self.stored_fluid.backend
        pmax = fluid.pmax()
        phase = self.stored_fluid.determine_phase(p, T)
        if p == 0:
            return 0
        if p >= pmax:
            pres = pmax/10
        else:
            pres = p
        
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, quality, T)
        else:
            fluid.update(CP.PT_INPUTS, pres, T)
        bulk_density = fluid.rhomolar()
        value = mpta.N_abs(eps0 = self.eps0,
                           beta = self.beta,
                           lam = self.lam,
                           gam = self.gam,
                           T = T,
                          dB = bulk_density,
                          fluid = fluid)
        return value
        
    def v_ads(self, p : float, T : float) -> float:
        """
        

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Adsorbed phase volume (m^3/kg).

        """
        
        return self.lam + self.gam * T
    
class DAModel(ModelIsotherm):
    def __init__(self,
                  sorbent : str,
                  stored_fluid : StoredFluid,
                  w0 : float,
                  f0 : float,
                  eps : float,
                  m : float = 2,
                  k : float = 2,
                  rhoa : float = None,
                  va_mode : str = "Constant",
                  rhoa_mode : str = "Constant",
                  f0_mode : str = "Dubinin"):
        if rhoa == None and rhoa_mode == "Constant":
            stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            rhoa = stored_fluid.backend.rhomolar()
        self.sorbent = sorbent
        self.stored_fluid = stored_fluid
        self.w0 = w0
        self.f0 = f0
        self.eps = eps
        self.m = m
        self.rhoa = rhoa
        self.rhoa_mode = rhoa_mode
        self.va_mode = va_mode
        self.f0_mode = f0_mode
        self.k = k
    
    def v_ads(self, p, T):
        if self.va_mode == "Excess":
            return 0
        
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)
        
        if self.f0_mode == "Constant":
            f0 = self.f0
        if self.f0_mode == "Dubinin":
            pc = self.stored_fluid.backend.p_critical()
            Tc = self.stored_fluid.backend.T_critical()
            if T < Tc:
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            else:
                p0 = ((T/Tc)**self.k) * pc
                self.stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
        return self.w0 * np.exp(-((sp.constants.R * T / \
                                     (self.eps))**self.m)\
                                   * ((np.log(f0/fug))**self.m))
                
    def n_absolute(self, p, T):
        if self.rhoa_mode == "Constant":
            rhoa = self.rhoa
        if self.rhoa_mode == "Ozawa":
            self.stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            Tb = self.stored_fluid.backend.T()
            vb = 1/self.stored_fluid.backend.rhomolar()
            ads_specific_volume = vb * np.exp((T-Tb)/T)
            rhoa = 1/ads_specific_volume
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)
        if self.f0_mode == "Constant":
            f0 = self.f0
        if self.f0_mode == "Dubinin":
            pc = self.stored_fluid.backend.p_critical()
            Tc = self.stored_fluid.backend.T_critical()
            if T < Tc:
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            else:
                p0 = ((T/Tc)**self.k) * pc
                self.stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
        return rhoa * self.w0 * np.exp(-((sp.constants.R * T / \
                                     (self.eps))**self.m)\
                                   * ((np.log(f0/fug))**self.m))
    
    def n_excess(self, p, T) :
        fluid = self.stored_fluid.backend
        fluid.update(CP.PT_INPUTS, p, T)
        rhomolar = fluid.rhomolar()
        return self.n_absolute(p, T) - rhomolar * self.v_ads(p,T)
    
    
    @classmethod
    def from_ExcessIsotherms(cls, 
                             ExcessIsotherms : List[ExcessIsotherm],
                             stored_fluid : StoredFluid = None,
                             sorbent: str = None,
                             w0guess: float = 0.001,
                             f0guess: float = 1470E6,
                             epsguess : float = 3000,
                             rhoaguess : float = None,
                             mguess : float = 2.0,
                             kguess : float = 2.0,
                             rhoa_mode : str = "Fit",
                             f0_mode : str = "Fit",
                             m_mode : str = "Fit",
                             k_mode : str = "Fit",
                             va_mode: str = "Fit",
                             pore_volume : float = 0.003):
        """
        This function takes a list of excessisotherms object,
        fits an MPTA model, and uses the fitted parameters to
        instantiate an MPTAModel object.

        Parameters
        ----------
        ExcessIsotherms : list[ExcessIsotherm]
            A list of excess isotherms object describing experimental
            adsorption measurement of the same material at different 
            temperatures.
        stored_fluid : StoredFluid, optional
            Object containing stored fluid properties and CoolProp backend.
            The default is None.
        sorbent : str, optional
            Name of the sorbent material. The default is None.
        eps0guess : float, optional
            Initial guess for characteristic adsorption energy (J/mol).
            The default is 2000.
        betaguess : float, optional
            Initial guess for the pore size heterogeneity parameter.
            The default is 2, but this value should only be between 0-10.
        lamguess : float, optional
            Initial guess for micropore volume (m^3/kg).
            The default is 0.001.
        gamguess : float, optional
            Initial guess for change in micropore volume w.r.t. temperature.
            (m^3/(kg K))
            The default is -3E-6.

        Returns
        -------
        MPTAModel
            Class that contains MPTA model parameters as well as 
            methods to get the adsorbed amount at a given pressure
            and temperature.

        """
        
        excess_isotherms = deepcopy(ExcessIsotherms)
        
        #Take values from excess isotherm if not supplied in argument
        if sorbent == None:
            sorbent = excess_isotherms[0].sorbent
        if stored_fluid == None:
            stored_fluid = StoredFluid(
                fluid_name=excess_isotherms[0].adsorbate, EOS="HEOS")
        
        
        
        def rhoa_switch(paramsvar, p, T, stored_fluid):
            if rhoa_mode == "Fit":
                return paramsvar["rhoa"]
            elif rhoa_mode == "Constant":
                return rhoaguess
            elif rhoa_mode == "Ozawa":
                stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
                Tb = stored_fluid.backend.T()
                vb = 1/stored_fluid.backend.rhomolar()
                ads_specific_volume = vb * np.exp((T-Tb)/T)
                return 1/ads_specific_volume
                
        def m_switch(paramsvar):
            if m_mode == "Constant":
                return mguess
            elif m_mode == "Fit":
                return paramsvar["m"]
        
        def k_switch(paramsvar):
            if k_mode == "Constant":
                return kguess
            elif k_mode == "Fit":
                return paramsvar["k"]
        
        def f0_switch(paramsvar, T, stored_fluid, k):
            if f0_mode == "Fit":
                return paramsvar["f0"]
            elif f0_mode == "Dubinin":
                pc = stored_fluid.backend.p_critical()
                Tc = stored_fluid.backend.T_critical()
                if T < Tc:
                    stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                else:
                    p0 = ((T/Tc)**k) * pc
                    stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                return f0
            
        loading_combined = []
        temperature_combined = []
        pressure_combined = []
        for i, isotherm in enumerate(excess_isotherms):
            pressure_data = isotherm.pressure
            loading_data = isotherm.loading
            temperature = isotherm.temperature
            loading_combined = np.append(loading_combined, loading_data)
            temperature_combined = np.append(temperature_combined, np.repeat(temperature,len(pressure_data)))
            pressure_combined = np.append(pressure_combined, pressure_data)
        
        params = lmfit.Parameters()
        params.add("w0", w0guess, True, 0, pore_volume)
        params.add("eps", epsguess, True, 300, 80000)
        if f0_mode == "Fit":
            params.add("f0", f0guess, True, 1E5)
        if rhoa_mode =="Fit":
            params.add("rhoa", rhoaguess, min = 0)
        if m_mode == "Fit":
            params.add("m", mguess , min = 1, max = 6)
        if k_mode == "Fit" and f0_mode == "Dubinin":
            params.add("k", kguess , min = 0, max = 6)
            
        def n_excess(p, T, params, stored_fluid):
            phase = stored_fluid.determine_phase(p, T)
            if phase != "Saturated":
                stored_fluid.backend.update(CP.PT_INPUTS, p, T)
            else:
                stored_fluid.backend.update(CP.QT_INPUTS, 1, T)
            fug = stored_fluid.backend.fugacity(0)
            rhof = stored_fluid.backend.rhomolar()
            k = k_switch(params)
            f0 = f0_switch(params, T, stored_fluid, k)
            m = m_switch(params)
            vads = params["w0"] * \
                np.exp(-((sp.constants.R * T / \
                          (params["eps"]))**m) * ((np.log(f0/fug))**m))
            rhoa = rhoa_switch(params, p, T, stored_fluid)
            va = vads if va_mode != "Excess" else 0
            return vads * rhoa - va * rhof
        
        def fit_penalty(params, dataP, dataAd, dataT, stored_fluid):
            value = params.valuesdict()
            difference = []
            for i in range(0, len(dataP)):
                difference.append(n_excess(dataP[i],dataT[i],value,stored_fluid) - dataAd[i])
            return difference
        
        fitting = lmfit.minimize(fit_penalty, params, args=(pressure_combined,
                                                                loading_combined, 
                                                                temperature_combined,
                                                                stored_fluid))
        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()
        
        f0_res = paramsdict["f0"] if f0_mode == "Fit" else f0guess
        k_res = paramsdict["k"] if k_mode == "Fit" else kguess
        m_res = paramsdict["m"] if m_mode == "Fit" else mguess 
        rhoa_res = paramsdict["rhoa"] if rhoa_mode == "Fit" else rhoaguess
        f0mode = "Constant" if f0_mode == "Fit" else f0_mode
        rhoamode = "Constant" if rhoa_mode == "Fit" else rhoa_mode
        
        return cls(sorbent = sorbent,
                      stored_fluid = stored_fluid,
                      w0  = paramsdict["w0"],
                      f0 = f0_res,
                      eps = paramsdict["eps"],
                      m = m_res,
                      k = k_res,
                      rhoa = rhoa_res,
                      rhoa_mode = rhoamode,
                      va_mode = va_mode,
                      f0_mode = f0mode)
    
    
    
class MDAModel(ModelIsotherm):
    def __init__(self,
                  sorbent : str,
                  stored_fluid : StoredFluid,
                  nmax : float,
                  f0 : float,
                  alpha : float,
                  beta : float,
                  va : float,
                  m : float = 2,
                  k : float = 2,
                  va_mode : str = "Constant",
                  f0_mode : str = "Constant"):
         self.sorbent = sorbent
         self.stored_fluid = stored_fluid
         self.f0 = f0
         self.alpha = alpha
         self.beta = beta
         self.va = va
         self.nmax = nmax
         self.m = m
         self.k = k
         self.va_mode = va_mode
         self.f0_mode = f0_mode
     
    def n_absolute(self, p, T):
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)
        
        if self.f0_mode == "Constant":
            f0 = self.f0
        if self.f0_mode == "Dubinin":
            pc = self.stored_fluid.backend.p_critical()
            Tc = self.stored_fluid.backend.T_critical()
            if T < Tc:
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            else:
                p0 = ((T/Tc)**self.k) * pc
                self.stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            
        return self.nmax * np.exp(-((sp.constants.R * T / \
                                     (self.alpha + self.beta * T))**self.m)\
                                   * ((np.log(f0/fug))**self.m))
    
    def v_ads(self, p, T):
        if self.va_mode == "Constant":
            return self.va
        if self.va_mode == "Ozawa":
            self.stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            Tb = self.stored_fluid.backend.T()
            vb = 1/self.stored_fluid.backend.rhomolar()
            ads_specific_volume = vb * np.exp((T-Tb)/T)
            ads_density = 1/ads_specific_volume
            na = self.n_absolute(p, T) 
            return na / ads_density
    
    def n_excess(self, p, T) :
        fluid = self.stored_fluid.backend
        fluid.update(CP.PT_INPUTS, p, T)
        rhomolar = fluid.rhomolar()
        return self.n_absolute(p, T) - rhomolar * self.v_ads(p,T)
    

    
    @classmethod
    def from_ExcessIsotherms(cls, 
                             ExcessIsotherms : List[ExcessIsotherm],
                             stored_fluid : StoredFluid = None,
                             sorbent: str = None,
                             nmaxguess: float = 71.6,
                             f0guess: float = 1470E6,
                             alphaguess: float = 3080, 
                             betaguess: float = 18.9,
                             vaguess : float = 0.00143,
                             mguess : float = 2.0,
                             kguess : float = 2.0,
                             va_mode : str = "Fit",
                             f0_mode : str = "Fit",
                             m_mode : str = "Fit",
                             k_mode : str = "Fit",
                             beta_mode : str = "Fit",
                             pore_volume : float = 0.003):
        """
        This function takes a list of excessisotherms object,
        fits an MPTA model, and uses the fitted parameters to
        instantiate an MPTAModel object.

        Parameters
        ----------
        ExcessIsotherms : list[ExcessIsotherm]
            A list of excess isotherms object describing experimental
            adsorption measurement of the same material at different 
            temperatures.
        stored_fluid : StoredFluid, optional
            Object containing stored fluid properties and CoolProp backend.
            The default is None.
        sorbent : str, optional
            Name of the sorbent material. The default is None.
        eps0guess : float, optional
            Initial guess for characteristic adsorption energy (J/mol).
            The default is 2000.
        betaguess : float, optional
            Initial guess for the pore size heterogeneity parameter.
            The default is 2, but this value should only be between 0-10.
        lamguess : float, optional
            Initial guess for micropore volume (m^3/kg).
            The default is 0.001.
        gamguess : float, optional
            Initial guess for change in micropore volume w.r.t. temperature.
            (m^3/(kg K))
            The default is -3E-6.

        Returns
        -------
        MPTAModel
            Class that contains MPTA model parameters as well as 
            methods to get the adsorbed amount at a given pressure
            and temperature.

        """
        
        excess_isotherms = deepcopy(ExcessIsotherms)
        
        #Take values from excess isotherm if not supplied in argument
        if sorbent == None:
            sorbent = excess_isotherms[0].sorbent
        if stored_fluid == None:
            stored_fluid = StoredFluid(
                fluid_name=excess_isotherms[0].adsorbate, EOS="HEOS")
        
        loading_combined = []
        temperature_combined = []
        pressure_combined = []
        
        def va_switch(paramsvar, p, T, stored_fluid, nabs):
            if va_mode == "Fit":
                return paramsvar["va"]
            elif va_mode == "Constant":
                return vaguess
            elif va_mode == "Ozawa":
                stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
                Tb = stored_fluid.backend.T()
                vb = 1/stored_fluid.backend.rhomolar()
                ads_specific_volume = vb * np.exp((T-Tb)/T)
                ads_density = 1/ads_specific_volume
                return nabs / ads_density
        
        def m_switch(paramsvar):
            if m_mode == "Constant":
                return mguess
            elif m_mode == "Fit":
                return paramsvar["m"]
        
        def k_switch(paramsvar):
            if k_mode == "Constant":
                return kguess
            elif k_mode == "Fit":
                return paramsvar["k"]
            
        def f0_switch(paramsvar, T, stored_fluid, k):
            if f0_mode == "Fit":
                return paramsvar["f0"]
            elif f0_mode == "Dubinin":
                pc = stored_fluid.backend.p_critical()
                Tc = stored_fluid.backend.T_critical()
                if T < Tc:
                    stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                else:
                    p0 = ((T/Tc)**k) * pc
                    stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                return f0
            
        
        min_nmax = 1
        for i, isotherm in enumerate(excess_isotherms):
            pressure_data = isotherm.pressure
            loading_data = isotherm.loading
            temperature = isotherm.temperature
            loading_combined = np.append(loading_combined, loading_data)
            temperature_combined = np.append(temperature_combined, np.repeat(temperature,len(pressure_data)))
            pressure_combined = np.append(pressure_combined, pressure_data)
            min_nmax = max(loading_data) if max(loading_data) > min_nmax else min_nmax
        params = lmfit.Parameters()
        params.add("nmax", nmaxguess, True, min_nmax, 300)
        if f0_mode == "Fit":
            params.add("f0", f0guess, True, 1E5)
        params.add("alpha", alphaguess, True, 500, 80000)
        params.add("beta", betaguess, beta_mode == "Fit", 0, 100)
        params.add("k", kguess, k_mode == "Fit", 1, 9)
        if va_mode =="Fit":
            params.add("va", vaguess, min = 0, max = pore_volume)
        if m_mode == "Fit":
            params.add("m", mguess , min = 1, max = 6)
        if k_mode == "Fit" and f0_mode == "Dubinin":
            params.add("k", kguess , min = 0, max = 6)
            
        def n_excess(p, T, params, stored_fluid):
            phase = stored_fluid.determine_phase(p, T)
            if phase != "Saturated":
                stored_fluid.backend.update(CP.PT_INPUTS, p, T)
            else:
                stored_fluid.backend.update(CP.QT_INPUTS, 1, T)
            fug = stored_fluid.backend.fugacity(0)
            rhof = stored_fluid.backend.rhomolar()
            k = k_switch(params)
            f0 = f0_switch(params, T, stored_fluid, k)
            m = m_switch(params)
            nabs = params["nmax"] * \
                np.exp(-((sp.constants.R * T / \
                          (params["alpha"] + params["beta"] * T))**m) * ((np.log(f0/fug))**m))
            va = va_switch(params, p, T, stored_fluid, nabs)
            return nabs - rhof * va
        
        def fit_penalty(params, dataP, dataAd, dataT, stored_fluid):
            value = params.valuesdict()
            difference = []
            for i in range(0, len(dataP)):
                difference.append(n_excess(dataP[i],dataT[i],value,stored_fluid) - dataAd[i])
            return difference
        
        fitting = lmfit.minimize(fit_penalty, params, args=(pressure_combined,
                                                                loading_combined, 
                                                                temperature_combined,
                                                                stored_fluid))
        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()
        
        f0_res = paramsdict["f0"] if f0_mode == "Fit" else f0guess
        va_res = paramsdict["va"] if va_mode == "Fit" else vaguess
        m_res = paramsdict["m"] if m_mode == "Fit" else mguess   
        k_res = paramsdict["k"] if k_mode == "Fit" else kguess    
        vamode = "Constant" if va_mode == "Fit" else va_mode
        f0mode = "Constant" if f0_mode == "Fit" else f0_mode
        
        
        return cls(sorbent = sorbent,
                   stored_fluid = stored_fluid,
                   nmax = paramsdict["nmax"],
                   f0 = f0_res,
                   alpha = paramsdict["alpha"],
                   beta = paramsdict["beta"],
                   va = va_res,
                   m = m_res,
                   k = k_res,
                   va_mode = vamode,
                   f0_mode = f0mode)
        
    
    
    

class SorbentMaterial:
    def __init__(self,
                 mass : float,
                 skeletal_density : float,
                 bulk_density : float,
                 specific_surface_area : float,
                 model_isotherm : ModelIsotherm,
                 molar_mass : float = 12.01E-3,
                 Debye_temperature : float = 1500):
        """
        

        Parameters
        ----------
        mass : float
            Mass of sorbent (kg).
        skeletal_density : float
            Skeletal density of the sorbent (kg/m^3).
        bulk_density : float
            Tapped/compacted bulk density of the sorbent (kg/m^3).
        model_isotherm : MPTAModel, optional
            Model of fluid adsorption on the sorbent. The default is None.


        """
        
        self.mass = mass
        self.skeletal_density = skeletal_density
        self.bulk_density = bulk_density
        self.model_isotherm = model_isotherm
        self.specific_surface_area = specific_surface_area
        self.molar_mass = molar_mass
        self.Debye_temperature = Debye_temperature
        
    def isosteric_heat(self, p, T):
        dn_dP = fd.partial_derivative(self.model_isotherm.n_absolute, 0,
                                      [p,T], 100)
        Vs = 1/self.skeletal_density + self.model_isotherm.v_ads(p,T)
        fluid = self.model_isotherm.stored_fluid.backend
        fluid.update(CP.PT_INPUTS, p, T)
        umolar = fluid.umolar()
        hmolar = fluid.hmolar()
        return hmolar +  self.model_isotherm.differential_heat(p, T) - umolar - (Vs/dn_dP)
        
