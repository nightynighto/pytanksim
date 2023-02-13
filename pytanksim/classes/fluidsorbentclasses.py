# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:17:45 2023

@author: nextf
"""

__all__ = ["StoredFluid", "MPTAModel", "SorbentMaterial", "MDAModel"]

import pytanksim.mpta as mpta
import numpy as np
import CoolProp as CP
import lmfit
import pytanksim.utils.tanksimutils as tsu
from pytanksim.classes.excessisothermclass import ExcessIsotherm
from copy import deepcopy
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
            "dh_dT" : backend.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP)
            }
    
    def saturation_property_dict(self,
                                 T:float):
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
        
        backend = self.TankParameters.backend
        backend.update(CP.QT_INPUTS, 0, T)
        return {
            "psat" : backend.p(),
            "dP_dT" : backend.first_saturation_deriv(CP.iP, CP.iT)
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
            if p < psat:
                return "Gas"
            elif p > psat and p > pcrit:
                return "Supercritical"
            elif p > psat and p< pcrit :
                return "Liquid"
            elif np.abs(p-psat) < psat * 1E-6:
                return "Saturated"
            
            
        

class MPTAModel:
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
 
    
class MDAModel(MPTAModel):
    def __init__(self,
                  sorbent : str,
                  stored_fluid : StoredFluid,
                  nmax : float,
                  p0 : float,
                  alpha : float,
                  beta : float,
                  va : float):
         self.sorbent = sorbent
         self.stored_fluid = stored_fluid
         self.p0 = p0
         self.alpha = alpha
         self.beta = beta
         self.va = va
         self.nmax = nmax
     
    def n_absolute(self, p, T):
         return self.nmax * np.exp(-((sp.constants.R * T / (self.alpha + self.beta * T))**2)\
                                   * ((np.log(self.p0/p))**2))
    def v_ads(self, p, T):
        return self.va
    

class SorbentMaterial:
    def __init__(self,
                 mass : float,
                 skeletal_density : float,
                 bulk_density : float,
                 model_isotherm : MPTAModel = None):
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
    
    
