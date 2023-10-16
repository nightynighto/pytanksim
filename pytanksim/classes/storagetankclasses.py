# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:42:02 2023

@author: nextf
"""

__all__ = ["StorageTank", "SorbentTank"]

from pytanksim.classes.fluidsorbentclasses import StoredFluid, SorbentMaterial
from pytanksim.utils.tanksimutils import Cs_gen
import CoolProp as CP
import numpy as np
import scipy as sp
import pandas as pd

class StorageTank:
    def __init__(self,
                 volume : float,  
                 stored_fluid : StoredFluid,
                 aluminum_mass : float = 0, 
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5,
                 thermal_resistance : float = 0,
                 surface_area : float = 0,
                 heat_transfer_coefficient : float = 0
                 ):
        """
        This class represents a tank for storing fluids.

        Parameters
        ----------
        volume : float
            Volume of the storage tank in m^3.
        aluminum_mass : float
           Mass of aluminum in the tank structure in kg.
        carbon_fiber_mass : float, optional
            Mass of carbon fiber used in tank structure in kg.
            The default is 0.
        stored_fluid : StoredFluid,
            An instance of the StoredFluid class which stores the properties
            of the fluid being stored in the tank.
        max_pressure : float, optional
            Maximum pressure (Pa) that can be stored inside of a tank.
            The default is None.
        vent_pressure : float, optional
            Pressure at which the tank is designed to start to vent. Pa.
            The default is None.
        min_supply_pressure: float, optional
            Minimum supply pressure of the tank when discharging in Pa.
            The default is 100000 Pa (Atmospheric Pressure).
        backend : CP.AbstractState, optional
            Coolprop backend to do thermophysical fluid property calculations.
            The default is None.
        

        """
        if (aluminum_mass or carbon_fiber_mass or steel_mass or volume) < 0:
            raise ValueError("Please input valid values for the mass and volume (>=0)")
        
        
        self.volume = volume
        self.aluminum_mass = aluminum_mass
        self.carbon_fiber_mass = carbon_fiber_mass
        self.steel_mass = steel_mass
        self.heat_capacity =  Cs_gen(mads = 0, 
                                     mcarbon = self.carbon_fiber_mass,
                                     malum = self.aluminum_mass,
                                     msteel = self.steel_mass)
        
        self.stored_fluid = stored_fluid
        self.min_supply_pressure = min_supply_pressure
        
        backend = self.stored_fluid.backend
        self.max_pressure = backend.pmax()/10
        
        if vent_pressure == None:
            self.vent_pressure = self.max_pressure
        else: self.vent_pressure = vent_pressure
        
        if self.max_pressure < self.vent_pressure:
            raise ValueError(
                "You set the venting pressure to be larger than the valid  \n" +
                "pressure range input for CoolProp.")
            
        self.surface_area = surface_area
        self.heat_transfer_coefficient = heat_transfer_coefficient
        if thermal_resistance >= 0:
            self.thermal_resistance = thermal_resistance
        elif (self.surface_area and self.heat_transfer_coefficient) >= 0:
            self.thermal_resistance = 1/(self.surface_area * self.heat_transfer_coefficient)
            
            
    
    def capacity(self, p, T, q = 0):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        
        return fluid.rhomolar() * self.volume
                
    def find_quality_at_saturation_capacity(self, T, capacity):
        fluid = self.stored_fluid.backend
        fluid.update(CP.QT_INPUTS, 0, T)
        rhol = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 1, T)
        rhog = fluid.rhomolar()
        A = np.array([[1, 1],
                      [1/rhog, 1/rhol]])
        b = [capacity, self.volume]
        res = np.linalg.solve(A, b)
        return res[0]/(res[0]+res[1])

    def internal_energy(self, p, T, q = 1):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.volume
        return ufluid * bulk_fluid_moles 
        
    def conditions_at_capacity_temperature(self, cap, T, p_guess, q_guess):
        bnds = ((10, 20E6),(0,1))
        def optim(x):
            return (self.capacity(x[0], T, x[1]) - cap)**2
        res = sp.optimize.minimize(optim,(p_guess, q_guess), bounds = bnds)
        if res.fun > 1:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
            psat = self.stored_fluid.backend.p()
            q = self.find_quality_at_saturation_capacity(T, cap)
            res.x[0] = psat
            res.x[1] = q
        return res
    
    def conditions_at_capacity_pressure(self, cap, p, T_guess, q_guess):
        bnds = ((15, 1500),(0,1))
        def optim(x):
            return (self.capacity(p, x[0], x[1]) - cap)**2
        res = sp.optimize.minimize(optim,(T_guess, q_guess), bounds = bnds)
        if res.fun > 1:
            self.stored_fluid.backend.update(CP.PQ_INPUTS, p, 0)
            Tsat = self.stored_fluid.backend.T()
            q = self.find_quality_at_saturation_capacity(Tsat, cap)
            res.x[0] = Tsat
            res.x[1] = q
        return res
           
    def calculate_dormancy(self, p, T, q, heating_power):
        init_cap = self.capacity(p, T, q)
        init_heat = self.internal_energy(p, T, q )
        vent_cond = self.conditions_at_capacity_pressure(init_cap, self.vent_pressure, T, q).x
        final_heat = self.internal_energy(self.vent_pressure, vent_cond[0], vent_cond[1])
        final_cap = self.capacity(self.vent_pressure, vent_cond[0], vent_cond[1])
        def heat_capacity_change(T1, T2):
            xgrid = np.linspace(T1, T2, 100)
            heatcapgrid = [self.heat_capacity(temper) for temper in xgrid]
            return sp.integrate.simps(heatcapgrid, xgrid)
                
        final_heat = final_heat + heat_capacity_change(T, vent_cond[0])
        
        return pd.DataFrame({"dormancy time" : (final_heat - init_heat)/heating_power,
                "final temperature" : vent_cond[0],
                "final quality" : vent_cond[1],
                "final pressure": self.vent_pressure,
                "capacity error": final_cap - init_cap,
                "total energy change" : final_heat - init_heat,
                "solid heat capacity contribution": heat_capacity_change(T, vent_cond[0])}, index = [0])
    
    
    
       
        
        
class SorbentTank(StorageTank):
    def __init__(self,
                 volume : float,  
                 sorbent_material : SorbentMaterial,
                 aluminum_mass : float = 0, 
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5,
                 thermal_resistance : float = 0,
                 surface_area : float = 0,
                 heat_transfer_coefficient : float = 0
                 ):
        """
        Init for the class SorbentTank.
        This class represents a fluid storage tank which contains sorbents.

        Parameters
        ----------
        volume : float
            The volume of the tank in m^3.
        aluminum_mass : float
            The mass of the aluminum that makes up the tank structure (kg).
        sorbent_material : SorbentMaterial,
            An object which contains all of the sorbent material properties.
        carbon_fiber_mass : float, optional
            Mass of the structural carbon fiber in the tank. The default is 0.
        max_pressure : float, optional
            Maximum pressure (Pa) that can be stored inside of a tank.
            The default is None.
        vent_pressure : float, optional
            Pressure at which the tank is designed to start to vent. Pa.
            The default is None.
        min_supply_pressure: float, optional
            Minimum supply pressure of the tank when discharging in Pa.
            The default is 100000 Pa (Atmospheric Pressure).
        overwrite_backend : bool, optional
            Switch to overwrite the model isotherm backend with one
            supplied from the fluid name and EOS name.
            The overwrite will be done if this switch is True.
            The default is False.

        """
        stored_fluid = sorbent_material.model_isotherm.stored_fluid
        super().__init__(volume = volume,
                         aluminum_mass = aluminum_mass,
                         stored_fluid = stored_fluid,
                         carbon_fiber_mass = carbon_fiber_mass,
                         min_supply_pressure = min_supply_pressure,
                         vent_pressure = vent_pressure,
                         thermal_resistance = thermal_resistance,
                         surface_area = surface_area,
                         steel_mass = steel_mass,
                         heat_transfer_coefficient = heat_transfer_coefficient)
        self.sorbent_material = sorbent_material
        self.heat_capacity =  Cs_gen(mads = self.sorbent_material.mass, 
                                     mcarbon = self.carbon_fiber_mass, 
                                     malum = self.aluminum_mass,
                                     msteel = self.steel_mass,
                                     Tads = self.sorbent_material.Debye_temperature,
                                     MWads = self.sorbent_material.molar_mass)
        
    
    def bulk_fluid_volume(self,
                          p : float,
                          T : float):
        """

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature(K).

        Returns
        -------
        float
            Bulk fluid volume within the tank (m^3).

        """
        
        tankvol = self.volume
        mads = self.sorbent_material.mass
        rhoskel = self.sorbent_material.skeletal_density
        vads = self.sorbent_material.model_isotherm.v_ads
        outputraw = tankvol - mads/rhoskel - vads(p,T) * mads
        output =  outputraw if outputraw >= 0 else 0
        return output
    
    def capacity(self, p, T, q = 0):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        return bulk_fluid_moles + adsorbed_moles
    
    def capacity_bulk(self, p, T, q = 0):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        return bulk_fluid_moles
    
    def internal_energy(self, p, T, q = 1):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        uadsorbed = self.sorbent_material.model_isotherm.internal_energy_adsorbed(p, T)
        return ufluid * bulk_fluid_moles + adsorbed_moles * (uadsorbed)
    
    def internal_energy_sorbent(self, p, T, q = 1):
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        uadsorbed = self.sorbent_material.model_isotherm.internal_energy_adsorbed(p, T)
        return adsorbed_moles * (uadsorbed)
    
    def internal_energy_bulk(self, p, T, q = 1):
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        return ufluid * bulk_fluid_moles
    
    def find_quality_at_saturation_capacity(self, T, capacity):
        fluid = self.stored_fluid.backend
        fluid.update(CP.QT_INPUTS, 0, T)
        rhol = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 1, T)
        rhog = fluid.rhomolar()
        p = fluid.p()
        bulk_capacity = capacity - self.sorbent_material.mass *\
            self.sorbent_material.model_isotherm.n_absolute(p, T)
        A = np.array([[1, 1],
                      [1/rhog, 1/rhol]])
        b = [bulk_capacity, self.bulk_fluid_volume(p, T)]
        res = np.linalg.solve(A, b)
        q =  res[0]/(res[0]+res[1])
        return q
    
    def find_temperature_at_saturation_quality(self, q, cap):
        def optim(x):
            self.stored_fluid.backend.update(CP.QT_INPUTS, q, x)
            p = self.stored_fluid.backend.p()
            return (self.capacity(p, x, q) - cap)**2
        res = sp.optimize.minimize_scalar(optim, method = "bounded", bounds = (15, 33.145))
        return res
    
    def calculate_dormancy(self, p, T, q, heating_power, init_time = 0):
        if init_time != 0:
            print("Warning: energy breakdown not accurate for non-zero initial time.")
        init_cap = self.capacity(p, T, q)
        init_ene = self.internal_energy(p, T, q )
        init_ene_ads = self.internal_energy_sorbent(p, T, q)
        init_ene_bulk = self.internal_energy_bulk(p, T, q)
        vent_cond = self.conditions_at_capacity_pressure(init_cap, self.vent_pressure, T, q).x
        final_ene = self.internal_energy(self.vent_pressure, vent_cond[0], vent_cond[1])
        final_cap = self.capacity(self.vent_pressure, vent_cond[0], vent_cond[1])
        final_ene_ads = self.internal_energy_sorbent(self.vent_pressure, vent_cond[0], vent_cond[1])
        final_ene_bulk = self.internal_energy_bulk(self.vent_pressure, vent_cond[0], vent_cond[1])
        
        def heat_capacity_change(T1, T2):
            xgrid = np.linspace(T1, T2, 100)
            heatcapgrid = [self.heat_capacity(temper) for temper in xgrid]
            return sp.integrate.simps(heatcapgrid, xgrid)
        
        final_ene += heat_capacity_change(T, vent_cond[0])
        
        res1 = self.find_temperature_at_saturation_quality(1,init_cap)
        res2 = self.find_temperature_at_saturation_quality(0,init_cap)
        if (res1.x > T and res1.fun < 1) or (res2.x > T and res2.fun < 1) or (vent_cond[1] != q):
            resfinal = res1.x if res1.fun < res2.fun else res2.x
            
            if vent_cond[1] != q:
                lower_bound = max(q, vent_cond[1])
                upper_bound = min(q, vent_cond[1])
            else:
                lower_bound = q if resfinal == res2.x else 1
                upper_bound = 0 if resfinal == res2.x else q
            total_surface_area = self.sorbent_material.mass *\
                self.sorbent_material.specific_surface_area * 1000
            qgrid = np.linspace(lower_bound, upper_bound, 100)
            Agrid = np.zeros_like(qgrid)
            ygrid = np.zeros_like(qgrid)
            for i, qual in enumerate(qgrid):
                temper = self.find_temperature_at_saturation_quality(qual,init_cap).x
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, temper)
                p = self.stored_fluid.backend.p()
                rhol = self.stored_fluid.backend.rhomolar()
                nl = (1 - qual) * self.capacity_bulk(p, temper, qual)
                vbulk = self.bulk_fluid_volume(p, temper)
                Agrid[i] = total_surface_area * (nl/(rhol * vbulk))
                ygrid[i] = self.sorbent_material.model_isotherm.areal_immersion_energy(temper)
            integ_res = sp.integrate.simps(ygrid, Agrid)
            integ_res = integ_res if lower_bound == q else  - integ_res
            final_ene = final_ene + integ_res
        else:
            integ_res = 0
        return pd.DataFrame({"dormancy time" : init_time + (final_ene - init_ene)/heating_power,
                "final temperature" : vent_cond[0],
                "final quality" : vent_cond[1],
                "final pressure": self.vent_pressure,
                "capacity error": final_cap - init_cap,
                "total energy change" : final_ene - init_ene,
                "sorbent energy contribution" : final_ene_ads - init_ene_ads,
                "bulk energy contribution" : final_ene_bulk - init_ene_bulk,
                "immersion heat contribution" : integ_res,
                "solid heat capacity contribution": heat_capacity_change(T, vent_cond[0]) }, index = [0],
                )
    
