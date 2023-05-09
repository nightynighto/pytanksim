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

class StorageTank:
    def __init__(self,
                 volume : float,  
                 aluminum_mass : float, 
                 stored_fluid : StoredFluid,
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 max_pressure : float = None,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5,
                 thermal_resistance : float = 0,
                 surface_area : float = 0
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
        assert volume > 0
        assert aluminum_mass >= 0
        
        
        
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
        
        if max_pressure == None:
            backend = self.stored_fluid.backend
            self.max_pressure = backend.pmax()
        else: self.max_pressure = max_pressure
        if vent_pressure == None:
            self.vent_pressure = self.max_pressure
        else: self.vent_pressure = vent_pressure
            
        self.surface_area = surface_area
        self.thermal_resistance = thermal_resistance
    
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
        return np.linalg.solve(A, b)

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
        

        
       
        
        
        
class SorbentTank(StorageTank):
    def __init__(self,
                 volume : float,  
                 aluminum_mass : float, 
                 sorbent_material : SorbentMaterial,
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 max_pressure : float = None,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5,
                 thermal_resistance : float = 0,
                 surface_area : float = 0):
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
                         min_supply_pressure= min_supply_pressure,
                         max_pressure = max_pressure,
                         vent_pressure = vent_pressure,
                         thermal_resistance = thermal_resistance,
                         surface_area = surface_area,
                         steel_mass = steel_mass)
        self.sorbent_material = sorbent_material
        self.heat_capacity =  Cs_gen(mads = self.sorbent_material.mass, 
                                     mcarbon = self.carbon_fiber_mass, 
                                     malum = self.aluminum_mass,
                                     msteel = self.steel_mass)
        
    
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
        return tankvol - mads/rhoskel - vads(p,T) * mads
    
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
        uadsorbed = self.sorbent_material.model_isotherm.isosteric_internal_energy(p, T)
        return ufluid * bulk_fluid_moles + adsorbed_moles * (ufluid - uadsorbed)
    

    
    
