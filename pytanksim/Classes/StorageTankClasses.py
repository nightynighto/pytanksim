# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:42:02 2023

@author: nextf
"""

from FluidSorbentClasses import StoredFluid, SorbentMaterial
from pytanksim.Utils.TankSimUtils import Cs_gen

class StorageTank:
    def __init__(self,
                 volume : float,  
                 aluminum_mass : float, 
                 stored_fluid : StoredFluid,
                 carbon_fiber_mass: float = 0,
                 max_pressure : float = None,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5
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
        self.heat_capacity =  Cs_gen(self.sorbent_mass, 
                                     self.carbon_fiber_mass, self.aluminum_mass)
        
        self.stored_fluid = stored_fluid
        
        if max_pressure == None:
            backend = self.stored_fluid.backend
            self.max_pressure = backend.pmax()

        if vent_pressure == None:
            self.vent_pressure = self.max_pressure
            

    

        
       
        
        
        
class SorbentTank(StorageTank):
    def __init__(self,
                 volume : float,  
                 aluminum_mass : float, 
                 sorbent_material : SorbentMaterial,
                 carbon_fiber_mass: float = 0,
                 max_pressure : float = None,
                 vent_pressure : float = None,
                 min_supply_pressure : float = 1E5):
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
        stored_fluid = sorbent_material.stored_fluid
        super().__init__(volume = volume,
                         aluminum_mass = aluminum_mass,
                         stored_fluid = stored_fluid,
                         carbon_fiber_mass = carbon_fiber_mass,
                         min_supply_pressure= min_supply_pressure,
                         max_pressure = max_pressure,
                         vent_pressure = vent_pressure)
        self.sorbent_material = sorbent_material
        
    
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
    
    