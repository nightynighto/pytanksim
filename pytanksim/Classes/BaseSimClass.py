# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:45:42 2023

@author: nextf
"""


__all__ = ["SimulationParams", "BoundaryFlux", "BaseSimulation"]


from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
from dataclasses import dataclass
from typing import Callable



@dataclass
class SimulationParams:
    """Class for storing parameters for the tank simulation"""
    
    init_pressure : float  # in Pa
    init_temperature : float # in K
    final_time: float # in seconds
    init_time : float = 0 # in seconds
    displayed_points : float = 200
    target_temp: float = 0
    
    @classmethod
    def from_SimResults(cls,
                        sim_results : SimResults,
                        final_time : float = None,
                        displayed_points : float = 200,
                        target_temp :  float = None):
        final_conditions = sim_results.get_final_conditions()
        return cls(
                init_pressure = final_conditions["pressure"],
                   init_temperature = final_conditions["temperature"],
                   init_time = final_conditions["time"],
                   final_time = final_time,
                   target_temp = target_temp,
                   displayed_points = displayed_points
                   )
        

    
class BoundaryFlux:
    def __init__(self,
                 mass_flow_in: float = 0,
                 mass_flow_out: float  = 0,                 
                 heating_power: float  = 0,
                 cooling_power: float  = 0, 
                 pressure_in: Callable[[float, float], float] = None,
                 temperature_in: Callable[[float, float], float] = None,
                 fill_rate: float = None):
        """
        

        Parameters
        ----------
        mass_flow_in : float, optional
            Mass flow going into the simulation boundary (kg/s). The default is 0.
        mass_flow_out : float, optional
            Mass flow going out of the simulation boundary (kg/s). The default is 0.
        heating_power : float, optional
            How much heat is going into the simulation boundary (W). The default is 0.
        cooling_power : float, optional
            How much heat is going out of the simulation boundary (W). The default is 0.
        pressure_in : Callable[[float, float], float], optional
            Pressure of the fluid flowing in (Pa) as a function of 
            control volume pressure and temperature. The default is None.
        temperature_in : Callable[[float, float], float], optional
            Temperature of the fluid flowing in (K) as a function of control volume
            pressure and temperature. The default is None.
        fill_rate : float, optional
            The net mass change inside of the simulation boundary
            per second (kg/s). The default is None.
        """
        
        def float_function_generator(floatingvalue):
            def float_function(p,T):
                return floatingvalue
            return float_function
        
        if isinstance(pressure_in, float):
            pressure_in = float_function_generator(pressure_in)
        
        if isinstance(temperature_in, float):
            temperature_in = float_function_generator(temperature_in)
        
        self.mass_flow_in = mass_flow_in
        self.mass_flow_out = mass_flow_out
        self.heating_power = heating_power
        self.cooling_power = cooling_power
        self.pressure_in = pressure_in
        self.temperature_in = temperature_in
        self.fill_rate = fill_rate
        
        
        
        if mass_flow_in and (pressure_in == None and temperature_in == None):
            raise ValueError("Please specify the pressure and temperature of the flow going in")
        if fill_rate and mass_flow_in and mass_flow_out and \
                fill_rate != mass_flow_in - mass_flow_out:
            raise ValueError("Filling rate is not consistent with the mass flow in and out.")

    
class BaseSimulation:
    sim_type = None
    sim_phase = None
    def __init__(self, 
                 simulation_params : SimulationParams,
                 storage_tank : StorageTank,
                 boundary_flux : BoundaryFlux):
        self.simulation_params = simulation_params
        self.storage_tank = storage_tank
        self.boundary_flux = boundary_flux
        if self.storage_tank.max_pressure:
            assert self.storage_tank.max_pressure >= self.simulation_params.init_pressure 
        assert self.storage_tank.min_supply_pressure <= self.simulation_params.init_pressure
        if isinstance(self.storage_tank, SorbentTank):
            self.has_sorbent = True
        self.has_sorbent = False
    
    def run(self):
        raise NotImplementedError
