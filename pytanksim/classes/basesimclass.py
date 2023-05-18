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
import CoolProp as CP
import numpy as np



@dataclass
class SimulationParams:
    """Class for storing parameters for the tank simulation"""
    
    init_temperature : float # in K
    final_time: float # in seconds
    init_time : float = 0 # in seconds
    displayed_points : float = 200
    target_temp: float = 0
    target_pres: float = 0
    init_pressure : float = 1E5  # in Pa
    init_ng : float = 0
    init_nl : float = 0
    inserted_amount : float = 0
    vented_amount : float = 0
    cooling_required : float = 0
    heating_required : float = 0
    vented_energy : float = 0
    flow_energy_in : float = 0
    cooling_additional : float = 0
    heating_additional : float = 0
    heat_leak_in : float = 0
    stop_at_target_pressure : bool = False
    stop_at_target_temp : bool = False
    target_capacity : float = 0
    
    
    @classmethod
    def from_SimResults(cls,
                        sim_results : SimResults,
                        displayed_points : float = 200,
                        final_time : float = None,
                        target_pres : float = None,
                        target_temp :  float = None,
                        stop_at_target_pressure : bool = False,
                        stop_at_target_temp : bool = False,
                        target_capacity: float = None
                        ):
        final_conditions = sim_results.get_final_conditions()
        
        if target_temp == None:
            target_temp = sim_results.sim_params.target_temp
        if target_pres == None:
            target_pres = sim_results.sim_params.target_pres
        if stop_at_target_temp == None:
            stop_at_target_temp = sim_results.sim_params.stop_at_target_temp
        if stop_at_target_pressure == None:
            stop_at_target_pressure = sim_results.sim_params.stop_at_target_pressure
        if target_capacity == None:
            target_capacity = sim_results.sim_params.target_capacity
        if final_time == None:
            final_time = sim_results.sim_params.final_time
        
        if final_conditions["moles_gas"] == final_conditions["moles_liquid"] == 0:
            fluid = sim_results.tank_params.stored_fluid.backend
            Tcrit = fluid.T_critical()
            final_minus_one = sim_results.get_final_conditions(-2)
            if final_minus_one["temperature"] > Tcrit:
                final_conditions["moles_gas"] = final_conditions["moles_supercritical"]
            else:
                final_conditions["moles_liquid"] = final_conditions["moles_supercritical"]
        return cls(
                init_pressure = final_conditions["pressure"],
                   init_temperature = final_conditions["temperature"],
                   init_time = final_conditions["time"],
                   init_ng = final_conditions["moles_gas"],
                   init_nl = final_conditions["moles_liquid"],
                   inserted_amount = final_conditions["inserted_amount"],
                   vented_amount = final_conditions["vented_amount"],
                   cooling_required = final_conditions["cooling_required"],
                   heating_required = final_conditions["heating_required"],
                   vented_energy = final_conditions["vented_energy"],
                   flow_energy_in = final_conditions["flow_energy_in"],
                   heat_leak_in = final_conditions["heat_leak_in"],
                   cooling_additional = final_conditions["cooling_additional"],
                   heating_additional = final_conditions["heating_additional"],
                   final_time = final_time,
                   target_temp = target_temp,
                   target_pres = target_pres,
                   stop_at_target_pressure = stop_at_target_pressure,
                   stop_at_target_temp = stop_at_target_temp,
                   target_capacity = target_capacity,
                   displayed_points = displayed_points,
                   )
        

    
class BoundaryFlux:
    def __init__(self,
                 mass_flow_in: float = 0.0,
                 mass_flow_out: float  = 0.0,                 
                 heating_power: float  = 0.0,
                 cooling_power: float  = 0.0, 
                 pressure_in: Callable[[float, float], float] = None,
                 temperature_in: Callable[[float, float], float] = None,
                 fill_rate: float = None,
                 environment_temp: float = 0,
                 enthalpy_in: Callable[[float], float] = 0.0,
                 enthalpy_out: Callable[[float], float] = 0.0):
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
        
        def float_function_generator_time(floatingvalue):
            def float_function(time):
                return floatingvalue
            return float_function
        
        
        if isinstance(pressure_in, float):
            pressure_in = float_function_generator(pressure_in)
        
        if isinstance(temperature_in, float):
            temperature_in = float_function_generator(temperature_in)
            
        if isinstance(mass_flow_in, float):
            mass_flow_in = float_function_generator_time(mass_flow_in)
        
        if isinstance(mass_flow_out, float):
            mass_flow_out = float_function_generator_time(mass_flow_out)
        
        if isinstance(enthalpy_in, float):
            enthalpy_in = float_function_generator_time(enthalpy_in)
        
        if isinstance(enthalpy_out, float):
            enthalpy_out = float_function_generator_time(enthalpy_out)
            
        if isinstance(heating_power, float):
            heating_power = float_function_generator_time(heating_power)
        
        if isinstance(cooling_power, float):
            cooling_power = float_function_generator_time(cooling_power)
        
        self.mass_flow_in = mass_flow_in
        self.mass_flow_out = mass_flow_out
        self.heating_power = heating_power
        self.cooling_power = cooling_power
        self.pressure_in = pressure_in
        self.temperature_in = temperature_in
        self.fill_rate = fill_rate
        self.environment_temp =environment_temp
        self.enthalpy_in = enthalpy_in
        
        
        if mass_flow_in != 0 and (pressure_in == None and temperature_in == None and enthalpy_in == None):
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
        if isinstance(self.storage_tank, SorbentTank):
            self.has_sorbent = True
        self.has_sorbent = False
        
    def heat_leak_in(self, T):
        if self.boundary_flux.environment_temp == 0:
            return 0
        else:
            return (self.boundary_flux.environment_temp - T)/self.storage_tank.thermal_resistance
    
    def run(self):
        raise NotImplementedError
    
    def enthalpy_in_calc(self, p, T):
        pin = self.boundary_flux.pressure_in(p, T)
        Tin = self.boundary_flux.temperature_in(p, T)
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        if Tin <= Tcrit:
            fluid.update(CP.QT_INPUTS, 0, Tin)
            psat = fluid.p()
            if np.abs(pin-psat) <= 1E-6 * psat:
                fluid.update(CP.QT_INPUTS, 0, Tin)
                return fluid.hmolar()
            else:
                fluid.update(CP.PT_INPUTS, pin, Tin)
                return fluid.hmolar()
        else:
            fluid.update(CP.PT_INPUTS, pin, Tin)
            return fluid.hmolar()
