# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:22:37 2023

@author: nextf
"""

__all__ = ["TwoPhaseFluidSim", "TwoPhaseFluidDefault", "TwoPhaseFluidVenting",
           "TwoPhaseFluidCooled", "TwoPhaseFluidControlledInlet", "TwoPhaseFluidHeatedDischarge"]

import CoolProp as CP
import numpy as np
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation


class TwoPhaseFluidSim(BaseSimulation):
    def _dv_dT(self, ng, nl, saturation_prop_gas, saturation_prop_liquid):
        term = np.zeros(2)
        term[0] = - (ng / (saturation_prop_gas["rhof"]**2)) *\
            (saturation_prop_gas["drho_dp"] * saturation_prop_gas["dps_dT"] + \
             saturation_prop_gas["drho_dT"])
        term[1] = - (nl / (saturation_prop_liquid["rhof"]**2)) *\
            (saturation_prop_liquid["drho_dp"] * saturation_prop_liquid["dps_dT"] + \
             saturation_prop_liquid["drho_dT"])
        
        return sum(term)
    
    def _dU_dT(self, ng, nl, saturation_prop_gas, saturation_prop_liquid):
        term = np.zeros(2)
        term[0] = ng * (saturation_prop_gas["du_dp"] * saturation_prop_gas["dps_dT"] \
                        + saturation_prop_gas["du_dT"])
        term[1] = nl * (saturation_prop_liquid["du_dp"] * saturation_prop_liquid["dps_dT"] \
                        + saturation_prop_liquid["du_dT"])
        return sum(term)
    
class TwoPhaseFluidDefault(TwoPhaseFluidSim):
    def solve_differentials(self, time, ng, nl, T):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = 1/ satur_prop_gas["rhof"]
        m22 = 1/ satur_prop_liquid["rhof"]
        m23 = self._dv_dT(ng, nl, satur_prop_gas, satur_prop_liquid)
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = self._dU_dT(ng, nl, satur_prop_gas, satur_prop_liquid)
        
        A = np.array([[m11 , m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
         
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(satur_prop_gas["psat"], T)
            Tinput = flux.temperature_in(satur_prop_gas["psat"],T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * satur_prop_gas["hf"] + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)
                
        b = np.array([b1,b2,b3])
        
        diffresults = np.linalg.solve(A, b)
        
        return np.append(diffresults, [ndotin,
        ndotin * hin,
        ndotout,
        ndotout * satur_prop_gas["hf"],
        self.boundary_flux.cooling_power(time),
        self.boundary_flux.heating_power(time),
        self.heat_leak_in(T)])
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            ng, nl, T = w[:3]
            diffresults = self.solve_differentials(t, ng , nl, T)
            return diffresults
        
        def events(t, w, sw):
            #Check that the minimum and pressure has not been reached
            fluid.update(CP.QT_INPUTS, 0, w[2])
            p = fluid.p()
            min_pres_event = p - self.storage_tank.min_supply_pressure
            max_pres_event = self.storage_tank.vent_pressure - p
            #Check that the target temperature has not been reached
            target_temp_event = self.simulation_params.target_temp - w[2]
            target_pres_event = self.simulation_params.target_pres - p
            #check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            #Check that the critical temperature hasn't been reached
            crit_temp_event = w[2] - fluid.T_critical()
            return np.array([min_pres_event, max_pres_event,
                             sat_gas_event, sat_liquid_event, 
                             target_temp_event, target_pres_event,
                             crit_temp_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            
            if state_info[0] != 0:
                print("\n Minimum pressure has been reached. \n Switch to heated discharge simulation.")
                raise TerminateSimulation
                
            if state_info[1] != 0:
                print("\n Maximum pressure has been reached. \n Either begin venting or cooling.")
                raise TerminateSimulation
                        
            if state_info[2] != 0 or state_info[3] != 0:
                print("\n Phase change has ended. Switch to one phase simulation.")
                raise TerminateSimulation

            if state_info[4] != 0 and solver.sw[0]:
                print("\n Target temperature reached.")
                raise TerminateSimulation

            if state_info[5] != 0 and solver.sw[1]:
                print("\n Target pressure reached.")
                raise TerminateSimulation

            if state_info[4] != 0 and state_info[5] != 0:
                print("\n Target conditions reached.")
                raise TerminateSimulation
                
            if state_info[6] != 0:
                print("\n Reached critical temperature. Switch to one phase simulation.")
                raise TerminateSimulation
            
                
            
     
        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.init_temperature,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in])
        
        
        sw0 = self.simulation_params.stop_at_target_temp
        
        sw1 = self.simulation_params.stop_at_target_pressure
            
        switches0 = [sw0, sw1]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
        pres = np.zeros_like(t)
        
        for i, time in enumerate(t):
            fluid.update(CP.QT_INPUTS, 0, y[i, 2])
            pres[i] = fluid.p()
            

            
        return SimResults(time = t, 
                          pressure = pres,
                          temperature = y[:, 2],
                          moles_adsorbed = 0,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical = 0,
                          inserted_amount = y[:, 3],
                          flow_energy_in = y[:,4],
                          vented_amount = y[:,5],
                          vented_energy = y[:,6],
                          cooling_additional = y[:,7],
                          heating_additional = y[:,8],
                          heat_leak_in = y[:,9],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseFluidVenting(TwoPhaseFluidSim):
    def solve_differentials(self, time):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 1)
        satur_prop_liquid = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 0)
        
        m11 = 1
        m12 = 1
        m13 = 1
        m21 = 1/ satur_prop_gas["rhof"]
        m22 = 1/ satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = satur_prop_gas["hf"]
        
        
        A = np.array([[m11 , m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
         
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(satur_prop_gas["psat"], self.simulation_params.init_temperature)
            Tinput = flux.temperature_in(satur_prop_gas["psat"],self.simulation_params.init_temperature)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin 
        b2 = 0
        b3 = ndotin * hin + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(self.simulation_params.init_temperature)
                
        b = np.array([b1,b2,b3])
        
        diffresults = np.linalg.solve(A, b)
        ndotout = diffresults[-1]
        diffresults = np.append(diffresults, ndotout * satur_prop_gas["hf"])
        return np.append(diffresults, [ndotin,
                                       ndotin * hin,
                                       self.boundary_flux.cooling_power(time),
                                       self.boundary_flux.heating_power(time),
                                       self.heat_leak_in(self.simulation_params.init_temperature)]) 
        
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]        
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
            return diffresults
        
        def events(t, w, sw):
            #check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            return np.array([sat_gas_event, sat_liquid_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            
            if state_info[0] != 0 or state_info[1] != 0:
                print("\n Phase change has ended. Switch to one phase simulation.")
                raise TerminateSimulation            
            
     
        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in
                       ])
        
        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ venting"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
            

            
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = 0,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical = 0,
                          vented_amount = y[:, 2],
                          vented_energy = y[:,3],
                          inserted_amount = y[:, 4],
                          flow_energy_in = y[:,5],
                          cooling_additional = y[:,6],
                          heating_additional = y[:,7],
                          heat_leak_in = y[:,8],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseFluidCooled(TwoPhaseFluidSim):
    def solve_differentials(self, time):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 1)
        satur_prop_liquid = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 0)
        
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = 1/ satur_prop_gas["rhof"]
        m22 = 1/ satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = 1
        
        
        A = np.array([[m11 , m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
         
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(satur_prop_gas["psat"], self.simulation_params.init_temperature)
            Tinput = flux.temperature_in(satur_prop_gas["psat"],self.simulation_params.init_temperature)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin 
        b2 = 0
        b3 = ndotin * hin + ndotout * satur_prop_gas["hf"] +\
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time) + \
                self.heat_leak_in(self.simulation_params.init_temperature)
                
        b = np.array([b1,b2,b3])
        
        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [ndotin,
                                       ndotin * hin,
                                       ndotout,
                                       ndotout * satur_prop_gas["hf"],
                                       self.boundary_flux.cooling_power(time),
                                       self.boundary_flux.heating_power(time),
                                       self.heat_leak_in(self.simulation_params.init_temperature)
                                       ]) 
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
            return diffresults
        
        def events(t, w, sw):
            #check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            return np.array([sat_gas_event, sat_liquid_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            
            if state_info[0] != 0 or state_info[1] != 0:
                print("\n Phase change has ended. Switch to one phase simulation.")
                raise TerminateSimulation
                            
        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.cooling_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in
                       ])
        

        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ cooling"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
            

            
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = 0,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical = 0,
                          cooling_required = y[:,2],
                          inserted_amount = y[:,3],
                          flow_energy_in = y[:,4],
                          vented_amount = y[:,5],
                          vented_energy = y[:,6],
                          cooling_additional = y[:,7],
                          heating_additional = y[:,8],
                          heat_leak_in = y[:,9],
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseFluidHeatedDischarge(TwoPhaseFluidSim):
    def solve_differentials(self, time):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 1)
        satur_prop_liquid = self.storage_tank.stored_fluid.saturation_property_dict(self.simulation_params.init_temperature, 0)
        
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = 1/ satur_prop_gas["rhof"]
        m22 = 1/ satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = -1
        
        
        A = np.array([[m11 , m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
         
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(satur_prop_gas["psat"], self.simulation_params.init_temperature)
            Tinput = flux.temperature_in(satur_prop_gas["psat"],self.simulation_params.init_temperature)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin 
        b2 = 0
        b3 = ndotin * hin + ndotout * satur_prop_gas["hf"] +\
            - self.boundary_flux.cooling_power(time) + self.boundary_flux.heating_power(time) + \
                self.heat_leak_in(self.simulation_params.init_temperature)
                
        b = np.array([b1,b2,b3])
        
        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [ndotin,
                                       ndotin * hin,
                                       ndotout,
                                       ndotout * satur_prop_gas["hf"],
                                       self.boundary_flux.cooling_power(time),
                                       self.boundary_flux.heating_power(time),
                                       self.heat_leak_in(self.simulation_params.init_temperature)]) 
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
            return diffresults
        
        def events(t, w, sw):
            #check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            return np.array([sat_gas_event, sat_liquid_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            
            if state_info[0] != 0 or state_info[1] != 0:
                print("\n Phase change has ended. Switch to one phase simulation.")
                raise TerminateSimulation
                             
        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.heating_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in
                       ])
        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ cooling"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = 0,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical = 0,
                          inserted_amount = y[:,3],
                          flow_energy_in = y[:,4],
                          vented_amount = y[:,5],
                          vented_energy = y[:,6],
                          cooling_additional = y[:,7],
                          heating_additional = y[:,8],
                          heat_leak_in = y[:,9],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = y[:,2] ,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseFluidControlledInlet(TwoPhaseFluidDefault):
    def solve_differentials(self, time, ng, nl, T):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = 1/ satur_prop_gas["rhof"]
        m22 = 1/ satur_prop_liquid["rhof"]
        m23 = self._dv_dT(ng, nl, satur_prop_gas, satur_prop_liquid)
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = self._dU_dT(ng, nl, satur_prop_gas, satur_prop_liquid)
        
        A = np.array([[m11 , m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
         
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            hin = flux.enthalpy_in(time)
        else:
            hin = 0
        
        if flux.mass_flow_out(time) != 0:
            hout = flux.enthalpy_out(time)
        else:
            hout = 0
        
        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)
                
        b = np.array([b1,b2,b3])
        
        diffresults = np.linalg.solve(A, b)
        
        return np.append(diffresults, [ndotin,
        ndotin * hin,
        ndotout,
        ndotout * hout,
        self.boundary_flux.cooling_power(time),
        self.boundary_flux.heating_power(time),
        self.heat_leak_in(T)])