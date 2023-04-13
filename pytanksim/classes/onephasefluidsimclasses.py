# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:31:47 2023

@author: nextf
"""

__all__ = ["OnePhaseFluidSim", "OnePhaseFluidDefault", "OnePhaseFluidVenting",
           "OnePhaseFluidCooled", "OnePhaseFluidControlledInlet", "OnePhaseFluidHeatedDischarge"]

import CoolProp as CP
import numpy as np
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation

class OnePhaseFluidSim(BaseSimulation):
    sim_phase = "One Phase"
    
    def _dn_dp(self, fluid_prop_dict):
        return fluid_prop_dict["drho_dp"] * self.storage_tank.volume
    
    def _dn_dT(self, fluid_prop_dict):
        return fluid_prop_dict["drho_dT"] * self.storage_tank.volume
    
    def _du_dp(self, fluid_prop_dict):
        term = np.zeros(2)
        term[0] = fluid_prop_dict["drho_dp"] * fluid_prop_dict["uf"]
        term[1] = fluid_prop_dict["du_dp"] * fluid_prop_dict["rhof"]
        return self.storage_tank.volume * (sum(term))
    
    def _du_dT(self, fluid_prop_dict):
        term = np.zeros(2)
        term[0] = fluid_prop_dict["drho_dT"] * fluid_prop_dict["uf"]
        term[1] = fluid_prop_dict["du_dT"] * fluid_prop_dict["rhof"]
        return self.storage_tank.volume * (sum(term))

class OnePhaseFluidDefault(OnePhaseFluidSim):
    sim_type = "Default"
    def solve_differentials(self, time, p, T):
        prop_dict = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        
        m11 = self._dn_dp(prop_dict)
        m12 = self._dn_dT(prop_dict)
        m21 = self._du_dp(prop_dict)
        m22 = self._du_dp(prop_dict)
        
        A = np.array([[m11 , m12],
                      [m21, m22]])
        
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(p, T)
            Tinput = flux.temperature_in(p,T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * fluid_props["hf"] + \
            self.boundary_flux.heating_power - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T)
                
        b = np.array([b1, b2])
        
        soln = np.linalg.solve(A, b)
        
        return np.append(soln, ndotin)
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            p, T, inserted = w
            return self.solve_differentials(t, p, T)
        
        def events(t, w, sw):
            ##Check saturation status
            if w[1] >= Tcrit:
                satstatus = w[0] - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[1])
                satpres = fluid.p()
                if np.abs(w[0]-satpres) > (1E-6 * satpres):
                    satstatus = w[0] - satpres
                else:
                    satstatus = 0
            return np.array([self.storage_tank.vent_pressure - w[0], satstatus,
                             w[0] - self.storage_tank.min_supply_pressure])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                if solver.sw[0]:
                    print("\n The simulation has hit maximum pressure! Switch to venting or cooling simulation")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
            if state_info[1] != 0 and solver.y[1] <= Tcrit:
                if solver.sw[1]:
                    print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                    raise TerminateSimulation
                solver.sw[1] = not solver.sw[1]
            if state_info[2] != 0:
                if solver.sw[2]:
                    print("\n The simulation has hit minimum supply pressure! Switch to heated discharge simulation")
                    raise TerminateSimulation
                solver.sw[2] = not solver.sw[2]
            
     
        w0 = np.array([self.simulation_params.init_pressure,
                       self.simulation_params.init_temperature,
                       self.simulation_params.inserted_amount])
        
        
        ##Initialize switches for event handling
        if self.simulation_params.init_pressure == self.storage_tank.vent_pressure:
            sw0 = False
        else:
            sw0 = True
        if self.simulation_params.init_temperature <= Tcrit:
            fluid.update(CP.QT_INPUTS, 0, self.init_temperature)
            psat_init =  fluid.p()
            if self.simulation_params.init_pressure == psat_init:
                sw1 = False
            else:
                sw1 = True
        else:
            sw1 = True
            
        if self.simulation_params.init_pressure == self.storage_tank.min_supply_pressure:
            sw2 = False
        else:
            sw2 = True
        
        
        switches0 = [sw0, sw1, sw2]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = np.array([1E-2,  1E-6, 1E-6])
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        n_phase = {"Gas" : np.zeros_like(t),
                   "Supercritical"  :np.zeros_like(t),
                   "Liquid" : np.zeros_like(t)}
        
        for i in range(0, len(t)):
            fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1])   
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
            n_phase[phase][i] = nfluid
            
        
        return SimResults(time = t, 
                          pressure = y[:,0],
                          temperature = y[:,1],
                          moles_adsorbed = 0,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          vented_amount = self.simulation_params.vented_amount,
                          vented_energy = self.simulation_params.vented_energy,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    

class OnePhaseFluidVenting(OnePhaseFluidSim):
    sim_type = "Venting"
    def solve_differentials(self, time, T):
        p = self.simulation_params.init_pressure
        prop_dict = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        m11 = self._dn_dT(prop_dict)
        m12 = 1
        m21 = self._du_dT(prop_dict)
        m22 = prop_dict["hf"]
        
        A = np.array([[m11, m12],
                      [m21, m22]])
        
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(p, T)
            Tinput = flux.temperature_in(p,T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        
        b1 = ndotin 
        b2 = ndotin * hin + \
            self.boundary_flux.heating_power - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T)
                
        b = np.array([b1, b2])
                
        soln = np.linalg.solve(A, b)
        
        soln_w_ndotin = np.append(soln, ndotin)
        
        return np.append(soln_w_ndotin, soln[1] * m22)
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p0 = self.simulation_params.init_pressure
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T, vented, inserted = w
            return self.solve_differentials(t, T)
        
        def events(t, w, sw):
            ##Check saturation status
            if w[0] >= Tcrit:
                satstatus = p0 - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p0-satpres) > (1E-6 * satpres):
                    satstatus = p0 - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0] - self.simulation_params.target_temp])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                if solver.sw[0]:
                    print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
            if state_info[1] != 0:
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            
     
        w0 = np.array([self.simulation_params.init_temperature,
                       self.simulation_params.vented_amount,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.vented_energy])
        
        

        if self.simulation_params.init_temperature <= Tcrit:
            fluid.update(CP.QT_INPUTS, 0, self.init_temperature)
            psat_init =  fluid.p()
            if self.simulation_params.init_pressure == psat_init:
                sw0 = False
            else:
                sw0 = True
        else:
            sw0 = True
            
        
        
        switches0 = [sw0]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = np.array([1E-2,  1E-6, 1E-6])
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        n_phase = {"Gas" : np.zeros_like(t),
                   "Supercritical"  :np.zeros_like(t),
                   "Liquid" : np.zeros_like(t)}
        
        for i in range(0, len(t)):
            fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1])   
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
            n_phase[phase][i] = nfluid
            
        
        return SimResults(time = t, 
                          pressure = p0,
                          temperature = y[:,0],
                          moles_adsorbed = 0,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          vented_amount = y[:,1],
                          vented_energy = y[:,3],
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class OnePhaseFluidCooled(OnePhaseFluidSim):
    sim_type = "Cooled"
    def solve_differentials(self, time, T):
        p = self.simulation_params.init_pressure
        prop_dict = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        m11 = self._dn_dT(prop_dict)
        m12 = 0
        m21 = self._du_dT(prop_dict)
        m22 = 1
        
        A = np.array([[m11, m12],
                      [m21, m22]])
        
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(p, T)
            Tinput = flux.temperature_in(p,T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * fluid_props["hf"] + \
            self.boundary_flux.heating_power\
                + self.heat_leak_in(T)
                
        b = np.array([b1, b2])
        
        diffresults = np.linalg.solve(A, b)
        
        return np.append(diffresults, ndotin)
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p0 = self.simulation_params.init_pressure
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T, cooling, inserted = w
            return self.solve_differentials(t, T)
        
        def events(t, w, sw):
            ##Check saturation status
            if w[0] >= Tcrit:
                satstatus = p0 - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p0-satpres) > (1E-6 * satpres):
                    satstatus = p0 - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0] - self.simulation_params.target_temp])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                if solver.sw[0]:
                    print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
            if state_info[1] != 0:
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            
     
        w0 = np.array([self.simulation_params.init_temperature,
                       self.simulation_params.cooling_required,
                       self.simulation_params.inserted_amount])
        
        

        if self.simulation_params.init_temperature <= Tcrit:
            fluid.update(CP.QT_INPUTS, 0, self.init_temperature)
            psat_init =  fluid.p()
            if self.simulation_params.init_pressure == psat_init:
                sw0 = False
            else:
                sw0 = True
        else:
            sw0 = True
            
        
        
        switches0 = [sw0]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics Cooled at Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = np.array([1E-2,  1E-6, 1E-6])
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        n_phase = {"Gas" : np.zeros_like(t),
                   "Supercritical"  :np.zeros_like(t),
                   "Liquid" : np.zeros_like(t)}
        
        for i in range(0, len(t)):
            fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1])   
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
            n_phase[phase][i] = nfluid
            
        
        return SimResults(time = t, 
                          pressure = p0,
                          temperature = y[:,0],
                          moles_adsorbed = 0,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          cooling_required = y[:,1],
                          heating_required = self.simulation_params.heating_required,
                          vented_amount = self.simulation_params.vented_amount,
                          vented_energy = self.simulation_params.vented_energy,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
        
class OnePhaseFluidHeatedDischarge(OnePhaseFluidSim):
    sim_type = "Heated"
    def solve_differentials(self, time, T):
        p = self.simulation_params.init_pressure
        prop_dict = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        m11 = self._dn_dT(prop_dict)
        m12 = 0
        m21 = self._du_dT(prop_dict)
        m22 = -1
        
        A = np.array([[m11, m12],
                      [m21, m22]])
        
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid = self.storage_tank.stored_fluid.backend
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            Pinput = flux.pressure_in(p, T)
            Tinput = flux.temperature_in(p,T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0    
        
        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * fluid_props["hf"] + \
             - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T)
                
        b = np.array([b1, b2])
        
        diffresults = np.linalg.solve(A, b)
        
        return np.append(diffresults, ndotin)
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p0 = self.simulation_params.init_pressure
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T, heating, inserted = w
            return self.solve_differentials(t, T)
        
        def events(t, w, sw):
            ##Check saturation status
            if w[0] >= Tcrit:
                satstatus = p0 - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p0-satpres) > (1E-6 * satpres):
                    satstatus = p0 - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0] - self.simulation_params.target_temp])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                if solver.sw[0]:
                    print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
            if state_info[1] != 0:
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            
     
        w0 = np.array([self.simulation_params.init_temperature,
                       self.simulation_params.heating_required,
                       self.simulation_params.inserted_amount])
        
        

        if self.simulation_params.init_temperature <= Tcrit:
            fluid.update(CP.QT_INPUTS, 0, self.init_temperature)
            psat_init =  fluid.p()
            if self.simulation_params.init_pressure == psat_init:
                sw0 = False
            else:
                sw0 = True
        else:
            sw0 = True
            
        
        
        switches0 = [sw0]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics Cooled at Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = np.array([1E-2,  1E-6, 1E-6])
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        n_phase = {"Gas" : np.zeros_like(t),
                   "Supercritical"  :np.zeros_like(t),
                   "Liquid" : np.zeros_like(t)}
        
        for i in range(0, len(t)):
            fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1])   
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
            n_phase[phase][i] = nfluid
            
        
        return SimResults(time = t, 
                          pressure = p0,
                          temperature = y[:,0],
                          moles_adsorbed = 0,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = y[:,1],
                          vented_amount = self.simulation_params.vented_amount,
                          vented_energy = self.simulation_params.vented_energy,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class OnePhaseFluidControlledInlet(OnePhaseFluidDefault):
    def solve_differentials(self, time, p, T):
        prop_dict = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        
        m11 = self._dn_dp(prop_dict)
        m12 = self._dn_dT(prop_dict)
        m21 = self._du_dp(prop_dict)
        m22 = self._du_dp(prop_dict)
        
        A = np.array([[m11 , m12],
                      [m21, m22]])
        
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW

        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            hin = self.boundary_flux.enthalpy_in(time)
        else:
            hin = 0    
        
        b1 = ndotin 
        b2 = ndotin * hin + \
            self.boundary_flux.heating_power - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T)
                
        b = np.array([b1, b2])
        
        soln = np.linalg.solve(A, b)
        
        return np.append(soln, ndotin)
        
        
        
        