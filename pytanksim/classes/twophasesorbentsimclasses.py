# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:00:50 2023

@author: nextf
"""

__all__ = ["TwoPhaseSorbentSim", "TwoPhaseSorbentDefault", "TwoPhaseSorbentVenting",
           "TwoPhaseSorbentCooled", "TwoPhaseSorbentControlledInlet", "TwoPhaseSorbentHeatedDischarge"]

import CoolProp as CP
import numpy as np
import pytanksim.utils.finitedifferences as fd
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation

class TwoPhaseSorbentSim(BaseSimulation):
    sim_phase = "Two Phase"
    
    #Generate the elements of the matrix representing the governing eqs.
    #First, from mass balance
    
    def _dn_dT(self, T, saturation_properties):
        p = saturation_properties["psat"]
        dps_dT = saturation_properties["dps_dT"]
        sorbent = self.storage_tank.sorbent_material
        isotherm = self.storage_tank.sorbent_material.model_isotherm
        term1 = fd.partial_derivative(isotherm.n_absolute, 0, [p, T], p*1E-5) *\
            dps_dT
        term2 = fd.partial_derivative(isotherm.n_absolute, 1, [p, T], T*1E-5)
        return sorbent.mass * (term1 + term2)

    ##Then, from the volume change
    
    def _dv_dn(self, saturation_properties):
        return 1/saturation_properties["rhof"]
    
    def _dv_dT(self, ng, nl, T, saturation_properties_gas, saturation_properties_liquid):
        sorbent = self.storage_tank.sorbent_material
        p = saturation_properties_gas["psat"]
        term = np.zeros(4)
        dps_dT = saturation_properties_gas["dps_dT"]
        drhog_dT, rhog, drhog_dp = map(saturation_properties_gas.get, ("drho_dT", "rhof", "drho_dp"))
        drhol_dT, rhol, drhol_dp = map(saturation_properties_liquid.get, ("drho_dT", "rhof", "drho_dp"))
        term[0] = sorbent.mass * \
            fd.partial_derivative(self.storage_tank.sorbent_material.model_isotherm.v_ads, 
                                        0, [p, T], p * 1E-5) * dps_dT
        term[1] = sorbent.mass * \
            fd.partial_derivative(self.storage_tank.sorbent_material.model_isotherm.v_ads, 
                                        1, [p, T], T * 1E-5)
        term[3] = (-ng/(rhog**2)) * (drhog_dp * dps_dT + drhog_dT)
        term[4] = (-nl/(rhol**2)) * (drhol_dp * dps_dT + drhol_dT)
        return sum(term)
    
    ##Finally, from the energy balance
    
    def _du_dn(self, saturation_properties):
        return saturation_properties["uf"]
    
    def _du_dT(self, ng, nl, T, saturation_properties_gas, saturation_properties_liquid):
        dps_dT = saturation_properties_gas["dps_dT"]
        p = saturation_properties_gas["psat"]
        sorbent = self.storage_tank.sorbent_material
        isotherm = self.storage_tank.sorbent_material.model_isotherm
        dps_dT = saturation_properties_gas["dps_dT"]
        dug_dT, ug, dug_dp = map(saturation_properties_gas.get, ("du_dT", "uf", "du_dp"))
        dul_dT, ul, dul_dp = map(saturation_properties_liquid.get, ("du_dT", "uf", "du_dp"))
        term = np.zeros(4)
        term[0] = sorbent.mass * isotherm.differential_energy(p, T) * \
            (fd.partial_derivative(isotherm.n_absolute, 0, [p, T], p*1E-5) * dps_dT + \
             fd.partial_derivative(isotherm.n_absolute, 1, [p, T], T*1E-5))
        term[1] = sorbent.mass * isotherm.n_absolute(p, T) * \
            fd.partial_derivative(isotherm.internal_energy_adsorbed, 1, [p, T], T*1E-5)
        term[2] = ng * (dug_dT + dug_dp * dps_dT) 
        term[3] = nl * (dul_dT + dul_dp * dps_dT)
        return sum(term)
    
class TwoPhaseSorbentDefault(TwoPhaseSorbentSim):
    sim_type = "Default"
    
    def solve_differentials(self, ng, nl, T, time):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        
        m11 = 1
        m12 = 1
        m13 = self._dn_dT(T, satur_prop_gas)
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = self._dv_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        m31 = self._du_dn(satur_prop_gas)
        m32 = self._du_dn(satur_prop_liquid)
        m33 = self._du_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        A = np.matrix([[m11, m12, m13],
                       [m21, m22, m23],
                       [m31, m32, m33]])
        
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        p = satur_prop_gas["psat"]
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        ##Get the input pressure at a condition
        if flux.mass_flow_in:
            Pinput = flux.pressure_in(p, T)
            Tinput = flux.temperature_in(p,T)
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
        return np.append(diffresults,
                         [ndotin,
                         ndotin * hin,
                         ndotout,
                         ndotout * satur_prop_gas["hf"],
                         self.boundary_flux.cooling_power(time),
                         self.boundary_flux.heating_power(time),
                         self.heat_leak_in(T) ]
                         )
        
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()

        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            ng, nl, T = w[:3]
            return self.solve_differentials(ng, nl, T, t)
        
        def events(t, w, sw):
            ##Check for boundary events
            ##First check criticality of the fluid
            T = w[2]
            crit = T-Tcrit
            ##Then check the minimum and maximum pressure of the tank
            fluid.update(CP.QT_INPUTS, 0, T)
            p = fluid.p()
            min_pres_event = p - self.storage_tank.min_supply_pressure
            max_pres_event = self.storage_tank.vent_pressure - p
            
            ##Check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            
            ##Check that the target conditions has not been reached
            target_pres_reach = p - self.simulation_params.target_pres
            target_temp_reach = T - self.simulation_params.target_temp
            return np.array([crit, min_pres_event, max_pres_event,
                             sat_gas_event, sat_liquid_event,
                             target_pres_reach, target_temp_reach])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                if solver.sw[0]:
                    print("\n The simulation has reached critical temperature, \n please switch to one phase simulation.")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
            
            if state_info[1] != 0:
                if solver.sw[1]:
                    print("\n The simulation has hit the minimum supply pressure. \n Switch to heated discharge.")
                    raise TerminateSimulation
                solver.sw[1] = not solver.sw[1]
                
            if state_info[2] != 0:
                if solver.sw[2]:
                    print("\n The simulation has hit maximum pressure! Switch to cooling or venting simulation")
                    raise TerminateSimulation
                solver.sw[2] = not solver.sw[2]
                
            if state_info[3] != 0 or state_info[4] != 0:
                if solver.sw[3]:
                    print("\n Phase change has ended. Switch to one phase simulation.")
                    raise TerminateSimulation
                solver.sw[3] = not solver.sw[3]
            
            if state_info[5] != 0 and state_info[6] != 0:
                print("\n Target conditions has been reached.")
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
        
        if self.simulation_params.init_temperature == Tcrit:
            sw0 = False
        else:
            sw0 = True
        
        fluid.update(CP.QT_INPUTS, 0, 1)
        psat_init = fluid.p()
        
        if self.storage_tank.min_supply_pressure == psat_init:
            sw1 = False
        else:
            sw1 = True
        
        if self.storage_tank.vent_pressure == psat_init:
            sw2 = False
        else:
            sw2 = True
        
        if self.simulation_params.init_nl == 0 or self.simulation_params.init_ng == 0:
            sw3 = False
        else:
            sw3 = True
            
        switches0 = [sw0, sw1, sw2, sw3]
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
        nads = np.zeros_like(t)
        
        for i in range(0, len(t)):
            fluid.update(CP.QT_INPUTS, 0, y[i, 2])
            pres[i] = fluid.p()
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(pres[i], y[i, 2]) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = pres,
                          temperature = y[:,2],
                          moles_adsorbed = nads,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical= 0,
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
    
class TwoPhaseSorbentCooled(TwoPhaseSorbentSim):
    sim_type = "Cooled"
    
    def _dn_dt(self, time):
        T = self.simulation_params.init_temperature
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)    
        m11 = 1
        m12 = 1
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        b1 = ndotin - ndotout
        b2 = 0
        A = np.array([[m11, m12],
                     [m21, m22]])
        b = np.array([b1, b2])
        return np.linalg.solve(A, b)
    
    def _cooling_power(self, time, dng_dt, dnl_dt):
        T = self.simulation_params.init_temperature
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        m31 = self._du_dn(satur_prop_gas)
        m32 = self._du_dn(satur_prop_liquid)
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time)  / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        Pinput =self.boundary_flux.pressure_in(self.simulation_params.init_pressure, self.simulation_params.init_temperature)
        Tinput = self.boundary_flux.temperature_in(self.simulation_params.init_pressure,self.simulation_params.init_temperature)
        ##Get the molar enthalpy of the inlet fluid
        fluid.update(CP.PT_INPUTS, Pinput, Tinput)
        hin = fluid.hmolar()
        return - dng_dt * m31 - dnl_dt * m32 + ndotin * hin - ndotout * satur_prop_gas["hf"] + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T) 
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self._dn_dt(t)
            cool_power = self._cooling_power(t, diffresults[0], diffresults[1])
            flux = self.boundary_flux
            MW = fluid.molar_mass()
            ndotin = flux.mass_flow_in(t)  / MW
            ndotout = flux.mass_flow_out(t) / MW
            Pinput =self.boundary_flux.pressure_in(self.simulation_params.init_pressure, self.simulation_params.init_temperature)
            Tinput = self.boundary_flux.temperature_in(self.simulation_params.init_pressure,self.simulation_params.init_temperature)
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
            fluid.update(CP.QT_INPUTS, 1, self.simulation_params.init_temperature)
            hout = fluid.hmolar()
            return np.append(diffresults,
                             [cool_power,
                              ndotin,
                              ndotin * hin,
                              ndotout,
                              ndotout * hout,
                              self.boundary_flux.cooling_power(t),
                              self.boundary_flux.heating_power(t),
                              self.heat_leak_in(self.simulation_params.init_temperature)]
                             )
        
        def events(t, w, sw):
            #check that either phase has not fully saturated
            sat_liquid_event = w[0]  
            sat_gas_event = w[1]
            return np.array([sat_gas_event, sat_liquid_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
        
            if state_info[0] != 0 or state_info[1] != 0:
                if solver.sw[0]:
                    print("\n Phase change has ended. Switch to one phase simulation.")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
                

        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.cooling_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in]) 
        
        if self.simulation_params.init_nl == 0 or self.simulation_params.init_ng == 0:
            sw0 = False
        else:
            sw0 = True
            
        switches0 = [sw0]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics Cooled with Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
            self.simulation_params.init_pressure, self.simulation_params.init_temperature) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = nads,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical= 0,
                          cooling_required = y[:, 2],
                          inserted_amount = y[:, 3],
                          flow_energy_in = y[:,4],
                          vented_amount = y[:,5],
                          vented_energy = y[:,6],
                          cooling_additional = y[:,7],
                          heating_additional = y[:,8],
                          heat_leak_in = y[:,9],
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseSorbentVenting(TwoPhaseSorbentSim):
    sim_type = "Venting"
    
    def solve_differentials(self, time):
        T = self.simulation_params.init_temperature
        fluid = self.storage_tank.stored_fluid.backend
        fluid.update(CP.QT_INPUTS, 0, T)
        psat = fluid.p()
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        m11 = 1
        m12 = 1 
        m13 = 1
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = 0
        m31 = self._du_dn(satur_prop_gas)
        m32 = self._du_dn(satur_prop_liquid)
        m33 = satur_prop_gas["hf"]
        
        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
        
        
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        
        if flux.mass_flow_in:
            Pinput = flux.pressure_in(psat, T)
            Tinput = flux.temperature_in(psat,T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0
        
        b1 = ndotin
        b2 = 0
        b3 = ndotin * hin + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T) 
        
        b = np.array([b1,b2,b3])
        
        soln = np.linalg.solve(A, b)
        
        return np.append(soln, [
            soln[-1] * satur_prop_gas["hf"],
            ndotin,
            ndotin * hin,
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
                if solver.sw[0]:
                    print("\n Phase change has ended. Switch to one phase simulation.")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
                
            
     
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
        
        
        if self.simulation_params.init_nl == 0 or self.simulation_params.init_ng == 0:
            sw0 = False
        else:
            sw0 = True
            
        switches0 = [sw0]
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

        
        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
           self.simulation_params.init_pressure, self.simulation_params.init_temperature) *\
               self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = nads,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical = 0,
                          vented_amount = y[:,2],
                          vented_energy = y[:, 3],
                          inserted_amount = y[:, 4],
                          flow_energy_in = y[:,5],
                          cooling_additional = y[:,6],
                          heating_additional = y[:,7],
                          heat_leak_in = y[:,8],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class TwoPhaseSorbentHeatedDischarge(TwoPhaseSorbentSim):
    sim_type = "Heated"
    
    def _dn_dt(self, time):
        T = self.simulation_params.init_temperature
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)    
        m11 = 1
        m12 = 1
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        b1 = ndotin - ndotout
        b2 = 0
        A = np.array([[m11, m12],
                     [m21, m22]])
        b = np.array([b1, b2])
        return np.linalg.solve(A, b)
    
    def _heating_power(self, time, dng_dt, dnl_dt):
        T = self.simulation_params.init_temperature
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        m31 = self._du_dn(satur_prop_gas)
        m32 = self._du_dn(satur_prop_liquid)
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time)  / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        Pinput =self.boundary_flux.pressure_in(self.simulation_params.init_pressure, self.simulation_params.init_temperature)
        Tinput = self.boundary_flux.temperature_in(self.simulation_params.init_pressure,self.simulation_params.init_temperature)
        ##Get the molar enthalpy of the inlet fluid
        fluid.update(CP.PT_INPUTS, Pinput, Tinput)
        hin = fluid.hmolar()
        return dng_dt * m31 + dnl_dt * m32 - ndotin * hin + ndotout * satur_prop_gas["hf"] + \
           - self.boundary_flux.heating_power(time) + self.boundary_flux.cooling_power(time) - self.heat_leak_in(T) 
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self._dn_dt(t)
            heating_power = self._heating_power(t, diffresults[0], diffresults[1])
            flux = self.boundary_flux
            MW = fluid.molar_mass()
            ndotin = flux.mass_flow_in(t)  / MW
            ndotout = flux.mass_flow_out(t) / MW
            Pinput =self.boundary_flux.pressure_in(self.simulation_params.init_pressure, self.simulation_params.init_temperature)
            Tinput = self.boundary_flux.temperature_in(self.simulation_params.init_pressure,self.simulation_params.init_temperature)
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
            fluid.update(CP.QT_INPUTS, 1, self.simulation_params.init_temperature)
            hout = fluid.hmolar()
            return np.append(diffresults, [
                heating_power,
                ndotin,
                ndotin * hin,
                ndotout,
                ndotout * hout,
                self.boundary_flux.cooling_power(t),
                self.boundary_flux.heating_power(t),
                self.heat_leak_in(self.simulation_params.init_temperature)
                ])
        
        def events(t, w, sw):
            #check that either phase has not fully saturated
            sat_liquid_event = w[0] 
            sat_gas_event = w[1]
            return np.array([sat_gas_event, sat_liquid_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
        
            if state_info[0] != 0 or state_info[1] != 0:
                if solver.sw[0]:
                    print("\n Phase change has ended. Switch to one phase simulation.")
                    raise TerminateSimulation
                solver.sw[0] = not solver.sw[0]
                

        w0 = np.array([self.simulation_params.init_ng,
                       self.simulation_params.init_nl,
                       self.simulation_params.heating_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in]) 
        
        if self.simulation_params.init_nl == 0 or self.simulation_params.init_ng == 0:
            sw0 = False
        else:
            sw0 = True
            
        switches0 = [sw0]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics Heated with Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        
        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
            self.simulation_params.init_pressure, self.simulation_params.init_temperature) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = self.simulation_params.init_pressure,
                          temperature = self.simulation_params.init_temperature,
                          moles_adsorbed = nads,
                          moles_gas = y[:, 0], 
                          moles_liquid = y[:, 1],
                          moles_supercritical= 0,
                          heating_required = y[:, 2],
                          inserted_amount = y[:, 3],
                          flow_energy_in = y[:,4],
                          vented_amount = y[:,5],
                          vented_energy = y[:,6],
                          cooling_additional = y[:,7],
                          heating_additional = y[:,8],
                          heat_leak_in = y[:,9],
                          cooling_required = self.simulation_params.cooling_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)

class TwoPhaseSorbentControlledInlet(TwoPhaseSorbentDefault):
    def solve_differentials(self, ng, nl, T, time):
        satur_prop_gas = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid =  self.storage_tank.stored_fluid.saturation_property_dict(T, 0)
        
        m11 = 1
        m12 = 1
        m13 = self._dn_dT(T, satur_prop_gas)
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = self._dv_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        m31 = self._du_dn(satur_prop_gas)
        m32 = self._du_dn(satur_prop_liquid)
        m33 = self._du_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        A = np.matrix([[m11, m12, m13],
                       [m21, m22, m23],
                       [m31, m32, m33]])
        
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        ##Get the input pressure at a condition
        if ndotin != 0:
            hin = self.boundary_flux.enthalpy_in(time)
        else:
            hin = 0
            
        if ndotout != 0:
            hout = self.boundary_flux.enthalpy_out(time)
        else:
            hout = 0
        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout\
             + self.boundary_flux.heating_power - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T) 
        b = np.array([b1,b2,b3])
        diffresults = np.linalg.solve(A, b)
        
        return np.append(diffresults,
                         [ndotin,
                         ndotin * hin,
                         ndotout,
                         ndotout * hout,
                         self.boundary_flux.cooling_power(time),
                         self.boundary_flux.heating_power(time),
                         self.heat_leak_in(T) ]
                         )
    
    
    
    
        