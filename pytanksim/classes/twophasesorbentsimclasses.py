# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:00:50 2023

@author: nextf
"""

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
    
class TwoPhaseSorbentSimDefault(TwoPhaseSorbentSim):
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
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
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
        b3 = ndotin * hin - ndotout * fluid_props["hf"] + \
            self.boundary_flux.heating_power - self.boundary_flux.cooling_power\
                + self.heat_leak_in(T) 
        b = np.array([b1,b2,b3])
        return np.linalg.solve(A, b)
    
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
            ng, nl, T = w
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
            
            ##Finally check that either phase has not fully saturated
            sat_liquid_event = w[0] / (w[0] + w[1])  
            sat_gas_event = w[0] / (w[0] + w[1])
            return np.array([crit, min_pres_event, max_pres_event,
                             sat_gas_event, sat_liquid_event])
                        
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
                
            if state_info[1] != 0 and solver.y[1] <= Tcrit:
                print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                raise TerminateSimulation     
            if state_info[2] != 0:
                print("\n The simulation has hit minimum supply pressure! Switch to heated discharge simulation")
                raise TerminateSimulation
            
     
        w0 = np.array([self.simulation_params.init_pressure,
                       self.simulation_params.init_temperature])
        switches0 = [True, True, True]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = np.array([1E-2,  1E-6])
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        
        print("Saving results...")
        nads = np.zeros_like(t)
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
            
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(y[i,0], y[i,1]) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = y[:,0],
                          temperature = y[:,1],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical= n_phase["Supercritical"],
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
        