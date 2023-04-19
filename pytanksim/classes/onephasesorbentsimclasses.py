# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:11:33 2023

@author: nextf
"""

__all__ = ["OnePhaseSorbentSim", "OnePhaseSorbentDefault", "OnePhaseSorbentVenting",
           "OnePhaseSorbentCooled", "OnePhaseSorbentControlledInlet", "OnePhaseSorbentHeatedDischarge"]

import CoolProp as CP
import numpy as np
import pytanksim.utils.finitedifferences as fd
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation

class OnePhaseSorbentSim(BaseSimulation):
    sim_phase = "One Phase"
    def _dn_dp(self, p, T, fluid_properties):
        drho_dp = fluid_properties["drho_dp"]
        term1 = drho_dp * self.storage_tank.bulk_fluid_volume(p, T)
        term2 = fd.partial_derivative(self.storage_tank.sorbent_material.model_isotherm.n_absolute, \
                                      0, [p, T], p * 1E-5) * \
            self.storage_tank.sorbent_material.mass
        return term1 + term2
    
    def _dn_dT(self, p, T, fluid_properties):
        rhof, drho_dT = map(fluid_properties.get, ("rhof", "drho_dT"))
        term = np.zeros(3)
        term[0] =  drho_dT * self.storage_tank.bulk_fluid_volume(p, T)
        term[1] = - rhof * fd.partial_derivative(self.storage_tank.sorbent_material.model_isotherm.v_ads,
                                     1, [p,T], T*1E-5) * self.storage_tank.sorbent_material.mass
        term[2] = fd.partial_derivative(self.storage_tank.sorbent_material.model_isotherm.n_absolute, 1, 
                                        [p,T], T*1E-5) * self.storage_tank.sorbent_material.mass
        return sum(term)

    def _dU_dp(self, p, T, fluid_properties):
        tank = self.storage_tank
        sorbent = self.storage_tank.sorbent_material
        drho_dp, du_dp, rhof, uf = map(fluid_properties.get, ("drho_dp", "du_dp", "rhof", "uf") )
        term = np.zeros(4)
        
        term[0] = drho_dp * tank.bulk_fluid_volume(p, T) * uf 
        term[1] = rhof * tank.bulk_fluid_volume(p, T) * du_dp
        term[2] = rhof * fd.partial_derivative(tank.bulk_fluid_volume, 0, [p, T], p*1E-5) * uf
        term[3] = sorbent.mass * fd.partial_derivative(sorbent.model_isotherm.n_absolute,
                                                          0, [p, T], p * 1E-5) *\
                sorbent.model_isotherm.differential_energy(p, T) 
        return sum(term)

    def _dU_dT(self, p, T, fluid_properties):
        du_dT, drho_dT, rhof, uf = map(fluid_properties.get, ("du_dT", "drho_dT", "rhof", "uf"))
        tank = self.storage_tank
        sorbent = self.storage_tank.sorbent_material
        term = np.zeros(6)
        term[0] = drho_dT * tank.bulk_fluid_volume(p, T) * uf 
        term[1] = rhof * tank.bulk_fluid_volume(p, T) * du_dT
        term[2] = rhof * fd.partial_derivative(tank.bulk_fluid_volume, 1, [p, T], T*1E-5) * uf
        term[3] = sorbent.mass * fd.partial_derivative(sorbent.model_isotherm.n_absolute,
                                                          1, [p, T], T * 1E-5) *\
             sorbent.model_isotherm.differential_energy(p, T)

        term[4] = fd.partial_derivative(sorbent.model_isotherm.internal_energy_adsorbed, 1, [p, T], T*1E-5) *\
            sorbent.mass * sorbent.model_isotherm.n_absolute(p, T)
        term[5] = tank.heat_capacity(T)
        return sum(term)


class OnePhaseSorbentDefault(OnePhaseSorbentSim):
    sim_type = "Default"
    
    def _dT_dt(self, p, T, time):
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW

        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            hin = self.enthalpy_in_calc(p, T)
        else:
            hin = 0    
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        
        k1 = ndotin - ndotout
        k2 = ndotin * hin - ndotout * fluid_props["hf"] + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)
        #print(hin, hgas)
        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        c = self._dU_dp(p, T, fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        #Put in the right hand side of the mass and energy balance equations
        return (k2 * a - c * k1)/(d*a - b*c)
    
    def _dP_dt(self, p, T, dTdt, time):
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW 
        k1 = ndotin - ndotout
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        return (k1 - b * dTdt)/a
    
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
            p, T, inserted, ene_in, cooling, heating, leak_in, vented, vent_ene = w
            phase = self.storage_tank.stored_fluid.determine_phase(p, T)
            dTdt = self._dT_dt(p, T, t) if phase != "Saturated" else 0
            dPdt = self._dP_dt(p, T, dTdt, t) if phase != "Saturated" else 0
            fluid = self.storage_tank.stored_fluid.backend
            MW = fluid.molar_mass() 
            ##Convert kg/s to mol/s
            flux = self.boundary_flux
            ndotin = flux.mass_flow_in(t)  / MW
            ndotout = flux.mass_flow_out(t) / MW
            ##Get the thermodynamic properties of the bulk fluid for later calculations
            if phase != "Saturated":
                fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
            else:
                fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
            ##Get the input pressure at a condition
            if flux.mass_flow_in(t) != 0:
                hin = self.enthalpy_in_calc(p, T)
            else:
                hin = 0
            output = np.array([dPdt, dTdt, ndotin, ndotin * hin,
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t), self.heat_leak_in(T),
                             ndotout, ndotout * fluid_props["hf"]])
            return output
        
        def events(t, w, sw):
            ##Check saturation status
            if w[1] > Tcrit:
                satstatus = w[0] - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[1])
                satpres = fluid.p()
                if np.abs(w[0]-satpres) > (5E-6 * satpres):
                    satstatus = w[0] - satpres
                else:
                    satstatus = 0
            
            
            return np.array([self.storage_tank.vent_pressure - w[0], 
                             satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[1] - self.simulation_params.target_temp,
                             w[0] - self.simulation_params.target_pres])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                print("\n The simulation has hit maximum pressure! Switch to venting or cooling simulation")
                raise TerminateSimulation
                
            if state_info[1] != 0 and solver.y[1] <= Tcrit:
                print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                raise TerminateSimulation
                
            if state_info[2] != 0:
                print("\n The simulation has hit minimum supply pressure! Switch to heated discharge simulation")
                raise TerminateSimulation
                
            if state_info[3] != 0 and solver.sw[0]:
                print("\n Target temperature reached")
                raise TerminateSimulation
                
            if state_info[4] != 0 and solver.sw[1]:
                print("\n Target pressure reached")
                raise TerminateSimulation
            
            if state_info[3] != 0 and state_info[4] != 0:
                print("\n The simulation target condition has been reached.")
                raise TerminateSimulation
        
            
            
     
        w0 = np.array([self.simulation_params.init_pressure,
                       self.simulation_params.init_temperature,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy
                       ])
        
        
        ##Initialize switches for event handling

        
        sw0 = self.simulation_params.stop_at_target_temp
        sw1 = self.simulation_params.stop_at_target_pressure
        
        switches0 = [sw0, sw1]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.atol = [100, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2]
        sim.rtol = 1E-2
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
            
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                        iterable = iterable - 1
                        phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
                q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i , 1])
            else:
                fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1]) 
                    
            n_phase[phase][i] = nfluid
            
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(y[i,0], y[i,1]) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = y[:,0],
                          temperature = y[:,1],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          flow_energy_in = y[:,3],
                          cooling_required = self.simulation_params.cooling_required,
                          cooling_additional = y[:, 4],
                          heating_required = self.simulation_params.heating_required,
                          heating_additional = y[:, 5],
                          heat_leak_in = y[:, 6],
                          vented_amount = y[:, 7],
                          vented_energy = y[:, 8],
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class OnePhaseSorbentVenting(OnePhaseSorbentSim):
    sim_type = "Venting"
    
    def _dT_dt_vent_vary_fillrate(self, T, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        if self.boundary_flux.mass_flow_in != 0:
            Pinput = self.boundary_flux.pressure_in(p, T)
            Tinput = self.boundary_flux.temperature_in(p, T)
            ##Get the molar enthalpy of the inlet fluid
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
        else:
            hin = 0
        
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        b = self._dn_dT(p, T, fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        hf = fluid_props["hf"]
        return(ndotin*(hin-hf)+self.boundary_flux.heating_power(time) -\
               self.boundary_flux.cooling_power(time) + self.heat_leak_in(T)) \
            /(d-(hf*b))  
        
    def _ndotout_vent_vary_fillrate(self, T, dTdt, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid.update(CP.PT_INPUTS, p, T)
        fluid_props =self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        b = self._dn_dT(p, T, fluid_props)
        return ndotin - b*dTdt
    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        fluid = self.storage_tank.stored_fluid.backend
        state = [0, self.simulation_params.final_time/1000]
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        
        p0 = self.simulation_params.init_pressure
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            
            dTdt =  self._dT_dt_vent_vary_fillrate(T, t) if phase != "Saturated" else 0
            ndotout = self._ndotout_vent_vary_fillrate(T, dTdt, t) if phase != "Saturated" else 0
            MW = self.storage_tank.stored_fluid.backend.molar_mass()
            ndotin = self.boundary_flux.mass_flow_in(t) / MW
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, p0, T)
            else:
                fluid.update(CP.QT_INPUTS, 0, T)
            hf = fluid.hmolar()
            p = self.simulation_params.init_pressure
            if self.boundary_flux.mass_flow_in != 0:
                Pinput = self.boundary_flux.pressure_in(p, T)
                Tinput = self.boundary_flux.temperature_in(p, T)
                ##Get the molar enthalpy of the inlet fluid
                fluid.update(CP.PT_INPUTS, Pinput, Tinput)
                hin = fluid.hmolar()
            else:
                hin = 0
            return np.array([dTdt, ndotout, ndotout * hf,
                             ndotin, ndotin * hin, 
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t),
                             self.boundary_flux.heat_leak_in(T)])
    
        def events(t, w, sw):
            if w[0] >= Tcrit:
                satstatus = p0 - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p0-satpres) > (1E-6 * satpres):
                    satstatus = p0 - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0]-self.simulation_params.target_temp])
    
        def handle_event(solver, event_info):
            if event_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation
                
            if event_info[1] != 0:
                print("\n Final refueling condition achieved, exiting simulation.")
                raise TerminateSimulation
    
        w0 = np.array([self.simulation_params.init_temperature,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in])
        
        
        switches0 = []
        
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Venting Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, 
                             self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        print("Saving results...")
        nads = np.zeros_like(t)
        n_phase = {
            "Gas" : np.zeros_like(t),
            "Liquid" : np.zeros_like(t),
            "Supercritical" : np.zeros_like(t)
            }
        
        for i in range(len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p0, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                        iterable = iterable - 1
                        phase = self.storage_tank.stored_fluid.determine_phase(p0, y[iterable,0])
                q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i , 0])
            else:
                fluid.update(CP.PT_INPUTS, p0, y[i,0])
            nfluid = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(p0, y[i,0]) 
                    
            n_phase[phase][i] = nfluid
            nads[i] =self.storage_tank.sorbent_material.model_isotherm.n_absolute(p0, y[i,0])\
                * self.storage_tank.sorbent_material.mass
            
        
        return SimResults(time = t, 
                          pressure = p0,
                          temperature = y[:,0],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical= n_phase["Supercritical"],
                          vented_amount = y[:, 1],
                          vented_energy = y[:, 2],
                          inserted_amount = y[:, 3],
                          flow_energy_in = y[:,4],
                          cooling_additional = y[:, 5],
                          heating_additional = y[:, 6],
                          heat_leak_in = y[:,7],
                          cooling_required = self.simulation_params.cooling_required,
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)

    
class OnePhaseSorbentCooled(OnePhaseSorbentSim):
    sim_type = "Cooled"
    def _dT_dt_cooled_const_pres(self, T, time):
        p = self.simulation_params.init_pressure
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        b = self._dn_dT(p, T, fluid_props)
        return (ndotin-ndotout)/b

    def _cooling_power_const_pres(self, T, dTdt, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        Pinput = self.boundary_flux.pressure_in(p, T)
        Tinput =self.boundary_flux.temperature_in(p, T)
        fluid.update(CP.PT_INPUTS, Pinput, Tinput)
        hin = fluid.hmolar()
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        hout = fluid_props["hf"]
        d = self._dU_dT(p, T, fluid_props)
        return - d * dTdt + ndotin * hin - ndotout * hout + self.heat_leak_in(T)\
            +self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)

    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p = self.simulation_params.init_pressure
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            phase = self.storage_tank.stored_fluid.determine_phase(p, T)
            dTdt =  self._dT_dt_cooled_const_pres(T, t) if phase != "Saturated" else 0
            cooling = self._cooling_power_const_pres(T, dTdt, t) if phase != "Saturated" else 0
            MW = self.storage_tank.stored_fluid.backend.molar_mass()
            Pinput = self.boundary_flux.pressure_in(p, T)
            Tinput =self.boundary_flux.temperature_in(p, T)
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
            
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T) \
                if phase != "Saturated" else self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
            ndotin =  self.boundary_flux.mass_flow_in(t)/MW
            ndotout = self.boundary_flux.mass_flow_out(t)/MW
            return np.array([dTdt, cooling, 
                             ndotin, ndotin * hin,
                             ndotout,
                             ndotout * fluid_props["hf"],
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t),
                             self.heat_leak_in(T),
                            ])
    
        def events(t, w, sw):
            if w[0] >= Tcrit:
                satstatus = p - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p-satpres) > (1E-6 * satpres):
                    satstatus = p - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0]-self.simulation_params.target_temp])
    
        def handle_event(solver, event_info):
            if event_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation

            if event_info[1] !=0 and p == self.simulation_params.target_pres:
                print("\n Final refueling temperature achieved, exiting simulation.")
                raise TerminateSimulation
        
        w0 = np.array([self.simulation_params.init_temperature,
                       self.simulation_params.cooling_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in])
        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase dynamics of constant P refuel w/ Cooling"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, 
                             self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        nads = np.zeros_like(t)
        n_phase = {
            "Gas": np.zeros_like(t),
            "Liquid" : np.zeros_like(t),
            "Supercritical" : np.zeros_like(t)
            } 
        for i in range(0, len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                        iterable = iterable - 1
                        phase = self.storage_tank.stored_fluid.determine_phase(p, y[iterable,0])
                q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i , 0])
            else:
                fluid.update(CP.PT_INPUTS, p, y[i,0])
            n_phase[phase][i] = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(p, y[i,0]) 
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(p, y[i,0]) *\
            self.storage_tank.sorbent_material.mass
        
        return SimResults(time = t, 
                          pressure = np.repeat(p, len(t)),
                          temperature = y[:,0],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical= n_phase["Supercritical"],
                          cooling_required  = y[:,1],
                          inserted_amount = y[:,2],
                          flow_energy_in = y[:,3],
                          vented_amount = y[:,4],
                          vented_energy = y[:,5],
                          cooling_additional = y[:,6],
                          heating_additional = y[:,7],
                          heat_leak_in = y[:,8],
                          heating_required = self.simulation_params.heating_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)

class OnePhaseSorbentControlledInlet(OnePhaseSorbentSim):
    sim_type = "Controlled Inlet"
    def _dT_dt(self, p, T, time):
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        ##Get the input pressure at a condition
        if flux.mass_flow_in(time) != 0:
            hin = self.boundary_flux.enthalpy_in(time)
        else:
            hin = 0
        
        if flux.mass_flow_out(time) != 0:
            hout = self.boundary_flux.enthalpy_out(time)
        else:
            hout = 0
        
        k1 = ndotin - ndotout
        k2 = ndotin * hin - ndotout * hout + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)
        #print(hin, hgas)
        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        c = self._dU_dp(p, T, fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        #Put in the right hand side of the mass and energy balance equations
        return (k2 * a - c * k1)/(d*a - b*c)
    
    def _dP_dt(self, p, T, dTdt, time):
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW 
        k1 = ndotin - ndotout
        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        return (k1 - b * dTdt)/a
    
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
            p, T = w[:2]
            phase = self.storage_tank.stored_fluid.determine_phase(p, T)
            dTdt = self._dT_dt(p, T, t) if phase != "Saturated" else 0
            dPdt = self._dP_dt(p, T, dTdt, t) if phase != "Saturated" else 0
            fluid = self.storage_tank.stored_fluid.backend
            MW = fluid.molar_mass() 
            ##Convert kg/s to mol/s
            flux = self.boundary_flux
            ndotin = flux.mass_flow_in(t)  / MW
            ndotout = flux.mass_flow_out(t) / MW
            ##Get the input pressure at a condition
            if flux.mass_flow_in(t) != 0:
                hin = self.boundary_flux.enthalpy_in(t)
            else:
                hin = 0
           
            if flux.mass_flow_out(t) != 0:
                hout = self.boundary_flux.enthalpy_out(t)
            else:
                hout = 0
            return np.array([dPdt, dTdt, ndotin, ndotin * hin,
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t), self.heat_leak_in(T),
                             ndotout, ndotout * hout])
        
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
            
            
            return np.array([self.storage_tank.vent_pressure - w[0], 
                             satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[1] - self.simulation_params.target_temp,
                             w[0] - self.simulation_params.target_pres])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                print("\n The simulation has hit maximum pressure! Switch to venting or cooling simulation")
                raise TerminateSimulation
                
            if state_info[1] != 0 and solver.y[1] <= Tcrit:
                print("\n The simulation has hit the saturation line! Switch to two-phase simulation")
                raise TerminateSimulation
                
            if state_info[2] != 0:
                print("\n The simulation has hit minimum supply pressure! Switch to heated discharge simulation")
                raise TerminateSimulation

            if state_info[3] != 0 and solver.sw[0]:
                print("\n Target temperature reached.")
                raise TerminateSimulation
                
            if state_info[4] != 0 and solver.sw[1]:
                print("\n Target pressure reached.")
                raise TerminateSimulation
            
            if state_info[3] != 0 and state_info[4] != 0:
                print("\n The simulation target condition has been reached.")
                raise TerminateSimulation
        
            
            
     
        w0 = np.array([self.simulation_params.init_pressure,
                       self.simulation_params.init_temperature,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy
                       ])
        
        
        ##Initialize switches for event handling
        sw0 = self.simulation_params.stop_at_target_temp
        sw1 = self.simulation_params.stop_at_target_pressure
        
        
        switches0 = [sw0, sw1]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-5
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
            phase = self.storage_tank.stored_fluid.determine_phase(y[i, 0], y[i, 1])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                        iterable = iterable - 1
                        phase = self.storage_tank.stored_fluid.determine_phase(y[iterable,0], y[iterable,1])
                q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i , 1])
            else:
                fluid.update(CP.PT_INPUTS, y[i,0], y[i,1])
            n_phase[phase][i] = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(y[i,0], y[i,1]) 
            
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(y[i,0], y[i,1]) *\
                self.storage_tank.sorbent_material.mass
            
        return SimResults(time = t, 
                          pressure = y[:,0],
                          temperature = y[:,1],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical = n_phase["Supercritical"],
                          inserted_amount = y[:, 2],
                          flow_energy_in = y[:,3],
                          cooling_required = self.simulation_params.cooling_required,
                          cooling_additional = y[:, 4],
                          heating_required = self.simulation_params.heating_required,
                          heating_additional = y[:, 5],
                          heat_leak_in = y[:, 6],
                          vented_amount = y[:, 7],
                          vented_energy = y[:, 8],
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
class OnePhaseSorbentHeatedDischarge(OnePhaseSorbentSim):
    sim_type = "Heated"
    def _dT_dt_heated_const_pres(self, T, time):
        p = self.simulation_params.init_pressure
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        b = self._dn_dT(p, T, fluid_props)
        return ndotin-ndotout/b

    def _heating_power_const_pres(self, T, dTdt, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotout= self.boundary_flux.mass_flow_out(time) / MW
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        Pinput = self.boundary_flux.pressure_in(p, T)
        Tinput =self.boundary_flux.temperature_in(p, T)
        fluid.update(CP.PT_INPUTS, Pinput, Tinput)
        hin = fluid.hmolar()
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        hout = fluid_props["hf"]
        d = self._dU_dT(p, T, fluid_props)
        return d * dTdt + ndotout * hout - ndotin * hin - self.heat_leak_in(T)\
             + self.boundary_flux.cooling_power(time) - self.boundary_flux.heating_power(time)

    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        pcrit = fluid.p_critical()
        p = self.simulation_params.init_pressure
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            phase = self.storage_tank.stored_fluid.determine_phase(p, T)
            dTdt =  self._dT_dt_heated_const_pres(T, t) if phase != "Saturated" else 0
            heating = self._heating_power_const_pres(T, dTdt, t) if phase != "Saturated" else 0
            MW = fluid.molar_mass()
            ndotout= self.boundary_flux.mass_flow_out(t) / MW
            ndotin = self.boundary_flux.mass_flow_in(t) / MW
            Pinput = self.boundary_flux.pressure_in(p, T)
            Tinput =self.boundary_flux.temperature_in(p, T)
            
            fluid.update(CP.PT_INPUTS, Pinput, Tinput)
            hin = fluid.hmolar()
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T) if phase != "Saturated" else \
                self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
            hout = fluid_props["hf"]
            return np.array([dTdt, heating, 
                             ndotin, ndotin * hin,
                             ndotout, ndotout * hout,
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t),
                             self.heat_leak_in(T)])
    
        def events(t, w, sw):
            if w[0] >= Tcrit:
                satstatus = p - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[0])
                satpres = fluid.p()
                if np.abs(p-satpres) > (1E-6 * satpres):
                    satstatus = p - satpres
                else:
                    satstatus = 0
            return np.array([satstatus, w[0]-self.simulation_params.target_temp])
    
        def handle_event(solver, event_info):
            if event_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation

            if event_info[1] !=0:
                print("\n Final temperature achieved, exiting simulation.")
                raise TerminateSimulation
        
        w0 = np.array([self.simulation_params.init_temperature, 
                       self.simulation_params.heating_required,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in])
        
        
        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time, sw0 = switches0 )
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase dynamics of constant P discharge w/ heating"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t,  y = sim.simulate(self.simulation_params.final_time, 
                             self.simulation_params.displayed_points)
        try:
           tqdm._instances.clear()
        except Exception:
           pass
        nads = np.zeros_like(t)
        n_phase = {
            "Gas": np.zeros_like(t),
            "Liquid" : np.zeros_like(t),
            "Supercritical" : np.zeros_like(t)
            } 
        for i in range(0, len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                        iterable = iterable - 1
                        phase = self.storage_tank.stored_fluid.determine_phase(p, y[iterable,0])
                q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i , 0])
            else:
                fluid.update(CP.PT_INPUTS, p, y[i,0])
            n_phase[phase][i] = fluid.rhomolar() * self.storage_tank.bulk_fluid_volume(p, y[i,0]) 
            nads[i] = self.storage_tank.sorbent_material.model_isotherm.n_absolute(p, y[i,0]) *\
            self.storage_tank.sorbent_material.mass
        return SimResults(time = t, 
                          pressure = np.repeat(p, len(t)),
                          temperature = y[:,0],
                          moles_adsorbed = nads,
                          moles_gas = n_phase["Gas"], 
                          moles_liquid = n_phase["Liquid"],
                          moles_supercritical= n_phase["Supercritical"],
                          heating_required = y[:,1],
                          inserted_amount = y[:,2],
                          flow_energy_in = y[:,3],
                          vented_amount = y[:,4],
                          vented_energy = y[:,5],
                          cooling_additional = y[:,6],
                          heating_additional = y[:,7],
                          heat_leak_in = y[:,7],
                          cooling_required = self.simulation_params.cooling_required,
                          sim_type= self.sim_type,
                          tank_params = self.storage_tank)
    
    
