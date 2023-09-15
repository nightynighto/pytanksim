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
import numpy as np

class OnePhaseSorbentSim(BaseSimulation):
    sim_phase = "One Phase"
    
    def _derivfunc(self, func, var, point, stepsize):
        pT = point[:2]
        def phase_func(x):
            pT[var] = x
            return self.storage_tank.stored_fluid.determine_phase(pT[0], pT[1])
        
        x0 = point[var]
        x1 = x0 + stepsize
        x2 = x0 - stepsize
        phase1 = phase_func(x0)
        phase2 = phase_func(x1)
        phase3 = phase_func(x2)
        qinit = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.partial_derivative(func, var, point, stepsize)
        elif phase1 == "Saturated":
            if (qinit == 0 and var == 1) or (qinit == 1 and var == 0):
                return fd.backward_partial_derivative(func, var, point, stepsize)
            else:
                return fd.forward_partial_derivative(func, var, point, stepsize)
        else:
            if phase1 == phase3:
                return fd.backward_partial_derivative(func, var, point, stepsize)
            elif phase1 == phase2:
                return fd.forward_partial_derivative(func, var, point, stepsize)
    
    def _dn_dp(self, p, T, fluid_properties):
        drho_dp = fluid_properties["drho_dp"]
        rhof = fluid_properties["rhof"]
        deriver = self._derivfunc
        stepsize = 100
        qinit = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1

        
        # term1 = drho_dp * self.storage_tank.bulk_fluid_volume(p, T)
        # term2 = - rhof * deriver(self.storage_tank.sorbent_material.model_isotherm.v_ads,
        #                               0, [p,T], stepsize) * self.storage_tank.sorbent_material.mass
        # term3 = deriver(self.storage_tank.sorbent_material.model_isotherm.n_absolute, \
        #                               0, [p, T], stepsize) * \
        #     self.storage_tank.sorbent_material.mass
        # return term1 + term2 + term3
        return deriver(self.storage_tank.capacity, 0, [p, T, qinit], 100)
    
    def _dn_dT(self, p, T, fluid_properties):
        rhof, drho_dT = map(fluid_properties.get, ("rhof", "drho_dT"))
        deriver = self._derivfunc
        qinit = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1
        # term = np.zeros(3)
        # term[0] =  drho_dT * self.storage_tank.bulk_fluid_volume(p, T)
        # term[1] = - rhof * deriver(self.storage_tank.sorbent_material.model_isotherm.v_ads,
        #                               1, [p,T], 1E-2) * self.storage_tank.sorbent_material.mass
        # term[2] = deriver(self.storage_tank.sorbent_material.model_isotherm.n_absolute, 1, 
        #                                 [p,T], 1E-2) * self.storage_tank.sorbent_material.mass
        # return sum(term)
        return deriver(self.storage_tank.capacity, 1, [p, T, qinit], 1E-2)


    def _dU_dp(self, p, T, fluid_properties):
        q = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1       
        nbulk = self.storage_tank.capacity_bulk(p, T, q)
        tank = self.storage_tank
        sorbent = tank.sorbent_material
        u = fluid_properties["uf"]
        du_dp = fluid_properties["du_dp"]
        rho = fluid_properties["rhof"]
        drho_dp = fluid_properties["drho_dp"]
        deriver = self._derivfunc
        # term = np.zeros(5)
        
        # stepsize = 100
        
        # term[0] = nbulk * du_dp
        
        # term[1] = -deriver(sorbent.model_isotherm.v_ads, 0, [p, T], stepsize) * rho * u
        
        # term[2] = drho_dp * u * tank.bulk_fluid_volume(p, T)
        
        # term[3] = sorbent.mass * deriver(sorbent.model_isotherm.n_absolute,
        #                                                   0, [p, T], stepsize) *\
        #         sorbent.model_isotherm.differential_energy(p, T, q)
        # term[4] = sorbent.mass * sorbent.model_isotherm.n_absolute(p, T) * \
        #   deriver(sorbent.model_isotherm.internal_energy_adsorbed, 0, [p, T, q], stepsize)
        return deriver(tank.internal_energy, 0, [p, T, q], 100)
        # return sum(term)
    
    def _dU_dT(self, p, T, fluid_properties):
        # du_dT = fluid_properties["du_dT"]
        # u = fluid_properties["uf"]
        # rho = fluid_properties["rhof"]
        # drho_dT = fluid_properties["drho_dT"]
        tank = self.storage_tank
        # sorbent = self.storage_tank.sorbent_material
        deriver = self._derivfunc
        q = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1   
        # term = np.zeros(6)
        # stepsize = 1E-1
        # term[0] = rho * tank.bulk_fluid_volume(p, T) * du_dT
        
        # term[1] = - deriver(sorbent.model_isotherm.v_ads, 1, [p, T], stepsize) * rho * u
        
        # term[2] = drho_dT * u * tank.bulk_fluid_volume(p, T)
        
        # term[3] = sorbent.mass *  deriver(sorbent.model_isotherm.n_absolute,
        #                                                   1, [p, T], stepsize) *\
        #       (sorbent.model_isotherm.differential_energy(p, T, q))

        # term[4] = sorbent.mass * sorbent.model_isotherm.n_absolute(p, T) * \
        # deriver(sorbent.model_isotherm.internal_energy_adsorbed, 1, [p, T, q], stepsize)                                         
        # term[5] = tank.heat_capacity(T)
        return deriver(tank.internal_energy, 1, [p, T, q], 1E-2) + tank.heat_capacity(T)
        # return sum(term)


class OnePhaseSorbentDefault(OnePhaseSorbentSim):
    sim_type = "Default"
    
    def _solve_differentials(self, p, T, time, phase):
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
        qinit = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1
        if phase != "Saturated":
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        else:
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, qinit)
        k1 = ndotin - ndotout
        k2 = ndotin * (hin ) - ndotout * (fluid_props["hf"]) + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)
                
        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        c = self._dU_dp(p, T, fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        #Put in the right hand side of the mass and energy balance equations
        A = np.array([[a, b],
                     [c, d]])
        b = np.array([k1,k2])
        return np.linalg.solve(A, b)

    
    def run(self):
        pbar = tqdm(total=1000, unit = "‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        q = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            p, T = w[:2]
            phase = self.storage_tank.stored_fluid.determine_phase(p, T)
            dPdt, dTdt = self._solve_differentials(p, T, t, phase)
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
                fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, q)
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
            if w[1] > Tcrit - 0.01:
                satstatus = w[0] - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[1])
                satpres = fluid.p()
                if np.abs(w[0]-satpres) > (1E-6 * satpres):
                    satstatus = w[0] - satpres
                else:
                    satstatus = 0
            
            capacity_event = self.storage_tank.capacity(w[0], w[1], q) - \
                self.simulation_params.target_capacity
            
            return np.array([self.storage_tank.vent_pressure - w[0], 
                             satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[1] - self.simulation_params.target_temp,
                             w[0] - self.simulation_params.target_pres,
                             capacity_event])
                        
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                print("\n The simulation has hit maximum pressure! Switch to venting or cooling simulation")
                raise TerminateSimulation
                
            if state_info[1] != 0 and solver.y[1] <= Tcrit - 0.01:
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
                
                
            if state_info[5] != 0:
                print("\n Target capacity reached.")
                raise TerminateSimulation
        
            
     
        w0 = np.array([self.simulation_params.init_pressure,
                       self.simulation_params.init_temperature,
                       self.simulation_params.inserted_amount,
                       self.simulation_params.flow_energy_in,
                       self.simulation_params.cooling_additional,
                       self.simulation_params.heating_additional,
                       self.simulation_params.heat_leak_in,
                       self.simulation_params.vented_amount,
                       self.simulation_params.vented_energy,
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
        # sim.rtol = 1E-5
        sim.atol = [1000, 1E-2,  1E-2, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2, 1E-2]
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
                phase = "Liquid" if self.simulation_params.init_nl > self.simulation_params.init_ng else "Gas"
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
                          tank_params = self.storage_tank,
                          sim_params = self.simulation_params)
    
class OnePhaseSorbentVenting(OnePhaseSorbentSim):
    sim_type = "Venting"
    
    def _dT_dt_vent_vary_fillrate(self, T, nh2, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        if self.boundary_flux.mass_flow_in(time) != 0:
            hin = self.enthalpy_in_calc(p, T)
        else:
            hin = 0
        phase = self.storage_tank.stored_fluid.determine_phase(p, T)
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        if phase != "Saturated":
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        else:
            q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, q)
        b = self._dn_dT(p, T, fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        hf = fluid_props["hf"]
        uf = fluid_props["uf"]
        return(ndotin*(hin-hf) + self.boundary_flux.heating_power(time) -\
               self.boundary_flux.cooling_power(time) + self.heat_leak_in(T)) \
            /(d-((hf)*b))  
        
    def _ndotout_vent_vary_fillrate(self, T, dTdt, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        phase = self.storage_tank.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        else:
            q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, q)
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
        if p0 <= pcrit:
            fluid.update(CP.PQ_INPUTS, p0, 0)
            Tsat = fluid.T()
        else:
            Tsat = 0
        
        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            nh2 = self.storage_tank.capacity(p0, T)
            phase = self.storage_tank.stored_fluid.determine_phase(p0, T)
            dTdt =  self._dT_dt_vent_vary_fillrate(T, nh2, t) 
            ndotout = self._ndotout_vent_vary_fillrate(T, dTdt, t) 
            MW = self.storage_tank.stored_fluid.backend.molar_mass()
            ndotin = self.boundary_flux.mass_flow_in(t) / MW
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, p0, T)
            else:
                q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
                fluid.update(CP.QT_INPUTS, q, T)
            hf = fluid.hmolar()
            if ndotin != 0:
                hin = self.enthalpy_in_calc(p0, T)
            else:
                hin = 0
            results = np.array([dTdt, ndotout, ndotout * hf,
                             ndotin, ndotin * hin, 
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t),
                             self.heat_leak_in(T)])
            return results
    
        def events(t, w, sw):
            satstatus = w[0] - Tsat
            if self.simulation_params.init_ng != 0 and self.simulation_params.init_nl != 0:
                q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            else:
                phaseinit = self.storage_tank.stored_fluid.determine_phase(p0, 
                            self.simulation_params.init_temperature)
                q = 0 if phaseinit == "Liquid" else 1
            capacity_event = self.storage_tank.capacity(p0, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus, w[0]-self.simulation_params.target_temp,
                             capacity_event])
    
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation
                
            if state_info[1] != 0:
                print("\n Final refueling condition achieved, exiting simulation.")
                raise TerminateSimulation
                
            if state_info[2] != 0:
                print("Target capacity reached.")
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
        sim.rtol = 1E-4
        sim.report_continuously = True
        
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
                          tank_params = self.storage_tank,
                          sim_params = self.simulation_params)

    
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
        nh2 = self.storage_tank.capacity(p, T)
        MW = fluid.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        hin = self.enthalpy_in_calc(p, T) if ndotin else 0
        fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T)
        hout = fluid_props["hf"]
        uf = fluid_props["uf"]
        d = self._dU_dT(p, T, fluid_props)
        return - d * dTdt + ndotin * (hin) - ndotout * (hout) + self.heat_leak_in(T)\
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
            
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T) \
                if phase != "Saturated" else self.storage_tank.stored_fluid.saturation_property_dict(T, 1)
            ndotin =  self.boundary_flux.mass_flow_in(t)/MW
            ndotout = self.boundary_flux.mass_flow_out(t)/MW
            hin = self.enthalpy_in_calc(p, T) if ndotin else 0
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
                    
            q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            capacity_event = self.storage_tank.capacity(p, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus, w[0]-self.simulation_params.target_temp,
                             capacity_event])
    
        def handle_event(solver, event_info):
            state_info = event_info[0]
            
            if state_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation

            if state_info[1] !=0 and p == self.simulation_params.target_pres:
                print("\n Final refueling temperature achieved, exiting simulation.")
                raise TerminateSimulation
            
            if state_info[2] != 0:
                print("\n Target capacity reached.")
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
                          tank_params = self.storage_tank,
                          sim_params = self.simulation_params)

class OnePhaseSorbentControlledInlet(OnePhaseSorbentSim):
    sim_type = "Controlled Inlet"
    def _solve_differentials(self, p, T, time, phase):
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass() 
        ##Convert kg/s to mol/s
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(time)  / MW
        ndotout = flux.mass_flow_out(time) / MW
        ##Get the thermodynamic properties of the bulk fluid for later calculations
        qinit = 0 if self.simulation_params.init_nl > self.simulation_params.init_ng else 1
        if phase != "Saturated":
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p,T)
        else:
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, qinit)
        ##Get the input pressure at a condition
        if ndotin != 0:
            hin = self.boundary_flux.enthalpy_in(time)
        else:
            hin = 0
        
        if ndotout != 0:
            hout = self.boundary_flux.enthalpy_out(time)
        else:
            hout = 0
        
        k1 = ndotin - ndotout
        k2 = ndotin * hin - ndotout * hout + \
            self.boundary_flux.heating_power(time) - self.boundary_flux.cooling_power(time)\
                + self.heat_leak_in(T)

        a = self._dn_dp(p, T, fluid_props)
        b = self._dn_dT(p, T, fluid_props)
        c = self._dU_dp(p, T,  fluid_props)
        d = self._dU_dT(p, T, fluid_props)
        #Put in the right hand side of the mass and energy balance equations
        A = [[a, b],
             [c,d]]
        b = [k1, k2]
        output = np.linalg.solve(A,b)
        return output
    
    
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
            dPdt, dTdt = self._solve_differentials(p, T, t, phase)
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
            output = np.array([dPdt, dTdt, ndotin, ndotin * hin,
                             self.boundary_flux.cooling_power(t),
                             self.boundary_flux.heating_power(t), self.heat_leak_in(T),
                             ndotout, ndotout * hout])
            return output
        
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
            
            q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            capacity_event = self.storage_tank.capacity(w[0], w[1], q)\
                - self.simulation_params.target_capacity
            return np.array([self.storage_tank.vent_pressure - w[0], 
                             satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[1] - self.simulation_params.target_temp,
                             w[0] - self.simulation_params.target_pres,
                             capacity_event])
                        
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
        
            if state_info[5] != 0 :
                print("\n Target capacity reached.")
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
        sim.rtol = 1E-4
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
                          tank_params = self.storage_tank,
                          sim_params = self.simulation_params)
    
class OnePhaseSorbentHeatedDischarge(OnePhaseSorbentSim):
    sim_type = "Heated"

    def _dT_dt_heated_const_pres(self, T, time):
        p = self.simulation_params.init_pressure
        MW = self.storage_tank.stored_fluid.backend.molar_mass()
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        ndotout = self.boundary_flux.mass_flow_out(time) / MW
        phase = self.storage_tank.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            qinit = 0 if self.simulation_params.init_ng < self.simulation_params.init_nl else 1
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, qinit)
        else:    
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T) 
        b = self._dn_dT(p, T, fluid_props)
        return ndotin-ndotout/b

    def _heating_power_const_pres(self, T, dTdt, time):
        p = self.simulation_params.init_pressure
        fluid = self.storage_tank.stored_fluid.backend
        MW = fluid.molar_mass()
        ndotout= self.boundary_flux.mass_flow_out(time) / MW
        ndotin = self.boundary_flux.mass_flow_in(time) / MW
        if ndotin != 0:
            hin = self.enthalpy_in_calc(p, T)
        else:
            hin = 0
        phase = self.storage_tank.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            qinit = 0 if self.simulation_params.init_ng < self.simulation_params.init_nl else 1
            fluid_props = self.storage_tank.stored_fluid.saturation_property_dict(T, qinit)
        else:    
            fluid_props = self.storage_tank.stored_fluid.fluid_property_dict(p, T) 
        hout = fluid_props["hf"]
        # uf = fluid_props["uf"]
        d = self._dU_dT(p, T, fluid_props)
        return d * dTdt + ndotout * (hout ) - ndotin * (hin ) - self.heat_leak_in(T)\
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
            dTdt =  self._dT_dt_heated_const_pres(T, t) 
            heating = self._heating_power_const_pres(T, dTdt, t)
            MW = fluid.molar_mass()
            ndotout= self.boundary_flux.mass_flow_out(t) / MW
            ndotin = self.boundary_flux.mass_flow_in(t) / MW
            if ndotin !=0:
                hin = self.enthalpy_in_calc(p, T)
            else:
                hin = 0
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
                    
            q = int(self.simulation_params.init_ng > self.simulation_params.init_nl)
            capacity_event = self.storage_tank.capacity(p, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus, w[0]-self.simulation_params.target_temp,
                             capacity_event])
    
        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] !=0 and solver.y[0] <= Tcrit:
                print("\n Saturation condition reached, switch to two-phase solver!")
                raise TerminateSimulation

            if state_info[1] !=0:
                print("\n Final temperature achieved, exiting simulation.")
                raise TerminateSimulation
            
            if state_info[2] != 0:
                print("\n Reached target capacity.")
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
        sim.atol = [0.01, 100, 1 ,1 ,1 ,1 ,1 ,1 ,1]
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
                          tank_params = self.storage_tank,
                          sim_params = self.simulation_params)
    
    
