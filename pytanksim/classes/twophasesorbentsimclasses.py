# -*- coding: utf-8 -*-
"""Module for simulating sorbent tanks in the two-phase region."""
"""
Copyright 2024 Muhammad Irfan Maulana Kusdhany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = ["TwoPhaseSorbentSim",
           "TwoPhaseSorbentDefault",
           "TwoPhaseSorbentVenting",
           "TwoPhaseSorbentCooled",
           "TwoPhaseSorbentHeatedDischarge"]

import CoolProp as CP
import numpy as np
import pytanksim.utils.finitedifferences as fd
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation
from pytanksim.utils import logger


class TwoPhaseSorbentSim(BaseSimulation):
    """Base class for sorbent tanks in the two-phase region."""

    sim_phase = "Two Phase"

    def _saturation_deriv(self, ptfunc, T, **kwargs):
        fluid = self.storage_tank.stored_fluid.backend

        def function_satur(T):
            fluid.update(CP.QT_INPUTS, 1, T)
            pres = fluid.p()
            return ptfunc(pres, T, **kwargs)
        Tcrit = fluid.T_critical()
        if T < (Tcrit - 0.001):
            return fd.pardev(function_satur, T, 0.001)
        else:
            return fd.backdev(function_satur, T, 0.001)

    def _dn_dT(self, T, saturation_properties):
        sorbent = self.storage_tank.sorbent_material
        isotherm = self.storage_tank.sorbent_material.model_isotherm
        return sorbent.mass * self._saturation_deriv(isotherm.n_absolute, T)

    def _dv_dn(self, saturation_properties):
        return 1/saturation_properties["rhof"]

    def _dv_dT(self, ng, nl, T, saturation_properties_gas,
               saturation_properties_liquid):
        sorbent = self.storage_tank.sorbent_material
        term = np.zeros(4)
        if ng < 0:
            ng = 0
        if nl < 0:
            nl = 0
        dps_dT = saturation_properties_gas["dps_dT"]
        drhog_dT, rhog, drhog_dp = map(saturation_properties_gas.get,
                                       ("drho_dT", "rhof", "drho_dp"))
        drhol_dT, rhol, drhol_dp = map(saturation_properties_liquid.get,
                                       ("drho_dT", "rhof", "drho_dp"))
        term[0] = sorbent.mass * \
            self._saturation_deriv(sorbent.model_isotherm.v_ads, T)
        term[2] = (-ng/(rhog**2)) * (drhog_dp * dps_dT + drhog_dT)
        term[3] = (-nl/(rhol**2)) * (drhol_dp * dps_dT + drhol_dT)
        return sum(term)

    def _du_dng(self, ng, nl, T, saturation_properties_gas):
        if ng < 0:
            ng = 0
        if nl < 0:
            nl = 0
        return saturation_properties_gas["uf"]

    def _du_dnl(self, ng, nl, T, saturation_properties_liquid):
        sorbent = self.storage_tank.sorbent_material
        total_surface_area = sorbent.specific_surface_area *\
            sorbent.mass * 1000
        du_dA = sorbent.model_isotherm.areal_immersion_energy(T)
        p = saturation_properties_liquid["psat"]
        bulkvol = self.storage_tank.bulk_fluid_volume(p, T)
        if ng < 0:
            ng = 0
        if nl < 0:
            nl = 0
        return saturation_properties_liquid["uf"] \
            + du_dA * total_surface_area / \
            (saturation_properties_liquid["rhof"]*bulkvol)

    def _du_dT(self, ng, nl, T, saturation_properties_gas,
               saturation_properties_liquid):
        dps_dT = saturation_properties_gas["dps_dT"]
        p = saturation_properties_gas["psat"]
        sorbent = self.storage_tank.sorbent_material
        total_surface_area = sorbent.specific_surface_area *\
            sorbent.mass * 1000
        dps_dT = saturation_properties_gas["dps_dT"]
        dug_dT, ug, dug_dp = map(saturation_properties_gas.get,
                                 ("du_dT", "uf", "du_dp"))
        dul_dT, ul, dul_dp = map(saturation_properties_liquid.get,
                                 ("du_dT", "uf", "du_dp"))
        rhol, drhol_dT, drhol_dp = map(saturation_properties_liquid.get,
                                       ("rhof", "drho_dT", "drho_dp"))
        du_dA = sorbent.model_isotherm.areal_immersion_energy(T)
        if ng < 0:
            ng = 0
        if nl < 0:
            nl = 0

        bulkvol = self.storage_tank.bulk_fluid_volume(p, T)
        dbulkvol_dT = self._saturation_deriv(self.storage_tank.
                                             bulk_fluid_volume, T)
        term = np.zeros(5)
        term[0] = self._saturation_deriv(self.storage_tank.
                                         internal_energy_sorbent, T)
        term[1] = ng * (dug_dT + dug_dp * dps_dT)
        term[2] = nl * (dul_dT + dul_dp * dps_dT)
        term[3] = self.storage_tank.heat_capacity(T)
        term[4] = - nl * total_surface_area * du_dA *\
            (drhol_dT * bulkvol + dbulkvol_dT * rhol) / ((rhol*bulkvol)**2)
        return sum(term)


class TwoPhaseSorbentDefault(TwoPhaseSorbentSim):
    """Simulate sorbent tanks in the two phase region without constraints."""

    sim_type = "Default"

    def solve_differentials(self, ng: float,
                            nl: float, T: float,
                            time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        time : float
            Current time step (in s).

        ng : float
            Current amount of gas in the tank (moles).

        nl : float
            Current amount of liquid in the tank (moles).

        T : float
            Current temperature (K).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        stored_fluid = self.storage_tank.stored_fluid
        satur_prop_gas = stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = stored_fluid.saturation_property_dict(T, 0)
        p = satur_prop_gas["psat"]
        m11 = 1
        m12 = 1
        m13 = self._dn_dT(T, satur_prop_gas)
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = self._dv_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        m31 = self._du_dng(ng, nl, T, satur_prop_gas)
        m32 = self._du_dnl(ng, nl, T, satur_prop_liquid)
        m33 = self._du_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)
        A = np.matrix([[m11, m12, m13],
                       [m21, m22, m23],
                       [m31, m32, m33]])
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)

        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional \
            + heat_leak
        b = np.array([b1, b2, b3])
        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults,
                         [ndotin,
                          ndotin * hin,
                          ndotout,
                          ndotout * hout,
                          cooling_additional,
                          heating_additional,
                          heat_leak])

    def run(self):
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as the end of
            the phase change, or if the simulation hits the maximum pressure of
            the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        try:
            tqdm._instances.clear()
        except Exception:
            pass

        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
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
            T = w[2]
            crit = T-Tcrit
            fluid.update(CP.QT_INPUTS, 0, T)
            p = fluid.p()
            min_pres_event = p - self.storage_tank.min_supply_pressure
            max_pres_event = self.storage_tank.vent_pressure - p
            sat_liquid_event = w[0]
            sat_gas_event = w[1]
            target_pres_reach = p - self.simulation_params.target_pres
            target_temp_reach = T - self.simulation_params.target_temp
            ng = w[0]
            nl = w[1]
            if ng < 0:
                ng = 0
            if nl < 0:
                nl = 0
            target_capacity_reach = self.storage_tank.capacity(p, T,
                                                               ng/(ng+nl)) \
                - self.simulation_params.target_capacity
            return np.array([crit, min_pres_event, max_pres_event,
                             sat_gas_event, sat_liquid_event,
                             target_pres_reach, target_temp_reach,
                             target_capacity_reach])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                self.stop_reason = "CritTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has reached crit. temp.,"
                                "\nplease switch to one phase simulation.")
                raise TerminateSimulation

            if state_info[1] != 0:
                self.stop_reason = "MinPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has hit min. pressure."
                                "\nSwitch to heated discharge.")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "MaxPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has hit maximum pressure!"
                                "\nSwitch to cooling or venting simulation")
                raise TerminateSimulation

            if state_info[3] != 0 or state_info[4] != 0:
                self.stop_reason = "PhaseChangeEnded"
                if self.simulation_params.verbose:
                    logger.warn("\nPhase change has ended."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[5] != 0 and solver.sw[0]:
                self.stop_reason = "TargetPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget pressure reached")
                raise TerminateSimulation

            if state_info[6] != 0 and solver.sw[1]:
                self.stop_reason = "TargetTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget temperature reached")
                raise TerminateSimulation

            if state_info[5] != 0 and state_info[6] != 0:
                self.stop_reason = "TargetCondsReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget conditions has been reached.")
                raise TerminateSimulation

            if state_info[6] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity has been reached.")
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

        sw0 = self.simulation_params.stop_at_target_pressure
        sw1 = self.simulation_params.stop_at_target_temp

        switches0 = [sw0, sw1]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics"
        sim = CVode(model)
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.discr = "BDF"
        sim.atol = [1, 1, 0.05, 1, 1, 1, 1, 1, 1, 1]
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        pres = np.zeros_like(t)
        nads = np.zeros_like(t)

        for i in range(0, len(t)):
            fluid.update(CP.QT_INPUTS, 0, y[i, 2])
            pres[i] = fluid.p()
            nads[i] = self.storage_tank.sorbent_material.\
                model_isotherm.n_absolute(pres[i], y[i, 2]) *\
                self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=pres,
                          temperature=y[:, 2],
                          moles_adsorbed=nads,
                          moles_gas=y[:, 0],
                          moles_liquid=y[:, 1],
                          moles_supercritical=0,
                          inserted_amount=y[:, 3],
                          flow_energy_in=y[:, 4],
                          vented_amount=y[:, 5],
                          vented_energy=y[:, 6],
                          cooling_additional=y[:, 7],
                          heating_additional=y[:, 8],
                          heat_leak_in=y[:, 9],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class TwoPhaseSorbentCooled(TwoPhaseSorbentSim):
    """Sorbent tank cooled at constant pressure in the two-phase region."""

    sim_type = "Cooled"

    def solve_differentials(self, time: float,
                            ng: float, nl: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        time : float
            Current time step (in s).

        ng : float
            Current amount of gas in the tank (moles).

        nl : float
            Current amount of liquid in the tank (moles).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        T = self.simulation_params.init_temperature
        stored_fluid = self.storage_tank.stored_fluid
        satur_prop_gas = stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = stored_fluid.saturation_property_dict(T, 0)
        p = satur_prop_gas["psat"]
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = 0
        m31 = self._du_dng(ng, nl, T, satur_prop_gas)
        m32 = self._du_dnl(ng, nl, T, satur_prop_liquid)
        m33 = 1
        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)

        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional \
            + heat_leak
        b = np.array([b1, b2, b3])
        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [
            ndotin,
            ndotin * hin,
            ndotout,
            ndotout * hout,
            cooling_additional,
            heating_additional,
            heat_leak
            ])

    def run(self):
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as the end of
            the phase change, or if the simulation hits the maximum pressure of
            the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
        state = [0, self.simulation_params.final_time/1000]

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            return self.solve_differentials(t, w[0], w[1])

        def events(t, w, sw):
            sat_liquid_event = w[0]
            sat_gas_event = w[1]
            p = self.simulation_params.init_pressure
            T = self.simulation_params.init_temperature
            ng = 0 if w[0] < 0 else w[0]
            nl = 0 if w[1] < 0 else w[1]
            target_capacity_reach = self.storage_tank.capacity(p, T,
                                                               ng/(ng+nl)) \
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_reach])

        def handle_event(solver, event_info):
            state_info = event_info[0]

            if state_info[0] != 0 or state_info[1] != 0:
                self.stop_reason = "PhaseChangeEnded"
                if self.simulation_params.verbose:
                    logger.warn("\nPhase change has ended."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached.")
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
                       self.simulation_params.heat_leak_in])

        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics Cooled with Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.rtol = 1E-3
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")

        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
            self.simulation_params.init_pressure,
            self.simulation_params.init_temperature) *\
            self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=nads,
                          moles_gas=y[:, 0],
                          moles_liquid=y[:, 1],
                          moles_supercritical=0,
                          cooling_required=y[:, 2],
                          inserted_amount=y[:, 3],
                          flow_energy_in=y[:, 4],
                          vented_amount=y[:, 5],
                          vented_energy=y[:, 6],
                          cooling_additional=y[:, 7],
                          heating_additional=y[:, 8],
                          heat_leak_in=y[:, 9],
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class TwoPhaseSorbentVenting(TwoPhaseSorbentSim):
    """Sorbent tank venting at constant pressure in the two-phase region."""

    sim_type = "Venting"

    def solve_differentials(self, ng: float,
                            nl: float, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        ng : float
            Current amount of gas in the tank (moles).

        nl : float
            Current amount of liquid in the tank (moles).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        T = self.simulation_params.init_temperature
        stored_fluid = self.storage_tank.stored_fluid
        satur_prop_gas = stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = stored_fluid.saturation_property_dict(T, 0)
        p = satur_prop_gas["psat"]
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)
        m11 = 1
        m12 = 1
        m13 = 1
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = 0
        m31 = self._du_dng(ng, nl, T, satur_prop_gas)
        m32 = self._du_dnl(ng, nl, T, satur_prop_liquid)
        m33 = hout

        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        hin = self.enthalpy_in_calc(p, T, time) if ndotin else 0
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)

        b1 = ndotin
        b2 = 0
        b3 = ndotin * hin + \
            heating_additional - cooling_additional\
            + heat_leak

        b = np.array([b1, b2, b3])

        diffresults = np.linalg.solve(A, b)
        ndotout = diffresults[-1]
        return np.append(diffresults, [
            ndotout * hout,
            ndotin,
            ndotin * hin,
            cooling_additional,
            heating_additional,
            heat_leak
            ])

    def run(self):
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as the end of
            the phase change, or if the simulation hits the maximum pressure of
            the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        try:
            tqdm._instances.clear()
        except Exception:
            pass

        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
        state = [0, self.simulation_params.final_time/1000]

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(w[0], w[1], t)
            return diffresults

        def events(t, w, sw):
            sat_liquid_event = w[0]
            sat_gas_event = w[1]
            p = self.simulation_params.init_pressure
            T = self.simulation_params.init_temperature
            ng = w[0]
            nl = w[1]
            if ng < 0:
                ng = 0
            if nl < 0:
                nl = 0
            target_capacity_reach = self.storage_tank.capacity(p,
                                                               T,
                                                               ng/(ng+nl)) \
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_reach])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 or state_info[1] != 0:
                self.stop_reason = "PhaseChangeEnded"
                if self.simulation_params.verbose:
                    logger.warn("\nPhase change has ended."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ venting"
        sim = CVode(model)
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.discr = "BDF"
        sim.rtol = 1E-6
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")

        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
           self.simulation_params.init_pressure,
           self.simulation_params.init_temperature) *\
            self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=nads,
                          moles_gas=y[:, 0],
                          moles_liquid=y[:, 1],
                          moles_supercritical=0,
                          vented_amount=y[:, 2],
                          vented_energy=y[:, 3],
                          inserted_amount=y[:, 4],
                          flow_energy_in=y[:, 5],
                          cooling_additional=y[:, 6],
                          heating_additional=y[:, 7],
                          heat_leak_in=y[:, 8],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class TwoPhaseSorbentHeatedDischarge(TwoPhaseSorbentSim):
    """Sorbent tank heated at constant pressure in the two-phase region."""

    sim_type = "Heated"

    def solve_differentials(self, time: float,
                            ng: float, nl: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        ng : float
            Current amount of gas in the tank (moles).

        nl : float
            Current amount of liquid in the tank (moles).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        T = self.simulation_params.init_temperature
        stored_fluid = self.storage_tank.stored_fluid
        satur_prop_gas = stored_fluid.saturation_property_dict(T, 1)
        satur_prop_liquid = stored_fluid.saturation_property_dict(T, 0)
        p = satur_prop_gas["psat"]
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = self._dv_dn(satur_prop_gas)
        m22 = self._dv_dn(satur_prop_liquid)
        m23 = 0
        m31 = self._du_dng(ng, nl, T, satur_prop_gas)
        m32 = self._du_dnl(ng, nl, T, satur_prop_liquid)
        m33 = -1
        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)

        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional \
            + heat_leak
        b = np.array([b1, b2, b3])
        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [
            ndotin,
            ndotin * hin,
            ndotout,
            ndotout * hout,
            cooling_additional,
            heating_additional,
            heat_leak
            ])

    def run(self):
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as the end of
            the phase change, or if the simulation hits the maximum pressure of
            the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
        state = [0, self.simulation_params.final_time/1000]

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            return self.solve_differentials(t, w[0], w[1])

        def events(t, w, sw):
            sat_liquid_event = w[0]
            sat_gas_event = w[1]
            ng = w[0]
            nl = w[1]
            if ng < 0:
                ng = 0
            if nl < 0:
                nl = 0
            p = self.simulation_params.init_pressure
            T = self.simulation_params.init_temperature
            target_capacity_reach = self.storage_tank.capacity(p, T,
                                                               ng/(ng+nl)) \
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_reach])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 or state_info[1] != 0:
                self.stop_reason = "PhaseChangeEnded"
                if self.simulation_params.verbose:
                    logger.warn("\nPhase change has ended."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached.")
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
                       self.simulation_params.heat_leak_in])
        switches0 = []
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics Heated with Constant Pressure"
        sim = CVode(model)
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.discr = "BDF"
        sim.rtol = 1E-3
        t,  y = sim.simulate(self.simulation_params.final_time,
                             self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")

        nads = self.storage_tank.sorbent_material.model_isotherm.n_absolute(
            self.simulation_params.init_pressure,
            self.simulation_params.init_temperature) *\
            self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=nads,
                          moles_gas=y[:, 0],
                          moles_liquid=y[:, 1],
                          moles_supercritical=0,
                          heating_required=y[:, 2],
                          inserted_amount=y[:, 3],
                          flow_energy_in=y[:, 4],
                          vented_amount=y[:, 5],
                          vented_energy=y[:, 6],
                          cooling_additional=y[:, 7],
                          heating_additional=y[:, 8],
                          heat_leak_in=y[:, 9],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)
