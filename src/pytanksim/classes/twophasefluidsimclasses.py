# -*- coding: utf-8 -*-
"""Module for simulating fluid tanks in the two-phase region."""
# This file is a part of the python package pytanksim.
#
# Copyright (c) 2024 Muhammad Irfan Maulana Kusdhany, Kyushu University
#
# pytanksim is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

__all__ = ["TwoPhaseFluidSim", "TwoPhaseFluidDefault", "TwoPhaseFluidVenting",
           "TwoPhaseFluidCooled", "TwoPhaseFluidHeatedDischarge"]

import CoolProp as CP
import numpy as np
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation
from pytanksim.utils import logger


class TwoPhaseFluidSim(BaseSimulation):
    """Base class for the simulation of fluid tanks in the two-phase region.

    Contains functions for calculating the governing ODEs.
    """

    sim_phase = "Two Phase"

    def _dv_dT(self, ng, nl, saturation_prop_gas, saturation_prop_liquid):
        term = np.zeros(2)
        term[0] = - (ng / (saturation_prop_gas["rhof"]**2)) *\
            (saturation_prop_gas["drho_dp"] * saturation_prop_gas["dps_dT"] +
             saturation_prop_gas["drho_dT"])
        term[1] = - (nl / (saturation_prop_liquid["rhof"]**2)) *\
            (saturation_prop_liquid["drho_dp"] *
             saturation_prop_liquid["dps_dT"] +
             saturation_prop_liquid["drho_dT"])
        return sum(term)

    def _dU_dT(self, ng, nl, T, saturation_prop_gas, saturation_prop_liquid):
        term = np.zeros(3)
        term[0] = ng * (saturation_prop_gas["du_dp"] *
                        saturation_prop_gas["dps_dT"]
                        + saturation_prop_gas["du_dT"])
        term[1] = nl * (saturation_prop_liquid["du_dp"] *
                        saturation_prop_liquid["dps_dT"]
                        + saturation_prop_liquid["du_dT"])
        term[2] = self.storage_tank.heat_capacity(T)
        return sum(term)


class TwoPhaseFluidDefault(TwoPhaseFluidSim):
    """Simulation of fluid tanks in the two-phase region w/o constraints."""

    sim_type = "Default"

    def solve_differentials(self, time: float,
                            ng: float, nl: float,
                            T: float) -> np.ndarray:
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
        p = satur_prop_liquid["psat"]
        m11 = 1
        m12 = 1
        m13 = 0
        m21 = 1 / satur_prop_gas["rhof"]
        m22 = 1 / satur_prop_liquid["rhof"]
        m23 = self._dv_dT(ng, nl, satur_prop_gas, satur_prop_liquid)
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = self._dU_dT(ng, nl, T, satur_prop_gas, satur_prop_liquid)

        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)

        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional\
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

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            ng, nl, T = w[:3]
            diffresults = self.solve_differentials(t, ng, nl, T)
            return diffresults

        def events(t, w, sw):
            fluid.update(CP.QT_INPUTS, 0, w[2])
            p = fluid.p()
            min_pres_event = p - self.storage_tank.min_supply_pressure
            max_pres_event = self.storage_tank.vent_pressure - p
            target_temp_event = self.simulation_params.target_temp - w[2]
            target_pres_event = self.simulation_params.target_pres - p
            sat_liquid_event = w[0]
            sat_gas_event = w[1]
            crit_temp_event = w[2] - fluid.T_critical()
            ng = w[0]
            nl = w[1]
            if ng < 0:
                ng = 0
            if nl < 0:
                nl = 0
            target_capacity_event = self.storage_tank.capacity(p, w[2], ng/(ng+nl))\
                - self.simulation_params.target_capacity
            return np.array([min_pres_event, max_pres_event,
                             sat_gas_event, sat_liquid_event,
                             target_temp_event, target_pres_event,
                             crit_temp_event, target_capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                self.stop_reason = "MinPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nMinimum pressure has been reached."
                                "\nSwitch to heated discharge simulation.")
                raise TerminateSimulation

            if state_info[1] != 0:
                self.stop_reason = "MaxPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nMaximum pressure has been reached. "
                                "\nEither begin venting or cooling.")
                raise TerminateSimulation

            if state_info[2] != 0 or state_info[3] != 0:
                self.stop_reason = "PhaseChangeEnded"
                if self.simulation_params.verbose:
                    logger.warn("\nPhase change has ended."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[4] != 0 and solver.sw[0]:
                self.stop_reason = "TargetTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget temperature reached.")
                raise TerminateSimulation

            if state_info[5] != 0 and solver.sw[1]:
                self.stop_reason = "TargetPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget pressure reached.")
                raise TerminateSimulation

            if state_info[4] != 0 and state_info[5] != 0:
                self.stop_reason = "TargetCondsReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget conditions reached.")
                raise TerminateSimulation

            if state_info[6] != 0:
                self.stop_reason = "CritTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nReached critical temperature."
                                "\nSwitch to one phase simulation.")
                raise TerminateSimulation

            if state_info[7] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nReached target capacity.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        pres = np.zeros_like(t)
        for i, time in enumerate(t):
            fluid.update(CP.QT_INPUTS, 0, y[i, 2])
            pres[i] = fluid.p()
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=pres,
                          temperature=y[:, 2],
                          moles_adsorbed=0,
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


class TwoPhaseFluidVenting(TwoPhaseFluidSim):
    """Fluid tank venting at constant pressure in the two-phase region."""

    sim_type = "Venting"

    def solve_differentials(self, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
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
        m21 = 1 / satur_prop_gas["rhof"]
        m22 = 1 / satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = hout

        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)

        b1 = ndotin
        b2 = 0
        b3 = ndotin * hin + \
            heating_additional - cooling_additional\
            + heat_leak

        b = np.array([b1, b2, b3])

        diffresults = np.linalg.solve(A, b)
        ndotout = diffresults[-1]
        diffresults = np.append(diffresults, )
        return np.append(diffresults, [ndotout * hout,
                                       ndotin,
                                       ndotin * hin,
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
        state = [0, self.simulation_params.final_time / 1000]

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
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
            target_capacity_event = self.storage_tank.\
                capacity(p, T, ng / (ng + nl))\
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_event])

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
                    logger.warn("\nReached target capacity.")
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
        sim.discr = "BDF"
        sim.rtol = 1E-6
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=0,
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


class TwoPhaseFluidCooled(TwoPhaseFluidSim):
    """Fluid tank being cooled at constant pressure in the two-phase region."""

    sim_type = "Cooled"

    def solve_differentials(self, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
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
        m21 = 1 / satur_prop_gas["rhof"]
        m22 = 1 / satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = 1

        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)

        b1 = ndotin
        b2 = 0
        b3 = ndotin * hin + ndotout * hout +\
            heating_additional - cooling_additional + \
            heat_leak

        b = np.array([b1, b2, b3])

        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [ndotin,
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

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
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
            target_capacity_event = self.storage_tank.capacity(p, T, ng/(ng + nl))\
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_event])

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
                    logger.warn("\nReached target capacity.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ cooling"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")

        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=0,
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


class TwoPhaseFluidHeatedDischarge(TwoPhaseFluidSim):
    """Fluid tank being heated at constant pressure in the two-phase region."""

    sim_type = "Heated"

    def solve_differentials(self, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
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
        m21 = 1 / satur_prop_gas["rhof"]
        m22 = 1 / satur_prop_liquid["rhof"]
        m23 = 0
        m31 = satur_prop_gas["uf"]
        m32 = satur_prop_liquid["uf"]
        m33 = -1

        A = np.array([[m11, m12, m13],
                      [m21, m22, m23],
                      [m31, m32, m33]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        hout = self.enthalpy_out_calc(satur_prop_gas, p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)

        b1 = ndotin - ndotout
        b2 = 0
        b3 = ndotin * hin - ndotout * hout +\
            - cooling_additional + heating_additional + \
            heat_leak

        b = np.array([b1, b2, b3])

        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [ndotin,
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

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            diffresults = self.solve_differentials(t)
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
            target_capacity_event = self.storage_tank.capacity(p, T, ng/(ng + nl))\
                - self.simulation_params.target_capacity
            return np.array([sat_gas_event, sat_liquid_event,
                             target_capacity_event])

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
                    logger.warn("\nReached target capacity.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "2 Phase Dynamics w/ Heating"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-6
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)

        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=self.simulation_params.init_pressure,
                          temperature=self.simulation_params.init_temperature,
                          moles_adsorbed=0,
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
                          heating_required=y[:, 2],
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)
