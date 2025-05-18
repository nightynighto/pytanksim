# -*- coding: utf-8 -*-
"""Module for the simulation of sorbent tanks in the one phase region."""
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

__all__ = ["OnePhaseSorbentSim",
           "OnePhaseSorbentDefault",
           "OnePhaseSorbentVenting",
           "OnePhaseSorbentCooled",
           "OnePhaseSorbentHeatedDischarge"]

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


class OnePhaseSorbentSim(BaseSimulation):
    """Base class for simulation of sorbent tanks in the one phase region.

    It includes functions to calculate the governing ODE
    """

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
        qinit = self.simulation_params.init_q
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.partial_derivative(func, var, point, stepsize)
        elif phase1 == "Saturated":
            if (qinit == 0 and var == 1) or (qinit == 1 and var == 0):
                return \
                    fd.backward_partial_derivative(func, var, point, stepsize)
            else:
                return \
                    fd.forward_partial_derivative(func, var, point, stepsize)
        else:
            if phase1 == phase3:
                return \
                    fd.backward_partial_derivative(func, var, point, stepsize)
            elif phase1 == phase2:
                return \
                    fd.forward_partial_derivative(func, var, point, stepsize)

    def _dn_dp(self, p, T, qinit):
        deriver = self._derivfunc
        return deriver(self.storage_tank.capacity, 0, [p, T, qinit], 100)

    def _dn_dT(self, p, T, qinit):
        deriver = self._derivfunc
        return deriver(self.storage_tank.capacity, 1, [p, T, qinit], 1E-2)

    def _dU_dp(self, p, T, qinit):
        tank = self.storage_tank
        deriver = self._derivfunc
        return deriver(tank.internal_energy, 0, [p, T, qinit], 100)

    def _dU_dT(self, p, T, qinit):
        tank = self.storage_tank
        deriver = self._derivfunc
        return deriver(tank.internal_energy, 1, [p, T, qinit], 1E-2)\
            + tank.heat_capacity(T)


class OnePhaseSorbentDefault(OnePhaseSorbentSim):
    """Simulates sorbent tanks in the one phase region without constraints."""

    sim_type = "Default"

    def solve_differentials(self, p: float,
                            T: float,
                            time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        p : float
            Current pressure (Pa).

        T : float
            Current temperature (K).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        stored_fluid = self.storage_tank.stored_fluid
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW

        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        qinit = self.simulation_params.init_q
        phase = stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid_props = stored_fluid.fluid_property_dict(p, T)
        else:
            fluid_props = stored_fluid.saturation_property_dict(T, qinit)

        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)
        hout = self.enthalpy_out_calc(fluid_props, p, T, time)

        k1 = ndotin - ndotout
        k2 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional + heat_leak

        a = self._dn_dp(p, T, qinit)
        b = self._dn_dT(p, T, qinit)
        c = self._dU_dp(p, T, qinit)
        d = self._dU_dT(p, T, qinit)

        A = np.array([[a, b],
                     [c, d]])
        b = np.array([k1, k2])

        diffresults = np.linalg.solve(A, b)
        return np.append(diffresults, [ndotin,
                                       ndotin * hin,
                                       cooling_additional,
                                       heating_additional,
                                       heat_leak,
                                       ndotout,
                                       ndotout * hout
                                       ])

    def run(self) -> SimResults:
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as hitting the
            saturation line, or hitting the maximum pressure limit of the tank.

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
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        q = self.simulation_params.init_q

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            p, T = w[:2]
            res = self.solve_differentials(p, T, t)
            return res

        def events(t, w, sw):
            if w[1] > (Tcrit - 0.01):
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
            critical_event = ((w[0]-pcrit)**2)/((0.01*pcrit)**2) +\
                ((w[1]-Tcrit)**2)/((0.01*Tcrit)**2) - 1
            return np.array([self.storage_tank.vent_pressure - w[0],
                             satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[1] - self.simulation_params.target_temp,
                             w[0] - self.simulation_params.target_pres,
                             capacity_event,
                             critical_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                self.stop_reason = "MaxPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has hit maximum pressure!"
                                "\nSwitch to venting or cooling simulation")
                raise TerminateSimulation

            if state_info[1] != 0 and solver.y[1] <= Tcrit - 0.01:
                self.stop_reason = "SaturLineReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has hit the saturation line!"
                                "\nSwitch to two-phase simulation")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "MinPresReached"
                if self.simulation_params.verbose:
                    logger.warn("\nThe simulation has hit min supply pressure!"
                                "\nSwitch to heated discharge simulation")
                raise TerminateSimulation

            if state_info[3] != 0 and solver.sw[0]:
                self.stop_reason = "TargetTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget temperature reached")
                raise TerminateSimulation

            if state_info[4] != 0 and solver.sw[1]:
                self.stop_reason = "TargetPresReached"
                if self.simulation_params.verbose:
                    ("\nTarget pressure reached")
                raise TerminateSimulation

            if state_info[3] != 0 and state_info[4] != 0:
                self.stop_reason = "TargetCondsReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget conditions has been reached.")
                raise TerminateSimulation

            if state_info[5] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached.")
                raise TerminateSimulation

            if state_info[6] != 0:
                self.stop_reason = "CritPointReached"
                if self.simulation_params.verbose:
                    logger.warn("\nReached critical point,"
                                " ODE too stiff to simulate.")
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

        sw0 = self.simulation_params.stop_at_target_temp
        sw1 = self.simulation_params.stop_at_target_pressure
        switches0 = [sw0, sw1]
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        # sim.atol = [1000, 1E-3,  1E2, 1E3, 1E3, 1E3, 1E3, 1E2, 1E3]
        sim.rtol = 1e-5
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        nads = np.zeros_like(t)
        n_phase = {"Gas": np.zeros_like(t),
                   "Supercritical": np.zeros_like(t),
                   "Liquid": np.zeros_like(t)}

        for i in range(0, len(t)):
            iterable = i
            phase = self.storage_tank.\
                stored_fluid.determine_phase(y[i, 0], y[i, 1])
            if phase == "Saturated":
                while phase == "Saturated" and iterable > -len(y[:, 0]):
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.\
                        determine_phase(y[iterable, 0],
                                        y[iterable, 1])
                if phase == "Saturated":
                    q = self.simulation_params.init_q
                    phase = "Gas" if q == 1 else "Liquid"
                if phase == "Supercritical":
                    q = 0 if y[iterable, 1] < Tcrit else 1
                else:
                    q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i, 1])
            else:
                fluid.update(CP.PT_INPUTS, y[i, 0], y[i, 1])
            nfluid = fluid.rhomolar() *\
                self.storage_tank.bulk_fluid_volume(y[i, 0], y[i, 1])
            n_phase[phase][i] = nfluid
            nads[i] = self.storage_tank.sorbent_material.\
                model_isotherm.n_absolute(y[i, 0], y[i, 1]) *\
                self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=y[:, 0],
                          temperature=y[:, 1],
                          moles_adsorbed=nads,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          cooling_additional=y[:, 4],
                          heating_required=self.simulation_params.
                          heating_required,
                          heating_additional=y[:, 5],
                          heat_leak_in=y[:, 6],
                          vented_amount=y[:, 7],
                          vented_energy=y[:, 8],
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class OnePhaseSorbentVenting(OnePhaseSorbentSim):
    """Sorbent tank venting at constant pressure in the one phase region."""

    sim_type = "Venting"

    def solve_differentials(self, T: float, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        T : float
            Current temperature (K).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        p = self.simulation_params.init_pressure
        stored_fluid = self.storage_tank.stored_fluid
        flux = self.boundary_flux
        MW = stored_fluid.backend.molar_mass()
        ndotin = flux.mass_flow_in(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        phase = stored_fluid.determine_phase(p, T)
        q = self.simulation_params.init_q
        if phase != "Saturated":
            fluid_props = stored_fluid.fluid_property_dict(p, T)
        else:
            fluid_props = stored_fluid.saturation_property_dict(T, q)

        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)
        hout = self.enthalpy_out_calc(fluid_props, p, T, time)
        m11 = self._dn_dT(p, T, q)
        m12 = 1
        m21 = self._dU_dT(p, T, q)
        m22 = hout
        A = np.array([[m11, m12],
                      [m21, m22]])
        b1 = ndotin
        b2 = ndotin * hin + heating_additional - cooling_additional + heat_leak
        b = np.array([b1, b2])
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

    def run(self) -> SimResults:
        """Run the dynamic simulation.

        Raises
        ------
        TerminateSimulation
            Stops the simulation when it detects an event such as hitting the
            saturation line, or hitting the maximum pressure limit of the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
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
            return self.solve_differentials(T, t)

        def events(t, w, sw):
            satstatus = w[0] - Tsat
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p0, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus,
                             w[0]-self.simulation_params.target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                if self.simulation_params.verbose:
                    logger.warn("\nSaturation condition reached!"
                                "\nSwitch to two-phase solver!")
                raise TerminateSimulation

            if state_info[1] != 0:
                self.stop_reason = "TargetTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget temperature reached")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached")
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

        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Venting Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-4
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        if self.simulation_params.verbose:
            logger.info("Saving results...")
        nads = np.zeros_like(t)
        n_phase = {
            "Gas": np.zeros_like(t),
            "Liquid": np.zeros_like(t),
            "Supercritical": np.zeros_like(t)
            }

        for i in range(len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p0, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.\
                        determine_phase(p0, y[iterable, 0])
                if phase == "Supercritical":
                    q = 0 if y[iterable, 0] < Tcrit else 1
                else:
                    q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i, 0])
            else:
                fluid.update(CP.PT_INPUTS, p0, y[i, 0])
            nfluid = fluid.rhomolar() * self.storage_tank.\
                bulk_fluid_volume(p0, y[i, 0])

            n_phase[phase][i] = nfluid
            nads[i] = self.storage_tank.sorbent_material.\
                model_isotherm.n_absolute(p0, y[i, 0])\
                * self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=p0,
                          temperature=y[:, 0],
                          moles_adsorbed=nads,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          vented_amount=y[:, 1],
                          vented_energy=y[:, 2],
                          inserted_amount=y[:, 3],
                          flow_energy_in=y[:, 4],
                          cooling_additional=y[:, 5],
                          heating_additional=y[:, 6],
                          heat_leak_in=y[:, 7],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class OnePhaseSorbentCooled(OnePhaseSorbentSim):
    """Sorbent tank cooled at constant pressure in the one phase region."""

    sim_type = "Cooled"

    def solve_differentials(self, T: float, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        T : float
            Current temperature (K).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        p = self.simulation_params.init_pressure
        stored_fluid = self.storage_tank.stored_fluid
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        q = self.simulation_params.init_q
        phase = stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid_props = stored_fluid.fluid_property_dict(p, T)
        else:
            fluid_props = stored_fluid.saturation_property_dict(T, q)
        hout = self.enthalpy_out_calc(fluid_props, p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)
        m11 = self._dn_dT(p, T, q)
        m12 = 0
        m21 = self._dU_dT(p, T, q)
        m22 = 1
        A = [[m11, m12],
             [m21, m22]]
        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * hout + heating_additional\
            - cooling_additional + heat_leak
        b = np.array([b1, b2])
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
            Stops the simulation when it detects an event such as hitting the
            saturation line, or hitting the maximum pressure limit of the tank.

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
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p = self.simulation_params.init_pressure

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            return self.solve_differentials(T, t)

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

            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus,
                             w[0]-self.simulation_params.target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]

            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                if self.simulation_params.verbose:
                    logger.warn("\nSaturation condition reached!"
                                "\nSwitch to two-phase solver!")
                raise TerminateSimulation

            if state_info[1] != 0 and p == self.simulation_params.target_pres:
                self.stop_reason = "TargetCondsReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget condition achieved")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget capacity reached.")
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
        model = Explicit_Problem(rhs, w0,
                                 self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase dynamics of constant P refuel w/ Cooling"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.atol = [0.001, 100, 1, 100, 1, 100, 100, 100, 100]
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        nads = np.zeros_like(t)
        n_phase = {
            "Gas": np.zeros_like(t),
            "Liquid": np.zeros_like(t),
            "Supercritical": np.zeros_like(t)
            }
        for i in range(0, len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.\
                        determine_phase(p, y[iterable, 0])
                if phase == "Supercritical":
                    q = 0 if y[iterable, 0] < Tcrit else 1
                else:
                    q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i, 0])
            else:
                fluid.update(CP.PT_INPUTS, p, y[i, 0])
            n_phase[phase][i] = fluid.rhomolar() *\
                self.storage_tank.bulk_fluid_volume(p, y[i, 0])
            nads[i] = self.storage_tank.sorbent_material.\
                model_isotherm.n_absolute(p, y[i, 0]) *\
                self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=np.repeat(p, len(t)),
                          temperature=y[:, 0],
                          moles_adsorbed=nads,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          cooling_required=y[:, 1],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          vented_amount=y[:, 4],
                          vented_energy=y[:, 5],
                          cooling_additional=y[:, 6],
                          heating_additional=y[:, 7],
                          heat_leak_in=y[:, 8],
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class OnePhaseSorbentHeatedDischarge(OnePhaseSorbentSim):
    """Sorbent tank heated at constant pressure in the one phase region."""

    sim_type = "Heated"

    def solve_differentials(self, T: float, time: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        T : float
            Current temperature (K).

        time : float
            Current time step (in s).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        p = self.simulation_params.init_pressure
        stored_fluid = self.storage_tank.stored_fluid
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        q = self.simulation_params.init_q
        phase = stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid_props = stored_fluid.fluid_property_dict(p, T)
        else:
            fluid_props = stored_fluid.saturation_property_dict(T, q)
        hout = self.enthalpy_out_calc(fluid_props, p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(p, T, time)
        m11 = self._dn_dT(p, T, q)
        m12 = 0
        m21 = self._dU_dT(p, T, q)
        m22 = -1
        A = [[m11, m12],
             [m21, m22]]
        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * hout + heating_additional\
            - cooling_additional + heat_leak
        b = np.array([b1, b2])
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
            Stops the simulation when it detects an event such as hitting the
            saturation line, or hitting the maximum pressure limit of the tank.

        Returns
        -------
        SimResults
            An object for storing and manipulating the results of the dynamic
            simulation.

        """
        pbar = tqdm(total=1000, unit="‰",
                    disable=not(self.simulation_params.verbose))
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
            return self.solve_differentials(T, t)

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
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p, w[0], q)\
                - self.simulation_params.target_capacity
            return np.array([satstatus,
                             w[0]-self.simulation_params.target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                if self.simulation_params.verbose:
                    logger.warn("\nSaturation condition reached."
                                "\nSwitch to two-phase solver!")
                raise TerminateSimulation

            if state_info[1] != 0:
                self.stop_reason = "TargetTempReached"
                if self.simulation_params.verbose:
                    logger.warn("\nTarget temperature reached")
                raise TerminateSimulation

            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                if self.simulation_params.verbose:
                    logger.warn("\nReached target capacity.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase dynamics of constant P discharge w/ heating"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.verbosity = 30 if self.simulation_params.verbose else 50
        sim.atol = [0.01, 100, 1, 1, 1, 1, 1, 1, 1]
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass
        nads = np.zeros_like(t)
        n_phase = {
            "Gas": np.zeros_like(t),
            "Liquid": np.zeros_like(t),
            "Supercritical": np.zeros_like(t)
            }
        for i in range(0, len(t)):
            phase = self.storage_tank.stored_fluid.determine_phase(p, y[i, 0])
            iterable = i
            if phase == "Saturated":
                while phase == "Saturated":
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.\
                        determine_phase(p, y[iterable, 0])
                if phase == "Supercritical":
                    q = 0 if y[iterable, 0] < Tcrit else 1
                else:
                    q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i, 0])
            else:
                fluid.update(CP.PT_INPUTS, p, y[i, 0])
            n_phase[phase][i] = fluid.rhomolar() *\
                self.storage_tank.bulk_fluid_volume(p, y[i, 0])
            nads[i] = self.storage_tank.sorbent_material.\
                model_isotherm.n_absolute(p, y[i, 0]) *\
                self.storage_tank.sorbent_material.mass
        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=np.repeat(p, len(t)),
                          temperature=y[:, 0],
                          moles_adsorbed=nads,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          heating_required=y[:, 1],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          vented_amount=y[:, 4],
                          vented_energy=y[:, 5],
                          cooling_additional=y[:, 6],
                          heating_additional=y[:, 7],
                          heat_leak_in=y[:, 7],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)
