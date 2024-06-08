# -*- coding: utf-8 -*-
"""Module for simulating one phase fluid storage tanks without sorbents."""

__all__ = ["OnePhaseFluidSim", "OnePhaseFluidDefault", "OnePhaseFluidVenting",
           "OnePhaseFluidCooled", "OnePhaseFluidHeatedDischarge"]

import CoolProp as CP
import numpy as np
from tqdm.auto import tqdm
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.exception import TerminateSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.basesimclass import BaseSimulation


class OnePhaseFluidSim(BaseSimulation):
    """Base class for one phase fluid simulations."""

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

    def _du_dT(self, T, fluid_prop_dict):
        term = np.zeros(2)
        term[0] = fluid_prop_dict["drho_dT"] * fluid_prop_dict["uf"]
        term[1] = fluid_prop_dict["du_dT"] * fluid_prop_dict["rhof"]
        return self.storage_tank.volume *\
            (sum(term)) + self.storage_tank.heat_capacity(T)


class OnePhaseFluidDefault(OnePhaseFluidSim):
    """Class for simulating fluid storage dynamics in the one phase region."""

    sim_type = "Default"

    def solve_differentials(self, time: float,
                            p: float, T: float) -> np.ndarray:
        """Find the right hand side of the governing ODE at a given time step.

        Parameters
        ----------
        time : float
            Current time step (in s).

        p : float
            Current pressure (Pa).

        T : float
            Current temperature (K).

        Returns
        -------
        np.ndarray
            An array containing the right hand side of the ODE.

        """
        stored_fluid = self.storage_tank.stored_fluid
        phase = stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            prop_dict = stored_fluid.fluid_property_dict(p, T)
        else:
            qinit = self.simulation_params.init_q
            prop_dict = stored_fluid.saturation_property_dict(T, qinit)

        m11 = self._dn_dp(prop_dict)
        m12 = self._dn_dT(prop_dict)
        m21 = self._du_dp(prop_dict)
        m22 = self._du_dT(T, prop_dict)

        A = np.array([[m11, m12],
                      [m21, m22]])

        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)
        hout = self.enthalpy_out_calc(prop_dict, p, T, time)

        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * hout + \
            heating_additional -\
            cooling_additional\
            + heat_leak

        b = np.array([b1, b2])
        soln = np.linalg.solve(A, b)
        return np.append(soln,
                         [ndotin,
                          ndotin * hin,
                          ndotout,
                          ndotout * hout,
                          cooling_additional,
                          heating_additional,
                          heat_leak])

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

        pbar = tqdm(total=1000, unit="‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        pcrit = fluid.p_critical()

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            p, T = w[:2]
            return self.solve_differentials(t, p, T)

        def events(t, w, sw):
            if w[1] >= Tcrit:
                satstatus = w[0] - pcrit
            else:
                fluid.update(CP.QT_INPUTS, 0, w[1])
                satpres = fluid.p()
                if np.abs(w[0]-satpres) > (1E-6 * satpres):
                    satstatus = w[0] - satpres
                else:
                    satstatus = 0
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(w[0], w[1], q) - \
                self.simulation_params.target_capacity
            return np.array([self.storage_tank.vent_pressure - w[0], satstatus,
                             w[0] - self.storage_tank.min_supply_pressure,
                             w[0] - self.simulation_params.target_pres,
                             w[1] - self.simulation_params.target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0:
                self.stop_reason = "MaxPresReached"
                print("\n The simulation has hit maximum pressure!")
                print("\n Switch to venting or cooling simulation")
                raise TerminateSimulation
            if state_info[1] != 0 and solver.y[1] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                print("\n The simulation has hit the saturation line!")
                print("\n Switch to two-phase simulation")
                raise TerminateSimulation
            if state_info[2] != 0:
                self.stop_reason = "MinPresReached"
                print("\n The simulation has hit minimum supply pressure!")
                print("/n Switch to heated discharge simulation")
                raise TerminateSimulation

            if state_info[3] != 0 and solver.sw[0]:
                self.stop_reason = "TargetPresReached"
                print("Target pressure reached")
                raise TerminateSimulation

            if state_info[4] != 0 and solver.sw[1]:
                self.stop_reason = "TargetTempReached"
                print("Target temperature reached")
                raise TerminateSimulation

            if state_info[3] != 0 and state_info[4] != 0:
                self.stop_reason = "TargetCondsReached"
                print("\n Target conditions reached.")
                raise TerminateSimulation

            if state_info[5] != 0:
                self.stop_reason = "TargetCapReached"
                print("\n Target capacity reached.")
                raise TerminateSimulation

        w0 = np.array([self.simulation_params.init_pressure,
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
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-10
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass

        print("Saving results...")
        n_phase = {"Gas": np.zeros_like(t),
                   "Supercritical": np.zeros_like(t),
                   "Liquid": np.zeros_like(t)}

        for i in range(0, len(t)):
            iterable = i
            phase = self.storage_tank.stored_fluid.\
                determine_phase(y[i, 0], y[i, 1])
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
            nfluid = fluid.rhomolar() * self.storage_tank.volume
            n_phase[phase][i] = nfluid

        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"

        return SimResults(time=t,
                          pressure=y[:, 0],
                          temperature=y[:, 1],
                          moles_adsorbed=0,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          vented_amount=y[:, 4],
                          vented_energy=y[:, 5],
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


class OnePhaseFluidVenting(OnePhaseFluidSim):
    """Simulate the dynamics of a fluid tank venting at constant pressure."""

    sim_type = "Venting"

    def solve_differentials(self, time: float, T: float) -> np.ndarray:
        """Solve for the right hand side of the governing ODE.

        Parameters
        ----------
        time : float
            Current time step in the simulation (s).

        T : float
            Current temperature (K).

        Returns
        -------
        np.ndarray
            Numpy array containing values for the RHS of the governing ODE.

        """
        p = self.simulation_params.init_pressure
        stored_fluid = self.storage_tank.stored_fluid
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux

        ndotin = flux.mass_flow_in(p, T, time) / MW

        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)

        phase = stored_fluid.determine_phase(p, T)
        qinit = self.simulation_params.init_q
        if phase != "Saturated":
            prop_dict = stored_fluid.fluid_property_dict(p, T)
        else:
            prop_dict = stored_fluid.saturation_property_dict(T, qinit)

        hout = self.enthalpy_out_calc(prop_dict, p, T, time)

        m11 = self._dn_dT(prop_dict)
        m12 = 1
        m21 = self._du_dT(T, prop_dict)
        m22 = hout

        A = np.array([[m11, m12],
                      [m21, m22]])

        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(T)

        b1 = ndotin
        b2 = ndotin * hin + \
            heating_additional - cooling_additional\
            + heat_leak

        b = np.array([b1, b2])
        soln = np.linalg.solve(A, b)
        return np.append(soln,
                         [soln[-1] * hout,
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

        pbar = tqdm(total=1000, unit="‰")
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
            T = w[0]
            return self.solve_differentials(t, T)

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
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p0, w[0], q) - \
                self.simulation_params.target_capacity
            return np.array([satstatus, w[0] -
                             self.simulation_params.target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                print("\n The simulation has hit the saturation line!\n"
                      " Switch to two-phase simulation")
                raise TerminateSimulation
            if state_info[1] != 0:
                self.stop_reason = "TargetTempReached"
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                print("\n Target capacity reached.")
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
        model = Explicit_Problem(rhs, w0,
                                 self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-10
        t,  y = sim.simulate(self.simulation_params.final_time,
                             self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass

        print("Saving results...")
        n_phase = {"Gas": np.zeros_like(t),
                   "Supercritical": np.zeros_like(t),
                   "Liquid": np.zeros_like(t)}

        for i in range(0, len(t)):
            iterable = i
            phase = self.storage_tank.stored_fluid.determine_phase(p0, y[i, 0])
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
            nfluid = fluid.rhomolar() * self.storage_tank.volume
            n_phase[phase][i] = nfluid

        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"

        return SimResults(time=t,
                          pressure=p0,
                          temperature=y[:, 0],
                          moles_adsorbed=0,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          inserted_amount=y[:, 3],
                          flow_energy_in=y[:, 4],
                          cooling_additional=y[:, 5],
                          heating_additional=y[:, 6],
                          heat_leak_in=y[:, 7],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          heating_required=self.simulation_params.
                          heating_required,
                          vented_amount=y[:, 1],
                          vented_energy=y[:, 2],
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class OnePhaseFluidCooled(OnePhaseFluidSim):
    """Simulates a tank being cooled to maintain constant pressure."""

    sim_type = "Cooled"

    def solve_differentials(self, time: float, T: float) -> np.ndarray:
        """Solve for the right hand side of the governing ODE.

        Parameters
        ----------
        time : float
            Current time step in the simulation (s).

        T : float
            Current temperature (K).

        Returns
        -------
        np.ndarray
            Numpy array containing values for the RHS of the governing ODE.

        """
        stored_fluid = self.storage_tank.stored_fluid
        p = self.simulation_params.init_pressure
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        phase = stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            prop_dict = stored_fluid.fluid_property_dict(p, T)
        else:
            q = self.simulation_params.init_q
            prop_dict = stored_fluid.saturation_property_dict(T, q)

        m11 = self._dn_dT(prop_dict)
        m12 = 0
        m21 = self._du_dT(T, prop_dict)
        m22 = 1

        A = np.array([[m11, m12],
                      [m21, m22]])

        ndotout = flux.mass_flow_out(p, T, time) / MW
        hout = self.enthalpy_out_calc(prop_dict, p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        cooling_additional = flux.cooling_power(p, T, time)
        heat_leak = self.heat_leak_in(T)

        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * hout + \
            heating_additional - cooling_additional \
            + heat_leak

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

        pbar = tqdm(total=1000, unit="‰")
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
            T = w[0]
            return self.solve_differentials(t, T)

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
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p0, w[0], q) - \
                self.simulation_params.target_capacity
            return np.array([satstatus, w[0] - self.simulation_params.
                             target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                print("\n The simulation has hit the saturation line!\n"
                      "Switch to two-phase simulation")
                raise TerminateSimulation
            if state_info[1] != 0:
                self.stop_reason = "TargetTempReached"
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                print("\n The simulation has hit the target capacity.")
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
        model = Explicit_Problem(rhs, w0, self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics Cooled at Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-10
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)
        try:
            tqdm._instances.clear()
        except Exception:
            pass

        print("Saving results...")
        n_phase = {"Gas": np.zeros_like(t),
                   "Supercritical": np.zeros_like(t),
                   "Liquid": np.zeros_like(t)}

        for i in range(0, len(t)):
            iterable = i
            phase = self.storage_tank.stored_fluid.determine_phase(p0, y[i, 0])
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
            nfluid = fluid.rhomolar() * self.storage_tank.volume
            n_phase[phase][i] = nfluid

        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=p0,
                          temperature=y[:, 0],
                          moles_adsorbed=0,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          vented_amount=y[:, 4],
                          vented_energy=y[:, 5],
                          cooling_additional=y[:, 6],
                          heating_additional=y[:, 7],
                          heat_leak_in=y[:, 8],
                          cooling_required=y[:, 1],
                          heating_required=self.simulation_params.
                          heating_required,
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)


class OnePhaseFluidHeatedDischarge(OnePhaseFluidSim):
    """Simulates a tank being heated to discharge at a constant pressure."""

    sim_type = "Heated"

    def solve_differentials(self, time: float, T: float) -> np.ndarray:
        """Solve for the right hand side of the governing ODE.

        Parameters
        ----------
        time : float
            Current time step in the simulation (s).

        T : float
            Current temperature (K).

        Returns
        -------
        np.ndarray
            Numpy array containing values for the RHS of the governing ODE.

        """
        p = self.simulation_params.init_pressure
        stored_fluid = self.storage_tank.stored_fluid
        MW = stored_fluid.backend.molar_mass()
        flux = self.boundary_flux
        ndotin = flux.mass_flow_in(p, T, time) / MW
        ndotout = flux.mass_flow_out(p, T, time) / MW
        hin = 0 if ndotin == 0 else self.enthalpy_in_calc(p, T, time)
        phase = stored_fluid.determine_phase(p, T)

        if phase != "Saturated":
            prop_dict = stored_fluid.fluid_property_dict(p, T)
        else:
            q = self.simulation_params.init_q
            prop_dict = stored_fluid.saturation_property_dict(T, q)

        m11 = self._dn_dT(prop_dict)
        m12 = 0
        m21 = self._du_dT(T, prop_dict)
        m22 = -1

        A = np.array([[m11, m12],
                      [m21, m22]])

        cooling_additional = flux.cooling_power(p, T, time)
        heating_additional = flux.heating_power(p, T, time)
        heat_leak = self.heat_leak_in(T)
        hout = self.enthalpy_out_calc(prop_dict, p, T, time)

        b1 = ndotin - ndotout
        b2 = ndotin * hin - ndotout * hout + \
            - cooling_additional + heating_additional\
            + heat_leak

        b = np.array([b1, b2])

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

        pbar = tqdm(total=1000, unit="‰")
        state = [0, self.simulation_params.final_time/1000]
        fluid = self.storage_tank.stored_fluid.backend
        Tcrit = fluid.T_critical()
        fluid.update(CP.QT_INPUTS, 0, Tcrit)
        pcrit = fluid.p()
        p0 = self.simulation_params.init_pressure
        if p0 <= pcrit:
            fluid.update(CP.PQ_INPUTS, p0, 0)
            Tsat = fluid.T()
        else:
            Tsat = 0
        Tsat -= 1E-5*Tsat

        def rhs(t, w, sw):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            T = w[0]
            return self.solve_differentials(t, T)

        def events(t, w, sw):
            satstatus = w[0] - Tsat
            q = self.simulation_params.init_q
            capacity_event = self.storage_tank.capacity(p0, w[0], q) - \
                self.simulation_params.target_capacity
            return np.array([satstatus, w[0] - self.simulation_params.
                             target_temp,
                             capacity_event])

        def handle_event(solver, event_info):
            state_info = event_info[0]
            if state_info[0] != 0 and solver.y[0] <= Tcrit:
                self.stop_reason = "SaturLineReached"
                print("\n The simulation has hit the saturation line!\n"
                      "Switch to two-phase simulation")
                raise TerminateSimulation
            if state_info[1] != 0:
                self.stop_reason = "TargetTempReached"
                print("\n The simulation has hit the target temperature.")
                raise TerminateSimulation
            if state_info[2] != 0:
                self.stop_reason = "TargetCapReached"
                print("\n Target capacity reached.")
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
        model = Explicit_Problem(rhs, w0,
                                 self.simulation_params.init_time,
                                 sw0=switches0)
        model.state_events = events
        model.handle_event = handle_event
        model.name = "1 Phase Dynamics Heated at Constant Pressure"
        sim = CVode(model)
        sim.discr = "BDF"
        sim.rtol = 1E-10
        t, y = sim.simulate(self.simulation_params.final_time,
                            self.simulation_params.displayed_points)

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        print("Saving results...")
        n_phase = {"Gas": np.zeros_like(t),
                   "Supercritical": np.zeros_like(t),
                   "Liquid": np.zeros_like(t)}

        for i in range(0, len(t)):
            iterable = i
            phase = self.storage_tank.stored_fluid.determine_phase(p0, y[i, 0])
            if phase == "Saturated":
                while phase == "Saturated" and iterable > -len(y[:, 0]):
                    iterable = iterable - 1
                    phase = self.storage_tank.stored_fluid.\
                        determine_phase(p0,
                                        y[iterable, 0])
                if phase == "Saturated":
                    q = self.simulation_params.init_q
                    phase = "Liquid" if q == 0 else "Gas"
                if phase == "Supercritical":
                    q = 0 if y[iterable, 0] < Tcrit else 1
                else:
                    q = 0 if phase == "Liquid" else 1
                fluid.update(CP.QT_INPUTS, q, y[i, 0])
            else:
                fluid.update(CP.PT_INPUTS, p0, y[i, 0])
            nfluid = fluid.rhomolar() * self.storage_tank.volume
            n_phase[phase][i] = nfluid

        if self.stop_reason is None:
            self.stop_reason = "FinishedNormally"
        return SimResults(time=t,
                          pressure=p0,
                          temperature=y[:, 0],
                          moles_adsorbed=0,
                          moles_gas=n_phase["Gas"],
                          moles_liquid=n_phase["Liquid"],
                          moles_supercritical=n_phase["Supercritical"],
                          inserted_amount=y[:, 2],
                          flow_energy_in=y[:, 3],
                          vented_amount=y[:, 4],
                          vented_energy=y[:, 5],
                          cooling_additional=y[:, 6],
                          heating_additional=y[:, 7],
                          heat_leak_in=y[:, 8],
                          cooling_required=self.simulation_params.
                          cooling_required,
                          heating_required=y[:, 1],
                          sim_type=self.sim_type,
                          tank_params=self.storage_tank,
                          sim_params=self.simulation_params,
                          stop_reason=self.stop_reason)
