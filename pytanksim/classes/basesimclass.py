# -*- coding: utf-8 -*-
"""Contains classes related exclusively to the dynamic simulations.

This includes SimParams, BoundaryFlux, and BaseSimulation.
"""
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

__all__ = ["SimParams", "BoundaryFlux", "BaseSimulation"]


from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
from dataclasses import dataclass
from typing import Callable, Dict, Union
import CoolProp as CP
import numpy as np


@dataclass
class SimParams:
    """A class to store simulation parameters.

    This data class stores the parameters of the tank at the start of the
    simulation as well as the conditions specified to stop the simulation.
    Additionally, it also stores the setting for the number of data points
    to be reported at the end of the simulation.

    Attributes
    ----------
    init_temperature : float
        The temperature (K) of the tank being simulated at the beginning of
        the simulation.

    init_pressure : float, optional
        The pressure of the tank being simulated (Pa) at the beginning of the
        simulation. The default value is 1E5. This parameter was made optional
        as the two-phase simulations did not require it to be filled, rather
        pytanksim will automatically calculate the saturation pressure given
        a starting temperature.

    final_time : float
        The time (seconds) at which the simulation is to be stopped.

    init_time : float, optional
        The time (seconds) at which the beginning of the simulation is set to.
        The default value is set to 0 seconds.

    displayed_points : int, optional
        The number of data points to be reported at the end of the simulation.
        The default is 200.

    target_temp : float, optional
        The target temperature (K) at which the simulation is to be stopped.
        The default value is 0, which effectively means the simulation
        does not have a set temperature at which the simulation is stopped.

    target_pres : float, optional
        The target pressure (Pa) at which the simulation is to be stopped.
        The default value is 0, which effectively means the simulation does
        not have a set pressure at which the simulation is stopped.

    stop_at_target_pressure : bool, optional
        If True, it will stop the simulation when the target pressure is met.
        The default is False.

    stop_at_target_temp : bool, optional
        If True, it will stop the simulation when the target temperature is
        met. The default is False.

    target_capacity : float, optional
        The amount of fluid (moles) stored in the tank at which the simulation
        is to be stopped. The default is 0.

    init_ng : float, optional
        The initial amount of gas (moles) stored in the tank at the beginning
        of the simulation. The default value is 0.

    init_nl : float, optional
        The initial amount of liquid (moles) stored in the tank at the
        beginning of the simulation. The default value is 0.

    init_q : float, optional
        The initial quality of the fluid being stored. It can vary between 0
        and 1. The default is None.

    Other Parameters
    ----------------
    inserted_amount : float, optional
        The amount of fluid which has been previously inserted into the tank
        (moles) at the beginning of the simulation. Used to track refueling
        processes across multiple simulations. The default value is 0.

    vented_amount : float, optional
        The amount of fluid which has been previously vented from the tank
        (moles) at the beginning of the simulation. Used to track discharging
        and boil-off processes across multiple simulations. The default value
        is 0.

    cooling_required : float, optional
        The cumulative amount of required cooling (J) to maintain a constant
        pressure prior to the start of a simulation. The default value is 0.
        Useful when restarting a stopped cooled refuel simulation.

    heating_required : float, optional
        The cumulative amount of required heating (J) to maintain a constant
        pressure prior to the start of a simulation. The default value is 0.
        Useful when restarting a stopped heated discharge simulation.

    vented_energy : float, optional
        Cumulative amount of enthaloy (J) contained in the fluid vented prior
        to the start of the simulation. The default is 0. Useful when
        stopping and restarting discharge simulations.

    flow_energy_in : float, optional
        Cumulative amount of enthalpy (J) contained in the fluid inserted
        prior to the start of the simulation. The default is 0. Useful when
        stopping and restarting refueling simulations.

    cooling_additional : float, optional
        The cumulative amount of user-specified cooling (J) prior to the start
        of a simulation. The default value is 0. Useful when stopping and
        restarting simulations with user-specified cooling.

    heating_additional : float, optional
        The cumulative amount of user-specified cooling (J) prior to the start
        of a simulation. The default value is 0. Useful when stopping and
        restarting simulations with user-specified heating.

    heat_leak_in : float, optional
        The cumulative amount of heat (J) which has leaked into the tank prior
        to the start of a simulation. The default value is 0. Useful when
        stopping and restarting simulations involving heat leakage.

    verbose : bool, optional
        Whether or not the simulation will print out its progress bars and
        give a notification once it has finished. The default value is True.

    """

    init_temperature: float
    final_time: float
    init_time: float = 0
    init_pressure: float = 1E5
    displayed_points: int = 200
    target_temp: float = 0
    target_pres: float = 0
    stop_at_target_pressure: bool = False
    stop_at_target_temp: bool = False
    target_capacity: float = 0
    init_ng: float = 0
    init_nl: float = 0
    init_q: float = None
    inserted_amount: float = 0
    vented_amount: float = 0
    cooling_required: float = 0
    heating_required: float = 0
    vented_energy: float = 0
    flow_energy_in: float = 0
    cooling_additional: float = 0
    heating_additional: float = 0
    heat_leak_in: float = 0
    verbose: bool = True

    @classmethod
    def from_SimResults(cls,
                        sim_results: SimResults,
                        displayed_points: float = None,
                        init_time: float = None,
                        final_time: float = None,
                        target_pres: float = None,
                        target_temp:  float = None,
                        stop_at_target_pressure: bool = None,
                        stop_at_target_temp: bool = None,
                        target_capacity: float = None,
                        verbose:  bool = None
                        ) -> "SimParams":
        """Take final conditions from a previous simulation as new parameters.

        Parameters
        ----------
        sim_results : SimResults
            An object containing previous simulation results.

        displayed_points : float, optional
            The number of data points to be reported at the end of the
            simulation. The default is 200.

        init_time : float, optional
            The time (seconds) at which the beginning of the simulation is set.
            The default value is None.

        final_time : float, optional
            The time (seconds) at which the simulation is to be stopped. If
            None, then the final_time setting from the previous simulation is
            used. The default is None.

        target_pres : float, optional
            The target pressure (Pa) at which the simulation is to be stopped.
            If None, then the target_pres setting from the previous simulation
            is used. The default is None.

        target_temp : float, optional
            The target temperature (K) at which the simulation is to be
            stopped. If None, then the target_temp setting from the previous
            simulation is used. The default is None.

        stop_at_target_pressure : bool, optional
            If True, it will stop the simulation when the target pressure is
            met. If None, then the stop_at_target_pressure setting from the
            previous simulation is used. The default is None.

        stop_at_target_temp : bool, optional
            If True, it will stop the simulation when the target temperature
            is  met. If None, then the stop_at_target_temp setting from the
            previous simulation is used. The default is None.

        target_capacity : float, optional
           The amount of fluid (moles) stored in the tank at which the
           simulation is to be stopped. If None, then the target_capacity
           value from the previous simulation is used. The default is None.

        Returns
        -------
        SimParams
            A SimParams object containing the final conditions taken from
            sim_results set as the new starting parameters.

        """
        final_conditions = sim_results.get_final_conditions()
        if target_temp is None:
            target_temp = sim_results.sim_params.target_temp
        if target_pres is None:
            target_pres = sim_results.sim_params.target_pres
        if stop_at_target_temp is None:
            stop_at_target_temp = sim_results.sim_params.stop_at_target_temp
        if stop_at_target_pressure is None:
            stop_at_target_pressure = sim_results.sim_params.\
                stop_at_target_pressure
        if target_capacity is None:
            target_capacity = sim_results.sim_params.target_capacity
        if final_time is None:
            final_time = sim_results.sim_params.final_time
        if verbose is None:
            verbose = sim_results.sim_params.verbose
        if final_conditions["moles_gas"]\
                == final_conditions["moles_liquid"] == 0:
            fluid = sim_results.tank_params.stored_fluid.backend
            Tcrit = fluid.T_critical()
            final_minus_one = sim_results.get_final_conditions(-2)
            if final_minus_one["temperature"] > Tcrit:
                final_conditions["moles_gas"] = \
                    final_conditions["moles_supercritical"]
            else:
                final_conditions["moles_liquid"] = \
                    final_conditions["moles_supercritical"]
        if displayed_points is None:
            prev_finish_time = sim_results.sim_params.final_time
            prev_init_time = sim_results.sim_params.init_time
            actual_finish_time = final_conditions["time"]
            prev_displayed_points = sim_results.sim_params.displayed_points
            displayed_points = prev_displayed_points * (prev_finish_time -
                                                        actual_finish_time) /\
                (prev_finish_time - prev_init_time)
            displayed_points = int(displayed_points)
        if init_time is None:
            init_time = final_conditions["time"]
        return cls(
            init_pressure=final_conditions["pressure"],
            init_temperature=final_conditions["temperature"],
            init_time=init_time,
            init_ng=final_conditions["moles_gas"],
            init_nl=final_conditions["moles_liquid"],
            inserted_amount=final_conditions["inserted_amount"],
            vented_amount=final_conditions["vented_amount"],
            cooling_required=final_conditions["cooling_required"],
            heating_required=final_conditions["heating_required"],
            vented_energy=final_conditions["vented_energy"],
            flow_energy_in=final_conditions["flow_energy_in"],
            heat_leak_in=final_conditions["heat_leak_in"],
            cooling_additional=final_conditions["cooling_additional"],
            heating_additional=final_conditions["heating_additional"],
            final_time=final_time,
            target_temp=target_temp,
            target_pres=target_pres,
            stop_at_target_pressure=stop_at_target_pressure,
            stop_at_target_temp=stop_at_target_temp,
            target_capacity=target_capacity,
            displayed_points=displayed_points,
            verbose=verbose)


class BoundaryFlux:
    """Stores information of the mass and energy fluxes on the tank boundaries.

    Attributes
    ----------
    mass_flow_in : Callable[[float, float, float], float], optional
        A function which returns mass flow into the tank (kg/s) as a function
        of tank pressure (Pa), tank temperature (K), and time (s). The default
        is a function which returns 0 everywhere.

    mass_flow_out : Callable[[float, float, float], float], optional
        A function which returns mass flow exiting the tank (kg/s) as a
        function of tank pressure (Pa), tank temperature (K), and time (s).
        The default is a function which returns 0 everywhere.

    heating_power : Callable[[float, float, float], float], optional
        A function which returns heating power added to the tank (W) as a
        function of tank pressure (Pa), tank temperature (K), and time (s).
        The default is a function which returns 0 everywhere.

    cooling_power : Callable[[float, float, float], float], optional
        A function which returns cooling power added to the tank (W) as a
        function of tank pressure (Pa), tank temperature (K), and time (s).
        The default is a function which returns 0 everywhere.

    pressure_in : Callable[[float, float, float], float], optional
        A function which returns the pressure (Pa) of the fluid being inserted
        into the tank as a  function of tank pressure (Pa), tank temperature
        (K), and time (s). The default is None.

    temperature_in : Callable[[float, float, float], float], optional
        A function which returns the temperature (K) of the fluid being
        inserted into the tank as a  function of tank pressure (Pa), tank
        temperature (K), and time (s). The default is None.

    environment_temp : float, optional
        The temperature (K) of the environment surrounding the tank.
        This value is used in the dynamic simulation to calculate heat leakage
        into the tank. The default is 0, in which case heat leakage into the
        tank is not considered.

    enthalpy_in : Callable[[float, float, float], float], optional
        A function which returns the enthalpy (J/mol) of the fluid being
        inserted into the tank as a  function of tank pressure (Pa), tank
        temperature (K), and time (s). The default is None.

    enthalpy_out : Callable[[float, float, float], float], optional
        A function which returns the enthalpy (J/mol) of the fluid exiting
        the tank as a  function of tank pressure (Pa), tank temperature (K),
        and time (s). The default is None.

    """

    def __init__(self,
                 mass_flow_in:  Union[Callable[[float, float, float], float],
                                      float] = 0.0,
                 mass_flow_out: Union[Callable[[float, float, float], float],
                                      float] = 0.0,
                 heating_power: Union[Callable[[float, float, float], float],
                                      float] = 0.0,
                 cooling_power: Union[Callable[[float, float, float], float],
                                      float] = 0.0,
                 pressure_in: Union[Callable[[float, float, float], float],
                                    float] = None,
                 temperature_in: Union[Callable[[float, float, float], float],
                                       float] = None,
                 environment_temp: float = 0,
                 enthalpy_in: Union[Callable[[float, float, float], float],
                                    float] = None,
                 enthalpy_out: Union[Callable[[float, float, float], float],
                                     float] = None) -> "BoundaryFlux":
        """Initialize a BoundaryFlux object.

        Parameters
        ----------
        mass_flow_in : Callable or float, optional
            A function which returns mass flow into the tank (kg/s) as a
            functionof tank pressure (Pa), tank temperature (K), and time (s).
            The default is a function which returns 0 everywhere. If a float is
            provided, it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def mass_flow_in_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    #Returned is the mass flow going into the tank (kg/s)
                    return mass_flow_in

        mass_flow_out : Callable or float, optional
            A function which returns mass flow exiting the tank (kg/s) as a
            function of tank pressure (Pa), tank temperature (K), and time (s).
            The default is a function which returns 0 everywhere. If a float is
            provided it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def mass_flow_out_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the mass flow going out of the tank (kg/s)
                    return mass_flow_out

        heating_power : Callable or float, optional
            A function which returns heating power added to the tank (W) as a
            function of tank pressure (Pa), tank temperature (K), and time (s).
            The default is a function which returns 0 everywhere. If a float is
            provided, it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def heating_power_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the heat put into the tank (W)
                    return heating_power

        cooling_power : Callable or float, optional
            A function which returns cooling power added to the tank (W) as a
            function of tank pressure (Pa), tank temperature (K), and time (s).
            The default is a function which returns 0 everywhere. If a float is
            provided,it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def cooling_power_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the heat taken out of the tank (W)
                    return cooling_power

        pressure_in : Callable or float, optional
            A function which returns the pressure (Pa) of the fluid being
            inserted into the tank as a  function of tank pressure (Pa), tank
            temperature (K), and time (s). The default is None. If a float is
            provided,it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def pressure_in_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the pressure (Pa) of the fluid going into
                    # the tank.
                    return pressure_in

        temperature_in : Callable or float, optional
            A function which returns the temperature (K) of the fluid being
            inserted into the tank as a  function of tank pressure (Pa), tank
            temperature (K), and time (s). The default is None. If a float is
            provided,it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def temperature_in_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the temperature (K) of the fluid going into
                    # the tank.
                    return temperature_in

        environment_temp : float, optional
            The temperature (K) of the environment surrounding the tank. This
            value is used in the dynamic simulation to calculate heat leakage
            into the tank. The default is 0, in which case heat leakage into
            the tank is not considered.

        enthalpy_in : Callable or float, optional
            A function which returns the enthalpy (J/mol) of the fluid being
            inserted into the tank as a  function of tank pressure (Pa), tank
            temperature (K), and time (s). The default is None. If a float is
            provided,it will be converted to a function which returns that
            value everywhere.

            If a callable is passed, it must have the signature::

                def enthalpy_in_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the enthalpy (J/mol) of the fluid going into
                    # the tank.
                    return enthalpy_in

        enthalpy_out : Callable or float, optional
            A function which returns the enthalpy (J/mol) of the fluid exiting
            the tank as a  function of tank pressure (Pa), tank temperature
            (K), and time (s). The default is None. If a float is provided, it
            will be converted to a function which returns that value
            everywhere.

            If a callable is passed, it must have the signature::

                def enthalpy_out_function(p, T, time):
                    # 'p' is tank pressure (Pa)
                    # 'T' is tank temperature (K)
                    # 'time' is the time elapsed within the simulation (s)
                    ....
                    # Returned is the enthalpy (J/mol) of the fluid going out
                    # of the tank.
                    return enthalpy_out

        Raises
        ------
        ValueError
            If the mass flow going in is specified but the parameters that
            specify its enthalpy (i.e., either pressure and temperature or
            its enthalpy value) are not specified.

        Returns
        -------
        BoundaryFlux
            An object which stores information of the mass and energy fluxes on
            the tank boundaries.

        """
        def float_function_generator(floatingvalue):
            def float_function(p, T, time):
                return float(floatingvalue)
            return float_function

        if mass_flow_in != 0.0 and ((pressure_in is None or
                                     temperature_in is None) and
                                    enthalpy_in is None):
            raise ValueError("Please specify the pressure and temperature of"
                             " the flow going in")

        if isinstance(pressure_in, (float, int)):
            pressure_in = float_function_generator(pressure_in)

        if isinstance(temperature_in, (float, int)):
            temperature_in = float_function_generator(temperature_in)

        if isinstance(mass_flow_in, (float, int)):
            mass_flow_in = float_function_generator(mass_flow_in)

        if isinstance(mass_flow_out, (float, int)):
            mass_flow_out = float_function_generator(mass_flow_out)

        if isinstance(enthalpy_in, (float, int)):
            enthalpy_in = float_function_generator(enthalpy_in)

        if isinstance(enthalpy_out, (float, int)):
            enthalpy_out = float_function_generator(enthalpy_out)

        if isinstance(heating_power, (float, int)):
            heating_power = float_function_generator(heating_power)

        if isinstance(cooling_power, (float, int)):
            cooling_power = float_function_generator(cooling_power)

        self.mass_flow_in = mass_flow_in
        self.mass_flow_out = mass_flow_out
        self.heating_power = heating_power
        self.cooling_power = cooling_power
        self.pressure_in = pressure_in
        self.temperature_in = temperature_in
        self.environment_temp = environment_temp
        self.enthalpy_in = enthalpy_in
        self.enthalpy_out = enthalpy_out


class BaseSimulation:
    """An abstract base class for dynamic simulations.

    Other simulation classes inherit some attributes and methods from this
    class.

    Attributes
    ----------
    sim_type : str
        Type of simulation (default, heated discharge, cooled refuel, etc.)

    sim_phase : int
        1 or 2 phases.

    simulation_params : SimParams
        Object which stores simulation parameters.

    storage_tank : StorageTank
        Object which stores the properties of the tank being simulated.

    boundary_flux: BoundaryFlux
        Object which stores the amount of energy entering and exiting the tank.

    stop_reason : str
        A string stating the reason for the simulation to have stopped.
        It will be passed to the SimResults object once the simulation
        finishes.
    """

    sim_type = None
    sim_phase = None
    stop_reason = None

    def __init__(self,
                 simulation_params: SimParams,
                 storage_tank: StorageTank,
                 boundary_flux: BoundaryFlux) -> "BaseSimulation":
        """Initialize the BaseSimulation class.

        Parameters
        ----------
        simulation_params : SimParams
            Object containing simulation-specific parameters.

        storage_tank : StorageTank
            Object containing attributes and methods specific to the
            storage tank being simulated.

        boundary_flux : BoundaryFlux
            Object containing information on the mass and energy going in
            and out of the tank during the simulation.

        Raises
        ------
        ValueError
            If the simulation is set to begin on the saturation line but the
            initial values for liquid and gas in the tank, or, alternatively,
            the initial vapor quality, was not specified.

        ValueError
            If both the initial values for liquid and gas in the tank is
            specified as well as the initial vapor quality, but the values
            don't match each other.

        Returns
        -------
        BaseSimulation
            A simulation object which can be run to get results.

        """
        self.simulation_params = simulation_params
        self.storage_tank = storage_tank
        self.boundary_flux = boundary_flux
        self.has_sorbent = False
        if isinstance(self.storage_tank, SorbentTank):
            self.has_sorbent = True
        spr = self.simulation_params
        init_p = spr.init_pressure
        init_T = spr.init_temperature
        Tcrit = self.storage_tank.stored_fluid.backend.T_critical()
        init_phase = self.storage_tank.\
            stored_fluid.determine_phase(init_p, init_T)
        spr.init_ng = spr.init_ng if spr.init_ng >= 0 else 0
        spr.init_nl = spr.init_nl if spr.init_nl >= 0 else 0
        if init_phase s== "Saturated":
            psat = self.storage_tank.stored_fluid.saturation_property_dict(
                init_T, 0)
            init_p = psat
        if spr.init_q is None:
            if init_phase == "Saturated":
                if spr.init_ng == spr.init_nl == 0:
                    raise ValueError("Starting point is saturated,"
                                     " must input the initial amount of liquid"
                                     " and gas in the tank or the vapor "
                                     "quality.")
                else:
                    spr.init_q = spr.init_ng/(spr.init_ng + spr.init_nl)
            elif init_phase == "Gas":
                spr.init_q = 1
            elif init_phase == "Liquid":
                spr.init_q = 0
            elif init_phase == "Supercritical":
                spr.init_q = 1 if init_T > Tcrit else 0
        else:
            if init_phase == "Saturated":
                if spr.init_ng == spr.init_nl == 0:
                    amount = self.storage_tank.capacity_bulk(init_p, init_T,
                                                             spr.init_q)
                    spr.init_ng = amount * spr.init_q
                    spr.init_nl = amount * (1-spr.init_q)
                else:
                    qfrominitvals = spr.init_ng/(spr.init_ng + spr.init_nl)
                    if qfrominitvals != spr.init_q:
                        raise ValueError("Vapor quality does not match "
                                         "inputted initial amounts for gas and"
                                         " liquid! \n"
                                         "Try inputting either only the "
                                         "initial amounts or only the"
                                         " quality.")

    def heat_leak_in(self, T: float) -> float:
        """Calculate the heat leakage rate from the environment into the tank.

        Parameters
        ----------
        T : float
            Temperature (K) of the storage tank.

        Returns
        -------
        float
            The rate of heat leakage into the tank from the environment (W).

        """
        if self.boundary_flux.environment_temp == 0 or \
                self.storage_tank.thermal_resistance == 0:
            return 0
        else:
            return (self.boundary_flux.environment_temp - T) / \
                self.storage_tank.thermal_resistance

    def run(self):
        """Abstract function which will be defined in the child classes.

        Raises
        ------
        NotImplementedError
            Raises an error since it is not implemented in this abstract base
            class.

        Returns
        -------
        None.

        """
        raise NotImplementedError

    def enthalpy_in_calc(self, p: float, T: float, time: float) -> float:
        """Calculate the enthalpy (J/mol) of fluid going into the tank.

        Parameters
        ----------
        p : float
            Pressure inside of the tank (Pa)
        T : float
            Temperature inside of the tank (K)
        time : float
            Time (s) in the simulation.

        Returns
        -------
        float
            Enthalpy of the fluid going into the tank (J/mol).

        """
        if self.boundary_flux.enthalpy_in is None:
            pin = self.boundary_flux.pressure_in(p, T, time)
            Tin = self.boundary_flux.temperature_in(p, T, time)
            fluid = self.storage_tank.stored_fluid.backend
            Tcrit = fluid.T_critical()
            if Tin <= Tcrit:
                fluid.update(CP.QT_INPUTS, 0, Tin)
                psat = fluid.p()
                if np.abs(pin-psat) <= 1E-6 * psat:
                    fluid.update(CP.QT_INPUTS, 0, Tin)
                    return fluid.hmolar()
                else:
                    fluid.update(CP.PT_INPUTS, pin, Tin)
                    return fluid.hmolar()
            else:
                fluid.update(CP.PT_INPUTS, pin, Tin)
                return fluid.hmolar()
        else:
            return self.boundary_flux.enthalpy_in(p, T, time)

    def enthalpy_out_calc(self, fluid_property_dict: Dict[str, float],
                          p: float, T: float, time: float) -> float:
        """Calculate the enthalpy (J/mol) of fluid going out of the tank.

        Parameters
        ----------
        fluid_property_dict : Dict[str,float]
            A dictionary of properties of the fluid being stored inside of the
            tank. In the case of the two phase simulation, it is the properties
            of the gas and not the liquid. For this function, this dictionary
            must return an enthalpy (J/mol) value given the key "hf".
        p : float
            Pressure inside of the tank (Pa)
        T : float
            Temperature inside of the tank (K)
        time : float
            Time (s) in the simulation.

        Returns
        -------
        float
            Enthalpy of the fluid going out of the tank (J/mol).

        """
        if self.boundary_flux.enthalpy_out is None:
            return fluid_property_dict["hf"]
        else:
            return self.boundary_flux.enthalpy_out(p, T, time)
