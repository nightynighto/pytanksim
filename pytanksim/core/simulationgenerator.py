# -*- coding: utf-8 -*-
"""Main module of pytanksim, used to generate simulations."""

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

__all__ = ["generate_simulation", "automatic_simulation"]

from pytanksim.classes.basesimclass import BoundaryFlux, SimParams,\
    BaseSimulation
from pytanksim.classes.simresultsclass import SimResults
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
from pytanksim.classes.onephasesorbentsimclasses import *
from pytanksim.classes.twophasesorbentsimclasses import *
from pytanksim.classes.onephasefluidsimclasses import *
from pytanksim.classes.twophasefluidsimclasses import *
from typing import Union

phase_to_str = {
    1: "One Phase",
    2: "Two Phase"
    }

sim_class_dict = {
    "One Phase Sorbent Default": OnePhaseSorbentDefault,
    "One Phase Sorbent Venting": OnePhaseSorbentVenting,
    "One Phase Sorbent Cooled": OnePhaseSorbentCooled,
    "One Phase Sorbent Heated": OnePhaseSorbentHeatedDischarge,
    "Two Phase Sorbent Default": TwoPhaseSorbentDefault,
    "Two Phase Sorbent Venting": TwoPhaseSorbentVenting,
    "Two Phase Sorbent Cooled": TwoPhaseSorbentCooled,
    "Two Phase Sorbent Heated": TwoPhaseSorbentHeatedDischarge,
    "One Phase Fluid Default": OnePhaseFluidDefault,
    "One Phase Fluid Venting": OnePhaseFluidVenting,
    "One Phase Fluid Cooled": OnePhaseFluidCooled,
    "One Phase Fluid Heated": OnePhaseFluidHeatedDischarge,
    "Two Phase Fluid Default": TwoPhaseFluidDefault,
    "Two Phase Fluid Venting": TwoPhaseFluidVenting,
    "Two Phase Fluid Cooled": TwoPhaseFluidCooled,
    "Two Phase Fluid Heated": TwoPhaseFluidHeatedDischarge
    }


def generate_simulation(
        storage_tank: Union[StorageTank, SorbentTank],
        boundary_flux: BoundaryFlux,
        simulation_params: SimParams,
        simulation_type: str = "Default",
        phase: int = 1
        ) -> BaseSimulation:
    """Generate a dynamic simulation object.

    Parameters
    ----------
    storage_tank : Union[StorageTank, SorbentTank]
        An object with the properties of the storage tank. Can either be of the
        class StorageTank or its child class SorbentTank.

    boundary_flux : BoundaryFlux
        An object containing information about the mass and energy entering and
        leaving the control volume of the tank.

    simulation_params : SimParams
        An object containing various parameters for the dynamic simulation.

    simulation_type : str, optional
        A string describing the type of the simulation to be run. The default
        is "Default". The valid types are:

            - ``Default`` : A regular dynamic simulation with no constraints.
            - ``Cooled`` : A simulation where the tank is cooled to maintain a
              constant pressure. Here, the cooling power becomes one of the
              output variables. Typically used for simulating refueling after
              the tank has reached maximum allowable working pressure, or for
              simulating zero boil-off systems which are actively cooled.
            - ``Heated``: A simulation where the tank is heated to maintain a
              constant pressure. Here, the heating power becomes one of the
              output variables. Typically used for simulating discharging when
              the tank has reached the minimum supply pressure of the fuel cell
              system.
            - ``Venting`` : A simulation where the tank vents the fluid stored
              inside to maintain a constant pressure. Here, the amount vented
              becomes an output variable. Typically used to simulate boil-off
              or refueling with a feed-and-bleed scheme.

    phase : int, optional
        Specifies whether the fluid being stored is a single phase (1) or a
        two-phase (2) liquid and gas mixture. The default is 1 for single
        phase.

    Returns
    -------
    A child class of BaseSimulation
        A simulation object which can be ``run()`` to output a SimResults
        object. Which class will be generated depends on the parameters
        provided to this function.

    """
    if isinstance(storage_tank, SorbentTank):
        hasSorbent = " Sorbent "
    else:
        hasSorbent = " Fluid "
    class_caller = phase_to_str[phase] + hasSorbent + simulation_type
    return sim_class_dict.\
        get(class_caller)(storage_tank=storage_tank,
                          boundary_flux=boundary_flux,
                          simulation_params=simulation_params)


def _gen_phase(res: SimResults,
               prev_phase: int) -> int:
    """
    Generate number of phases for the next simulation in automated simulations.

    Parameters
    ----------
    res : SimResults
        Results of the previous simulation.

    prev_phase : int
        The number of fluid phases in the previous simulation. If the fluid
        was a single phase, it's 1. If the fluid was on the saturation line and
        there was a vapor-liquid equilibrium, then it's 2.

    Returns
    -------
    int
        Number of fluid phases in the next simulation.

    """
    if prev_phase == 1:
        phase = 2 if res.stop_reason == "SaturLineReached" else 1
    else:
        phase = 1 if res.stop_reason == "PhaseChangeEnded" else 2
    return phase


def _gen_type(res: SimResults, handle_max_pres: str,
              handle_min_pres: str) -> str:
    """
    Generate the next simulation type in a series of automated simulations.

    Parameters
    ----------
    res : SimResults
        Results of the previous simulation.

    handle_max_pres : str
        A string indicating how the simulation is to continue if the tank has
        reached its maximum allowable working pressure. "Cooled" means that the
        tank will not vent any gas, but will be actively cooled down. "Venting"
        means that the tank will begin to vent the exact amount of fluid inside
        to maintain the maximum pressure.

    handle_min_pres : str
        A string indicating how the simulation is to continue if the tank has
        reached its minimum supply pressure. "Heated" means exactly enough heat
        will be provided to the tank to maintain the minimum supply pressure.
        "Continue" means the simulation will restart without changing any
        parameters.

    Returns
    -------
    str
        The simulation type.

    """
    if res.stop_reason == "MaxPresReached":
        return handle_max_pres
    elif res.stop_reason == "MinPresReached":
        if handle_min_pres == "Continue":
            return "Default"
        else:
            return handle_min_pres
    else:
        return "Default"


def automatic_simulation(
        storage_tank: Union[StorageTank, SorbentTank],
        boundary_flux: BoundaryFlux,
        simulation_params: SimParams,
        stop_at_max_pres: bool = False,
        stop_at_min_pres: bool = False,
        handle_max_pres: str = "Cooled",
        handle_min_pres: str = "Heated") -> SimResults:
    """
    Automatically run and restart simulations until a target is reached.

    Parameters
    ----------
    storage_tank : Union[StorageTank, SorbentTank]
        An object with the properties of the storage tank. Can either be of the
        class StorageTank or its child class SorbentTank.

    boundary_flux : BoundaryFlux
        An object containing information about the mass and energy entering and
        leaving the control volume of the tank.

    simulation_params : SimParams
        An object containing various parameters for the dynamic simulation.

    stop_at_max_pres : bool, optional
        Whether or not the simulation is to be stopped when the tank hits its
        maximum allowable working pressure. The default is False.

    stop_at_min_pres : bool, optional
        Whether or not the simulation is to be stopped when the tank hits its
        minimum supply pressure. The default is False.

    handle_max_pres : str, optional
        A string indicating how the simulation is to continue if the tank has
        reached its maximum allowable working pressure. "Cooled" means that the
        tank will not vent any gas, but will be actively cooled down. "Venting"
        means that the tank will begin to vent the exact amount of fluid inside
        to maintain the maximum pressure. The default is "Cooled".

    handle_min_pres : str, optional
        A string indicating how the simulation is to continue if the tank has
        reached its minimum supply pressure. "Heated" means exactly enough heat
        will be provided to the tank to maintain the minimum supply pressure.
        "Continue" means the simulation will restart without changing any
        parameters. The default is "Heated".

    Returns
    -------
    SimResults
        An object for storing and manipulating the results of the dynamic
        simulations.

    """
    spr = simulation_params
    init_p = spr.init_pressure
    init_T = spr.init_temperature
    init_phase = storage_tank.\
        stored_fluid.determine_phase(init_p, init_T)

    valid_stop_reasons = ["FinishedNormally", "TargetTempReached",
                          "TargetPresReached", "TargetCondsReached",
                          "TargetCapReached", "CritPointReached"]
    if stop_at_max_pres:
        valid_stop_reasons.append("MaxPresReached")
    if stop_at_min_pres:
        valid_stop_reasons.append("MinPresReached")

    res_list = []
    simtype = "Default"
    phase = 1 if init_phase != "Saturated" else 2
    sim = generate_simulation(storage_tank, boundary_flux, spr,
                              phase=phase,
                              simulation_type=simtype)
    res = sim.run()
    res_list.append(res)
    while not (res.stop_reason in valid_stop_reasons):
        phase = _gen_phase(res, phase)
        simtype = _gen_type(res, handle_max_pres, handle_min_pres)
        spr = SimParams.from_SimResults(res)
        sim = generate_simulation(storage_tank, boundary_flux, spr,
                                  phase=phase,
                                  simulation_type=simtype)
        res = sim.run()
        res_list.append(res)
    return SimResults.combine(res_list)
