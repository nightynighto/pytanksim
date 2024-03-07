# -*- coding: utf-8 -*-
"""Main module of pytanksim, used to generate simulations."""

__all__ = ["generate_simulation"]

from pytanksim.classes.basesimclass import BoundaryFlux, SimParams,\
    BaseSimulation
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
