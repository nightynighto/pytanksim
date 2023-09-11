# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:58:53 2023

@author: nextf
"""

__all__ = ["generate_simulation"]

from pytanksim.classes.excessisothermclass import ExcessIsotherm
from pytanksim.classes.basesimclass import BoundaryFlux, SimParams
from pytanksim.classes.fluidsorbentclasses import StoredFluid, MPTAModel, SorbentMaterial
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
from pytanksim.classes.onephasesorbentsimclasses import *
from pytanksim.classes.twophasesorbentsimclasses import *
from pytanksim.classes.onephasefluidsimclasses import *
from pytanksim.classes.twophasefluidsimclasses import *

from typing import Union


phase_to_str = {
    1 : "One Phase",
    2 : "Two Phase"
    }
sim_class_dict = {
    "One Phase Sorbent Default" : OnePhaseSorbentDefault,
    "One Phase Sorbent Venting" : OnePhaseSorbentVenting,
    "One Phase Sorbent Cooled" : OnePhaseSorbentCooled,
    "One Phase Sorbent Controlled Inlet": OnePhaseSorbentControlledInlet,
    "One Phase Sorbent Heated Discharge": OnePhaseSorbentHeatedDischarge,
    "Two Phase Sorbent Default" : TwoPhaseSorbentDefault,
    "Two Phase Sorbent Venting" : TwoPhaseSorbentVenting,
    "Two Phase Sorbent Cooled" : TwoPhaseSorbentCooled,
    "Two Phase Sorbent Controlled Inlet": TwoPhaseSorbentControlledInlet,
    "Two Phase Sorbent Heated Discharge": TwoPhaseSorbentHeatedDischarge,
    "One Phase Fluid Default" : OnePhaseFluidDefault,
    "One Phase Fluid Venting" : OnePhaseFluidVenting,
    "One Phase Fluid Cooled" : OnePhaseFluidCooled,
    "One Phase Fluid Controlled Inlet": OnePhaseFluidControlledInlet,
    "One Phase Fluid Heated Discharge": OnePhaseFluidHeatedDischarge,
    "Two Phase Fluid Default" : TwoPhaseFluidDefault,
    "Two Phase Fluid Venting" : TwoPhaseFluidVenting,
    "Two Phase Fluid Cooled" : TwoPhaseFluidCooled,
    "Two Phase Fluid Controlled Inlet": TwoPhaseFluidControlledInlet,
    "Two Phase Fluid Heated Discharge": TwoPhaseFluidHeatedDischarge
    }

def generate_simulation(
        storage_tank : Union[StorageTank, SorbentTank],
        boundary_flux : BoundaryFlux,
        simulation_params : SimParams,
        simulation_type = "Default",
        phase = 1,
        ):
    if isinstance(storage_tank, SorbentTank):
        hasSorbent = " Sorbent "
    else: hasSorbent = " Fluid "
    
    class_caller = phase_to_str[phase] + hasSorbent + simulation_type
    
    return sim_class_dict.get(class_caller)(storage_tank = storage_tank,
                                            boundary_flux = boundary_flux,
                                            simulation_params = simulation_params)
    
    