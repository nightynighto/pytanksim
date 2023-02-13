# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:58:53 2023

@author: nextf
"""

__all__ = ["generate_simulation"]

from pytanksim.classes.excessisothermclass import ExcessIsotherm
from pytanksim.classes.basesimclass import BoundaryFlux, SimulationParams
from pytanksim.classes.fluidsorbentclasses import StoredFluid, MPTAModel, SorbentMaterial
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
from pytanksim.classes.onephasesorbentsimclasses import *
from typing import Union


phase_to_str = {
    1 : "One Phase",
    2 : "Two Phase"
    }
sim_class_dict = {
    "One Phase Sorbent Default" : OnePhaseSorbentDefault,
    "One Phase Sorbent Venting" : OnePhaseSorbentVenting,
    "One Phase Sorbent Cooled" : OnePhaseSorbentCooled,
    "One Phase Sorbent Controlled Inlet": OnePhaseSorbentControlledInlet
    }

def generate_simulation(
        storage_tank : Union[StorageTank, SorbentTank],
        boundary_flux : BoundaryFlux,
        simulation_params : SimulationParams,
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
    
    