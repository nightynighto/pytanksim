# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:44:21 2023

@author: nextf
"""
import numpy as np
from typing import List


class SimResults:
    def __init__(self,
                 pressure : List[float]|np.ndarray,
                 temperature : List[float]|np.ndarray,
                 time :  list|np.ndarray,
                 moles_adsorbed : list|np.ndarray,
                 moles_gas,
                 moles_liquid,
                 moles_supercritical,
                 tank_params,
                 sim_type,
                 cooling_required = 0,
                 heating_required = 0,
                 vented_amount = 0,
                 inserted_amount = 0, 
                 vented_energy = 0,):
        
       
                
        
        if cooling_required == 0:
            cooling_required = np.zeros_like(time)
        if inserted_amount == 0:
            inserted_amount = np.zeros_like(time)
        if heating_required == 0:
            heating_required = np.zeros_like(time)
        if vented_energy == 0 and vented_amount == 0 :
            vented_energy = np.zeros_like(time)
            vented_amount = np.zeros_like(time)
        if vented_energy == 0 and vented_amount != 0:
            raise ValueError("Please tell the object the energy of the vented fluid.")
        if vented_energy != 0 and vented_amount == 0:
            raise ValueError("Please tell the object how much has been vented.")
             
        
        
        self.results_dict = {
            "pressure" : pressure,
            "temperature" : temperature,
            "time" : time,
            "moles_adsorbed" : moles_adsorbed,
            "moles_gas" : moles_gas,
            "moles_supercritical" : moles_supercritical,
            "cooling_required" : cooling_required,
            "vented_amount" : vented_amount,
            "vented_energy" : vented_energy,
            "inserted_amount" : inserted_amount
            }
        
        length = len(pressure)
        for key, value in self.results_dict.items():
            assert len(value) == length
            
        self.tank_params = tank_params
        self.sim_type = sim_type
        
    
    def get_final_conditions(self):
        final_dict = {}
        for key, value in self.results_dict.items():
            final_dict[key] = value[-1]
        
        
    