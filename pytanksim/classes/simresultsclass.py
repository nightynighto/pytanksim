# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:44:21 2023

@author: nextf
"""
__all__ = ["SimResults"]

import numpy as np
from typing import List
import pandas as pd


class SimResults:
    def __init__(self,
                 pressure : "List[float]|np.ndarray",
                 temperature : "List[float]|np.ndarray",
                 time :  "list|np.ndarray",
                 moles_adsorbed : "list|np.ndarray",
                 moles_gas,
                 moles_liquid,
                 moles_supercritical,
                 tank_params,
                 sim_type,
                 inserted_amount,
                 cooling_required = 0,
                 heating_required = 0,
                 vented_amount = 0, 
                 vented_energy = 0,):
        
       
                
        
        # if isinstance(cooling_required, float) and cooling_required == 0:
        #     cooling_required = np.zeros_like(time)
        # if isinstance(inserted_amount, float) and inserted_amount == 0:
        #     inserted_amount = np.zeros_like(time)
        # if isinstance(heating_required, float) and heating_required == 0:
        #     heating_required = np.zeros_like(time)
        # if vented_energy == 0 and vented_amount == 0 :
        #     vented_energy = np.zeros_like(time)
        #     vented_amount = np.zeros_like(time)
        # if vented_energy == 0 and vented_amount != 0:
        #     raise ValueError("Please tell the object the energy of the vented fluid.")
        # if vented_energy != 0 and vented_amount == 0:
        #     raise ValueError("Please tell the object how much has been vented.")
             
        
        
        self.results_dict = {
            "pressure" : pressure,
            "temperature" : temperature,
            "time" : time,
            "moles_adsorbed" : moles_adsorbed,
            "moles_gas" : moles_gas,
            "moles_supercritical" : moles_supercritical,
            "cooling_required" : cooling_required,
            "heating_required" : heating_required,
            "vented_amount" : vented_amount,
            "vented_energy" : vented_energy,
            "inserted_amount" : inserted_amount
            }
        
        # length = len(pressure)
        # for key, value in self.results_dict.items():
        #     assert len(value) == length
            
        self.tank_params = tank_params
        self.sim_type = sim_type
        self.results_df = pd.DataFrame.from_dict(self.results_dict)
        
    
    def get_final_conditions(self, idx = -1):
        final_dict = {}
        for key, value in self.results_dict.items():
            if isinstance(value, (str, list, tuple, np.ndarray)):
                final_dict[key] = value[idx]
            else:
                final_dict[key] = value
        return final_dict
        
        
    def to_csv(self, filename):
       self.results_df.to_csv(filename)
     
    