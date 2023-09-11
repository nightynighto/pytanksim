# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:44:21 2023

@author: nextf
"""
__all__ = ["SimResults"]

import numpy as np
from typing import List
import pandas as pd
import csv
from pytanksim.classes.storagetankclasses import SorbentTank


class SimResults:
    
    fancy_colnames_dict ={ 
        "time" : "Time (s)",
        "pressure" : "Pressure (Pa)",
        "temperature" : "Temperature (K)",
        "moles_adsorbed" : "Amount Adsorbed (mol)" ,
        "moles_gas" : "Vapor Amount (mol)",
        "moles_liquid" : "Liquid Amount (mol)",
        "moles_supercritical" : "Supercritical Amount (mol)",
        "cooling_required" : "Cumulative Cooling Energy Required (J)",
        "heating_required" :"Cumulative Heating Energy Required (J)",
        "vented_amount" : "Amount Vented (mol)",
        "vented_energy" : "Cumulative Fluid Enthalpy Out (J)",
        "inserted_amount" : "Amount Inserted (mol)",
        "flow_energy_in" : "Cumulative Fluid Enthalpy In (J)",
        "cooling_additional" : "Cumulative Additional Cooling Provided (J)",
        "heating_additional" : "Cumulative Additional Heating Provided (J)",
        "heat_leak_in" : "Cumulative Heat Leakage (J)",
        "kg_adsorbed" : "Amount Adsorbed (kg)" ,
        "kg_gas" : "Vapor Amount (kg)",
        "kg_liquid" : "Liquid Amount (kg)",
        "kg_supercritical" : "Supercritical Amount (kg)",
        "vented_kg" : "Amount Vented (kg)",
        "inserted_kg" : "Amount Inserted (kg)",
        "moles_adsorbed_per_s" : "d(Adsorbed Amount)/dt (mol/s)",
        "moles_gas_per_s" :  "d(Vapor Amount)/dt (mol/s)",
        "moles_liquid_per_s" : "d(Liquid Amount)/dt (mol/s)",
        "moles_supercritical_per_s" : "d(Supercritical Amount)/dt (mol/s)",
        "cooling_required_W" : "Cooling Power Required (W)",
        "heating_required_W" : "Heating Power Required (W)",
        "vented_amount_per_s" : "Venting Rate (mol/s)",
        "vented_energy_W" : "Rate of Fluid Enthalpy Flow Out (W)",
        "inserted_amount_per_s" : "Refueling Rate (mol/s)",
        "flow_energy_in_W" : "Rate of Fluid Enthalpy Flow In (W)",
        "cooling_additional_W" : "Additional Cooling Power (W)",
        "heating_additional_W" : "Additional Cooling Power (W)",
        "heat_leak_in_W" : "Heat Leakage Rate (W)",
        "kg_adsorbed_per_s" : "d(Adsorbed Amount)/dt (kg/s)",
        "kg_gas_per_s" :  "d(Vapor Amount)/dt (kg/s)",
        "kg_liquid_per_s" : "d(Liquid Amount)/dt (kg/s)",
        "kg_supercritical_per_s" : "d(Supercritical Amount)/dt (kg/s)",
        "vented_kg_per_s" : "Venting Rate (kg/s)",
        "inserted_kg_per_s" : "Refueling Rate (kg/s)"
        }
    
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
                 sim_params,
                 inserted_amount = 0,
                 flow_energy_in = 0,
                 cooling_required = 0,
                 heating_required = 0,
                 cooling_additional = 0,
                 heating_additional = 0,
                 heat_leak_in = 0,
                 vented_amount = 0, 
                 vented_energy = 0,
                 ):
        


        self.results_dict = {
            "time" : time,
            "pressure" : pressure,
            "temperature" : temperature,
            "moles_adsorbed" : moles_adsorbed,
            "moles_gas" : moles_gas,
            "moles_liquid" : moles_liquid,
            "moles_supercritical" : moles_supercritical,
            "cooling_required" : cooling_required,
            "heating_required" : heating_required,
            "vented_amount" : vented_amount,
            "vented_energy" : vented_energy,
            "inserted_amount" : inserted_amount,
            "flow_energy_in" : flow_energy_in,
            "cooling_additional" : cooling_additional,
            "heating_additional" : heating_additional,
            "heat_leak_in" : heat_leak_in
            }
        
        # length = len(pressure)
        # for key, value in self.results_dict.items():
        #     assert len(value) == length
        self.sim_params = sim_params 
        self.tank_params = tank_params
        self.sim_type = sim_type
        self.results_df = pd.DataFrame.from_dict(self.results_dict)
        self.results_df = self.results_df.drop_duplicates(subset = ["time"])

        
        df_diff = self.results_df.diff()
        def time_diff(colname):
            return df_diff[colname]/df_diff["time"]
        self.results_df["moles_adsorbed_per_s"] = time_diff("moles_adsorbed")
        self.results_df["moles_gas_per_s"] = time_diff("moles_gas")
        self.results_df["moles_liquid_per_s"] = time_diff("moles_liquid")
        self.results_df["moles_supercritical_per_s"] = time_diff("moles_supercritical")
        self.results_df["cooling_required_W"] = time_diff("cooling_required")
        self.results_df["heating_required_W"] = time_diff("heating_required")
        self.results_df["vented_amount_per_s"] = time_diff("vented_amount")
        self.results_df["vented_energy_W"] = time_diff("vented_energy")
        self.results_df["inserted_amount_per_s"] = time_diff("inserted_amount")
        self.results_df["flow_energy_in_W"] = time_diff("flow_energy_in")
        self.results_df["cooling_additional_W"] = time_diff("cooling_additional")
        self.results_df["heating_additional_W"] = time_diff("heating_additional")
        self.results_df["heat_leak_in_W"] = time_diff("heat_leak_in")
        df_export = self.results_df.copy()
        molarmass = self.tank_params.stored_fluid.backend.molar_mass()
        overwrite_df = df_export.filter(regex = "moles|amount")
        overwrite_df = overwrite_df * molarmass
        column_names = list(overwrite_df.columns)
        column_names_new = [sub.replace("amount","kg").replace("moles","kg")
                            for sub in column_names]
        overwrite_df.columns = column_names_new
        self.results_df = pd.concat([df_export,overwrite_df], axis = 1)
        

        
    
    def get_final_conditions(self, idx = -1):
        final_dict = {}
        for key, value in self.results_dict.items():
            if isinstance(value, (str, list, tuple, np.ndarray)):
                final_dict[key] = value[idx]
            else:
                final_dict[key] = value
        return final_dict
        
        
    def to_csv(self, filename, convert_moles_to_kg = True):
        df_export = self.results_df.copy()
        new_colnames = [SimResults.fancy_colnames_dict[colname] for 
                             colname in list(df_export.columns)]
        df_export.columns = new_colnames
        print(df_export)
        header_info = [
            ["Has Sorbent?", isinstance(self.tank_params,SorbentTank)],
            ["Sorbent Name", self.tank_params.sorbent_material.model_isotherm.sorbent 
             if isinstance(self.tank_params,SorbentTank) else "NA"],
            ["Fluid Name", self.tank_params.stored_fluid.fluid_name],
            ["Aluminum Mass (kg)", self.tank_params.aluminum_mass],
            ["Carbon Fiber Mass (kg)", self.tank_params.carbon_fiber_mass],
            ["Steel Mass (kg)", self.tank_params.steel_mass],
            ["Minimum Supply Pressure (Pa)", self.tank_params.min_supply_pressure],
            ["Venting Pressure (Pa)", self.tank_params.vent_pressure]
            ]
        with open(filename, "w", newline="") as f:
            data = csv.writer(f)
            data.writerows(header_info)
            
        df_export.to_csv(filename, header=True, mode="a")
     
    @classmethod
    def combine_SimResults(cls, 
                           sim_results_list : "List[SimResults]"
                           ) -> "SimResults":
        list_of_df =  [result.results_df for result in sim_results_list]
        concat_df = pd.concat(list_of_df, ignore_index = True)
        concat_df = concat_df.sort_values("time")
        return cls()
    