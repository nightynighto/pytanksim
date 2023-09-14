# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:01:40 2023

@author: nextf
"""

import pytanksim as pts
import numpy as np


stored_fluid = pts.StoredFluid(fluid_name = "Hydrogen",
                                EOS = "HEOS")

temperatures = [298,318,338]
excesslist = []
for i, temper in enumerate(temperatures):
    filename = "AC2-" + str(temper) + "K.csv"
    excesslist.append( pts.ExcessIsotherm.from_csv(filename = filename, 
                                                adsorbate = "Methane",
                                                sorbent = "AC2",
                                                temperature = temper) )
    
stored_fluid = pts.StoredFluid(fluid_name = "Methane",
                                EOS = "HEOS")

model_isotherm_mda = pts.classes.MDAModel.from_ExcessIsotherms(excesslist, sorbent = "AC2",
                                                                stored_fluid = stored_fluid)

tankvol = np.pi * ((0.025)**2) * 0.13 
surface_area = 2 * np.pi * 0.05 * 0.185 + 2 * np.pi * ((0.05)**2) 
steel_volume =  np.pi * ((0.05)**2) * 0.03 + np.pi * ((0.05)**2) * 0.025  + \
                2 * np.pi * 0.05 * 0.13      
steel_density = 7861 #kg/m3
steel_mass = steel_density * steel_volume
                    
hc = 5


rhoskel = 1900
rhopack = 730
mads = rhopack * tankvol

sorbent_material = pts.SorbentMaterial(model_isotherm = model_isotherm_mda,
                                        skeletal_density = rhoskel,
                                        bulk_density = rhopack,
                                        mass = mads,
                                        specific_surface_area = 2261)


storage_tank = pts.SorbentTank(
                    volume = tankvol,
                    aluminum_mass = 0,
                    carbon_fiber_mass = 0,
                    steel_mass = steel_mass,
                    max_pressure = float(50E5),
                    vent_pressure = 50E5,
                    min_supply_pressure = 0,
                    sorbent_material = sorbent_material,
                    surface_area = surface_area,
                    thermal_resistance = 1/(hc * surface_area)
    )




