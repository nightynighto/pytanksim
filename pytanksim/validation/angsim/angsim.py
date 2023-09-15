# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:01:40 2023

@author: nextf
"""

import pytanksim as pts
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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
                                                                stored_fluid = stored_fluid,
                                                                pore_volume = 0.00123
                                                                )

for i, temper in enumerate(temperatures):
    pressure = excesslist[i].pressure
    mda_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    plt.figure()
    plt.plot(pressure, mda_result, label="MDA result")
    plt.scatter(pressure, excesslist[i].loading, label ="Excess")
    plt.legend()
    plt.title(str(temper))
    plt.show()

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


doserpres = pd.read_csv("AC2-doserpres.csv")
dosertemp = pd.read_csv("AC2-dosertemp.csv")
tankpres = pd.read_csv("AC2-tankpres.csv")
tanktemp = pd.read_csv("AC2-tanktemp.csv")

doserpres_f = CubicSpline(doserpres["t (s)"], doserpres["P (Pa)"], bc_type = "natural")
dosertemp_f = CubicSpline(dosertemp["t (s)"], dosertemp["T (K)"], bc_type = "natural")

def pin(time):
    return float(doserpres_f(time))
def Tin(time):
    return float(dosertemp_f(time))

doserflow =pd.DataFrame(data = [[0, 0.051],
             [43, 0.047],
             [116, 0.042],
             [139,0.035],
             [157,0.027],
             [175,0.003]],
                       columns = ["t (s)", "flow (mol/s)"])

doserflow["flow (kg/s)"] = doserflow["flow (mol/s)"] * stored_fluid.backend.molar_mass()
doserflow_f = CubicSpline(doserflow["t (s)"], doserflow["flow (kg/s)"], bc_type = "natural")
def flowin(time):
    return float(doserflow_f(time))

boundary_flux = pts.BoundaryFlux(
                mass_flow_in = flowin,
                mass_flow_out = 0.0,
                environment_temp = 289,
                pressure_in = pin,
                temperature_in =Tin
    )

simulation_params = pts.SimParams(
                    init_time = 0,
                    init_temperature = 257.8,
                    inserted_amount = 0,
                    init_pressure = 100,
                    final_time = 175
    )


simulation = pts.generate_simulation(storage_tank = storage_tank, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params,
                                      simulation_type = "Default")

results = simulation.run()
results.to_csv("ANGsim.csv")



