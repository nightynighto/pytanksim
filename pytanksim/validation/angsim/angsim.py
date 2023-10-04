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
import CoolProp as CP


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

model_isotherm_mda = pts.classes.DAModel.from_ExcessIsotherms(excesslist, sorbent = "AC2",
                                                                stored_fluid = stored_fluid,
                                                                k_mode = "Constant",
                                                                va_mode = "Excess",
                                                                f0_mode = "Dubinin",
                                                                rhoa_mode = "Ozawa",
                                                                pore_volume = 0.00123,
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

tankvol = 0.000292
surface_area = 2 * np.pi * 0.05 * 0.185 + 2 * np.pi * ((0.05)**2) 
steel_volume =  np.pi * ((0.05)**2) * 0.03 + np.pi * ((0.05)**2) * 0.025  + \
                np.pi * ((0.05**2)-(0.025**2)) * 0.13      
steel_density = 7861 #kg/m3
steel_mass = steel_density * steel_volume
                    
hc = 5


rhopack = 700
mads = rhopack * tankvol
rhoskel = 1900

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



inletconds =pd.DataFrame(data = [[0, 0.051, 1.089, 257.8],
              [43, 0.047, 0.112, 264.8],
              [116, 0.042, 0.051, 271.4],
              [139, 0.035, 0.032, 276.6],
              [157, 0.027, 0.019, 280.8],
              [175, 0.003, 0.002, 287.1]],
                        columns = ["t (s)", "flow (mol/s)", "flow (L/s)", "T (K)"])

pressures = np.zeros(len(inletconds))
flows = np.zeros(len(inletconds))

for index, row in inletconds.iterrows():
    # rhomolar = row["flow (mol/s)"] / (0.001 * row["flow (L/s)"])
    stored_fluid.backend.update(CP.PT_INPUTS, 1E5, 273.15)
    rhomass = stored_fluid.backend.rhomass()
    flows[index] = rhomass * row["flow (L/s)"] / 60
    

    
inletpres_f = CubicSpline(inletconds["t (s)"], pressures, bc_type = "natural")
inlettemp_f = CubicSpline(inletconds["t (s)"], inletconds["T (K)"], bc_type = "natural")

def pin(p, T, time):
    return 0.1 + float(inletpres_f(time))
    # return float(doserpres_f(time))
def Tin(p, T, time):
    return float(inlettemp_f(time))
    # return float(dosertemp_f(time))

inletconds["flow (kg/s)"] = flows
# inletconds["flow (kg/s)"] = inletconds ["flow (mol/s)"] * stored_fluid.backend.molar_mass()
inletflow_f = CubicSpline(inletconds["t (s)"], inletconds["flow (kg/s)"], bc_type = "natural")
def flowin(p, T, time):
    return float(inletflow_f(time))

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
                    init_pressure = 1000,
                    final_time = 175
    )


simulation = pts.generate_simulation(storage_tank = storage_tank, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params,
                                      simulation_type = "Default")

results = simulation.run()
results.to_csv("ANGsim.csv")


# fig, ax = plt.subplots(2, sharex = True)
# ax[0].set_xlabel("Time (s)")
# ax[0].set_ylabel("Pressure (MPa)")


# ax[0].scatter(tankpres["t (s)"], tankpres["P (Pa)"] * 1E-6, label = "Experiment", color = "r")
# ax[0].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6, 
#         label = "pytanksim", color ="b")
# ax[0].legend()

# ax[1].set_xlabel("t (s)")
# ax[1].set_ylabel("T (K)")
# ax[1].scatter(tanktemp["t (s)"], tanktemp["T (K)"], label = "Experiment", color = "r")
# ax[1].plot(results.results_df["time"], results.results_df["temperature"], 
#               label = "pytanksim", color="b")

# plt.savefig("ANG-Validation.svg")



