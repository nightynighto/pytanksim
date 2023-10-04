# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:10:09 2023

@author: nextf
"""


import pytanksim as pts
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import CoolProp as CP

stored_fluid = pts.StoredFluid(fluid_name = "Methane",
                                EOS = "HEOS")


def rhoads(T):
    return 422.62 / (np.exp(2.5 * 1E-3 * (T - 111.2)))

def Ps(T):
    pcr = stored_fluid.backend.p_critical()
    Tcr = stored_fluid.backend.T_critical()
    return pcr*((T/Tcr)**2)

def A(p, T):
    ps = Ps (T)
    return sp.constants.R * T * np.log(ps/p)

def nexc(p, T, q=1):
    bracket = A(p,T)/(25040*0.35)
    q = 3.3 * 1E-4 *  rhoads(T) * np.exp(-(bracket**1.8))
    return q / stored_fluid.backend.molar_mass()


# excesslist =[]  
temperatures = [303]
# pressures = np.linspace(100, 3.5E6, 100)
# for i, temper in enumerate(temperatures):
#     loading = [nexc(p, temper) for p in pressures]
#     excesslist.append( pts.ExcessIsotherm( loading = loading,
#                                           pressure = pressures,
#                                                 adsorbate = "Methane",
#                                                 sorbent = "AC2",
#                                                 temperature = temper) )
adsorptiondata = pd.read_csv("RGM1-303K.csv")
MW = stored_fluid.backend.molar_mass()
adsorptiondata["Loading (mol/kg)"] = adsorptiondata["Loading (kg/kg)"] / MW
excesslist = [pts.ExcessIsotherm(adsorbate = "Methane",
                                  temperature = 303,
                                  sorbent = "RGM1",
                                  loading = adsorptiondata["Loading (mol/kg)"],
                                  pressure = adsorptiondata["P (Pa)"])]




model_isotherm_mda = pts.classes.DAModel.from_ExcessIsotherms(excesslist, 
                                                              sorbent = "RGM1",
                                                              va_mode = "Excess",
                                                              k_mode = "Constant",
                                                                rhoa_mode = "Ozawa",
                                                                f0_mode = "Dubinin")

# model_isotherm_mda = pts.classes.DAModel(
#                                     f0 = 1E5,
#                                     stored_fluid = stored_fluid,
#                                     sorbent = "RGM1",
#                                     va_mode = "Excess",
#                                     w0 = 3.3E-4,
#                                     eps = 25040 * 0.35,
#                                     k = 2,
#                                     m = 1.8,
#                                     rhoa_mode = "Ozawa",
#                                     f0_mode = "Dubinin")

for i, temper in enumerate(temperatures):
    # pressure = excesslist[i].pressure
    pressure = np.linspace(100, 35E5, 50)
    mda_result = []
    da_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    plt.figure(figsize = (3.543, 2 * 3.543 /3))
    plt.xlabel("P (MPa)")
    plt.ylabel("Excess CH$_4$ (mol/kg)")
    plt.plot(pressure * 1E-6, mda_result, color = "#b96f74", label = "DA Fit")
    plt.scatter(excesslist[i].pressure * 1E-6, excesslist[i].loading, 
                label ="Experimental", color = "#b96f74")
    plt.legend()
    plt.savefig("DAFit.jpeg", format = "jpeg", dpi = 1000, bbox_inches = "tight")

tankvol = 1.82 * 1E-3
surface_area =  np.pi * 0.1116 * 0.202 + 2 * np.pi * ((0.1116/2)**2)  - \
    np.pi * (0.003175**2) + np.pi * 0.031 * 0.03
al_volume =  np.pi * ((0.1116/2)**2) * 0.202 - np.pi * ((0.1066/2)**2) * 0.197 +\
    np.pi * ((0.031/2)**2) * 0.03 - np.pi * ((0.026/2)**2) * 0.0275
al_density = 2700
al_mass = al_density * al_volume
                    
hc = 5


rhopack = 500
mads = rhopack * tankvol
rhoskel =  mads / (0.35 * tankvol)

sorbent_material = pts.SorbentMaterial(model_isotherm = model_isotherm_mda,
                                        skeletal_density = rhoskel,
                                        bulk_density = rhopack,
                                        mass = mads,
                                        specific_surface_area = 1308)


storage_tank = pts.SorbentTank(
                    volume = tankvol,
                    max_pressure = float(50E5),
                    vent_pressure = 50E5,
                    min_supply_pressure = 1E5,
                    sorbent_material = sorbent_material,
                    surface_area = surface_area,
                    thermal_resistance = 1/(hc * surface_area)
    )

# pin = 3.5E6
def pin( p, T, time):
    return p
Tin = 300
# def Tin(p, T, time):
#     return T
vol_flow_in = 5E-3 / 60
stored_fluid.backend.update(CP.PT_INPUTS, 1E5, 273.15)
rhomass = stored_fluid.backend.rhomass()

flowin = vol_flow_in * rhomass


boundary_flux = pts.BoundaryFlux(
                # mass_flow_in = flowin,
                mass_flow_out = flowin,
                environment_temp = 300,
                pressure_in = pin,
                temperature_in =Tin
    )

simulation_params = pts.SimParams(
                    init_time = 0,
                    init_temperature = 300.5,
                    inserted_amount = 0,
                    init_pressure = 36E5,
                    final_time = 1000
    )


simulation = pts.generate_simulation(storage_tank = storage_tank, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params,
                                      simulation_type = "Default")

results = simulation.run()
results.to_csv("ANGsim.csv")

validdata = pd.read_csv("ANGsimvalid.csv")

fig, ax = plt.subplots(2, sharex = True)
fig.set_figwidth(3.543)
ax[0].set_ylabel("Pressure (MPa)")


ax[0].scatter(validdata["t (s)"], validdata["P (Pa)"] * 1E-6, label = "Experiment",
              color = "#DC3220")
ax[0].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6, 
        label = "pytanksim", color ="#005AB5")
ax[0].legend()

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("T (K)")
ax[1].scatter(validdata["t (s)"], validdata["T (K)"], label = "Experiment",
              color = "#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["temperature"], 
              label = "pytanksim", color="#005AB5")

plt.savefig("ANG-Validation.jpeg", format = "jpeg", dpi = 1000, bbox_inches = "tight")



