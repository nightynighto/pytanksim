# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:52:49 2023

@author: nextf
"""

import pytanksim as pts
import numpy as np
from scipy.special import comb
import pandas as pd
import matplotlib.pyplot as plt
import itertools



stored_fluid = pts.StoredFluid(fluid_name = "Hydrogen",
                                EOS = "HEOS")

temperatures = [30, 35, 40, 45, 60, 77, 93, 113, 153, 213, 298]
excesslist = []
for i, temper in enumerate(temperatures):
    filename = "AX21-" + str(temper) + ".csv"
    excesslist.append( pts.ExcessIsotherm.from_csv(filename = filename, 
                                                adsorbate = "Hydrogen",
                                                sorbent = "AX21",
                                                temperature = temper) )


model_isotherm_mda = pts.classes.MDAModel.from_ExcessIsotherms(excesslist, sorbent = "AX21",
                                                                stored_fluid = stored_fluid,
                                                                m_mode = "Constant")

palette = itertools.cycle(["#8e463a",
"#71b54a",
"#9349cd",
"#c59847",
"#7b8cd1",
"#d14f38",
"#56a5a1",
"#cc4d97",
"#56733f",
"#593e78",
"#c07685"])

symbols = itertools.cycle(["o", "^", "D", "s", "p", "P", "X", "*", "v", "1", (6, 2,0)])
plt.figure(figsize = (5.51181, 2 * 5.51181 /3))
plt.xlim(0, 8)
plt.xlabel("P (MPa)")
plt.ylabel("Excess H$_2$ (mol/kg)")
for i, temper in enumerate(temperatures):
    # pressure = excesslist[i].pressure
    pressure = np.linspace(100, 100E5, 300)
    mda_result = []
    da_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    c = next(palette)
    plt.plot(pressure * 1E-6, mda_result, color = c)
    plt.scatter(excesslist[i].pressure * 1E-6, excesslist[i].loading,
                label =str(temper)+"K", color = c, marker = next(symbols))
plt.legend(loc = "right")
plt.savefig("MDAFit.jpeg", format = "jpeg", dpi = 1000, bbox_inches = "tight")

tankvol = 0.0024946
rhoskel = 2300
rhopack = 269
mads = 0.671

sorbent_material = pts.SorbentMaterial(model_isotherm = model_isotherm_mda,
                                        skeletal_density = rhoskel,
                                        bulk_density = rhopack,
                                        mass = mads,
                                        specific_surface_area = 2800)


storage_tank = pts.SorbentTank(
                    volume = tankvol,
                    aluminum_mass = 0,
                    carbon_fiber_mass = 0,
                    steel_mass = 3.714,
                    vent_pressure = 10E6,
                    min_supply_pressure = 0,
                    sorbent_material = sorbent_material,
                    surface_area = 0.1277,
                    thermal_resistance = 1/(28 * 0.1277)
    )






def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

def mfin(p, T, time):
    return  2.048E-5 - 2.048E-5 * smoothstep(time, 952.5, 953, 6 )
    
def mfout(p, T, time):
    return 2.186E-5 * smoothstep(time, 3821.5, 3822, 6)

entin = 3928600 * stored_fluid.backend.molar_mass()
entout =  3946400 * stored_fluid.backend.molar_mass()

boundary_flux = pts.BoundaryFlux(
                mass_flow_in = mfin,
                mass_flow_out = mfout,
                environment_temp = 302.5,
                enthalpy_in = entin,
                enthalpy_out = entout, 
    )

simulation_params = pts.SimParams(
                    init_time = 0,
                    init_temperature = 302.4,
                    inserted_amount = 0,
                    init_pressure = 32E3,
                    final_time = 4694
    )


simulation = pts.generate_simulation(storage_tank = storage_tank, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params)

results = simulation.run()
results.to_csv("AH2sim.csv")

fig, ax = plt.subplots(2, sharex = True)
fig.set_figwidth(3.543)

ax[0].set_ylabel("Pressure (MPa)")

test20pres = pd.read_csv("test20-pres.csv")
test20temp = pd.read_csv("test20-temp.csv")


ax[0].scatter(test20pres["t (s)"], test20pres["P (Pa)"] * 1E-6, label = "Experiment",
              color = "#DC3220")
ax[0].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6, 
        label = "pytanksim", color ="#005AB5")
ax[0].legend()

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("T (K)")
ax[1].scatter(test20temp["t (s)"], test20temp["T (K)"], label = "Experiment",
              color = "#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["temperature"], 
              label = "pytanksim", color="#005AB5")

plt.savefig("AH2-Validation.jpeg", format = "jpeg", dpi = 1000, bbox_inches = "tight")

