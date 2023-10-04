# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:15:18 2023

@author: nextf
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:52:49 2023

@author: nextf
"""

import pytanksim as pts
import pandas as pd
import matplotlib.pyplot as plt



stored_fluid = pts.StoredFluid(fluid_name = "Hydrogen",
                                EOS = "HEOS")


tankvol = 0.151



storage_tank_fluid = pts.StorageTank(volume = tankvol, aluminum_mass = 12,
                                     carbon_fiber_mass = 49,
                                     stored_fluid = stored_fluid,
                                     steel_mass = 0, max_pressure = 350E5,
                                     vent_pressure = 350E5,
                                     min_supply_pressure = 100)




def Pin(p, T, time):
    return p * 1.25


mfin = 0.013


boundary_flux = pts.BoundaryFlux(
                mass_flow_in = mfin,
                mass_flow_out = 0.0,
                pressure_in = Pin,
                temperature_in = 20.3689
    )

simulation_params = pts.SimParams(
                    init_time = 0,
                    init_temperature = 50,
                    init_pressure = 8E5,
                    target_pres = 350E5,
                    target_temp = 20,
                    final_time = 800,
                    displayed_points = 100
    )






simulation_fluid = pts.generate_simulation(storage_tank = storage_tank_fluid, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params,
                                      simulation_type = "Default")

results_1 = simulation_fluid.run()

p2 = pts.SimParams.from_SimResults(results_1)

simulation_fluid = pts.generate_simulation(storage_tank = storage_tank_fluid, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = p2,
                                      simulation_type = "Default",
                                      phase = 2)

results_2 = simulation_fluid.run()

p3 = pts.SimParams.from_SimResults(results_2)

simulation_fluid = pts.generate_simulation(storage_tank = storage_tank_fluid, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = p3,
                                      simulation_type = "Default",
                                      phase = 1)

results_3 = simulation_fluid.run()

combined_results = pts.SimResults.combine([results_1, results_2, results_3])
combined_results.to_csv("SLH2sim.csv")
combined_results.plot("t",["p","T"])



fig, ax = plt.subplots(2, sharex = True)
fig.set_figwidth(3.543)

ax[0].set_ylabel("Pressure (MPa)")

validpres = pd.read_csv("SLH2valid-pres.csv")
validtemp = pd.read_csv("SLH2valid-temp.csv")


ax[0].plot(validpres["Moles"], validpres["P (Pa)"] * 1E-6, label = "ANL Simulation",
           color = "#DC3220", linestyle = "-.")
ax[0].plot(combined_results.results_df["inserted_amount"], 
           combined_results.results_df["pressure"] * 1E-6, 
        label = "pytanksim", color ="#005AB5", alpha = 0.5)
ax[0].legend()

ax[1].set_xlabel("Inserted Amount (Moles)")
ax[1].set_ylabel("T (K)")
ax[1].plot(validtemp["Moles"], validtemp["T (K)"], label = "ANL Simulation",
           color = "#DC3220", linestyle = "-.")
ax[1].plot(combined_results.results_df["inserted_amount"], combined_results.results_df["temperature"], 
              label = "pytanksim", color="#005AB5", alpha = 0.5)

plt.savefig("SLH2-Validation.jpeg", format = "jpeg", dpi = 1000, bbox_inches = "tight")



