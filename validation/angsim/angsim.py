# -*- coding: utf-8 -*-
"""Copyright 2024 Muhammad Irfan Maulana Kusdhany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import pytanksim as pts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import CoolProp as CP
import matplotlib.transforms as mtransforms


stored_fluid = pts.StoredFluid(fluid_name="Methane",
                               EOS="HEOS")
plt.style.use(["science", "nature"])

adsorptiondata = pd.read_csv("RGM1-303K.csv")
MW = stored_fluid.backend.molar_mass()
adsorptiondata["Loading (mol/kg)"] = adsorptiondata["Loading (kg/kg)"] / MW
excesslist = [pts.ExcessIsotherm(adsorbate="Methane",
                                 temperature=303,
                                 sorbent="RGM1",
                                 loading=adsorptiondata["Loading (mol/kg)"],
                                 pressure=adsorptiondata["P (Pa)"])]

model_isotherm_da = pts.classes.DAModel.from_ExcessIsotherms(
                                        excesslist,
                                        sorbent="RGM1",
                                        va_mode="Excess",
                                        k_mode="Constant",
                                        rhoa_mode="Ozawa",
                                        f0_mode="Dubinin")

pressure = np.linspace(100, 35E5, 50)
da_result = []
for index, pres in enumerate(pressure):
    da_result.append(model_isotherm_da.n_excess(pres, 303))
plt.figure(figsize=(3.543, 2 * 3.543 / 3))
plt.xlabel("P (MPa)")
plt.ylabel("Excess CH$_4$ (mol/kg)")
plt.plot(pressure * 1E-6, da_result, color="#b96f74", label="DA Fit")
plt.scatter(excesslist[0].pressure * 1E-6, excesslist[0].loading,
            label="Experimental", color="#b96f74")
plt.legend()
plt.savefig("DAFit.jpeg", format="jpeg", dpi=1000, bbox_inches="tight")

tankvol = 1.82 * 1E-3
surface_area = np.pi * 0.1116 * 0.202 + 2 * np.pi * ((0.1116/2)**2) - \
    np.pi * (0.003175**2) + np.pi * 0.031 * 0.03
al_volume = np.pi * ((0.1116/2)**2) * 0.202 - np.pi * ((0.1066/2)**2) * 0.197 +\
    np.pi * ((0.031/2)**2) * 0.03 - np.pi * ((0.026/2)**2) * 0.0275
al_density = 2700
al_mass = al_density * al_volume
hc = 5
rhopack = 500
mads = rhopack * tankvol
rhoskel = mads/(0.35 * tankvol)

sorbent_material = pts.SorbentMaterial(model_isotherm=model_isotherm_da,
                                       skeletal_density=rhoskel,
                                       bulk_density=rhopack,
                                       mass=mads,
                                       specific_surface_area=1308)


storage_tank = pts.SorbentTank(
                    volume=tankvol,
                    vent_pressure=50E5,
                    min_supply_pressure=1E5,
                    sorbent_material=sorbent_material,
                    surface_area=surface_area,
                    heat_transfer_coefficient=hc
    )

vol_flow_out = 5E-3 / 60
stored_fluid.backend.update(CP.PT_INPUTS, 1E5, 273.15)
rhomass = stored_fluid.backend.rhomass()

flowout = vol_flow_out * rhomass

boundary_flux = pts.BoundaryFlux(
                mass_flow_out=flowout,
                environment_temp=300
    )

simulation_params = pts.SimParams(
                    init_time=0,
                    init_temperature=300.5,
                    inserted_amount=0,
                    init_pressure=36E5,
                    final_time=1000
    )


simulation = pts.generate_simulation(storage_tank=storage_tank,
                                     boundary_flux=boundary_flux,
                                     simulation_params=simulation_params,
                                     simulation_type="Default")

results = simulation.run()
results.to_csv("ANGsim.csv")

validdata = pd.read_csv("ANGsimvalid.csv")

fig, ax = plt.subplots(2,  figsize=((3.543, 5/4*3.543)))
ax[0].set_ylabel("Pressure (MPa)")

ax[0].set_xlabel("Time (s)")
ax[0].scatter(validdata["t (s)"], validdata["P (Pa)"] * 1E-6,
              label="Experiment",
              color="#DC3220")
ax[0].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6,
           label="pytanksim", color="#005AB5")
ax[0].legend()

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("T (K)")
ax[1].scatter(validdata["t (s)"], validdata["T (K)"], label="Experiment",
              color="#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["temperature"],
           label="pytanksim", color="#005AB5")

for ind, ax in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, 3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif', weight="bold")

plt.savefig("ANG-Validation.jpeg", format="jpeg", dpi=1000,
            bbox_inches="tight")
