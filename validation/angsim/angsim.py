# -*- coding: utf-8 -*-
# This file is a part of the python package pytanksim.
#
# Copyright (c) 2024 Muhammad Irfan Maulana Kusdhany, Kyushu University
#
# pytanksim is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


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
small = 8.5
medium = 9.3
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Helvetica'
})
plt.rc('font', size=medium)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=medium)

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

fig, ax = plt.subplots(3,  figsize=((3.543, 7.5/4*3.543)))

pressure = np.linspace(100, 35E5, 50)
da_result = []
for index, pres in enumerate(pressure):
    da_result.append(model_isotherm_da.n_excess(pres, 303))
ax[0].set_xlabel("P (MPa)")
ax[0].set_ylabel("Excess CH$_4$ (mol/kg)")
ax[0].plot(pressure * 1E-6, da_result, color="#8e463a", label="DA Fit")
ax[0].scatter(excesslist[0].pressure * 1E-6, excesslist[0].loading,
            label="Experiment", color="#8e463a")
ax[0].legend()
ax[1].set_ylabel("Pressure (MPa)")

ax[1].set_xlabel("Time (s)")
ax[1].scatter(validdata["t (s)"], validdata["P (Pa)"] * 1E-6,
              label="Experiment",
              color="#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6,
           label="pytanksim", color="#005AB5")
ax[1].legend()

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("T (K)")
ax[2].scatter(validdata["t (s)"], validdata["T (K)"], label="Experiment",
              color="#DC3220")
ax[2].plot(results.results_df["time"], results.results_df["temperature"],
           label="pytanksim", color="#005AB5")

for ind, ax in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, 3/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif', weight="bold")

plt.tight_layout()
plt.savefig("ANG-Validation.jpeg", format="jpeg", dpi=1000,
            bbox_inches="tight")
