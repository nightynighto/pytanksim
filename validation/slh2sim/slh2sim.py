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
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.transforms as mtransforms

stored_fluid = pts.StoredFluid(fluid_name="Hydrogen",
                               EOS="HEOS")

tankvol = 0.151

storage_tank_fluid = pts.StorageTank(volume=tankvol,
                                     aluminum_mass=12,
                                     carbon_fiber_mass=49,
                                     stored_fluid=stored_fluid,
                                     steel_mass=0,
                                     vent_pressure=350E5,
                                     min_supply_pressure=100)


def Pin(p: float, T: float, time: float) -> float:
    return p * 1.25


mfin = 0.013

boundary_flux = pts.BoundaryFlux(
                mass_flow_in=mfin,
                mass_flow_out=0.0,
                pressure_in=Pin,
                temperature_in=20.3689
    )

simulation_params = pts.SimParams(
                    init_time=0,
                    init_temperature=50,
                    init_pressure=8E5,
                    target_pres=60E5,
                    target_temp=20,
                    final_time=800,
                    displayed_points=100,
                    stop_at_target_pressure=True,
                    verbose=False
    )

simulation_results = pts.automatic_simulation(storage_tank_fluid,
                                              boundary_flux,
                                              simulation_params)


simulation_results.to_csv("SLH2sim.csv")
simulation_results.plot("min", ["p", "T"])

plt.style.use(["science","nature"])
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

fig, ax = plt.subplots(3, figsize=(3.543, 7.5/4*3.543))

for ind, axis in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, -5/72, fig.dpi_scale_trans)
    axis.text(0.0, 1.0, label, transform=axis.transAxes + trans,
              fontsize='medium', va='bottom', fontfamily='serif',
              weight="bold")

ax[0].set_ylabel("Pressure (MPa)")
ax[0].set_xlabel("Usable Hydrogen (kg)")

validpres = pd.read_csv("SLH2valid-pres.csv")
validtemp = pd.read_csv("SLH2valid-temp.csv")

MW = stored_fluid.backend.molar_mass()
ax[0].plot(validpres["Moles"] * MW, validpres["P (Pa)"] * 1E-6,
           label="ANL Simulation",
           color="#DC3220", linestyle="-.")
ax[0].plot(simulation_results.df['min'],
           simulation_results.df['p'] * 1E-6,
           label="pytanksim", color="#005AB5", alpha=0.95)
ax[0].legend()

ax[1].set_xlabel("Usable Hydrogen (kg)")
ax[1].set_ylabel("T (K)")

ax[1].plot(validtemp["Moles"]*MW, validtemp["T (K)"], label="ANL Simulation",
           color="#DC3220", linestyle="-.")
ax[1].plot(simulation_results.df['min'],
           simulation_results.df['T'],
           label="pytanksim", color="#005AB5", alpha=0.95)
ax[1].legend()

ax[2].set_xlabel("Usable Hydrogen (kg)")
ax[2].set_ylabel("Hydrogen Mass (kg)")
ax[2].set_ylim(0, 12)

ax[2].plot(simulation_results.df['min'],
           simulation_results.df['mg'],
           label="Gaseous Hydrogen", color="green", linestyle="-.")

ax[2].plot(simulation_results.df['min'],
           simulation_results.df['ml'] + simulation_results.df['ms'],
           label="Liquid Hydrogen", color="blue", linestyle="dashed")
ax[2].legend()

plt.tight_layout()
plt.savefig("SLH2-Validation.jpeg", format="jpeg", dpi=1000)
