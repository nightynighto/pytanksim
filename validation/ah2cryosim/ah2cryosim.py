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
import itertools
import scienceplots
import matplotlib.transforms as mtransforms

stored_fluid = pts.StoredFluid(fluid_name="Hydrogen",
                               EOS="HEOS")

temperatures = [30, 35, 40, 45, 60, 77, 93, 113, 153, 213, 298]
excesslist = []
for i, temper in enumerate(temperatures):
    filename = "AX21-" + str(temper) + ".csv"
    excesslist.append(pts.ExcessIsotherm.from_csv(filename=filename,
                                                  adsorbate="Hydrogen",
                                                  sorbent="AX21",
                                                  temperature=temper))


model_isotherm_mda = pts.classes.MDAModel.from_ExcessIsotherms(
                                        excesslist,
                                        sorbent="AX21",
                                        stored_fluid=stored_fluid,
                                        m_mode="Constant",
                                        verbose=True)

rhoskel = 2300
rhopack = 269
mads = 0.671

sorbent_material = pts.SorbentMaterial(model_isotherm=model_isotherm_mda,
                                       skeletal_density=rhoskel,
                                       bulk_density=rhopack,
                                       mass=mads,
                                       specific_surface_area=3000)

tankvol = 0.0024946

def smoothstep(x, xmin=0, xmax=1):
    x = np.clip((x-xmin)/(xmax-xmin), 0, 1)
    return - 20 * (x**7) + 70 * (x**6) - 84 * (x**5) + 35 * (x**4)

def h_e(p: float, T: float, time: float, env_temp: float) -> float:
    return 37  - 22 * smoothstep(time, 1620, 1620.5)

storage_tank = pts.SorbentTank(
                    volume=tankvol,
                    aluminum_mass=0,
                    carbon_fiber_mass=0,
                    steel_mass=3.5,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    sorbent_material=sorbent_material,
                    surface_area=0.1277,
                    heat_transfer_coefficient=h_e
    )


def mfin(p: float, T: float, time: float) -> float:
    return 0.024E-3 - 0.024E-3 * smoothstep(time, 1620, 1620.5)

def pin(p: float, T: float, time: float) -> float:
    return p


boundary_flux = pts.BoundaryFlux(
                mass_flow_in=mfin,
                temperature_in=295,
                pressure_in=pin,
                environment_temp=82
    )

simulation_params = pts.SimParams(
                    init_time=0,
                    init_temperature=82,
                    init_pressure=0.14E6,
                    final_time=4800,
                    verbose=False,
                    displayed_points=3600
    )


simulation = pts.generate_simulation(storage_tank=storage_tank,
                                     boundary_flux=boundary_flux,
                                     simulation_params=simulation_params)

results = simulation.run()
results.to_csv("AH2Cryosim.csv")

plt.style.use(["science", "nature"])
small = 8.5
medium = 9
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Helvetica'
})
plt.rc('font', size=medium)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=small)

results = pts.SimResults.from_csv("AH2Cryosim.csv")

fig, ax = plt.subplots(3, figsize=((3.543, 7.5/4*3.543)))

for ind, axis in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, -5/72, fig.dpi_scale_trans)
    axis.text(0.0, 1.0, label, transform=axis.transAxes + trans,
              fontsize='medium', va='bottom', fontfamily='serif',
              weight="bold")

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

symbols = itertools.cycle(["o", "^", "D", "s", "p", "P", "X",
                           "*", "v", "1", (6, 2, 0)])
ax[0].set_xlim(0, 7)
ax[0].set_ylim(0, 60)
ax[0].set_xlabel("P (MPa)")
ax[0].set_ylabel("Excess H$_2$ (mol/kg)")
for i, temper in enumerate(temperatures):
    pressure = np.linspace(100, 70E5, 300)
    mda_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    c = next(palette)
    ax[0].plot(pressure * 1E-6, mda_result, color=c)
    ax[0].scatter(excesslist[i].pressure * 1E-6, excesslist[i].loading,
                  label=str(temper)+" K", color=c, marker=next(symbols))
ax[0].legend(ncol=3, columnspacing=0.1, labelspacing=0.3, handletextpad=0.1)

ax[1].set_ylabel("Pressure (MPa)")
ax[1].set_xlabel("Time (min)")

testpres = pd.read_csv("cryo_pres.csv", skiprows=6, header=None)
testtemp = pd.read_csv("cryo_temp.csv", skiprows=6, header=None)


ax[1].scatter(testpres[0]/60, testpres[1],
              label="Experiment",
              color="#DC3220")
ax[1].plot(results.results_df["time"]/60, results.results_df["pressure"] *
           1E-6, label="pytanksim", color="#005AB5")
ax[1].legend()

ax[2].set_xlabel("Time (min)")
ax[2].set_ylabel("Temperature (K)")
ax[2].set_ylim(77, 110)
ax[2].scatter(testtemp[0]/60, testtemp[1], label="Experiment",
              color="#DC3220")
ax[2].plot(results.results_df["time"]/60, results.results_df["temperature"],
           label="pytanksim", color="#005AB5")
plt.tight_layout()
plt.savefig("AH2Cryo-Validation.jpeg", format="jpeg", dpi=1000)
