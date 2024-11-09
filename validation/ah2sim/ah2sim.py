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
                                        verbose=False)

rhoskel = 2300
rhopack = 269
mads = 0.671

sorbent_material = pts.SorbentMaterial(model_isotherm=model_isotherm_mda,
                                       skeletal_density=rhoskel,
                                       bulk_density=rhopack,
                                       mass=mads,
                                       specific_surface_area=2800)

tankvol = 0.0024946

storage_tank = pts.SorbentTank(
                    volume=tankvol,
                    aluminum_mass=0,
                    carbon_fiber_mass=0,
                    steel_mass=3.714,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    sorbent_material=sorbent_material,
                    surface_area=0.1277,
                    heat_transfer_coefficient=28
    )


def smoothstep(x, xmin=0, xmax=1):
    x = np.clip((x-xmin)/(xmax-xmin), 0, 1)
    return - 20 * (x**7) + 70 * (x**6) - 84 * (x**5) + 35 * (x**4)

def mfin(p: float, T: float, time: float) -> float:
    return 2.048E-5 - 2.048E-5 * smoothstep(time, 952.5, 953)


def mfout(p: float, T: float, time: float) -> float:
    return 2.186E-5 * smoothstep(time, 3821.5, 3822)


entin = 3928600 * stored_fluid.backend.molar_mass()
entout = 3946400 * stored_fluid.backend.molar_mass()

boundary_flux = pts.BoundaryFlux(
                mass_flow_in=mfin,
                mass_flow_out=mfout,
                environment_temp=302.5,
                enthalpy_in=entin,
                enthalpy_out=entout
    )

simulation_params = pts.SimParams(
                    init_time=0,
                    init_temperature=302.4,
                    init_pressure=32E3,
                    final_time=4694,
                    verbose=False
    )


simulation = pts.generate_simulation(storage_tank=storage_tank,
                                     boundary_flux=boundary_flux,
                                     simulation_params=simulation_params)

results = simulation.run()
results.to_csv("AH2sim.csv")

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

results = pts.SimResults.from_csv("AH2sim.csv")

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
ax[1].set_xlabel("Time (s)")

test20pres = pd.read_csv("test20-pres.csv")
test20temp = pd.read_csv("test20-temp.csv")


ax[1].scatter(test20pres["t (s)"], test20pres["P (Pa)"] * 1E-6,
              label="Experiment",
              color="#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6,
           label="pytanksim", color="#005AB5")
ax[1].legend()

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Temperature (K)")
ax[2].scatter(test20temp["t (s)"], test20temp["T (K)"], label="Experiment",
              color="#DC3220")
ax[2].plot(results.results_df["time"], results.results_df["temperature"],
           label="pytanksim", color="#005AB5")
plt.tight_layout()
plt.savefig("AH2-Validation.jpeg", format="jpeg", dpi=1000)
