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
import itertools
import scienceplots
import matplotlib.transforms as mtransforms

plt.style.use(["science", "nature"])


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
plt.figure(figsize=(3.543, 2/3*3.543))
plt.xlim(0, 7)
plt.ylim(0, 60)
plt.xlabel("P (MPa)")
plt.ylabel("Excess H$_2$ (mol/kg)")
for i, temper in enumerate(temperatures):
    pressure = np.linspace(100, 70E5, 300)
    mda_result = []
    da_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    c = next(palette)
    plt.plot(pressure * 1E-6, mda_result, color=c)
    plt.scatter(excesslist[i].pressure * 1E-6, excesslist[i].loading,
                label=str(temper)+"K", color=c, marker=next(symbols))
plt.legend(ncol=3)
plt.savefig("MDAFit.jpeg", format="jpeg", dpi=1000, bbox_inches="tight")

tankvol = 0.0024946
rhoskel = 2300
rhopack = 269
mads = 0.671

sorbent_material = pts.SorbentMaterial(model_isotherm=model_isotherm_mda,
                                       skeletal_density=rhoskel,
                                       bulk_density=rhopack,
                                       mass=mads,
                                       specific_surface_area=2800)


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

results = pts.SimResults.from_csv("AH2sim.csv")

fig, ax = plt.subplots(2, figsize=((3.543, 5/4*3.543)))

for ind, axis in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, -5/72, fig.dpi_scale_trans)
    axis.text(0.0, 1.0, label, transform=axis.transAxes + trans,
              fontsize='medium', va='bottom', fontfamily='serif',
              weight="bold")

ax[0].set_ylabel("Pressure (MPa)")
ax[0].set_xlabel("Time (s)")

test20pres = pd.read_csv("test20-pres.csv")
test20temp = pd.read_csv("test20-temp.csv")


ax[0].scatter(test20pres["t (s)"], test20pres["P (Pa)"] * 1E-6,
              label="Experiment",
              color="#DC3220")
ax[0].plot(results.results_df["time"], results.results_df["pressure"] * 1E-6,
           label="pytanksim", color="#005AB5")
ax[0].legend()

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Temperature (K)")
ax[1].scatter(test20temp["t (s)"], test20temp["T (K)"], label="Experiment",
              color="#DC3220")
ax[1].plot(results.results_df["time"], results.results_df["temperature"],
           label="pytanksim", color="#005AB5")
plt.tight_layout()
plt.savefig("AH2-Validation.jpeg", format="jpeg", dpi=1000)
