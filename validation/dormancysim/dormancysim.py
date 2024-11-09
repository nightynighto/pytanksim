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
import matplotlib.pyplot as plt
import itertools
import scienceplots

stored_fluid = pts.StoredFluid(fluid_name="Hydrogen",
                               EOS="HEOS")

temperatures_ax = [30, 35, 40, 45, 60, 77, 93, 113, 153, 213, 298]
excesslist = []
for i, temper in enumerate(temperatures_ax):
    filename = "AX21-" + str(temper) + ".csv"
    excesslist.append(pts.ExcessIsotherm.from_csv(filename=filename,
                                                  adsorbate="Hydrogen",
                                                  sorbent="AX21",
                                                  temperature=temper))

temperatures_mof = [30, 40, 50, 60, 77, 100, 125, 200, 300]
excesslist_mof = []
for i, temper in enumerate(temperatures_mof):
    filename = "MOF5-" + str(temper) + ".csv"
    excesslist_mof.append(pts.ExcessIsotherm.from_csv(filename=filename,
                                                      adsorbate="Hydrogen",
                                                      sorbent="AX21",
                                                      temperature=temper))

temperatures_mofp = [77, 103, 132, 295]
excesslist_mofp = []
for i, temper in enumerate(temperatures_mofp):
    filename = "MOF5P-" + str(temper) + ".csv"
    excesslist_mofp.append(pts.ExcessIsotherm.from_csv(filename=filename,
                                                       adsorbate="Hydrogen",
                                                       sorbent="MOF-5 Pellet",
                                                       temperature=temper))

model_isotherm_ax = pts.classes.MDAModel.from_ExcessIsotherms(
                                            excesslist,
                                            sorbent="AX21",
                                            stored_fluid=stored_fluid,
                                            m_mode="Constant")

model_isotherm_mof = pts.classes.MDAModel.from_ExcessIsotherms(
                                        excesslist_mof,
                                        sorbent="MOF-5",
                                        stored_fluid=stored_fluid,
                                        mguess=9)

model_isotherm_mofp = pts.classes.MDAModel.from_ExcessIsotherms(
                                        excesslist_mofp,
                                        sorbent="MOF-5 Pellet",
                                        stored_fluid=stored_fluid,
                                        mguess=9)

tankvol = 0.19
rhopack_ax = 269
rhopack_mof = 130
rhopack_mofp = 520
mads_ax = tankvol * rhopack_ax
mads_mof = tankvol * rhopack_mof
mads_mofp = tankvol * rhopack_mofp


AX21 = pts.SorbentMaterial(model_isotherm=model_isotherm_ax,
                           skeletal_density=2300,
                           bulk_density=rhopack_ax,
                           mass=mads_ax,
                           specific_surface_area=2800)


def cs_mof(T):
    return 1000*(0.524-8.885E-3 * T + 9.624E-5 * (T**2) - 3.469E-7 * (T**3) +
                 4.417E-10*(T**4))


MOF5 = pts.SorbentMaterial(model_isotherm=model_isotherm_mof,
                           skeletal_density=2030,
                           bulk_density=rhopack_mof,
                           mass=mads_mof,
                           specific_surface_area=2762,
                           heat_capacity_function=cs_mof)

MOF5P = pts.SorbentMaterial(model_isotherm=model_isotherm_mofp,
                            skeletal_density=2030,
                            bulk_density=rhopack_mofp,
                            mass=mads_mofp,
                            specific_surface_area=2263,
                            heat_capacity_function=cs_mof)

tank_empty = pts.StorageTank(
                    volume=tankvol,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    stored_fluid=stored_fluid,
    )

tank_ax = pts.SorbentTank(
                    volume=tankvol,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    sorbent_material=AX21,
    )

tank_mof = pts.SorbentTank(
                    volume=tankvol,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    sorbent_material=MOF5,
    )

tank_mofp = pts.SorbentTank(
                    volume=tankvol,
                    vent_pressure=10E6,
                    min_supply_pressure=0,
                    sorbent_material=MOF5P
    )

tank_list = [tank_empty, tank_ax, tank_mof, tank_mofp]
dormancy_usable_list = []
dormancy_5kg_list = []
start_pres_usable = []
start_pres_5kg = []


for tank in tank_list:
    min_cap = tank.capacity(5E5, 130, 0)
    mw = stored_fluid.backend.molar_mass()
    usable_cap = min_cap + 40*tankvol/mw
    max_pres = tank.conditions_at_capacity_temperature(usable_cap,
                                                       80,
                                                       10E5,
                                                       1).x[0]
    start_pres_usable.append(max_pres)
    tank.vent_pressure = max_pres * 1.25
    if isinstance(tank, pts.SorbentTank):
        tankname = tank.sorbent_material.model_isotherm.sorbent
    else:
        tankname = "Fluid"
    print(tankname)
    print("Pressure when full (assuming 40 kg/m3 usable capacity)")
    print(max_pres)
    dormancy_usable = tank.calculate_dormancy(max_pres, 80, 3600*24)
    dormancy_usable_list.append(dormancy_usable)
    dormancy_usable.to_csv(tankname+"-Dormancy-Usable.csv")
    max_pres = tank.conditions_at_capacity_temperature(5/mw, 80, 10E5, 1).x[0]
    start_pres_5kg.append(max_pres)
    tank.vent_pressure = 100E5
    print("Pressure when full (assuming 5 kg total capacity)")
    print(max_pres)
    dormancy_5kg = tank.calculate_dormancy(max_pres, 80, 3600*24)
    dormancy_5kg_list.append(dormancy_5kg)
    dormancy_5kg.to_csv(tankname+"-Dormancy-5kg.csv")

start_pres_usable = np.array(start_pres_usable)
vent_pres_usable = start_pres_usable * 1.25
vent_temp_usable = np.array([float(dorm["final temperature"])
                             for dorm in dormancy_usable_list])
start_pres_5kg = np.array(start_pres_5kg)
vent_temp_5kg = np.array([float(dorm["final temperature"])
                          for dorm in dormancy_5kg_list])

vent_label_list_usable = []
start_label_list_usable = []
vent_label_list_5kg = []
start_label_list_5kg = []
for i in range(len(tank_list)):
    start_label_list_usable.append(
        f"{start_pres_usable[i]/1E5:.0f}"+" bar,\n80 K")
    vent_label_list_usable.append(
        f"{vent_pres_usable[i]/1E5:.0f}" + " bar,\n"
        + f"{vent_temp_usable[i]:.1f}" + " K")
    start_label_list_5kg.append(
        f"{start_pres_5kg[i]/1E5:.0f}" + " bar,\n80 K")
    vent_label_list_5kg.append(
        "100 bar,\n" +
        f"{vent_temp_5kg[i]:.0f}" + " K")

plt.style.use(["science", "nature"])
small = 9
medium = 10
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Helvetica'
})
plt.rc('font', size=medium)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=medium)

fig, ax = plt.subplots(2, 2, figsize=(7.48, 8),
                       gridspec_kw={'height_ratios': [3.5, 5]})

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
ax = ax.flat
symbols = itertools.cycle(["o", "^", "D", "s", "p", "P", "X",
                           "*", "v", "1", (6, 2, 0)])
ax[0].set_xlim(0, 7)
ax[0].set_ylim(0, 60)
ax[0].set_xlabel("P (MPa)")
ax[0].set_ylabel("Excess H$_2$ (mol/kg)")
for i, temper in enumerate(temperatures_mof):
    pressure = np.linspace(100, 70E5, 300)
    mda_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mof.n_excess(pres, temper))
    c = next(palette)
    ax[0].plot(pressure * 1E-6, mda_result, color=c)
    ax[0].scatter(excesslist_mof[i].pressure * 1E-6, excesslist_mof[i].loading,
                  label=str(temper)+" K", color=c, marker=next(symbols), s=20)
ax[0].legend(ncol=3, loc="upper right", columnspacing=0.5,
             handletextpad=0.1)
ax[0].text(0, 1.02, r"\textbf{MOF-5 (Powder)}",
           transform=ax[0].transAxes, fontsize=11, va='bottom', weight="bold")


ax[1].set_xlim(0, 10)
ax[1].set_ylim(0, 30)
ax[1].set_xlabel("P (MPa)")
ax[1].set_ylabel("Excess H$_2$ (mol/kg)")
for i, temper in enumerate(temperatures_mofp):
    pressure = np.linspace(100, 100E5, 300)
    mda_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mofp.n_excess(pres, temper))
    c = next(palette)
    ax[1].plot(pressure * 1E-6, mda_result, color=c)
    ax[1].scatter(excesslist_mofp[i].pressure * 1E-6,
                  excesslist_mofp[i].loading,
                  label=str(temper)+" K", color=c,
                  marker=next(symbols), s=20)
ax[1].legend(ncol=2, loc="upper right", columnspacing=0.5,
             handletextpad=0.1)
ax[1].text(0, 1.02, r"\textbf{MOF-5 (Pellet)}",
           transform=ax[1].transAxes, va='bottom',
           fontsize=11, weight="bold")
for n, axis in enumerate(ax):
    axis.text(-0.15, 1.02, r"\textbf{"+chr(ord('`')+(n+1))+".)" + "}",
              transform=axis.transAxes, va='bottom',
              fontsize=11, weight="bold")

samplenames = ["CcH2", "AX-21\n(Powder)", "MOF-5\n(Powder)", "MOF-5\n(Pellet)"]
tick_pos = range(len(samplenames))

counts_usable = {
    "$\Delta U_a$": np.array([float(
        dorm["sorbent energy contribution"]/(3600*24)) if
        "sorbent energy contribution" in dorm.columns else float(0)
        for dorm in dormancy_usable_list]),
    "$\Delta U_f$": np.array([float(
        dorm["bulk energy contribution"]/(3600*24)) if
        "sorbent energy contribution" in dorm.columns else
        float((dorm["total energy change"]
               - dorm["solid heat capacity contribution"])
              / (3600 * 24))
        for dorm in dormancy_usable_list]),
    "$\Delta U_s$": np.array([float(
        dorm["solid heat capacity contribution"]/(3600*24))
                              for dorm in dormancy_usable_list])
    }

counts_5kg = {
    "$\Delta U_a$": np.array([float(
        dorm["sorbent energy contribution"]/(3600*24)) if
        "sorbent energy contribution" in dorm.columns else float(0)
        for dorm in dormancy_5kg_list]),
    "$\Delta U_f$": np.array([float(
        dorm["bulk energy contribution"]/(3600*24)) if
        "sorbent energy contribution" in dorm.columns else
        float((dorm["total energy change"]
               - dorm["solid heat capacity contribution"])/(3600*24))
        for dorm in dormancy_5kg_list]),
    "$\Delta U_s$": np.array([float(
        dorm["solid heat capacity contribution"]/(3600*24))
        for dorm in dormancy_5kg_list])
    }

bottom = np.zeros(4)
for heat, heat_count in counts_5kg.items():
    p = ax[2].bar(samplenames, heat_count, label=heat, bottom=bottom)
    bottom += heat_count
    if heat == "$\Delta U_s$":
        ax[2].bar_label(p, vent_label_list_5kg, padding=9)
for i, label in enumerate(start_label_list_5kg):
    col = "black" if i == 0 else "white"
    ax[2].annotate(label, (i, 0.15), ha="center", va="bottom", color=col)

ax[2].set_ylim(0, 60)
ax[2].set_ylabel("Dormancy (watt days)")
ax[2].set_xticks(tick_pos)
ax[2].set_xticklabels(samplenames)
ax[2].legend(loc='upper left')
ax[2].text(0, 1.02, r"\textbf{5 kg Total Hydrogen in Tank}",
           transform=ax[2].transAxes, va='bottom',
           fontsize=11, weight="bold")

bottom = np.zeros(4)
for heat, heat_count in counts_usable.items():
    p = ax[3].bar(samplenames, heat_count, label=heat, bottom=bottom)
    bottom += heat_count
    if heat == "$\Delta U_s$":
        ax[3].bar_label(p, vent_label_list_usable, padding=9)

for i, label in enumerate(start_label_list_usable):
    col = "black" if i == 0 else "white"
    ax[3].annotate(label, (i, 0.15), ha="center", va="bottom", color=col)
ax[3].set_ylim(0, 14)
ax[3].set_ylabel("Dormancy (watt days)")
ax[3].set_xticks(tick_pos)
ax[3].set_xticklabels(samplenames)
ax[3].legend()
ax[3].text(0, 1.02, r"\textbf{40 kg/$\mathbf{m^3}$ Usable Density w/ "
           "50 K Swing}",
           transform=ax[3].transAxes, va='bottom',
           fontsize=11, weight="bold")

plt.tight_layout()
plt.savefig("dormancysim.jpeg", dpi=1000)
