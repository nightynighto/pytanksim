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
import matplotlib.pyplot as plt
import CoolProp as CP
import numpy as np
import scipy as sp
import scienceplots
import matplotlib.transforms as mtransforms

cng_capacity_mass = 2.74877 * 75
cng_density = CP.CoolProp.PropsSI("D", 'T', 300, 'P', 24.82113E6,
                                  'NaturalGasSample.mix')
tankvol = cng_capacity_mass/cng_density

mix = CP.AbstractState("HEOS", 'NaturalGasSample.mix')
mix.update(CP.PT_INPUTS, 24.82113E6, 300)
cp = mix.cpmass()
cv = mix.cvmass()
rho = mix.rhomass()
mu = mix.viscosity()
gam = cp/cv
crit_rat = (2/(gam+1))**(gam/(gam-1))
mdot = 5 * 2.567/60
D = 6


def residual(d):
    # Calculate diameter of orifice to allow CNG refueling rate similar to
    # conventional refueling stations.
    Re = 4*mdot/(np.pi*mu*D*1E-3)
    beta = d/D
    fact = (19000*beta/Re)**0.8
    C = 0.5961 + 0.0261 * (beta**2) - 0.216 * (beta ** 8) + \
        0.000521*((10E6*beta/Re)**0.7) + (0.0188 + 0.0063 * fact) *\
        (beta**3.5) * \
        max([(10E6/Re)**0.3, 22.7-4700*(Re/10E6)]) \
        + (0.043+0.08-0.123) * (1-0.11*fact) * (beta**4)/(1-beta**4) \
        + 0.011 * (0.75-beta) * max([2.8-D/25.4, 0])
    pr = 24.82113E6
    A = np.pi*((d*1E-3/2)**2)
    Cd = C / (np.sqrt(1-(beta**4)))
    mdotcalc = Cd * A * np.sqrt(gam * pr * rho *
                                ((2/(gam+1))**((gam+1)/(gam-1))))
    return mdotcalc-mdot


d = sp.optimize.root_scalar(residual, bracket=[1E-1, 5]).root
beta = d/D

fluid_name_list = ["Hydrogen&Methane"] * 5 + ["Methane", "Hydrogen"]
h2_comp_list = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
ch4_comp_list = 1 - h2_comp_list

for ind, name in enumerate(fluid_name_list):
    fracs = [h2_comp_list[ind], ch4_comp_list[ind]] \
        if name == "Hydrogen&Methane" else None
    fluid = pts.StoredFluid(fluid_name=name,
                            EOS="HEOS",
                            mole_fractions=fracs)
    fluid.backend.specify_phase(CP.iphase_gas)
    pr = 24.82113E6
    Tr = 300
    A = np.pi * ((d * 1E-3 / 2)**2)
    fluid.backend.update(CP.PT_INPUTS, pr, Tr)
    cp = fluid.backend.cpmass()
    cv = fluid.backend.cvmass()
    rho = fluid.backend.rhomass()
    mu = fluid.backend.viscosity()
    gam = cp/cv
    ent = fluid.backend.hmolar()
    crit_rat = (2/(gam+1))**(gam/(gam-1))
    fluid = pts.StoredFluid(fluid_name=name,
                            EOS="PR",
                            mole_fractions=fracs)
    fluid.backend.specify_phase(CP.iphase_gas)
    tank = pts.StorageTank(volume=tankvol,
                           stored_fluid=fluid,
                           vent_pressure=24.82113E6,
                           min_supply_pressure=100)

    def mass_flow_in(p, T, t):     
        def C(mdot):
            # Calculate the coefficient of discharge
            Re = 4*mdot/(np.pi*mu*D*1E-3)
            if Re == 0:
                return 0.5961
            else:
                fact = (19000*beta/Re)**0.8
                return 0.5961 + 0.0261 * (beta**2) - 0.216 * (beta ** 8) + \
                    0.000521*((10E6*beta/Re)**0.7) + (0.0188 + 0.0063 * fact) *\
                    (beta**3.5) * max([(10E6/Re)**0.3, 22.7-4700*(Re/10E6)]) \
                    + (0.043+0.08-0.123) * (1-0.11*fact) *\
                    (beta**4)/(1-beta**4) + 0.011 * (0.75-beta) * \
                    max([2.8-D/25.4, 0])

        def mdot(C, p):
            # Calculate mass flow rate based on the coefficient of discharge
            Cd = C / np.sqrt(1-beta**4)
            if p > pr:
                p = pr
            if p/pr <= crit_rat:
                return Cd * A * np.sqrt(gam * pr * rho *
                                        ((2/(gam+1))**((gam+1)/(gam-1))))
            else:
                return Cd * A * np.sqrt(2 * pr * rho * (gam/(gam-1)) *
                                        ((p/pr)**(2/gam)-(p/pr)**(
                                            (gam+1)/gam)))

        def root(mdottry, p):
            Ctry = C(mdottry)
            mcalc = mdot(Ctry, p)
            return mdottry-mcalc
        # Calculate the mass flow rate recursively
        return sp.optimize.root_scalar(root, p, bracket=[0, 3]).root

    boundary_flux = pts.BoundaryFlux(
        mass_flow_in=mass_flow_in,
        enthalpy_in=ent
        )

    simulation_params = pts.SimParams(init_temperature=300,
                                      init_pressure=101325,
                                      final_time=60*30)

    simulation_fluid = pts.generate_simulation(
        storage_tank=tank,
        boundary_flux=boundary_flux,
        simulation_params=simulation_params,
        simulation_type="Default")

    results = simulation_fluid.run()
    filename = name+".csv" if name != "Hydrogen&Methane" else \
        str(h2_comp_list[ind])+"H2"+str(ch4_comp_list[ind])+"CH4.csv"
    results.to_csv(filename)

plt.style.use(["science", "nature"])

fig1, ax1 = plt.subplots(2, figsize=(3.543, 3.543 * 4/3))
fig2, ax2 = plt.subplots(3, figsize=(3.543, 3.543 * 2))

for ind, name in enumerate(fluid_name_list):
    filename = name+".csv" if name != "Hydrogen&Methane" else \
        str(h2_comp_list[ind])+"H2"+str(ch4_comp_list[ind])+"CH4.csv"
    results, tank, params = pts.SimResults.from_csv(filename, True)
    df = results.df
    labelname = name if name != "Hydrogen&Methane" else \
        str(h2_comp_list[ind]) + " H$_2$ + " + str(ch4_comp_list[ind]) + \
        " CH$_4$"
    if name == "Hydrogen":
        eneden = 33.3
    elif name == "Methane":
        eneden = 13.9
    else:
        eneden = h2_comp_list[ind]*33.3 + ch4_comp_list[ind] * 13.9
    ax1[0].plot(df["t"], df["p"]/1E6, label=labelname)
    ax1[1].plot(df["t"], df["T"], label=labelname)
    ax2[0].plot(df["t"], df["min_dot"], label=labelname)
    ax2[1].plot(df["t"], df["mg"]+df["ms"], label=labelname)
    ax2[2].plot(df["t"], eneden*(df["mg"]+df["ms"])/1000, label=labelname)

for ind, ax in enumerate(ax1):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, 3/72, fig1.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif', weight="bold")

for ind, ax in enumerate(ax2):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, 3/72, fig2.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif', weight="bold")

ax1[0].set_ylabel("Pressure (MPa)")
ax1[1].set_ylabel("Temperature (K)")
ax1[0].set_xlabel("Time (s)")
ax1[1].set_xlabel("Time (s)")
ax1[0].legend()
ax2[0].set_ylabel("Mass Flow Rate (kg/s)")
ax2[1].set_ylabel("Fuel Stored (kg)")
ax2[2].set_ylabel("Energy Stored (MWh)")
ax2[0].set_xlabel("Time (s)")
ax2[1].set_xlabel("Time (s)")
ax2[2].set_xlabel("Time (s)")
handles, labels = ax2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=3)
fig1.savefig("CH4fig1.jpeg", dpi=1000)
fig2.savefig("CH4fig2.jpeg", dpi=1000)
