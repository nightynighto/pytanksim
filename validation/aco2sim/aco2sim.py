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
import matplotlib.transforms as mtransforms
import CoolProp as CP
from scipy.integrate import simps

stored_fluid = pts.StoredFluid(fluid_name="CO2",
                               EOS="HEOS")

temperatures = [273, 298]
excesslist = []
for i, temper in enumerate(temperatures):
    filename = "CF-" + str(temper) + ".csv"
    excesslist.append(pts.ExcessIsotherm.from_csv(filename=filename,
                                                  adsorbate="CO2",
                                                  sorbent="AX21",
                                                  temperature=temper))

model_isotherm_mda = pts.classes.MDAModel.from_ExcessIsotherms(
                                        excesslist,
                                        sorbent="CF",
                                        stored_fluid=stored_fluid,
                                        f0_mode="Dubinin",
                                        va_mode="Ozawa",
                                        f0guess=6E5)

sorbent_material = pts.SorbentMaterial(model_isotherm=model_isotherm_mda,
                                       skeletal_density=2300,
                                       bulk_density=(1/2300 + 0.00219)**(-1),
                                       specific_surface_area=3452)
MW = stored_fluid.backend.molar_mass()
rho_low = stored_fluid.fluid_property_dict(1E5, 298)["rhof"]
charge_time = 10 * 3600
total_energy = 100000  # kWh
cap = 42.13 * charge_time / MW
mdot_work = cap * MW/(charge_time)

# Define the tank with the desired capacity for the given TSA conditions
lpa_tank = pts.SorbentTank(
                    vent_pressure=1E5,
                    min_supply_pressure=0,
                    sorbent_material=sorbent_material,
                    set_capacity=cap,
                    full_pressure=1E5,
                    full_temperature=308.15,
                    empty_pressure=1E5,
                    empty_temperature=473.15
    )

# Calculate the investment cost for sorbents based on the sorbent mass
sorbent_cost = lpa_tank.sorbent_material.mass * 47.2
sorbent_mass_per_kwh = lpa_tank.sorbent_material.mass/total_energy
sorbent_cost_per_kwh = sorbent_cost/total_energy

# Simulate the discharging of the sorbent tank
bf1 = pts.BoundaryFlux(mass_flow_out=mdot_work)
sp1 = pts.SimParams(init_temperature=308.15,
                    final_time=charge_time,
                    init_pressure=1E5)
sim1 = pts.generate_simulation(lpa_tank, bf1, sp1, simulation_type="Heated")
res1 = sim1.run()
res1.to_csv("sorbent_discharge.csv")
res1 = pts.SimResults.from_csv("sorbent_discharge.csv")

# Simulate the charging of the sorbent tank
bf2 = pts.BoundaryFlux(mass_flow_in=mdot_work,
                       pressure_in=1E5,
                       temperature_in=303.15)
sp2 = pts.SimParams(init_temperature=473.15,
                    final_time=charge_time,
                    init_pressure=1E5)
sim2 = pts.generate_simulation(lpa_tank, bf2, sp2, simulation_type="Cooled")
res2 = sim2.run()
res2.to_csv("sorbent_charge.csv")
res2 = pts.SimResults.from_csv("sorbent_charge.csv")


# Find the maximum heat exchanger area for both charging and discharging
water = pts.StoredFluid(fluid_name="Water", EOS="HEOS")
MW_water = water.backend.molar_mass()
rho_water = MW_water*water.saturation_property_dict(303.15, 0)["rhof"]

def heat_exchanger(T_tank, heat_load, T_water=303.15,
                   p_water=1E5, h=300, pinch=3, eff=0.85):
    pinch = - pinch if T_tank > T_water else pinch
    T_pinch = T_tank + pinch
    try:
        hinit = water.fluid_property_dict(p_water, T_water)["hf"]
    except(ValueError):
        hinit = water.saturation_property_dict(T_water, 1)["hf"]
    try:
        hfinal_ideal = water.fluid_property_dict(p_water, T_pinch)["hf"]
    except(ValueError):
        print(T_pinch)
        hfinal_ideal = water.saturation_property_dict(p_water, T_pinch)["hf"]
    q = (hfinal_ideal-hinit)*eff
    hfinal_actual = q + hinit
    water.backend.update(CP.HmolarP_INPUTS, hfinal_actual, p_water)
    T_actual = water.backend.T()
    pinch_check = T_actual - T_pinch if T_water > T_tank else\
        T_pinch - T_actual
    if pinch_check < 0:
        print("pinch!", T_actual)
        T_actual = T_tank + pinch
        try:
            hfinal_actual = water.fluid_property_dict(p_water, T_actual)["hf"]
        except(ValueError):
            hfinal_actual = water.saturation_property_dict(
                T_water, T_actual)["hf"]
    eff_actual = (hfinal_actual - hinit)/(hfinal_ideal-hinit)
    mfrate = MW_water * heat_load/np.abs(hfinal_actual-hinit)
    area = heat_load/(np.abs(T_water-T_actual)*h)
    cost = 2143 * (area**0.514)
    return eff_actual, mfrate, cost


cost_charge = 0
mfrate_all_charge = np.zeros_like(res2.df["t"])
i = 0
for i in range(len(res2.df["t"])):
    T = res2.df["T"][i]
    heat = res2.df["Qcoolreq_dot"][i]
    eff_actual, mfrate, cost_i = heat_exchanger(T, heat, T_water=299.65)
    if eff_actual > 0.85+0.85*1E-3:
        print(T)
        break
    cost_charge = cost_i if cost_i > cost_charge else cost_charge
    mfrate_all_charge[i] = mfrate

area_charge = (cost_charge/2143)**(1/0.514)
area_charge_per_kwh = area_charge/total_energy
cost_charge_per_kwh = cost_charge/total_energy

mfrate_total_charge = simps(mfrate_all_charge, res2.df["t"])
cooling_tower_max_cap = max(mfrate_all_charge)
ct_max_cap_gpm = 264.172*60*cooling_tower_max_cap/rho_water
cooling_tower_cost = ct_max_cap_gpm * 50 * 1.12 * 1.1 * 1.8
cooling_tower_cost_per_kwh = cooling_tower_cost/total_energy

cost_discharge = 0
mfrate_all_discharge = np.zeros_like(res1.df["t"])
for i in range(len(res1.df["t"])):
    T = res1.df["T"][i]
    heat = res1.df["Qheatreq_dot"][i]
    eff_actual, mfrate, cost_i = heat_exchanger(T, heat, T_water=523.15)
    if eff_actual > 0.85+0.85*1E-3:
        print(T)
        break
    cost_discharge = cost_i if cost_i > cost_discharge else cost_discharge
    mfrate_all_discharge[i] = mfrate

area_discharge = (cost_discharge/2143)**(1/0.514)
area_discharge_per_kwh = area_discharge/total_energy
cost_discharge_per_kwh = cost_discharge/total_energy
mfrate_total_discharge = simps(mfrate_all_discharge, res1.df["t"])

palette = itertools.cycle(["#8e463a", "#71b54a"])
symbols = itertools.cycle(["o", "^"])

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
fig = plt.figure(figsize=(7.4801, 3 / 4 * 7.4801))
ax = []
ax.append(plt.subplot2grid((60, 80), [0, 0], 53, 30))
ax.append(plt.subplot2grid((60, 80), [0, 40], 23, 40))
ax.append(plt.subplot2grid((60, 80), [30, 40], 23, 50))


ax[0].set_xlim(0, 0.5)
ax[0].set_ylim(0, 20)
ax[0].set_xlabel("P (MPa)")
ax[0].set_ylabel("Excess CO$_2$ (mol/kg)")
for i, temper in enumerate(temperatures):
    pressure = np.linspace(100, 5E5, 300)
    mda_result = []
    da_result = []
    for index, pres in enumerate(pressure):
        mda_result.append(model_isotherm_mda.n_excess(pres, temper))
    c = next(palette)
    ax[0].plot(pressure * 1E-6, mda_result, color=c)
    ax[0].scatter(excesslist[i].pressure * 1E-6, excesslist[i].loading,
                  label=str(temper)+"K", color=c, marker=next(symbols), s=20)
ax[0].legend(ncol=3)
twin1 = ax[1].twinx()
twin2 = ax[1].twinx()

p1, = ax[1].plot(res1.df["t"]/3600, res1.df["T"], label="Temperature",
                 color="#e41a1c")
p2, = twin1.plot(res1.df["t"]/3600, res1.df["Qheatreq_dot"]*1E-6,
                 label="Heating Power", color="#377eb8", linestyle="dashed")
p3, = twin2.plot(res1.df["t"]/3600, mfrate_all_discharge, label="Steam Flow",
                 color="#4daf4a", linestyle="dashdot")
ax[1].set_xlabel("Time (h)")
ax[1].set_ylabel("Temperature (K)")
twin1.set_ylabel("Required Heating Power (MW)")
twin1.set_ylim(13, 30)
twin2.set_ylabel("Steam Mass Flow Rate (kg/s)")
twin2.set_ylim(0, 400)
twin2.spines.right.set_position(("axes", 1.15))
ax[1].legend(handles=[p1, p2, p3])
twin3 = ax[2].twinx()
twin4 = ax[2].twinx()
p4, = ax[2].plot(res2.df["t"]/3600, res2.df["T"], label="Temperature",
                 color="#e41a1c")
p5, = twin3.plot(res2.df["t"]/3600, res2.df["Qcoolreq_dot"]*1E-6,
                 label="Cooling Power", color="#377eb8", linestyle="dashed")
p6, = twin4.plot(res1.df["t"]/3600, mfrate_all_charge, label="Water Flow",
                 color="#4daf4a", linestyle="dashdot")
ax[2].set_xlabel("Time (h)")
ax[2].set_ylabel("Temperature (K)")
twin3.set_ylabel("Required Cooling Power (MW)")
twin3.set_ylim(13, 30)
twin4.set_ylabel("Water Mass Flow Rate (kg/s)")
twin4.set_ylim(0, 900)
twin4.spines.right.set_position(("axes", 1.15))
ax[2].legend(handles=[p4, p5, p6], loc="upper center")
for ind, axis in enumerate(ax):
    label = r"\textbf{"+chr(ord('`')+(ind+1))+".)" + "}"
    trans = mtransforms.ScaledTranslation(-30/72, 3/72, fig.dpi_scale_trans)
    axis.text(-0.03, 1.0, label, transform=axis.transAxes + trans,
              fontsize='medium', va='bottom', fontfamily='serif',
              weight="bold")
plt.savefig("aco2valid.jpeg", format="jpeg", dpi=1000)
