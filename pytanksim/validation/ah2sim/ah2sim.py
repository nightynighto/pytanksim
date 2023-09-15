# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:52:49 2023

@author: nextf
"""

import pytanksim as pts
import numpy as np
from scipy.special import comb



stored_fluid = pts.StoredFluid(fluid_name = "Hydrogen",
                                EOS = "HEOS")

temperatures = [30, 35, 40, 45, 60, 77, 93, 113, 153, 213, 298]
excesslist = []
for i, temper in enumerate(temperatures):
    filename = "AX21-" + str(temper) + ".csv"
    excesslist.append( pts.ExcessIsotherm.from_csv(filename = filename, 
                                                adsorbate = "Hydrogen",
                                                sorbent = "AX21",
                                                temperature = temper) )


model_isotherm_mda = pts.classes.MDAModel.from_ExcessIsotherms(excesslist, sorbent = "AX21",
                                                                stored_fluid = stored_fluid,
                                                                m_mode = "Constant")


tankvol = 0.0024946
rhoskel = 2300
rhopack = 269
mads = 0.671

sorbent_material = pts.SorbentMaterial(model_isotherm = model_isotherm_mda,
                                        skeletal_density = rhoskel,
                                        bulk_density = rhopack,
                                        mass = mads,
                                        specific_surface_area = 2800)


storage_tank = pts.SorbentTank(
                    volume = tankvol,
                    aluminum_mass = 0,
                    carbon_fiber_mass = 0,
                    steel_mass = 3.714,
                    max_pressure = float(35E6),
                    vent_pressure = 10E6,
                    min_supply_pressure = 0,
                    sorbent_material = sorbent_material,
                    surface_area = 0.1277,
                    thermal_resistance = 1/(28 * 0.1277)
    )






def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

def mfin(time):
    return  2.048E-5 - 2.048E-5 * smoothstep(time, 952.5, 953, 6 )
    
def mfout(time):
    return 2.186E-5 * smoothstep(time, 3821.5, 3822, 6)

entin = 3928600 * stored_fluid.backend.molar_mass()
entout =  3946400 * stored_fluid.backend.molar_mass()

boundary_flux = pts.BoundaryFlux(
                mass_flow_in = mfin,
                mass_flow_out = mfout,
                environment_temp = 302.5,
                enthalpy_in = entin,
                enthalpy_out = entout, 
    )

simulation_params = pts.SimParams(
                    init_time = 0,
                    init_temperature = 302.4,
                    inserted_amount = 0,
                    init_pressure = 32E3,
                    final_time = 4694
    )


simulation = pts.generate_simulation(storage_tank = storage_tank, 
                                      boundary_flux = boundary_flux,
                                      simulation_params = simulation_params,
                                      simulation_type = "Controlled Inlet")

results = simulation.run()
results.to_csv("AH2sim.csv")
