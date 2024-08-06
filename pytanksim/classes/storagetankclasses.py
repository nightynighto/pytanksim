# -*- coding: utf-8 -*-
"""Contains classes which store the properties of the storage tanks.

The StorageTank and SorbentTank classes are part of this module.
"""
"""
Copyright 2024 Muhammad Irfan Maulana Kusdhany

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

__all__ = ["StorageTank", "SorbentTank"]

from pytanksim.classes.fluidsorbentclasses import StoredFluid, SorbentMaterial
import CoolProp as CP
import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import OptimizeResult
from typing import Callable


def Cs_gen(mads: float, mcarbon: float, malum: float,
           msteel: float, Tads: float = 1500,
           MWads: float = 12.01E-3, func: Callable[[float], float] = None
           ) -> Callable[[float, float], float]:
    """Generate a function to find the heat capacity at a given temperature.

    Based on Debye's model. Combines contributions from the various materials
    making up the storage tank.

    Parameters
    ----------
    mads : float
        Mass of sorbent (kg).

    mcarbon : float
        Carbon fiber mass (kg).

    malum : float
        Aluminum mass (kg).

    msteel : float
        Steel mass (kg).

    Tads : float, optional
        Debye temperature of the sorbent material (K). The default is 1500,
        which is the value for carbon.

    MWads : float, optional
        The molecular weight of the sorbent material (mol/kg). The default is
        12.01E-3, which is the value for carbon.

    func : Callable[[float],float], optional
        Custom function that returns the specific heat capacity (J/(kg K)) of
        the sorbent material given its temperature.

    Returns
    -------
    (Callable[[float, float], float])
        A function which takes the tank's temperature as an input and returns
        the heat capacity of the tank (J/K)

    """
    R = sp.constants.R

    def Cdebye(T, theta):
        N = 50
        grid = np.linspace(0, theta/T, N)
        y = np.zeros_like(grid)

        def integrand(x):
            return(x**4) * np.exp(x) / ((np.exp(x)-1)**2)

        for i in range(1, N):
            y[i] = integrand(grid[i])
        return 9 * R * ((T/theta)**3) * sp.integrate.simps(y, grid)
    carbon_molar_mass = 12.01E-3
    alum_molar_mass = 26.98E-3
    iron_molar_mass = 55.845E-3

    if func is not None:
        def Cads(T):
            return func(T) * mads
    else:
        def Cads(T):
            return (mads/MWads)*Cdebye(T, Tads)

    def Cs(T):
        return Cads(T) + (mcarbon / carbon_molar_mass) *\
            Cdebye(T, 1500) + (malum/alum_molar_mass) * Cdebye(T, 389.4) +\
            (msteel/iron_molar_mass) * Cdebye(T, 500)
    return Cs


class StorageTank:
    """Stores the properties of the storage tank.

    It also has methods to calculate useful quantities such as tank dormancy
    given a constant heat leakage rate, the internal energy of the fluid being
    stored at various conditions, etc.

    Attributes
    ----------
    volume : float
        Internal volume of the storage tank (m^3).

    stored_fluid : StoredFluid
        Object to calculate the thermophysical properties of the fluid
        being stored.

    aluminum_mass : float, optional
        The mass of aluminum making up the tank walls (kg). The default is
        0.

    carbon_fiber_mass : float, optional
        The mass of carbon fiber making up the tank walls (kg). The default
        is 0.

    steel_mass : float, optional
        The mass of steel making up the tank walls (kg). The default is 0.

    vent_pressure : float, optional
        The pressure (Pa) at which the fluid being stored must be vented.
        The default is None. If None, the value will be taken as the
        maximum value where the CoolProp backend can calculate the
        properties of the fluid being stored.

    min_supply_pressure : float, optional
        The minimum supply pressure (Pa) for discharging simulations.The
        default is 1E5.

    thermal_resistance : float, optional
        The thermal resistance of the tank walls (K/W). The default is 0.
        If 0, the value will not be considered in simulations. If the
        arguments 'surface_area' and 'heat_transfer' are passed,
        'thermal_resistance' will be calculated based on those two arguments
        as long as the user does not pass a value to 'thermal_resistance'.

    surface_area : float, optional
        The surface area of the tank that is in contact with the
        environment (m^2). The default is 0.

    heat_transfer_coefficient : float, optional
        The heat transfer coefficient of the tank surface (W/(m^2 K)).
        The default is 0.
    """

    def __init__(self,
                 stored_fluid: StoredFluid,
                 aluminum_mass: float = 0,
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 vent_pressure: float = None,
                 min_supply_pressure: float = 1E5,
                 thermal_resistance: float = 0,
                 surface_area: float = 0,
                 heat_transfer_coefficient: float = 0,
                 volume: float = None,
                 set_capacity: float = None,
                 full_pressure: float = None,
                 empty_pressure: float = None,
                 full_temperature: float = None,
                 empty_temperature: float = None,
                 full_quality: float = 1,
                 empty_quality: float = 1
                 ) -> "StorageTank":
        """Initialize a StorageTank object.

        Parameters
        ----------
        stored_fluid : StoredFluid
            Object to calculate the thermophysical properties of the fluid
            being stored.

        aluminum_mass : float, optional
            The mass of aluminum making up the tank walls (kg). The default is
            0.

        carbon_fiber_mass : float, optional
            The mass of carbon fiber making up the tank walls (kg). The default
            is 0.

        steel_mass : float, optional
            The mass of steel making up the tank walls (kg). The default is 0.

        vent_pressure : float, optional
            The pressure (Pa) at which the fluid being stored must be vented.
            The default is None. If None, the value will be taken as the
            maximum value where the CoolProp backend can calculate the
            properties of the fluid being stored.

        min_supply_pressure : float, optional
            The minimum supply pressure (Pa) for discharging simulations.The
            default is 1E5.

        thermal_resistance : float, optional
            The thermal resistance of the tank walls (K/W). The default is 0.
            If 0, the value will not be considered in simulations. If the
            arguments 'surface_area' and 'heat_transfer' are passed,
            'thermal_resistance' will be calculated based on those two
            arguments as long as the user does not pass a value to
            'thermal_resistance'.

        surface_area : float, optional
            The surface area of the tank that is in contact with the
            environment (m^2). The default is 0.

        heat_transfer_coefficient : float, optional
            The heat transfer coefficient of the tank surface (W/(m^2 K)).
            The default is 0.

        volume : float, optional
            Internal volume of the storage tank (m^3). The default is None.
            This value is required unless the set capacity and operating
            conditions are defined, in which case the volume is calculated from
            the capacity and operating conditions.

        set_capacity : float, optional
            Set internal capacity of the storage tank (mol). The default is
            None. If specified, this will override the user-specified tank
            volume.

        full_pressure : float, optional
            Pressure (Pa) of the tank when it is considered full. The default
            is None.

        empty_pressure : float, optional
            Pressure (Pa) of the tank when it is considered empty. The default
            is None.

        full_temperature : float, optional
            Temperature (K) of the tank when it is considered full. The
            default is None.

        empty_temperature : float, optional
            Temperature (K) of the tank when it is considered empty. The
            default is None.

        full_quality : float, optional
            Vapor quality of the tank when it is considered full. The default
            is 1 (Gas).

        empty_quality : float, optional
            Vapor quality of the tank when it is considered empty. The default
            is 1 (Gas).

        Raises
        ------
        ValueError
            If any of the mass values provided are less than 0.

        ValueError
            If the vent pressure set is higher than what can be calculated by
            'CoolProp'.

        ValueError
            If neither the volume nor the complete capacity and the pressure
            and temperature swing conditions were provided.

        Returns
        -------
        StorageTank
            A storage tank object which can be passed as arguments to dynamic
            simulations and can calculate certain properties on its own.

        """
        if (aluminum_mass or carbon_fiber_mass or steel_mass) < 0:
            raise ValueError("Please input valid values for the mass")
        if volume is None and (set_capacity or full_pressure or
                               full_temperature or empty_pressure or
                               empty_temperature) is None:
            raise ValueError("Please input the complete capacity + pressure "
                             "and temperature swing information, or input "
                             "the tank volume")

        self.volume = volume
        self.aluminum_mass = aluminum_mass
        self.carbon_fiber_mass = carbon_fiber_mass
        self.steel_mass = steel_mass
        self.heat_capacity = Cs_gen(mads=0,
                                    mcarbon=self.carbon_fiber_mass,
                                    malum=self.aluminum_mass,
                                    msteel=self.steel_mass)

        self.stored_fluid = stored_fluid
        self.min_supply_pressure = min_supply_pressure

        backend = self.stored_fluid.backend
        self.max_pressure = backend.pmax()/10

        if vent_pressure is None:
            self.vent_pressure = self.max_pressure
        else:
            self.vent_pressure = vent_pressure

        if self.max_pressure < self.vent_pressure and\
                stored_fluid.EOS == "HEOS":
            raise ValueError(
                "You set the venting pressure to be larger than the valid \n" +
                "pressure range input for CoolProp.")

        self.surface_area = surface_area
        self.heat_transfer_coefficient = heat_transfer_coefficient
        if (self.surface_area and self.heat_transfer_coefficient) > 0 \
                and thermal_resistance == 0:
            self.thermal_resistance = 1 / \
                (self.surface_area * self.heat_transfer_coefficient)
        else:
            self.thermal_resistance = thermal_resistance
        if set_capacity is not None:
            def min_func(vol):
                self.volume = vol
                cap_full = self.capacity(full_pressure, full_temperature,
                                         full_quality)
                cap_empty = self.capacity(empty_pressure, empty_temperature,
                                          empty_quality)
                return ((cap_full-cap_empty)-set_capacity)**2
            vol = sp.optimize.minimize_scalar(min_func, bounds=(0, 1E16),
                                              method="bounded")
            if self.capacity(full_pressure, full_temperature, full_quality) <\
                    set_capacity:
                raise ValueError("Difference between full and empty"
                                 " conditions too small. Tank volume not"
                                 " converged (i.e. solution >1E16).")
            self.volume = vol.x

    def capacity(self, p: float, T: float, q: float = 0,
                 unit: str = "mol") -> float:
        """Return the amount of fluid stored in the tank at given conditions.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. Can vary between 0 and 1.
            The default is 0.

        unit : str, optional
            Unit of the capacity to be returned. Valid units are "mol" and
            "kg". The default is "mol".

        Returns
        -------
        float
            Amount of fluid stored.

        """
        if p == 0:
            return 0
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        cap_mol = fluid.rhomolar() * self.volume
        if unit == "mol":
            return cap_mol
        elif unit == "kg":
            return cap_mol * self.stored_fluid.backend.molar_mass()

    def capacity_bulk(self, p: float, T: float, q: float = 0,
                      unit: str = "mol") -> float:
        """Calculate the amount of bulk fluid in the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. Can vary between 0 and 1.
            The default is 0.

        unit : str, optional
            Unit of the capacity to be returned. Valid units are "mol" and
            "kg". The default is "mol".

        Returns
        -------
        float
            Amount of bulk fluid stored.

        """
        return self.capacity(p, T, q, unit)

    def find_quality_at_saturation_capacity(self, T: float,
                                            capacity: float) -> float:
        """Find vapor quality at the given temperature and capacity.

        Parameters
        ----------
        T : float
            Temperature (K)

        capacity : float
            Amount of fluid in the tank (moles).

        Returns
        -------
        float
            Vapor quality of the fluid being stored. This is assuming that the
            fluid is on the saturation line.

        """
        fluid = self.stored_fluid.backend
        fluid.update(CP.QT_INPUTS, 0, T)
        rhol = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 1, T)
        rhog = fluid.rhomolar()
        A = np.array([[1, 1],
                      [1/rhog, 1/rhol]])
        b = [capacity, self.volume]
        res = np.linalg.solve(A, b)
        return res[0]/(res[0]+res[1])

    def internal_energy(self, p: float, T: float,
                        q: float = 1) -> float:
        """Calculate the internal energy of the fluid inside of the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. The default is 1.

        Returns
        -------
        float
            Internal energy of the fluid being stored (J).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.volume
        return ufluid * bulk_fluid_moles

    def conditions_at_capacity_temperature(self, cap: float,
                                           T: float, p_guess: float,
                                           q_guess: float) -> OptimizeResult:
        """Find conditions corresponding to a given capacity and temperature.

        Parameters
        ----------
        cap : float
            Amount of fluid inside the tank (moles).

        T : float
            Temperature (K).

        p_guess : float
            Initial guess for pressure value (Pa) to be optimized.

        q_guess : float
            Initial guess for vaport quality value to be optimized.

        Returns
        -------
        OptimizeResult
            The optimization result represented as a OptimizeResult object.
            The relevant attribute for this method is x, the solution array.
            x[0] contains the pressure value and x[1] contains the vapor
            quality value.

        """
        pmax = self.stored_fluid.backend.pmax()

        def optim(pres):
            return (self.capacity(pres, T, q_guess) - cap)**2
        res = sp.optimize.minimize_scalar(optim, bounds=(1E-16, pmax),
                                          method='bounded')

        x = [res.x, q_guess]
        res.x = x
        if res.fun > 1:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
            psat = self.stored_fluid.backend.p()
            q = self.find_quality_at_saturation_capacity(T, cap)
            res.x[0] = psat
            res.x[1] = q
        return res

    def conditions_at_capacity_pressure(self, cap: float, p: float,
                                        T_guess: float,
                                        q_guess: float) -> OptimizeResult:
        """Find conditions corresponding to a given capacity and temperature.

        Parameters
        ----------
        cap : float
            Amount of fluid inside the tank (moles).

        P : float
            Pressure (Pa).

        T_guess : float
            Initial guess for temperature value (K) to be optimized.

        q_guess : float
            Initial guess for vaport quality value to be optimized.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result represented as a OptimizeResult object.
            The relevant attribute for this package is x, the solution array.
            x[0] contains the temperature value and x[1] contains the vapor
            quality value.

        """
        fluid = self.stored_fluid.backend
        Tmin = fluid.Tmin()
        Tmax = fluid.Tmax()

        def optim(temper):
            return (self.capacity(p, temper, q_guess) - cap)**2

        res = sp.optimize.minimize_scalar(optim, bounds=(Tmin, Tmax),
                                          method='bounded')
        x = [res.x, q_guess]
        res.x = x
        if res.fun > 1:
            self.stored_fluid.backend.update(CP.PQ_INPUTS, p, 0)
            Tsat = self.stored_fluid.backend.T()
            q = self.find_quality_at_saturation_capacity(Tsat, cap)
            res.x[0] = Tsat
            res.x[1] = q
        return res

    def calculate_dormancy(self, p: float, T: float,
                           heating_power: float, q: float = 0) -> pd.DataFrame:
        """Calculate dormancy time given a constant heating rate.

        Parameters
        ----------
        p : float
            Initial tank pressure (Pa).

        T : float
            Initial tank temperature (K).

        heating_power : float
            The heating power going into the tank during parking (W).

        q : float, optional
            Initial vapor quality of the tank. The default is 0 (pure liquid).

        Returns
        -------
        pd.DataFrame
            Pandas dataframe containing calculation conditions and results.
            Each key stores a floating point number.
            The dictionary keys and their respective values are:

            - "init pressure": initial pressure
            - "init temperature": initial temperature
            - "init quality": initial vapor quality
            - "dormancy time": time until tank needs to be vented in seconds
            - "final temperature": temperature of the tank as venting begins
            - "final quality": vapor quality at the time of venting
            - "final pressure": pressure at the time of venting
            - "capacity error": error between final and initial capacity
            - "total energy change": difference in internal energy between the
              initial and final conditions
            - "solid heat capacity contribution": the amount of heat absorbed
              by the tank walls

        """
        init_cap = self.capacity(p, T, q)
        init_heat = self.internal_energy(p, T, q)
        vent_cond = self.conditions_at_capacity_pressure(init_cap,
                                                         self.vent_pressure,
                                                         T, q).x
        final_heat = self.internal_energy(self.vent_pressure,
                                          vent_cond[0], vent_cond[1])
        final_cap = self.capacity(self.vent_pressure,
                                  vent_cond[0], vent_cond[1])

        def heat_capacity_change(T1, T2):
            xgrid = np.linspace(T1, T2, 100)
            heatcapgrid = [self.heat_capacity(temper) for temper in xgrid]
            return sp.integrate.simps(heatcapgrid, xgrid)

        final_heat += heat_capacity_change(T, vent_cond[0])
        return pd.DataFrame({"init pressure": p,
                             "init temperature": T,
                             "init quality": q,
                             "dormancy time": (final_heat -
                                               init_heat)/heating_power,
                             "final temperature": vent_cond[0],
                             "final quality": vent_cond[1],
                             "final pressure": self.vent_pressure,
                             "capacity error": final_cap - init_cap,
                             "total energy change": final_heat - init_heat,
                             "solid heat capacity contribution":
                                 heat_capacity_change(T, vent_cond[0])},
                            index=[0])


class SorbentTank(StorageTank):
    """Stores properties of a fluid storage tank filled with sorbents.

    Attributes
    ----------
    volume : float
        Internal volume of the storage tank (m^3).

    sorbent_material : SorbentMaterial
        An object storing the properties of the sorbent material used in
        the tank.

    aluminum_mass : float, optional
        The mass of aluminum making up the tank walls (kg). The default is
        0.

    carbon_fiber_mass : float, optional
        The mass of carbon fiber making up the tank walls (kg). The default
        is 0.

    steel_mass : float, optional
        The mass of steel making up the tank walls (kg). The default is 0.

    vent_pressure : float, optional
        Maximum pressure at which the tank has to be vented (Pa). The
        default is None.

    min_supply_pressure : float, optional
        The minimum supply pressure (Pa) for discharging simulations. The
        default is 1E5.

    thermal_resistance : float, optional
        The thermal resistance of the tank walls (K/W). The default is 0.
        If 0, the value will not be considered in simulations. If the
        arguments 'surface_area' and 'heat_transfer' are passed,
        'thermal_resistance' will be calculated based on those two
        arguments as long as the user does not pass a value to
        'thermal_resistance'.

    surface_area : float, optional
        Outer surface area of the tank in contact with the environment
        (m^2). The default is 0.

    heat_transfer_coefficient : float, optional
        The heat transfer coefficient of the tank surface (W/(m^2 K)).
        The default is 0.

    """

    def __init__(self,
                 sorbent_material: SorbentMaterial,
                 aluminum_mass: float = 0,
                 carbon_fiber_mass: float = 0,
                 steel_mass: float = 0,
                 vent_pressure: float = None,
                 min_supply_pressure: float = 1E5,
                 thermal_resistance: float = 0,
                 surface_area: float = 0,
                 heat_transfer_coefficient: float = 0,
                 volume: float = None,
                 set_capacity: float = None,
                 full_pressure: float = None,
                 empty_pressure: float = None,
                 full_temperature: float = None,
                 empty_temperature: float = None,
                 full_quality: float = 1,
                 empty_quality: float = 1,
                 set_sorbent_fill: float = 1
                 ) -> "SorbentTank":
        """Initialize a SorbentTank object.

        Parameters
        ----------
        sorbent_material : SorbentMaterial
            An object storing the properties of the sorbent material used in
            the tank.

        aluminum_mass : float, optional
            The mass of aluminum making up the tank walls (kg). The default is
            0.

        carbon_fiber_mass : float, optional
            The mass of carbon fiber making up the tank walls (kg). The default
            is 0.

        steel_mass : float, optional
            The mass of steel making up the tank walls (kg). The default is 0.

        vent_pressure : float, optional
            Maximum pressure at which the tank has to be vented (Pa). The
            default is None.

        min_supply_pressure : float, optional
            The minimum supply pressure (Pa) for discharging simulations. The
            default is 1E5.

        thermal_resistance : float, optional
            The thermal resistance of the tank walls (K/W). The default is 0.
            If 0, the value will not be considered in simulations. If the
            arguments 'surface_area' and 'heat_transfer' are passed,
            'thermal_resistance' will be calculated based on those two
            arguments as long as the user does not pass a value to
            'thermal_resistance'.

        surface_area : float, optional
            Outer surface area of the tank in contact with the environment
            (m^2). The default is 0.

        heat_transfer_coefficient : float, optional
            The heat transfer coefficient of the tank surface (W/(m^2 K)).
            The default is 0.

        volume : float, optional
            Internal volume of the storage tank (m^3). The default is None.
            This value is required unless the set capacity and operating
            conditions are defined, in which case the volume is calculated from
            the capacity and operating conditions.

        set_capacity : float, optional
            Set internal capacity of the storage tank (mol). The default is
            None. If specified, this will override the user-specified tank
            volume.

        full_pressure : float, optional
            Pressure (Pa) of the tank when it is considered full. The default
            is None.

        empty_pressure : float, optional
            Pressure (Pa) of the tank when it is considered empty. The default
            is None.

        full_temperature : float, optional
            Temperature (K) of the tank when it is considered full. The
            default is None.

        empty_temperature : float, optional
            Temperature (K) of the tank when it is considered empty. The
            default is None.

        full_quality : float, optional
            Vapor quality of the tank when it is considered full. The default
            is 1 (Gas).

        empty_quality : float, optional
            Vapor quality of the tank when it is considered empty. The default
            is 1 (Gas).

        set_sorbent_fill : float, optional
            Ratio of tank volume filled with sorbent. The default is 1
            (completely filled with sorbent).


        Returns
        -------
        SorbentTank
            Object which stores various properties of a storage tank containing
            sorbents. It also has some useful methods related to the tank, most
            notably dormancy calculation.

        """
        stored_fluid = sorbent_material.model_isotherm.stored_fluid
        self.sorbent_material = sorbent_material
        super().__init__(volume=volume,
                         aluminum_mass=aluminum_mass,
                         stored_fluid=stored_fluid,
                         carbon_fiber_mass=carbon_fiber_mass,
                         min_supply_pressure=min_supply_pressure,
                         vent_pressure=vent_pressure,
                         thermal_resistance=thermal_resistance,
                         surface_area=surface_area,
                         steel_mass=steel_mass,
                         heat_transfer_coefficient=heat_transfer_coefficient,
                         set_capacity=set_capacity,
                         full_pressure=full_pressure,
                         empty_pressure=empty_pressure,
                         full_temperature=full_temperature,
                         empty_temperature=empty_temperature,
                         full_quality=full_quality,
                         empty_quality=empty_quality)
        self.heat_capacity = Cs_gen(mads=self.sorbent_material.mass,
                                    mcarbon=self.carbon_fiber_mass,
                                    malum=self.aluminum_mass,
                                    msteel=self.steel_mass,
                                    Tads=self.sorbent_material.
                                    Debye_temperature,
                                    MWads=self.sorbent_material.molar_mass,
                                    func=self.sorbent_material.
                                    heat_capacity_function)
        if set_capacity is not None:
            def min_func(v):
                self.volume = v
                sorbent_vol = set_sorbent_fill * v
                self.sorbent_material.mass = sorbent_vol *\
                    self.sorbent_material.bulk_density
                cap_full = self.capacity(full_pressure, full_temperature,
                                         full_quality)
                cap_empty = self.capacity(empty_pressure, empty_temperature,
                                          empty_quality)
                return ((cap_full-cap_empty)-set_capacity)**2
            vol = sp.optimize.minimize_scalar(min_func, bounds=(0, 1E10),
                                     method="Bounded")
            self.volume = vol.x
            sorbent_vol = set_sorbent_fill * self.volume
            self.sorbent_material.mass = sorbent_vol *\
                self.sorbent_material.bulk_density

    def bulk_fluid_volume(self,
                          p: float,
                          T: float) -> float:
        """Calculate the volume of bulk fluid inside of the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature(K).

        Returns
        -------
        float
            Bulk fluid volume within the tank (m^3).

        """
        tankvol = self.volume
        mads = self.sorbent_material.mass
        rhoskel = self.sorbent_material.skeletal_density
        vads = self.sorbent_material.model_isotherm.v_ads
        outputraw = tankvol - mads/rhoskel - vads(p, T) * mads
        output = outputraw if outputraw >= 0 else 0
        return output

    def capacity(self, p: float, T: float, q: float = 0) -> float:
        """Return the amount of fluid stored in the tank at given conditions.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. Can vary between 0 and 1.
            The default is 0.

        Returns
        -------
        float
            Amount of fluid stored (moles).

        """
        if p == 0:
            return 0
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        return bulk_fluid_moles + adsorbed_moles

    def capacity_bulk(self, p: float, T: float, q: float = 0) -> float:
        """Calculate the amount of bulk fluid in the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. Can vary between 0 and 1.
            The default is 0.

        Returns
        -------
        float
            Amount of bulk fluid stored (moles).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)

        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        return bulk_fluid_moles

    def internal_energy(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the internal energy of the fluid inside of the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. The default is 1.

        Returns
        -------
        float
            Internal energy of the fluid being stored (J).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        uadsorbed = self.sorbent_material.model_isotherm.\
            internal_energy_adsorbed(p, T)
        return ufluid * bulk_fluid_moles + adsorbed_moles * (uadsorbed)

    def internal_energy_sorbent(self, p: float,
                                T: float, q: float = 1) -> float:
        """Calculate the internal energy of the adsorbed fluid in the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. The default is 1.

        Returns
        -------
        float
            Internal energy of the adsorbed fluid in the tank (J).

        """
        adsorbed_moles = self.sorbent_material.model_isotherm.n_absolute(p, T) * \
            self.sorbent_material.mass
        uadsorbed = self.sorbent_material.model_isotherm.\
            internal_energy_adsorbed(p, T)
        return adsorbed_moles * (uadsorbed)

    def internal_energy_bulk(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the internal energy of the bulk fluid in the tank.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the fluid being stored. The default is 1.

        Returns
        -------
        float
            Internal energy of the bulk fluid in the tank (J).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        ufluid = fluid.umolar()
        bulk_fluid_moles = fluid.rhomolar() * self.bulk_fluid_volume(p, T)
        return ufluid * bulk_fluid_moles

    def find_quality_at_saturation_capacity(self, T: float,
                                            capacity: float) -> float:
        """Find vapor quality at the given temperature and capacity.

        Parameters
        ----------
        T : float
            Temperature (K)

        capacity : float
            Amount of fluid in the tank (moles).

        Returns
        -------
        float
            Vapor quality of the fluid being stored. This is assuming that the
            fluid is on the saturation line.

        """
        fluid = self.stored_fluid.backend
        fluid.update(CP.QT_INPUTS, 0, T)
        rhol = fluid.rhomolar()
        fluid.update(CP.QT_INPUTS, 1, T)
        rhog = fluid.rhomolar()
        p = fluid.p()
        bulk_capacity = capacity - self.sorbent_material.mass *\
            self.sorbent_material.model_isotherm.n_absolute(p, T)
        A = np.array([[1, 1],
                      [1/rhog, 1/rhol]])
        b = [bulk_capacity, self.bulk_fluid_volume(p, T)]
        res = np.linalg.solve(A, b)
        q = res[0]/(res[0]+res[1])
        return q

    def find_temperature_at_saturation_quality(self, q: float,
                                               cap: float) -> OptimizeResult:
        """Find temperature at a given capacity and vapor quality value.

        Parameters
        ----------
        q : float
            Vapor quality. Can vary between 0 and 1.
        cap : float
            Amount of fluid stored in the tank (moles).

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result represented as a OptimizeResult object.
            The relevant attribute for this function is x, the optimized
            temperature value.

        """

        def optim(x):
            self.stored_fluid.backend.update(CP.QT_INPUTS, q, x)
            p = self.stored_fluid.backend.p()
            return (self.capacity(p, x, q) - cap)**2

        fluid = self.stored_fluid.backend
        Tmin = fluid.Tmin()
        Tmax = fluid.T_critical()
        res = sp.optimize.minimize_scalar(optim, method="bounded",
                                          bounds=(Tmin, Tmax))
        return res

    def calculate_dormancy(self, p: float, T: float,
                           heating_power: float,
                           q: float = 0) -> pd.DataFrame:
        """Calculate dormancy time given a constant heating rate.

        Parameters
        ----------
        p : float
            Initial tank pressure (Pa).

        T : float
            Initial tank temperature (K).

        heating_power : float
            The heating power going into the tank during parking (W).

        q : float, optional
            Initial vapor quality of the tank. The default is 0 (pure liquid).

        Returns
        -------
        pd.DataFrame
            Pandas dataframe containing calculation conditions and results.
            Each key stores a floating point number.
            The dictionary keys and their respective values are:

            - "init pressure": initial pressure
            - "init temperature": initial temperature
            - "init quality": initial vapor quality
            - "dormancy time": time until tank needs to be vented in seconds
            - "final temperature": temperature of the tank as venting begins
            - "final quality": vapor quality at the time of venting
            - "final pressure": pressure at the time of venting
            - "capacity error": error between final and initial capacity
            - "total energy change": difference in internal energy between the
              initial and final conditions
            - "sorbent energy contribution": the amount of heat taken by
              the adsorbed phase via desorption
            - "bulk energy contribution": the amount of heat absorbed by the
              bulk phase
            - "immersion heat contribution": how much heat has been absorbed
              by un-immersing the sorbent material in the fluid
            - "solid heat capacity contribution": the amount of heat absorbed
              by the tank walls

        """
        init_pres = p
        init_cap = self.capacity(p, T, q)
        init_ene = self.internal_energy(p, T, q)
        init_ene_ads = self.internal_energy_sorbent(p, T, q)
        init_ene_bulk = self.internal_energy_bulk(p, T, q)
        vent_cond = self.conditions_at_capacity_pressure(init_cap,
                                                         self.vent_pressure,
                                                         T, q).x
        final_ene = self.internal_energy(self.vent_pressure,
                                         vent_cond[0], vent_cond[1])
        final_cap = self.capacity(self.vent_pressure,
                                  vent_cond[0], vent_cond[1])
        final_ene_ads = self.internal_energy_sorbent(self.vent_pressure,
                                                     vent_cond[0],
                                                     vent_cond[1])
        final_ene_bulk = self.internal_energy_bulk(self.vent_pressure,
                                                   vent_cond[0], vent_cond[1])

        def heat_capacity_change(T1, T2):
            xgrid = np.linspace(T1, T2, 100)
            heatcapgrid = [self.heat_capacity(temper) for temper in xgrid]
            return sp.integrate.simps(heatcapgrid, xgrid)

        final_ene += heat_capacity_change(T, vent_cond[0])

        res1 = self.find_temperature_at_saturation_quality(1, init_cap)
        res2 = self.find_temperature_at_saturation_quality(0, init_cap)
        if T > self.stored_fluid.backend.T_critical() or\
                p > self.stored_fluid.backend.p_critical():
            integ_res = 0
        elif (res1.x > T and res1.fun < 1) or (res2.x > T and res2.fun < 1)\
                or (vent_cond[1] != q):

            if vent_cond[1] != q:
                lower_bound = max(q, vent_cond[1])
                upper_bound = min(q, vent_cond[1])
            else:
                consider_res1 = True if res1.fun < 1 and \
                    T < res1.x < vent_cond[0] else False
                consider_res2 = True if res2.fun < 1 and \
                    T < res2.x < vent_cond[0] else False
                if consider_res1 and consider_res2:
                    Tcheck = max(res1.x, res2.x)
                    resfinal = 1 if res1.x > res2.x else 0
                elif consider_res2 and (not consider_res1):
                    Tcheck = res2.x
                    resfinal = 0
                elif consider_res1 and (not consider_res2):
                    Tcheck = res1.x
                    resfinal = 1
                vent_cond[1] = resfinal
                if Tcheck > 0.998 * self.stored_fluid.backend.T_critical():
                    Tcheck = 0.998 * self.stored_fluid.backend.T_critical()
                    resfinal = self.find_quality_at_saturation_capacity(
                        Tcheck,
                        init_cap)
                lower_bound = max(q, resfinal)
                upper_bound = min(q, resfinal)
            total_surface_area = self.sorbent_material.mass *\
                self.sorbent_material.specific_surface_area * 1000
            qgrid = np.linspace(lower_bound, upper_bound, 100)
            Agrid = np.zeros_like(qgrid)
            ygrid = np.zeros_like(qgrid)
            for i, qual in enumerate(qgrid):
                temper = self.find_temperature_at_saturation_quality(qual,
                                                                     init_cap)\
                    .x
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, temper)
                p = self.stored_fluid.backend.p()
                rhol = self.stored_fluid.backend.rhomolar()
                nl = (1 - qual) * self.capacity_bulk(p, temper, qual)
                vbulk = self.bulk_fluid_volume(p, temper)
                Agrid[i] = total_surface_area * (nl/(rhol * vbulk))
                ygrid[i] = self.sorbent_material.model_isotherm.\
                    areal_immersion_energy(temper)
            integ_res = sp.integrate.simps(ygrid, Agrid)
            integ_res = -integ_res if q < resfinal else integ_res
            final_ene = final_ene + integ_res
        else:
            integ_res = 0
        return pd.DataFrame(
            {"init pressure": init_pres,
             "init temperature": T,
             "init quality": q,
             "dormancy time": (final_ene - init_ene)/heating_power,
             "final temperature": vent_cond[0],
             "final quality": vent_cond[1],
             "final pressure": self.vent_pressure,
             "capacity error": final_cap - init_cap,
             "total energy change": final_ene - init_ene,
             "sorbent energy contribution": final_ene_ads - init_ene_ads,
             "bulk energy contribution": final_ene_bulk - init_ene_bulk,
             "immersion heat contribution": integ_res,
             "solid heat capacity contribution": heat_capacity_change(
                                                                T,
                                                                vent_cond[0])},
            index=[0])
