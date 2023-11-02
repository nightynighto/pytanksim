# -*- coding: utf-8 -*-
"""Contains the SimResults class.

It is used for storing and post-processing the results of dynamic simulations.
"""
__all__ = ["SimResults"]

import numpy as np
from typing import List, Union, TYPE_CHECKING
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pytanksim.classes.storagetankclasses import StorageTank, SorbentTank
if TYPE_CHECKING:
    from pytanksim.classes.basesimclass import SimParams


class SimResults:
    """Class for storing the results of dynamic simulations.

    It comes with methods for exporting the results to csv, plotting the
    results, and for combining the results of multiple simulations.

    Attributes
    ----------
    df : pd.DataFrame
        A dataframe containing the results of dynamic simulations. See notes
        for the column names and the variables each column has.

    Notes
    -----
    Below is a list of the pandas DataFrame column names and a short
    description of the variable stored inside each series.

        - ``t``: time (seconds)
        - ``p``: pressure (Pa)
        - ``T``: temperature (K)
        - ``na``: amount of fluid adsorbed (moles)
        - ``ng``: amount of fluid in gaseous form (moles)
        - ``nl``: amount of fluid in liquid form (moles)
        - ``ns``: amount of fluid in supercritical form (moles)
        - ``Qcoolreq``: cumulative amount of cooling required (J)
        - ``Qheatreq``: cumulative amount of heating required (J)
        - ``nout``: cumulative amount of fluid vented (moles)
        - ``Hout``: cumulative amount of vented fluid enthalpy (J)
        - ``nin``: cumulative amount of fluid inserted (moles)
        - ``Hin``: cumulative amount of inserted fluid enthalpy (J)
        - ``Qcooladd``: cumulative amount of user specified cooling (J)
        - ``Qheatadd``: cumulative amount of user specified heating (J)
        - ``Qleak``: cumulative amount of heat leakage into the tank (J)
        - ``ma``: mass of fluid adsorbed (kg)
        - ``mg``: mass of fluid in gaseous form (kg)
        - ``ml``: mass of fluid in liquid form (kg)
        - ``ms``: mass of fluid in supercritical form (kg)
        - ``mout``: cumulative mass of fluid vented (kg)
        - ``min``: cumulative mass of fluid inserted (kg)
        - ``na_dot``: the amount of fluid (moles) being adsorbed per
          second.
        - ``ng_dot``: the first derivative of the amount of fluid in
          gaseous form w.r.t. time. Its unit is mol/s.
        - ``nl_dot``: the first derivative of the amount of fluid in
          liquid form w.r.t. time. Its unit is mol/s
        - ``ns_dot``: the first derivative of the amount of fluid in
          supercritical form w.r.t. time. Its unit is mol/s.
        - ``Qcoolreq_dot``: the cooling power (W) required to maintain a
          constant pressure during refuel.
        - ``Qheatreq_dot``: the heating power (W) required to maintain a
          constant pressure during discharge.
        - ``nout_dot``: the rate at which fluid is being vented from the
          tank (mol/s).
        - ``Hout_dot``: the rate at which enthalpy is taken away by fluid
          leaving the tank (W).
        - ``nin_dot``: the rate at which fluid is entering the tank
          (mol/s).
        - ``Hin_dot``: the rate at which enthalpy is added by fluid
          fluid entering the tank (W).
        - ``Qcooladd_dot``: the user specified cooling power (W).
        - ``Qheatadd_dot``: the user specified heating power (W).
        - ``Qleak_dot``: the rate of heat leakage into the tank (W).
        - ``ma_dot``: the mass of fluid (kg) being adsorbed per second.
        - ``mg_dot``: the first derivative of the mass of fluid in
          gaseous form w.r.t. time. Its unit is kg/s.
        - ``ml_dot``: the first derivative of the mass of fluid in
          liquid form w.r.t. time. Its unit is kg/s.
        - ``ms_dot``: the first derivative of the mass of fluid in
          supercritical form w.r.t. time. Its unit is kg/s.
        - ``mout_dot``: the rate at which fluid is being vented from the
          tank (kg/s).
        - ``min_dot``: the rate at which fluid is being inserted into the
          tank (kg/s).

    """

    fancy_colnames_dict = {
        "time": "Time (s)",
        "pressure": "Pressure (Pa)",
        "temperature": "Temperature (K)",
        "moles_adsorbed": "Amount Adsorbed (mol)",
        "moles_gas": "Vapor Amount (mol)",
        "moles_liquid": "Liquid Amount (mol)",
        "moles_supercritical": "Supercritical Amount (mol)",
        "cooling_required": "Cumulative Cooling Energy Required (J)",
        "heating_required": "Cumulative Heating Energy Required (J)",
        "vented_amount": "Amount Vented (mol)",
        "vented_energy": "Cumulative Fluid Enthalpy Out (J)",
        "inserted_amount": "Amount Inserted (mol)",
        "flow_energy_in": "Cumulative Fluid Enthalpy In (J)",
        "cooling_additional": "Cumulative Additional Cooling Provided (J)",
        "heating_additional": "Cumulative Additional Heating Provided (J)",
        "heat_leak_in": "Cumulative Heat Leakage (J)",
        "kg_adsorbed": "Amount Adsorbed (kg)",
        "kg_gas": "Vapor Amount (kg)",
        "kg_liquid": "Liquid Amount (kg)",
        "kg_supercritical": "Supercritical Amount (kg)",
        "vented_kg": "Amount Vented (kg)",
        "inserted_kg": "Amount Inserted (kg)",
        "moles_adsorbed_per_s": "d(Adsorbed Amount)/dt (mol/s)",
        "moles_gas_per_s":  "d(Vapor Amount)/dt (mol/s)",
        "moles_liquid_per_s": "d(Liquid Amount)/dt (mol/s)",
        "moles_supercritical_per_s": "d(Supercritical Amount)/dt (mol/s)",
        "cooling_required_W": "Cooling Power Required (W)",
        "heating_required_W": "Heating Power Required (W)",
        "vented_amount_per_s": "Venting Rate (mol/s)",
        "vented_energy_W": "Rate of Fluid Enthalpy Flow Out (W)",
        "inserted_amount_per_s": "Refueling Rate (mol/s)",
        "flow_energy_in_W": "Rate of Fluid Enthalpy Flow In (W)",
        "cooling_additional_W": "Additional Cooling Power (W)",
        "heating_additional_W": "Additional Cooling Power (W)",
        "heat_leak_in_W": "Heat Leakage Rate (W)",
        "kg_adsorbed_per_s": "d(Adsorbed Amount)/dt (kg/s)",
        "kg_gas_per_s":  "d(Vapor Amount)/dt (kg/s)",
        "kg_liquid_per_s": "d(Liquid Amount)/dt (kg/s)",
        "kg_supercritical_per_s": "d(Supercritical Amount)/dt (kg/s)",
        "vented_kg_per_s": "Venting Rate (kg/s)",
        "inserted_kg_per_s": "Refueling Rate (kg/s)"
        }

    short_colnames_inv = {
        "time": "t",
        "pressure": "p",
        "temperature": "T",
        "moles_adsorbed": "na",
        "moles_gas": "ng",
        "moles_liquid": "nl",
        "moles_supercritical": "ns",
        "cooling_required": "Qcoolreq",
        "heating_required": "Qheatreq",
        "vented_amount": "nout",
        "vented_energy": "Hout",
        "inserted_amount": "nin",
        "flow_energy_in": "Hin",
        "cooling_additional": "Qcooladd",
        "heating_additional": "Qheatadd",
        "heat_leak_in": "Qleak",
        "kg_adsorbed": "ma",
        "kg_gas": "mg",
        "kg_liquid": "ml",
        "kg_supercritical": "ms",
        "vented_kg": "mout",
        "inserted_kg": "min",
        "moles_adsorbed_per_s": "na_dot",
        "moles_gas_per_s":  "ng_dot",
        "moles_liquid_per_s": "nl_dot",
        "moles_supercritical_per_s": "ns_dot",
        "cooling_required_W": "Qcoolreq_dot",
        "heating_required_W": "Qheatreq_dot",
        "vented_amount_per_s": "nout_dot",
        "vented_energy_W": "Hout_dot",
        "inserted_amount_per_s": "nin_dot",
        "flow_energy_in_W": "Hin_dot",
        "cooling_additional_W": "Qcooladd_dot",
        "heating_additional_W": "Qheatadd_dot",
        "heat_leak_in_W": "Qleak_dot",
        "kg_adsorbed_per_s": "ma_dot",
        "kg_gas_per_s":  "mg_dot",
        "kg_liquid_per_s": "ml_dot",
        "kg_supercritical_per_s": "ms_dot",
        "vented_kg_per_s": "mout_dot",
        "inserted_kg_per_s": "min_dot"
        }

    short_colnames = {v: k for k, v in short_colnames_inv.items()}

    def __init__(self,
                 pressure: Union[List[float], np.ndarray],
                 temperature: Union[List[float], np.ndarray],
                 time: Union[List[float], np.ndarray],
                 moles_adsorbed: Union[List[float], np.ndarray],
                 moles_gas: Union[List[float], np.ndarray],
                 moles_liquid: Union[List[float], np.ndarray],
                 moles_supercritical: Union[List[float], np.ndarray],
                 tank_params: Union[StorageTank, SorbentTank],
                 sim_type: str,
                 sim_params: "SimParams",
                 stop_reason: str,
                 inserted_amount: Union[List[float], np.ndarray] = 0,
                 flow_energy_in: Union[List[float], np.ndarray] = 0,
                 cooling_required: Union[List[float], np.ndarray] = 0,
                 heating_required: Union[List[float], np.ndarray] = 0,
                 cooling_additional: Union[List[float], np.ndarray] = 0,
                 heating_additional: Union[List[float], np.ndarray] = 0,
                 heat_leak_in: Union[List[float], np.ndarray] = 0,
                 vented_amount: Union[List[float], np.ndarray] = 0,
                 vented_energy: Union[List[float], np.ndarray] = 0
                 ) -> "SimResults":
        """Initialize a SimResults object.

        Parameters
        ----------
        pressure : Union[List[float], np.ndarray]
            A list or numpy array containing the pressure values inside of the
            tank (Pa) as it changes over time.

        temperature : Union[List[float], np.ndarray]
            A list or numpy array containing the temperature values inside of
            the tank (K) as it changes over time.

        time : Union[List[float], np.ndarray]
            A list or numpy array containing the simulation time points (s) at
            which results are reported.

        moles_adsorbed : Union[List[float], np.ndarray]
            A list or numpy array containing the amount of fluid that is
            adsorbed (moles) at given points in time.

        moles_gas : Union[List[float], np.ndarray]
            A list or numpy array containing the amount of fluid stored in
            gaseous form (moles) at given points in time.

        moles_liquid : Union[List[float], np.ndarray]
            A list or numpy array containing the amount of fluid stored in
            liquid form (moles) at given points in time.

        moles_supercritical : Union[List[float], np.ndarray]
            A list or numpy array containing the amount of supercritical fluid
            in the tank (moles) at given points in time.

        tank_params : Union[StorageTank, SorbentTank]
            An object containing the parameters of the storage tank used for
            the dynamic simulation.

        sim_type : str
            A string describing the type of simulation that was conducted.

        sim_params : SimParams
            An object containing the parameters used for the simulation.

        stop_reason : str
            A string describing why the simulation was terminated.

        inserted_amount: Union[List[float], np.ndarray], optional
            The cumulative amount of fluid inserted into the tank (moles)
            throughout the dynamic simulation. The default is 0.

        flow_energy_in : Union[List[float], np.ndarray], optional
            The cumulative amount of enthalpy brought by fluid flowing into the
            tank (J) throughout the dynamic simulation. The default is 0.

        cooling_required : Union[List[float], np.ndarray], optional
            The cumulative amount of cooling required (J) to maintain a
            constant pressure during refueling. The default is 0.

        heating_required : Union[List[float], np.ndarray], optional
            The cumulative amount of heating required (J) to maintain a
            constant pressure during discharging. The default is 0.

        cooling_additional : Union[List[float], np.ndarray], optional
            The cumulative amount of additional cooling (J) inputted to the
            simulation via a user-defined function. The default is 0.

        heating_additional : Union[List[float], np.ndarray], optional
            The cumulative amount of additional heating (J) inputted to the
            simulation via a user-defined function. The default is 0.

        heat_leak_in : Union[List[float], np.ndarray], optional
            The cumulative amount of heat (J) which has leaked into the tank
            from the environment. The default is 0.

        vented_amount : Union[List[float], np.ndarray], optional
            The cumulative amount of fluid vented (moles) throughout the
            dynamic simulation. The default is 0.

        vented_energy : Union[List[float], np.ndarray], optional
            The cumulative amount of enthalpy taken by fluid flowing out of the
            tank (J) throughout the dynamic simulation. The default is 0.

        Returns
        -------
        SimResults
            An object containing the results of a dynamic simulation run by
            pytanksim. It has functions for exporting and plotting.

        """
        self.results_dict = {
            "time": time,
            "pressure": pressure,
            "temperature": temperature,
            "moles_adsorbed": moles_adsorbed,
            "moles_gas": moles_gas,
            "moles_liquid": moles_liquid,
            "moles_supercritical": moles_supercritical,
            "cooling_required": cooling_required,
            "heating_required": heating_required,
            "vented_amount": vented_amount,
            "vented_energy": vented_energy,
            "inserted_amount": inserted_amount,
            "flow_energy_in": flow_energy_in,
            "cooling_additional": cooling_additional,
            "heating_additional": heating_additional,
            "heat_leak_in": heat_leak_in
            }
        self.sim_params = sim_params
        self.tank_params = tank_params
        self.sim_type = sim_type
        self.results_df = pd.DataFrame.from_dict(self.results_dict)
        self.results_df = self.results_df.drop_duplicates(subset=["time"])
        self.stop_reason = stop_reason

        df_diff = self.results_df.diff()

        def time_diff(colname):
            return df_diff[colname]/df_diff["time"]

        self.results_df["moles_adsorbed_per_s"] = time_diff("moles_adsorbed")
        self.results_df["moles_gas_per_s"] = time_diff("moles_gas")
        self.results_df["moles_liquid_per_s"] = time_diff("moles_liquid")
        self.results_df["moles_supercritical_per_s"] = \
            time_diff("moles_supercritical")
        self.results_df["cooling_required_W"] = time_diff("cooling_required")
        self.results_df["heating_required_W"] = time_diff("heating_required")
        self.results_df["vented_amount_per_s"] = time_diff("vented_amount")
        self.results_df["vented_energy_W"] = time_diff("vented_energy")
        self.results_df["inserted_amount_per_s"] = time_diff("inserted_amount")
        self.results_df["flow_energy_in_W"] = time_diff("flow_energy_in")
        self.results_df["cooling_additional_W"] = \
            time_diff("cooling_additional")
        self.results_df["heating_additional_W"] = \
            time_diff("heating_additional")
        self.results_df["heat_leak_in_W"] = time_diff("heat_leak_in")
        df_export = self.results_df.copy()
        molarmass = self.tank_params.stored_fluid.backend.molar_mass()
        overwrite_df = df_export.filter(regex="moles|amount")
        overwrite_df = overwrite_df * molarmass
        column_names = list(overwrite_df.columns)
        column_names_new = [sub.replace("amount", "kg").replace("moles", "kg")
                            for sub in column_names]
        overwrite_df.columns = column_names_new
        self.df = self.results_df.copy().\
            rename(columns=self.short_colnames_inv)

    def get_final_conditions(self, idx: int = -1) -> dict:
        """Output final tank conditions at the end of the simulation.

        Parameters
        ----------
        idx : int, optional
            The index of the simulation results array from which the values are
            to be taken. The default is -1 (the last time point in the
            simulation).

        Returns
        -------
        dict
            A dictionary containing tank conditions at'idx'.

        """
        final_dict = {}
        for key, value in self.results_dict.items():
            if isinstance(value, (str, list, tuple, np.ndarray)):
                final_dict[key] = value[idx]
            else:
                final_dict[key] = value
        return final_dict

    def to_csv(self, filename: str, convert_moles_to_kg: bool = True):
        """Export simulation results to a csv file.

        Parameters
        ----------
        filename : str
            The desired filepath for the csv file to be created.

        convert_moles_to_kg : bool, optional
            If 'True', the fluid amounts and their derivative values will be
            reported on a mass basis using kg as the unit. If 'False', the
            amounts will be reported in moles. The default is True.

        """
        df_export = self.results_df.copy()
        new_colnames = [SimResults.fancy_colnames_dict[colname] for
                        colname in list(df_export.columns)]
        df_export.columns = new_colnames
        print(df_export)
        header_info = [
            ["Has Sorbent?", isinstance(self.tank_params, SorbentTank)],
            ["Sorbent Name", self.tank_params.
             sorbent_material.model_isotherm.sorbent
             if isinstance(self.tank_params, SorbentTank) else "NA"],
            ["Fluid Name", self.tank_params.stored_fluid.fluid_name],
            ["Aluminum Mass (kg)", self.tank_params.aluminum_mass],
            ["Carbon Fiber Mass (kg)", self.tank_params.carbon_fiber_mass],
            ["Steel Mass (kg)", self.tank_params.steel_mass],
            ["Minimum Supply Pressure (Pa)", self.tank_params.
             min_supply_pressure],
            ["Venting Pressure (Pa)", self.tank_params.vent_pressure],
            ["Stoppage Reason", self.stop_reason]
            ]
        with open(filename, "w", newline="") as f:
            data = csv.writer(f)
            data.writerows(header_info)
        df_export.to_csv(filename, header=True, mode="a")

    def plot(self, x_axis: str, y_axes: Union[str, List[str]],
             colors: Union[str, List[str]] =
             ["r", "b", "g"]) -> Union[np.ndarray, plt.Axes]:
        """Plot the results of the simulation.

        Parameters
        ----------
        x_axis : str
            A string specifying what variable should be on the x-axis.
            See notes for valid inputs.

        y_axes : Union[str, List[str]]
            A string or a list of strings specifying what is to be plotted on
            the y-axis. See notes for valid inputs

        colors : Union[str, List[str]], optional
            A string or a list of strings specifying colors for the lines in
            the plot. The default is ["r", "b", "g"].

        Raises
        ------
        ValueError
            If more than 3 y-variables are specified to be plotted.

        Returns
        -------
        Union[np.ndarray, plt.Axes]
            A matplolib axis or a numpy array of several axes.

        Notes
        -----
        Below is a list of valid string inputs for ``x_axis`` and ``y_axes``
        along with the variables they represent.

            - ``t``: time (seconds)
            - ``p``: pressure (Pa)
            - ``T``: temperature (K)
            - ``na``: amount of fluid adsorbed (moles)
            - ``ng``: amount of fluid in gaseous form (moles)
            - ``nl``: amount of fluid in liquid form (moles)
            - ``ns``: amount of fluid in supercritical form (moles)
            - ``Qcoolreq``: cumulative amount of cooling required (J)
            - ``Qheatreq``: cumulative amount of heating required (J)
            - ``nout``: cumulative amount of fluid vented (moles)
            - ``Hout``: cumulative amount of vented fluid enthalpy (J)
            - ``nin``: cumulative amount of fluid inserted (moles)
            - ``Hin``: cumulative amount of inserted fluid enthalpy (J)
            - ``Qcooladd``: cumulative amount of user specified cooling (J)
            - ``Qheatadd``: cumulative amount of user specified heating (J)
            - ``Qleak``: cumulative amount of heat leakage into the tank (J)
            - ``ma``: mass of fluid adsorbed (kg)
            - ``mg``: mass of fluid in gaseous form (kg)
            - ``ml``: mass of fluid in liquid form (kg)
            - ``ms``: mass of fluid in supercritical form (kg)
            - ``mout``: cumulative mass of fluid vented (kg)
            - ``min``: cumulative mass of fluid inserted (kg)
            - ``na_dot``: the amount of fluid (moles) being adsorbed per
              second.
            - ``ng_dot``: the first derivative of the amount of fluid in
              gaseous form w.r.t. time. Its unit is mol/s.
            - ``nl_dot``: the first derivative of the amount of fluid in
              liquid form w.r.t. time. Its unit is mol/s
            - ``ns_dot``: the first derivative of the amount of fluid in
              supercritical form w.r.t. time. Its unit is mol/s.
            - ``Qcoolreq_dot``: the cooling power (W) required to maintain a
              constant pressure during refuel.
            - ``Qheatreq_dot``: the heating power (W) required to maintain a
              constant pressure during discharge.
            - ``nout_dot``: the rate at which fluid is being vented from the
              tank (mol/s).
            - ``Hout_dot``: the rate at which enthalpy is taken away by fluid
              leaving the tank (W).
            - ``nin_dot``: the rate at which fluid is entering the tank
              (mol/s).
            - ``Hin_dot``: the rate at which enthalpy is added by fluid
              fluid entering the tank (W).
            - ``Qcooladd_dot``: the user specified cooling power (W).
            - ``Qheatadd_dot``: the user specified heating power (W).
            - ``Qleak_dot``: the rate of heat leakage into the tank (W).
            - ``ma_dot``: the mass of fluid (kg) being adsorbed per second.
            - ``mg_dot``: the first derivative of the mass of fluid in
              gaseous form w.r.t. time. Its unit is kg/s.
            - ``ml_dot``: the first derivative of the mass of fluid in
              liquid form w.r.t. time. Its unit is kg/s.
            - ``ms_dot``: the first derivative of the mass of fluid in
              supercritical form w.r.t. time. Its unit is kg/s.
            - ``mout_dot``: the rate at which fluid is being vented from the
              tank (kg/s).
            - ``min_dot``: the rate at which fluid is being inserted into the
              tank (kg/s).

        """
        if isinstance(y_axes, str):
            y_axes = [y_axes]
        if isinstance(colors, str):
            colors = [colors]
        if len(y_axes) > 3:
            raise ValueError("You cannot fit more than 3 "
                             "y-variables in a single plot")
        y_axes = [SimResults.short_colnames[y] for y in y_axes]
        x_axis = SimResults.short_colnames[x_axis]

        if len(y_axes) > 0:
            fig, ax = plt.subplots()
            ax.set_xlabel(SimResults.fancy_colnames_dict[x_axis])
            ax.set_ylabel(SimResults.fancy_colnames_dict[y_axes[0]],
                          color=colors[0])
            p1, = ax.plot(self.results_df[x_axis], self.results_df[y_axes[0]],
                          label=SimResults.fancy_colnames_dict[y_axes[0]],
                          color=colors[0])
            handles = [p1]
            ax.yaxis.get_offset_text().set_color(colors[0])
            axlist = []
            if len(y_axes) > 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(SimResults.fancy_colnames_dict[y_axes[1]],
                               color=colors[1])
                ax2.yaxis.get_offset_text().set_color(colors[1])
                p2, = ax2.plot(self.results_df[x_axis],
                               self.results_df[y_axes[1]],
                               label=SimResults.fancy_colnames_dict[y_axes[1]],
                               color=colors[1])
                handles = [p1, p2]
                axlist.append(ax2)
                if len(y_axes) > 2:
                    ax3 = ax.twinx()
                    ax3.set_ylabel(SimResults.fancy_colnames_dict[y_axes[2]],
                                   color=colors[2])
                    ax3.yaxis.get_offset_text().set_color(colors[2])
                    p3, = ax3.plot(self.results_df[x_axis],
                                   self.results_df[y_axes[2]],
                                   label=SimResults.
                                   fancy_colnames_dict[y_axes[2]],
                                   color=colors[2])
                    ax3.spines.right.set_position(("axes", 1.2))
                    ax3.yaxis.get_offset_text().set_position((1.3, 1.1))
                    fig.subplots_adjust(right=0.75)
                    handles = [p1, p2, p3]
                    axlist.append(ax3)
            ax.legend(handles=handles)
            axlist.append(ax)
            return np.array(axlist) if len(axlist) > 1 else ax

    @classmethod
    def combine(cls,
                sim_results_list: "List[SimResults]") -> "SimResults":
        """Combine the results of several simulations into a single object.

        Parameters
        ----------
        sim_results_list : "List[SimResults]"
            A list of SimResults objects from several different simulations.

        Returns
        -------
        SimResults
            A single object containing the combined simulation results.

        """
        list_of_df = [result.results_df for result in sim_results_list]
        concat_df = pd.concat(list_of_df, ignore_index=True)
        concat_df = concat_df.drop_duplicates(subset="time")
        concat_df = concat_df.sort_values("time")
        return cls(
                    pressure=concat_df["pressure"].to_numpy(),
                    temperature=concat_df["temperature"].to_numpy(),
                    time=concat_df["time"].to_numpy(),
                    moles_adsorbed=concat_df["moles_adsorbed"].to_numpy(),
                    moles_gas=concat_df["moles_gas"].to_numpy(),
                    moles_liquid=concat_df["moles_liquid"].to_numpy(),
                    moles_supercritical=concat_df["moles_supercritical"]
                    .to_numpy(),
                    tank_params=sim_results_list[-1].tank_params,
                    sim_type="Combined",
                    sim_params=None,
                    stop_reason=sim_results_list[-1].stop_reason,
                    inserted_amount=concat_df["inserted_amount"].to_numpy(),
                    flow_energy_in=concat_df["flow_energy_in"].to_numpy(),
                    cooling_required=concat_df["cooling_required"].to_numpy(),
                    heating_required=concat_df["heating_required"].to_numpy(),
                    cooling_additional=concat_df["cooling_additional"]
                    .to_numpy(),
                    heating_additional=concat_df["heating_additional"]
                    .to_numpy(),
                    heat_leak_in=concat_df["heat_leak_in"].to_numpy(),
                    vented_amount=concat_df["vented_amount"].to_numpy(),
                    vented_energy=concat_df["vented_energy"].to_numpy())
