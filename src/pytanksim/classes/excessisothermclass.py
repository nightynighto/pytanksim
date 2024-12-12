# -*- coding: utf-8 -*-
"""Contains the ExcessIsotherm class."""
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
__all__ = ["ExcessIsotherm"]

import numpy as np
from typing import List


class ExcessIsotherm:
    """Stores experimental excess isotherm measurement results.

    This class can be provided values directly in Python or it can import
    the values from a csv file.

    Attributes
    ----------
    adsorbate : str
        Name of the adsorbate gas.

    sorbent : str
        Name of the sorbent material.

    temperature : float
        Temperature (K) at which the isotherm was measured.

    loading : List[float]
        A list of excess adsorption values (mol/kg).

    pressure : list[float]
        A list of pressures (Pa) corresponding to points at which the excess
        adsorption values were measured.

    """

    def __init__(self,
                 adsorbate: str,
                 sorbent: str,
                 temperature: float,
                 loading: List[float],
                 pressure: List[float]) -> "ExcessIsotherm":
        """Initialize the ExcessIsotherm class.

        Parameters
        ----------
        adsorbate : str
            Name of the adsorbate gas.

        sorbent : str
            Name of the sorbent material.

        temperature : float
            Temperature (K) at which the isotherm was measured.

        loading : List[float]
            A list of excess adsorption values (mol/kg).

        pressure : list[float]
            A list of pressures (Pa) corresponding to points at which the
            excess adsorption values were measured.

        Raises
        ------
        ValueError
            If the lengths of the loading and pressure data don't match.

        Returns
        -------
        ExcessIsotherm
            A class which stores experimental excess adsorption data.

        """
        self.adsorbate = adsorbate
        self.sorbent = sorbent
        self.temperature = temperature
        self.loading = loading
        self.pressure = pressure

        if not(len(loading) == len(pressure)):
            raise ValueError("The lengths of loading data and pressure"
                             "data don't match!")

    @classmethod
    def from_csv(cls,
                 filename: str,
                 adsorbate: str,
                 sorbent: str,
                 temperature: float) -> "ExcessIsotherm":
        """
        Import loading and pressure data from a csv file.

        Parameters
        ----------
        filename : str
            Path leading to the file from which the data is to be imported.

        adsorbate : str
            Name of adsorbate gas.

        sorbent : str
            Name of sorbent material.

        temperature : float
            Temperature (K) at which the data was measured.

        Returns
        -------
        ExcessIsotherm
            A class which stores experimental excess adsorption data.

        """
        dataP = np.loadtxt(filename, dtype="float", usecols=[0], skiprows=1,
                           delimiter=",", encoding="utf-8")
        dataAds = np.loadtxt(filename, dtype="float", usecols=[1], skiprows=1,
                             delimiter=",", encoding="utf-8")
        return cls(adsorbate, sorbent, temperature,
                   loading=dataAds, pressure=dataP)
