# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:44:06 2023

@author: nextf
"""

__all__ = ["ExcessIsotherm"]

import numpy as np
from typing import List


class ExcessIsotherm:
    def __init__(self,
                 adsorbate : str,
                 sorbent : str,
                 temperature : float,
                 loading: List[float] = None,
                 pressure: List[float] = None):
        """
        Initializes the ExcessIsotherm class.
        This class stores excess isotherm measurement results,
        and can import it from files.

        Parameters
        ----------
        adsorbate : str
            Name of adsorbate fluid.
        sorbent : str
            Name of sorbent sample.
        temperature : float
            Temperature of isotherm measurement in K.
        loading : List[float] | np.ndarray[float], optional
            Array of amount adsorbed on the adsorbate in mol/kg. The default is None.
        pressure : List[float] | np.ndarray[float], optional
            Array of measurement pressure in Pa. The default is None.

        Returns
        -------
        None.

        """
        
    
        
        self.adsorbate = adsorbate
        self.sorbent = sorbent
        self.temperature = temperature
        self.loading = loading
        self.pressure = pressure
        
        assert len(loading) == len(pressure)
        
    
    @classmethod
    def from_csv(cls,
                 filename : str,
                 adsorbate : str,
                 sorbent : str, 
                 temperature : float):
        
        dataP = np.loadtxt(filename,dtype="float",usecols=[0],skiprows=1,delimiter=",",encoding="utf-8")
        dataAds = np.loadtxt(filename,dtype="float",usecols=[1],skiprows=1,delimiter=",",encoding="utf-8")
        return cls(adsorbate, sorbent, temperature,
                   loading = dataAds, pressure = dataP)