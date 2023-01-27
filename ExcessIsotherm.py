# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:44:06 2023

@author: nextf
"""

import numpy as np

class ExcessIsotherm:
    def __init__(self, adsorbate, sorbent, temperature,
                 loading=None, pressure=None):
        self.adsorbate = adsorbate
        self.sorbent = sorbent
        self.temperature = temperature
        self.loading = loading
        self.pressure = pressure
    
    @classmethod
    def from_csv(cls, filename, adsorbate, sorbent, temperature):
        dataP = np.loadtxt(filename,dtype="float",usecols=[0],skiprows=1,delimiter=",",encoding="utf-8")
        dataAds = np.loadtxt(filename,dtype="float",usecols=[1],skiprows=1,delimiter=",",encoding="utf-8")
        return cls(adsorbate, sorbent, temperature, dataAds, dataP)