# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:21:18 2023

@author: nextf
"""

from setuptools import setup

setup(name = "pyTankSim",
      version = "1.0.0",
      description = "Simulates thermodynamics of fluid tank refueling and discharging.",
      packages =[
       "pytanksim",
       "pytanksim.utils",
       "pytanksim.classes"
       "pytanksim.mpta"
          ],
      author = "Muhammad Irfan Maulana Kusdhany",
      author_email = "ikusdhany@kyudai.jp",
      install_requires =[
          "scipy",
          "CoolProp",
          "numpy",
          "assimulo",
          "lmfit",
          "pandas"
          ],
      package_dir = {"" : "pytanksim"},
      python_requires='>3.7'
      )