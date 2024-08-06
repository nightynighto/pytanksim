from setuptools import setup

setup(name = "pyTankSim",
      version = "1.0.0",
      description = "Simulates thermodynamics of fluid tank refueling and discharging.",
      packages =[
       "pytanksim",
       "pytanksim.utils",
       "pytanksim.classes",
          ],
      author = "Muhammad Irfan Maulana Kusdhany",
      author_email = "ikusdhany@kyudai.jp",
      install_requires =[
          "scipy",
          "CoolProp",
          "numpy",
          "assimulo",
          "lmfit",
          "pandas",
          "matplotlib",
          "tqdm"
          ],
      license="Apache Software License",
      python_requires='>3.7'
      )
