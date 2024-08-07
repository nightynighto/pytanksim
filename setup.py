from setuptools import setup

setup(name = "pytanksim",
      version = "1.0.0",
      description = "Simulates thermodynamics of fluid tank refueling and discharging.",
      packages =[
       "pytanksim",
       "pytanksim.utils",
       "pytanksim.classes",
          ],
      author = "Muhammad Irfan Maulana Kusdhany, Kyushu University",
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
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows"
    ],
      license="LGPLv3+",
      license_files=('COPYING.txt', 'COPYING.LESSER.txt', 'NOTICE.txt'),
      python_requires='>3.7'
      )
