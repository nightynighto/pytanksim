# pytanksim
### Python package for simulating sorbent-filled hydrogen (and other fluids) storage tanks

pytanksim is a package which calculates the mass and energy balances in fluid storage tanks and simulates the dynamics during processes such as refueling, discharging, dormancy, and boil-off.

It is also equipped to fit gas adsorption data on nanoporous materials to adsorption models, which then allows the inclusion of sorbent materials in the dynamic simulation.

## Getting Started

### Dependencies

The installation of pytanksim requires:
- python versions 3.7 and above
- CoolProp
- pandas 
- numpy>=1.6.1
- scipy
- tqdm
- assimulo>=3.0
- matplotlib

### Installation

Pytanksim is listed on PyPI, so it should be possible in theory to install it with a single command via pip. Unfortunately, one of its dependencies, assimulo, tend to cause issues if one were to try installing pytanksim directly via pip, since the PyPI version of assimulo is not up to date and seems to be broken for newer versions of python, therefore causing pytanksim's install to fail. However, the build of assimulo on Anaconda seems to be well-maintained and working fine.

Thus, the easiest way to get pytanksim running if you don't have python installed on your computer, is to install python via [Anaconda](https://www.anaconda.com) or [Miniconda](https://conda.io/miniconda.html). In Anaconda or Miniconda, you can create a virtual environment that contains python version 3.7 using the following command on the Anaconda prompter (just change the myenv name to whatever you want to name your working environment):

```bash
conda create -n myenv python=3.7 

```

Then, activate this new virtual environment.

```bash
conda activate myenv

```

You can then write the following to install a working version of assimulo in this virtual environment:

```bash
conda install conda-forge::assimulo

```

Finally, you can install pytanksim to this virtual environment.


```bash
pip install pytanksim

```

You should be able to use pytanksim in this virtual environment after this!

If you need an Integrated Development Environment (IDE), you need to install the IDE through Anaconda within the virtual environment as well. For more information on how to work with virtual environments via Anaconda, you may want to see the [Anaconda website](https://docs.anaconda.com/working-with-conda/environments/).

As an alternative to working with Anaconda, you may also install assimulo directly from [the source](https://github.com/modelon-community/Assimulo/blob/master/INSTALL) and then you can pip install pytanksim without having to use a virtual environment, but that process is generally more involved.

### Validation Case Studies

Several case studies are available under the validation folder which illustrates pytanksim's various functionalities.
First time users are recommended to go through the example case studies.

#### Datasets

The data used in the case studies are taken from the following papers:

- Petitpas, G., Bénard, P., Klebanoff, L. E., Xiao, J., & Aceves, S. (2014).
A comparative analysis of the cryo-compression and cryo-adsorption hydrogen
storage methods. International Journal of Hydrogen Energy, 39(20),
10564–10584. https://doi.org/10.1016/j.ijhydene.2014.04.200 . Copyright 2014 Elsevier.

- Ahluwalia, R. K., & Peng, J. K. (2008). Dynamics of cryogenic hydrogen
storage in insulated pressure vessels for automotive applications.
International Journal of Hydrogen Energy, 33(17), 4622–4633.
https://doi.org/10.1016/j.ijhydene.2008.05.090 . Copyright 2008 Elsevier.

- Sahoo, P. K., John, M., Newalkar, B. L., Choudhary, N. V., &
Ayappa, K. G. (2011). Filling Characteristics for an Activated Carbon
Based Adsorbed Natural Gas Storage System. Industrial & Engineering
Chemistry Research, 50(23), 13000–13011. https://doi.org/10.1021/ie200241x . 
Copyright 2011 American Chemical Society.

- Xiao, J., Zhou, Z., Cossement, D., Bénard, P., & Chahine, R. (2013).
Lumped parameter model for charge–discharge cycle of adsorptive hydrogen
storage system. International Journal of Heat and Mass Transfer, 64,
245–253. https://doi.org/10.1016/j.ijheatmasstransfer.2013.04.029 . Copyright 2013 Elsevier.

- Richard, M.-A., Bénard, P., & Chahine, R. (2009). Gas adsorption process
in activated carbon over a wide temperature range above the critical point.
Part 1: Modified Dubinin-Astakhov model. Adsorption, 15, 43–51. Copyright 2009 Springer.

- Purewal, J., Liu, D., Sudik, A., Veenstra, M., Yang, J., Maurer, S., Müller,
U., & Siegel, D. J. (2012). Improved Hydrogen Storage and Thermal Conductivity
in High-Density MOF-5 Composites. The Journal of Physical Chemistry C, 116(38),
20199–20212. https://doi.org/10.1021/jp305524f. Copyright 2012 American Chemical Society.

- Zhou, W., Wu, H., Hartman, M. R., & Yildirim, T. (2007). Hydrogen and Methane
Adsorption in Metal−Organic Frameworks: A High-Pressure Volumetric Study.
The Journal of Physical Chemistry C, 111(44), 16131–16137.
https://doi.org/10.1021/jp074889i . Copyright 2007 American Chemical Society.

- Kusdhany, M.I.M., Ma, Z., Mufundirwa, A., Li, H.-W., Sasaki, K., Hayashi, A., 
Lyth, S.M., 2022. Hydrogen and carbon dioxide uptake on scalable and 
inexpensive microporous carbon foams. 
Microporous and Mesoporous Materials 343, 112141. https://doi.org/10.1016/j.micromeso.2022.112141


Please cite them if you use the data from the case studies in your own work.
If you wish to redistribute the data for your own work, you are 
advised to contact the original authors or publishers for the copyright
information.

## License

pytanksim is available under an LGPL v3+ open source license. See [FSF's website](https://www.gnu.org/licenses/licenses.html#LGPL) for more detail regarding what this license allows.

## Contact Information

If you have any questions or inquiries regarding this package, please contact Muhammad Irfan Maulana Kusdhany, Kyushu University, at [ikusdhany@kyudai.jp](mailto:ikusdhany@kyudai.jp).

## Acknowledgement
This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2132, and by an external research grant from Mitsubishi Fuso Truck \& Bus Corporation.