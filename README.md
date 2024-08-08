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


## License

pytanksim is available under an LGPL v3+ open source license. See [FSF's website](https://www.gnu.org/licenses/licenses.html#LGPL) for more detail regarding what this license allows.

## Contact Information

If you have any questions or inquiries regarding this package, please contact Muhammad Irfan Maulana Kusdhany, Kyushu University, at [ikusdhany@kyudai.jp](mailto:ikusdhany@kyudai.jp).

## Acknowledgement
This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2132, and by an external research grant from Mitsubishi Fuso Truck \& Bus Corporation.