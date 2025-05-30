pytanksim.classes
=================

.. py:module:: pytanksim.classes

.. autoapi-nested-parse::

   Contains the various classes making up pytanksim.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/pytanksim/classes/basesimclass/index
   /autoapi/pytanksim/classes/excessisothermclass/index
   /autoapi/pytanksim/classes/fluidsorbentclasses/index
   /autoapi/pytanksim/classes/onephasefluidsimclasses/index
   /autoapi/pytanksim/classes/onephasesorbentsimclasses/index
   /autoapi/pytanksim/classes/simresultsclass/index
   /autoapi/pytanksim/classes/storagetankclasses/index
   /autoapi/pytanksim/classes/twophasefluidsimclasses/index
   /autoapi/pytanksim/classes/twophasesorbentsimclasses/index


Classes
-------

.. autoapisummary::

   pytanksim.classes.ExcessIsotherm
   pytanksim.classes.StoredFluid
   pytanksim.classes.SorbentMaterial
   pytanksim.classes.ModelIsotherm
   pytanksim.classes.MDAModel
   pytanksim.classes.DAModel
   pytanksim.classes.StorageTank
   pytanksim.classes.SorbentTank
   pytanksim.classes.SimResults
   pytanksim.classes.SimParams
   pytanksim.classes.BoundaryFlux
   pytanksim.classes.BaseSimulation
   pytanksim.classes.OnePhaseSorbentSim
   pytanksim.classes.OnePhaseSorbentDefault
   pytanksim.classes.OnePhaseSorbentVenting
   pytanksim.classes.OnePhaseSorbentCooled
   pytanksim.classes.OnePhaseSorbentHeatedDischarge
   pytanksim.classes.TwoPhaseSorbentSim
   pytanksim.classes.TwoPhaseSorbentDefault
   pytanksim.classes.TwoPhaseSorbentVenting
   pytanksim.classes.TwoPhaseSorbentCooled
   pytanksim.classes.TwoPhaseSorbentHeatedDischarge
   pytanksim.classes.OnePhaseFluidSim
   pytanksim.classes.OnePhaseFluidDefault
   pytanksim.classes.OnePhaseFluidVenting
   pytanksim.classes.OnePhaseFluidCooled
   pytanksim.classes.OnePhaseFluidHeatedDischarge
   pytanksim.classes.TwoPhaseFluidSim
   pytanksim.classes.TwoPhaseFluidDefault
   pytanksim.classes.TwoPhaseFluidVenting
   pytanksim.classes.TwoPhaseFluidCooled
   pytanksim.classes.TwoPhaseFluidHeatedDischarge


Package Contents
----------------

.. py:class:: ExcessIsotherm(adsorbate, sorbent, temperature, loading, pressure)

   Stores experimental excess isotherm measurement results.

   This class can be provided values directly in Python or it can import
   the values from a csv file.

   .. attribute:: adsorbate

      Name of the adsorbate gas.

      :type: str

   .. attribute:: sorbent

      Name of the sorbent material.

      :type: str

   .. attribute:: temperature

      Temperature (K) at which the isotherm was measured.

      :type: float

   .. attribute:: loading

      A list of excess adsorption values (mol/kg).

      :type: List[float]

   .. attribute:: pressure

      A list of pressures (Pa) corresponding to points at which the excess
      adsorption values were measured.

      :type: list[float]

   Initialize the ExcessIsotherm class.

   :param adsorbate: Name of the adsorbate gas.
   :type adsorbate: str
   :param sorbent: Name of the sorbent material.
   :type sorbent: str
   :param temperature: Temperature (K) at which the isotherm was measured.
   :type temperature: float
   :param loading: A list of excess adsorption values (mol/kg).
   :type loading: List[float]
   :param pressure: A list of pressures (Pa) corresponding to points at which the
                    excess adsorption values were measured.
   :type pressure: list[float]

   :raises ValueError: If the lengths of the loading and pressure data don't match.

   :returns: A class which stores experimental excess adsorption data.
   :rtype: ExcessIsotherm


   .. py:method:: from_csv(filename, adsorbate, sorbent, temperature)
      :classmethod:


      Import loading and pressure data from a csv file.

      :param filename: Path leading to the file from which the data is to be imported.
      :type filename: str
      :param adsorbate: Name of adsorbate gas.
      :type adsorbate: str
      :param sorbent: Name of sorbent material.
      :type sorbent: str
      :param temperature: Temperature (K) at which the data was measured.
      :type temperature: float

      :returns: A class which stores experimental excess adsorption data.
      :rtype: ExcessIsotherm



.. py:class:: StoredFluid(fluid_name, EOS = 'HEOS', mole_fractions = None)

   A class to calculate the properties of the fluid being stored.

   .. attribute:: fluid_name

      The name of the fluid being stored which corresponds to fluid names
      in the package CoolProp.

      :type: str

   .. attribute:: EOS

      The name of the equation of state to be used for the calculations
      of fluid properties by the package CoolProp.

      :type: str

   .. attribute:: backend

      The CoolProp backend used for calculation of fluid properties at
      various conditions.

      :type: CoolProp.AbstractState

   Initialize a StoredFluid object.

   :param fluid_name: Name of the fluid. Valid fluid names that work with CoolProp can be
                      found here:
                      http://www.coolprop.org/fluid_properties/PurePseudoPure.html
   :type fluid_name: str, optional
   :param EOS: Name of the equation of state to be used for calculations.
               Default is the Helmholtz Equation of State (HEOS).
   :type EOS: str, optional
   :param mole_fraction: List of mole fractions of components in a mixture.
   :type mole_fraction: List

   :returns: A class to calculate the properties of the fluid being stored.
   :rtype: StoredFluid


   .. py:method:: fluid_property_dict(p, T)

      Generate a dictionary of fluid properties using CoolProp.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K)
      :type T: float

      :returns: Dictionary containing several fluid properties needed for various
                calculations in pytanksim.
      :rtype: Dict[str, float]

      .. rubric:: Notes

      Below is a list of keys and the variables they contain for the output
      dictionary.

      - ``hf``: enthalpy (J/mol)
      - ``drho_dp``: first partial derivative of density (mol/m^3) w.r.t.
        pressure (Pa)
      - ``drho_dT``: first partial derivative of density (mol/m^3) w.r.t.
        temperature (K)
      - ``rhof``: density (mol/m^3)
      - ``dh_dp``: first partial derivative of enthalpy (J/mol) w.r.t.
        pressure (Pa)
      - ``dh_dT``: first partial derivative of enthalpy (J/mol) w.r.t.
        temperature (K)
      - ``uf``: internal energy (J/mol)
      - ``du_dp``: first partial derivative of internal energy (J/mol) w.r.t.
        pressure (Pa)
      - ``du_dT``: first partial derivative of internal energy (J/mol)
        w.r.t. temperature (K)
      - ``MW``: molar mass (kg/mol)



   .. py:method:: saturation_property_dict(T, Q = 0)

      Generate a dictionary of fluid properties at saturation.

      :param T: Temperature in K.
      :type T: float
      :param Q: Vapor quality of the fluid being stored.
      :type Q: float

      :returns: A dictionary containing the fluid properties at saturation
                at a given temperature.
      :rtype: Dict[str, float]

      .. rubric:: Notes

      Below is a list of keys and the variables they contain for the output
      dictionary.

      - ``psat``: saturation vapor pressure (Pa)
      - ``dps_dT``: first derivative of the saturation vapor pressure (Pa)
        w.r.t. temperature (K).
      - ``hf``: enthalpy (J/mol)
      - ``drho_dp``: first partial derivative of density (mol/m^3) w.r.t.
        pressure (Pa)
      - ``drho_dT``: first partial derivative of density (mol/m^3) w.r.t.
        temperature (K)
      - ``rhof``: density (mol/m^3)
      - ``dh_dp``: first partial derivative of enthalpy (J/mol) w.r.t.
        pressure (Pa)
      - ``dh_dT``: first partial derivative of enthalpy (J/mol) w.r.t.
        temperature (K)
      - ``uf``: internal energy (J/mol)
      - ``du_dp``: first partial derivative of internal energy (J/mol) w.r.t.
        pressure (Pa)
      - ``du_dT``: first partial derivative of internal energy (J/mol)
        w.r.t. temperature (K)
      - ``MW``: molar mass (kg/mol)



   .. py:method:: determine_phase(p, T)

      Determine the phase of the fluid being stored.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: String that could either be "Supercritical", "Gas", "Liquid",
                or "Saturated" depending on the bulk fluid phase.
      :rtype: str



.. py:class:: SorbentMaterial(skeletal_density, bulk_density, specific_surface_area, model_isotherm, mass = 0, molar_mass = 0.01201, Debye_temperature = 1500, heat_capacity_function = None)

   Class containing the properties of a sorbent material.

   .. attribute:: mass

      Mass of sorbent (kg).

      :type: float

   .. attribute:: skeletal_density

      Skeletal density of the sorbent (kg/m^3).

      :type: float

   .. attribute:: bulk_density

      Tapped/compacted bulk density of the sorbent (kg/m^3).

      :type: float

   .. attribute:: specific_surface_area

      Specific surface area of the sorbent (m^2/g).

      :type: float

   .. attribute:: model_isotherm

      Model of fluid adsorption on the sorbent.

      :type: ModelIsotherm

   .. attribute:: molar_mass

      Molar mass of the sorbent material in kg/mol. The default is 12.01E-3
      which corresponds to carbon materials.

      :type: float, optional

   .. attribute:: Debye_temperature

      The Debye temperature (K) determining the specific heat of the sorbent
      at various temperatures. The default is 1500, the value for carbon.

      :type: float, optional

   .. attribute:: heat_capacity_function

      A function which takes in the temperature (K) of the sorbent and
      returns its specific heat capacity (J/(kg K)). If specified, this
      function will override the Debye model for specific heat calculation.
      The default is None.

      :type: Callable[[float], float], optional

   Initialize the SorbentMaterial class.

   :param skeletal_density: Skeletal density of the sorbent (kg/m^3).
   :type skeletal_density: float
   :param bulk_density: Tapped/compacted bulk density of the sorbent (kg/m^3).
   :type bulk_density: float
   :param specific_surface_area: Specific surface area of the sorbent (m^2/g).
   :type specific_surface_area: float
   :param model_isotherm: Model of fluid adsorption on the sorbent.
   :type model_isotherm: ModelIsotherm
   :param mass: Mass of sorbent (kg). The default is None.
   :type mass: float, optional
   :param molar_mass: Molar mass of the sorbent material. The default is 12.01E-3 which
                      corresponds to carbon materials.
   :type molar_mass: float, optional
   :param Debye_temperature: The Debye temperature determining the specific heat of the sorbent
                             at various temperatures. The default is 1500, the value for carbon.
   :type Debye_temperature: float, optional
   :param heat_capacity_function: A function which takes in the temperature (K) of the sorbent and
                                  returns its specific heat capacity (J/(kg K)). If specified, this
                                  function will override the Debye model for specific heat
                                  calculation. The default is None.
   :type heat_capacity_function: Callable, optional

   :returns: Class containing the properties of a sorbent material.
   :rtype: SorbentMaterial


.. py:class:: ModelIsotherm

   A base class for model isotherm objects.

   Contains methods to calculate various thermodynamic properties of
   the adsorbed phase.



   .. py:method:: pressure_from_absolute_adsorption(n_abs, T, p_max_guess = 35000000.0)

      Calculate a pressure value corresponding to an adsorbed amount.

      :param n_abs: Amount adsorbed (mol/kg).
      :type n_abs: float
      :param T: Temperature (K).
      :type T: float
      :param p_max_guess: Maximum pressure (Pa) for the optimization. The default is 20E6.
                          If the value provided is larger than the maximum that can be
                          handled by the CoolProp backend, it will take the maximum that
                          can be handled by the CoolProp backend.
      :type p_max_guess: float, optional

      :returns: Pressure (Pa) corresponding to the specified adsorbed amount
                and temperature value.
      :rtype: float



   .. py:method:: isosteric_enthalpy(p, T, q = 1)

      Calculate isosteric adsorbed enthalpy (J/mol).

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: Isosteric enthalpy of adsorption (J/mol).
      :rtype: float



   .. py:method:: isosteric_internal_energy(p, T, q = 1)

      Calculate the isosteric internal energy of the adsorbed phase.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: Isosteric internal energy of the adsorbed phase (J/mol).
      :rtype: float



   .. py:method:: _derivfunc(func, var, point, qinit, stepsize)

      Calculate the first partial derivative.

      It automatically decides the direction of the derivative so that the
      evaluations are done for fluids at the same phases. Otherwise, there
      will be discontinuities in the fluid properties at different phases
      which causes the resulting derivative values to be invalid.




   .. py:method:: _derivfunc_second(func, point, qinit, stepsize)

      Calculate the second partial derivative.

      It automatically decides the direction of the derivative so that the
      evaluations are done for fluids at the same phases. Otherwise, there
      will be discontinuities in the fluid properties at different phases
      which causes the resulting derivative values to be invalid.




   .. py:method:: isosteric_energy_temperature_deriv(p, T, q = 1, stepsize = 0.001)

      Calculate first derivative of isosteric internal energy w.r.t. T.

      This function calculates the first partial derivative of the isosteric
      internal energy of the adsorbed phase (J/mol) w.r.t. temperature (K).

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional
      :param stepsize: Stepsize for numerical derivative. The default is 1E-3.
      :type stepsize: float, optional

      :returns: The first partial derivative of the isosteric internal energy
                of the adsorbed phase (J/mol) w.r.t. temperature (K).
      :rtype: float



   .. py:method:: differential_energy(p, T, q = 1)

      Calculate the differential energy of adsorption (J/mol).

      The calculation is based on Myers & Monson [1]_.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: The differential energy of adsorption (J/mol).
      :rtype: float

      .. rubric:: Notes

      .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
         the case for absolute adsorption as the basis for thermodynamic
         analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
         doi: 10.1007/s10450-014-9604-1.



   .. py:method:: differential_heat(p, T, q = 1)

      Calculate the differential heat of adsorption (J/mol).

      The calculation is based on Myers & Monson [1]_.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: The differential heat of adsorption (J/mol).
      :rtype: float

      .. rubric:: Notes

      .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
         the case for absolute adsorption as the basis for thermodynamic
         analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
         doi: 10.1007/s10450-014-9604-1.



   .. py:method:: internal_energy_adsorbed(p, T, q = 1)

      Calculate the molar integral internal energy of adsorption (J/mol).

      The calculation is based on Myers & Monson [1]_.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: The differential energy of adsorption (J/mol).
      :rtype: float

      .. rubric:: Notes

      .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
         the case for absolute adsorption as the basis for thermodynamic
         analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
         doi: 10.1007/s10450-014-9604-1.



   .. py:method:: areal_immersion_energy(T)

      Calculate the areal energy of immersion (J/m^2).

      The calculation is based on the one written in Rouquerol et al. [1]_.

      :param T: Temperature (K).
      :type T: float

      :returns: Areal energy of immersion (J/m^2)
      :rtype: float



.. py:class:: MDAModel(sorbent, stored_fluid, nmax, f0, alpha, beta, va, m = 2, k = 2, va_mode = 'Constant', f0_mode = 'Constant')

   Bases: :py:obj:`ModelIsotherm`


   A class for the Modified Dubinin-Astakhov model for adsorption.

   A key modification compared to the DA model is the use of the enthalpic and
   entropic factors to calculate the adsorption energy as a function of
   temperature instead of treating it as a constant.

   Initialize the MDAModel class.

   :param sorbent: Name of the sorbent material.
   :type sorbent: str
   :param stored_fluid: Object to calculate the thermophysical properties of the adsorbate.
   :type stored_fluid: StoredFluid
   :param nmax: Maximum adsorbed amount (mol/kg) at saturation.
   :type nmax: float
   :param f0: Fugacity at saturation (Pa).
   :type f0: float
   :param alpha: The empirical enthalpic factor for determining the characteristic
                 energy of adsorption.
   :type alpha: float
   :param beta: The empirical entropic factor for determining the characteristic
                energy of adsorption.
   :type beta: float
   :param va: The volume of the adsorbed phase (m^3/kg).
   :type va: float
   :param m: The empirical heterogeneity parameter for the Dubinin-Astakhov
             model. The default is 2.
   :type m: float, optional
   :param k: The empirical heterogeneity parameter for Dubinin's approximation
             of the saturation fugacity above critical temperatures. The default
             is 2.
   :type k: float, optional
   :param va_mode: Determines how the adsorbed phase density is calculated. "Ozawa"
                   uses Ozawa's approximation to calculate the adsorbed phase density.
                   "Constant" assumes a constant adsorbed phase volume. The default is
                   "Constant".
   :type va_mode: str, optional
   :param f0_mode: Determines how the fugacity at saturation is calculated. "Dubinin"
                   uses Dubinin's approximation. "Constant" assumes a constant value
                   for the fugacity at saturation. The default is "Constant".
   :type f0_mode: str, optional

   :returns: An MDAModel object. It can calculate the excess and absolute
             adsorbed amounts at various pressures and temperatures, and it can
             provide thermophysical properties of the adsorbed phase.
   :rtype: MDAModel


   .. py:method:: dlnf0_dT(T)

      Calculate derivative of log saturation fugacity w.r.t. temperature.

      :param T: Temperature (K)
      :type T: float

      :returns: Derivative of log saturation fugacity w.r.t. temperature
      :rtype: float



   .. py:method:: f0_fun(T)

      Calculate saturation fugacity as a function of temperature.

      :param T: Temperature (K).
      :type T: float

      :returns: Saturation fugacity (Pa).
      :rtype: float



   .. py:method:: n_absolute(p, T)

      Calculate the absolute adsorbed amount at given conditions.

      :param p: Pressure (Pa)
      :type p: float
      :param T: Temperature (K)
      :type T: float

      :returns: Absolute adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: v_ads(p, T)

      Calculate the adsorbed phase volume at the given condtions.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: Adsorbed phase volume (m^3/kg)
      :rtype: float



   .. py:method:: n_excess(p, T, q = 1)

      Calculate the excess adsorbed amount at the given conditions.

      :param p: Pressure (Pa)
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 and 1. The
                default is 1.
      :type q: float, optional

      :returns: Excess adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: internal_energy_adsorbed(p, T, q = 1)

      Calculate the molar integral internal energy of adsorption (J/mol).

      The calculation is based on Myers & Monson [1]_.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: The molar integral energy of adsorption (J/mol).
      :rtype: float

      .. rubric:: Notes

      .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
          the case for absolute adsorption as the basis for thermodynamic
          analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
          doi: 10.1007/s10450-014-9604-1.



   .. py:method:: differential_energy_na(na, T)

      Calculate differential energy of adsorption analytically.

      :param na: Amount adsorbed (mol/kg)
      :type na: float
      :param T: Temperature (K)
      :type T: float

      :returns: Differential energy of adsorption (J/mol)
      :rtype: float



   .. py:method:: differential_energy(p, T, q = 1)

      Calculate differential energy of adsorption analytically.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality.
      :type q: float

      :returns: Differential energy of adsorption (J/mol).
      :rtype: float



   .. py:method:: from_ExcessIsotherms(ExcessIsotherms, stored_fluid = None, sorbent = None, nmaxguess = 71.6, f0guess = 1470000000.0, alphaguess = 3080, betaguess = 18.9, vaguess = 0.00143, mguess = 2.0, kguess = 2.0, va_mode = 'Fit', f0_mode = 'Fit', m_mode = 'Fit', k_mode = 'Fit', beta_mode = 'Fit', pore_volume = 0.003, verbose = True)
      :classmethod:


      Fit the MDA model from a list of excess adsorption data.

      :param ExcessIsotherms: A list of ExcessIsotherm objects which contain measurement
                              data at various temperatures.
      :type ExcessIsotherms: List[ExcessIsotherm]
      :param stored_fluid: Object for calculating the properties of the adsorbate. The default
                           is None. If None, the StoredFluid object inside of one of the
                           ExcessIsotherm objects passed will be used.
      :type stored_fluid: StoredFluid, optional
      :param sorbent: Name of sorbent material. The default is None. If None, name will
                      be taken from one of the ExcessIsotherm objects passed.
      :type sorbent: str, optional
      :param nmaxguess: The initial guess for the maximum adsorbed amount (mol/kg). The
                        default is 71.6.
      :type nmaxguess: float, optional
      :param f0guess: The initial guess for the fugacity at saturation (Pa). The default
                      is 1470E6.
      :type f0guess: float, optional
      :param alphaguess: The initial guess for the enthalpy factor determining the
                         characteristic energy of adsorption. The default is 3080.
      :type alphaguess: float, optional
      :param betaguess: The initial guess for the entropy factor determining the
                        characteristic energy of adsorption. The default is 18.9.
      :type betaguess: float, optional
      :param vaguess: Initial guess for the adsorbed phase volume (m^3/kg). The default
                      is 0.00143.
      :type vaguess: float, optional
      :param mguess: The initial guess for the heterogeneity parameter of the
                     Dubinin-Astakhov equation. The default is 2.0.
      :type mguess: float, optional
      :param kguess: The initial guess for the heterogeneity parameter of Dubinin's
                     approximation method for saturation fugacity. The default is 2.0.
      :type kguess: float, optional
      :param va_mode: Determines how the volume of the adsorbed phase (va) is
                      calculated. If "Fit", va is a constant to be fitted
                      statistically. If "Ozawa", Ozawa's approximation is used to
                      calculate va and va is not a fitting parameter. If "Constant",
                      the user supplied value for vaguess is taken as the volume.
                      The default is "Fit".
      :type va_mode: str, optional
      :param f0_mode: Determines how the fugacity at saturation (f0) is calculated. If
                      "Fit" then f0 is a constant to be statistically fitted to the data.
                      If "Dubinin" then Dubinin's approximation is used. If "Constant"
                      then the user supplied value for f0guess is used. The default is
                      "Fit".
      :type f0_mode: str, optional
      :param m_mode: Determines whether the heterogeneity parameter of the Dubinin-
                     Astakhov equation is taken as a user-supplied constant (if
                     "Constant") or a fitted parameter (if "Fit"). The default is "Fit".
      :type m_mode: str, optional
      :param k_mode: Determines whether the heterogeneity parameter of Dubinin's
                     approximation for the fugacity above the critical temperature is
                     taken as a user-supplied constant value (if "Constant") or as a
                     statistically fitted parameter (if "Fit"). The default is "Fit".
      :type k_mode: str, optional
      :param beta_mode: Determines whether the entropic factor determining the
                        characteristic energy of adsorption is taken as a user-supplied
                        constant (if "Constant") or as a fitted parameter (if "Fit"). The
                        default is "Fit".
      :type beta_mode: str, optional
      :param pore_volume: The experimentally measured pore volume of the sorbent material
                          (m^3/kg). It serves as the maximum possible physical value for the
                          parameters w0 and va. The default is 0.003.
      :type pore_volume: float, optional
      :param verbose: Determines whether or not the complete fitting quality report is
                      logged for the user. The default is True.
      :type verbose: bool, optional

      :returns: An MDAModel object. It can calculate the excess and absolute
                adsorbed amounts at various pressures and temperatures, and it can
                provide thermophysical properties of the adsorbed phase.
      :rtype: MDAModel



.. py:class:: DAModel(sorbent, stored_fluid, w0, f0, eps, m = 2, k = 2, rhoa = None, va = None, va_mode = 'Constant', rhoa_mode = 'Constant', f0_mode = 'Dubinin')

   Bases: :py:obj:`ModelIsotherm`


   A class for the Dubinin-Astakhov model for adsorption in micropores.

   .. attribute:: sorbent

      Name of sorbent material.

      :type: str

   .. attribute:: stored_fluid

      Object containing properties of the adsorbate.

      :type: StoredFluid

   .. attribute:: w0

      The volume of the adsorbed phase at saturation (m^3/kg).

      :type: float

   .. attribute:: f0

      The fugacity at adsorption saturation (Pa).

      :type: float

   .. attribute:: eps

      Characteristic energy of adsorption (J/mol).

      :type: float

   .. attribute:: m

      The empirical heterogeneity parameter for the Dubinin-Astakhov
      model. The default is 2.

      :type: float, optional

   .. attribute:: k

      The empirical heterogeneity parameter for Dubinin's approximation
      of the saturation fugacity above critical temperatures. The default
      is 2.

      :type: float, optional

   .. attribute:: rhoa

      The density of the adsorbed phase (mol/m^3). The default is None.
      If None, the value will be taken as the liquid density at 1 bar.

      :type: float, optional

   .. attribute:: va

      The volume of the adsorbed phase (m^3/kg). The default is None.
      If None and va_mode is "Constant", the va_mode will be switched to
      "Excess" and the va will be assumed to be 0.

      :type: float, optional

   .. attribute:: va_mode

      Determines how the adsorbed phase volume is calculated. "Excess"
      assumes that the adsorbed phase volume is 0, so the model
      calculates excess adsorption instead of absolute adsorption.
      "Constant" assumes a constant adsorbed phase volume. "Vary" will
      assume that the adsorbed phase volume varies according to the pore
      filling mechanism posited by the Dubinin-Astakhov equation. The
      default is "Constant", but if the parameter va is not specified it
      will switch to "Excess".

      :type: str, optional

   .. attribute:: rhoa_mode

      Determines how the adsorbed phase density is calculated. "Ozawa"
      uses Ozawa's approximation to calculate the adsorbed phase density.
      "Constant" assumes a constant adsorbed phase volume. The default is
      "Constant".

      :type: str, optional

   .. attribute:: f0_mode

      Determines how the fugacity at saturation is calculated. "Dubinin"
      uses Dubinin's approximation. "Constant" assumes a constant value
      for the fugacity at saturation. The default is "Dubinin".

      :type: str, optional

   Initialize the DAModel class.

   :param sorbent: Name of sorbent material.
   :type sorbent: str
   :param stored_fluid: Object containing properties of the adsorbate.
   :type stored_fluid: StoredFluid
   :param w0: The volume of the adsorbed phase at saturation (m^3/kg).
   :type w0: float
   :param f0: The fugacity at adsorption saturation (Pa).
   :type f0: float
   :param eps: Characteristic energy of adsorption (J/mol).
   :type eps: float
   :param m: The empirical heterogeneity parameter for the Dubinin-Astakhov
             model. The default is 2.
   :type m: float, optional
   :param k: The empirical heterogeneity parameter for Dubinin's approximation
             of the saturation fugacity above critical temperatures. The default
             is 2.
   :type k: float, optional
   :param va: The volume of the adsorbed phase (m^3/kg). The default is None.
   :type va: float, optional
   :param rhoa: The density of the adsorbed phase (mol/m^3). The default is None.
                If None, the value will be taken as the liquid density at 1 bar.
   :type rhoa: float, optional
   :param va_mode: Determines how the adsorbed phase volume is calculated. "Excess"
                   assumes that the adsorbed phase volume is 0, so the model
                   calculates excess adsorption instead of absolute adsorption.
                   "Constant" assumes a constant adsorbed phase volume. "Vary" will
                   assume that the adsorbed phase volume varies according to the pore
                   filling mechanism posited by the Dubinin-Astakhov equation. The
                   default is "Constant", but if the parameter va is not specified it
                   will switch to "Excess".
   :type va_mode: str, optional
   :param rhoa_mode: Determines how the adsorbed phase density is calculated. "Ozawa"
                     uses Ozawa's approximation to calculate the adsorbed phase density.
                     "Constant" assumes a constant adsorbed phase volume. The default is
                     "Constant".
   :type rhoa_mode: str, optional
   :param f0_mode: Determines how the fugacity at saturation is calculated. "Dubinin"
                   uses Dubinin's approximation. "Constant" assumes a constant value
                   for the fugacity at saturation. The default is "Dubinin".
   :type f0_mode: str, optional

   :returns: A DAModel object which can calculate excess and absolute adsorption
             at various conditions as well as the thermophysical properties of
             the adsorbed phase.
   :rtype: DAModel


   .. py:method:: f0_calc(T)

      Calculate the fugacity at saturation (Pa) at a given temperature.

      :param T: Temperature (K).
      :type T: float

      :returns: Fugacity at saturation (Pa).
      :rtype: float



   .. py:method:: dlnf0_dT(T)

      Calculate derivative of log saturation fugacity w.r.t. temperature.

      :param T: Temperature (K)
      :type T: float

      :returns: Derivative of log saturation fugacity w.r.t. temperature
      :rtype: float



   .. py:method:: rhoa_calc(T)

      Calculate the density of the adsorbed phase at a given temperature.

      :param T: Temperature (K).
      :type T: float

      :returns: The density of the adsorbed phase (mol/m^3).
      :rtype: float



   .. py:method:: v_ads(p, T)

      Calculate the volume of the adsorbed phase (m^3/kg).

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: Volume of the adsorbed phase (m^3/kg).
      :rtype: float



   .. py:method:: n_absolute(p, T)

      Calculate the absolute adsorbed amount at a given condition.

      :param p: Pressure(Pa).
      :type p: float
      :param T: Temperature(K).
      :type T: float

      :returns: Absolute adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: n_excess(p, T, q = 1)

      Calculate the excess adsorbed amount at a given condition.

      :param p: Pressure (Pa)
      :type p: float
      :param T: Temperature (K)
      :type T: float
      :param q: The vapor quality of the bulk adsorbate. Can vary between 0 and 1.
                The default is 1.
      :type q: float, optional

      :returns: Excess adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: differential_energy_na(na, T)

      Calculate differential energy of adsorption analytically.

      :param na: Amount adsorbed (mol/kg)
      :type na: float
      :param T: Temperature (K)
      :type T: float

      :returns: Differential energy of adsorption (J/mol)
      :rtype: float



   .. py:method:: differential_energy(p, T, q)

      Calculate differential energy of adsorption analytically.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality.
      :type q: float

      :returns: Differential energy of adsorption (J/mol).
      :rtype: float



   .. py:method:: internal_energy_adsorbed(p, T, q = 1)

      Calculate the molar integral internal energy of adsorption (J/mol).

      The calculation is based on Myers & Monson [1]_.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the bulk fluid. Can vary between 0 to 1.
                The default is 1.
      :type q: float, optional

      :returns: The differential energy of adsorption (J/mol).
      :rtype: float

      .. rubric:: Notes

      .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
         the case for absolute adsorption as the basis for thermodynamic
         analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
         doi: 10.1007/s10450-014-9604-1.



   .. py:method:: from_ExcessIsotherms(ExcessIsotherms, stored_fluid = None, sorbent = None, w0guess = 0.001, f0guess = 1470000000.0, epsguess = 3000, vaguess = 0.001, rhoaguess = None, mguess = 2.0, kguess = 2.0, rhoa_mode = 'Fit', f0_mode = 'Fit', m_mode = 'Fit', k_mode = 'Fit', va_mode = 'Excess', pore_volume = 0.003, verbose = True)
      :classmethod:


      Fit the DA model to a list of ExcessIsotherm data.

      :param ExcessIsotherms: A list containing ExcessIsotherm objects which contain measurement
                              data at various temperatures.
      :type ExcessIsotherms: List[ExcessIsotherm]
      :param stored_fluid: Object for calculating the properties of the adsorbate. The default
                           is None. If None, the StoredFluid object inside of one of the
                           ExcessIsotherm objects passed will be used.
      :type stored_fluid: StoredFluid, optional
      :param sorbent: Name of sorbent material. The default is None. If None, name will
                      be taken from one of the ExcessIsotherm objects passed.
      :type sorbent: str, optional
      :param w0guess: The initial guess for the adsorbed phase volume at saturation
                      (m^3/kg). The default is 0.001.
      :type w0guess: float, optional
      :param f0guess: The initial guess for the fugacity at saturation (Pa). The default
                      is 1470E6.
      :type f0guess: float, optional
      :param epsguess: The initial guess for the characteristic energy of adsorption
                       (J/mol). The default is 3000.
      :type epsguess: float, optional
      :param vaguess: The initial guess for the volume of the adsorbed phase (m^3/kg).
                      The default is 0.001.
      :type vaguess: float, optional
      :param rhoaguess: The initial guess for the adsorbed phase density (mol/m^3).
                        The default is None. If None, it will be taken as the liquid
                        density at 1 bar.
      :type rhoaguess: float, optional
      :param mguess: The initial guess for the heterogeneity parameter of the
                     Dubinin-Astakhov equation. The default is 2.0.
      :type mguess: float, optional
      :param kguess: The initial guess for the heterogeneity parameter of Dubinin's
                     approximation method for saturation fugacity. The default is 2.0.
      :type kguess: float, optional
      :param rhoa_mode: Determines how the density of the adsorbed phase (rhoa) is
                        calculated. If "Fit", rhoa is a constant to be fitted
                        statistically. If "Ozawa", Ozawa's approximation is used to
                        calculate rhoa and rhoa is not a fitting parameter. If "Constant",
                        the user supplied value for rhoaguess is taken as the density.
                        The default is "Fit".
      :type rhoa_mode: str, optional
      :param f0_mode: Determines how the fugacity at saturation (f0) is calculated. If
                      "Fit" then f0 is a constant to be statistically fitted to the data.
                      If "Dubinin" then Dubinin's approximation is used. If "Constant"
                      then the user supplied value for f0guess is used. The default is
                      "Fit".
      :type f0_mode: str, optional
      :param m_mode: Determines whether the heterogeneity parameter of the Dubinin-
                     Astakhov equation is taken as a user-supplied constant (if
                     "Constant") or a fitted parameter (if "Fit"). The default is "Fit".
      :type m_mode: str, optional
      :param k_mode: Determines whether the heterogeneity parameter of Dubinin's
                     approximation for the fugacity above the critical temperature is
                     taken as a user-supplied constant value (if "Constant") or as a
                     statistically fitted parameter (if "Fit"). The default is "Fit".
      :type k_mode: str, optional
      :param va_mode: Determines how the volume of the adsorbed phase is calculated. If
                      "Fit", the value is a statistically fitted constant. If "Constant",
                      the value is the user defined value vaguess. If "Vary", the value
                      varies w.r.t. pressure according to the micropore filling
                      mechanism posited by the Dubinin-Astakhov model. The default is
                      "Excess".
      :type va_mode: str, optional
      :param pore_volume: The experimentally measured pore volume of the sorbent material
                          (m^3/kg). It serves as the maximum possible physical value for the
                          parameters w0 and va. The default is 0.003.
      :type pore_volume: float, optional
      :param verbose: Determines whether or not the complete fitting quality report is
                      logged for the user. The default is True.
      :type verbose: bool, optional

      :returns: A DAModel object which can calculate excess and absolute adsorption
                at various conditions as well as the thermophysical properties of
                the adsorbed phase.
      :rtype: DAModel



.. py:class:: StorageTank(stored_fluid, aluminum_mass = 0, carbon_fiber_mass = 0, steel_mass = 0, vent_pressure = None, min_supply_pressure = 100000.0, thermal_resistance = 0, surface_area = 0, heat_transfer_coefficient = 0, volume = None, set_capacity = None, full_pressure = None, empty_pressure = None, full_temperature = None, empty_temperature = None, full_quality = 1, empty_quality = 1)

   Stores the properties of the storage tank.

   It also has methods to calculate useful quantities such as tank dormancy
   given a constant heat leakage rate, the internal energy of the fluid being
   stored at various conditions, etc.

   .. attribute:: volume

      Internal volume of the storage tank (m^3).

      :type: float

   .. attribute:: stored_fluid

      Object to calculate the thermophysical properties of the fluid
      being stored.

      :type: StoredFluid

   .. attribute:: aluminum_mass

      The mass of aluminum making up the tank walls (kg). The default is
      0.

      :type: float, optional

   .. attribute:: carbon_fiber_mass

      The mass of carbon fiber making up the tank walls (kg). The default
      is 0.

      :type: float, optional

   .. attribute:: steel_mass

      The mass of steel making up the tank walls (kg). The default is 0.

      :type: float, optional

   .. attribute:: vent_pressure

      The pressure (Pa) at which the fluid being stored must be vented.
      The default is None. If None, the value will be taken as the
      maximum value where the CoolProp backend can calculate the
      properties of the fluid being stored.

      :type: float, optional

   .. attribute:: min_supply_pressure

      The minimum supply pressure (Pa) for discharging simulations.The
      default is 1E5.

      :type: float, optional

   .. attribute:: thermal_resistance

      The thermal resistance of the tank walls (K/W). The default is 0.
      If 0, the value will not be considered in simulations.

      :type: Callable, optional

   .. attribute:: surface_area

      The surface area of the tank that is in contact with the
      environment (m^2). The default is 0.

      :type: float, optional

   .. attribute:: heat_transfer_coefficient

      The heat transfer coefficient of the tank surface (W/(m^2 K)).
      The default is 0.

      :type: Callable, optional

   Initialize a StorageTank object.

   :param stored_fluid: Object to calculate the thermophysical properties of the fluid
                        being stored.
   :type stored_fluid: StoredFluid
   :param aluminum_mass: The mass of aluminum making up the tank walls (kg). The default is
                         0.
   :type aluminum_mass: float, optional
   :param carbon_fiber_mass: The mass of carbon fiber making up the tank walls (kg). The default
                             is 0.
   :type carbon_fiber_mass: float, optional
   :param steel_mass: The mass of steel making up the tank walls (kg). The default is 0.
   :type steel_mass: float, optional
   :param vent_pressure: The pressure (Pa) at which the fluid being stored must be vented.
                         The default is None. If None, the value will be taken as the
                         maximum value where the CoolProp backend can calculate the
                         properties of the fluid being stored.
   :type vent_pressure: float, optional
   :param min_supply_pressure: The minimum supply pressure (Pa) for discharging simulations.The
                               default is 1E5.
   :type min_supply_pressure: float, optional
   :param thermal_resistance: A function which returns the thermal resistance of the tank
                              walls (K/W) as a function of tank pressure (Pa), tank
                              temperature (K), time (s), and temperature of surroundings (K). The
                              default is 0. If a float is provided, it will be converted to a
                              function which returns that value everywhere. If both this and
                              the arguments 'surface_area' and 'heat_transfer_coefficient' are
                              passed, two values of thermal resistance will be calculated and
                              the highest value between the two will be taken at each time step.
                              Thus, to avoid confusion, one should either: (a) use the other two
                              arguments together, or (b) use this one, but not both at the same
                              time.

                              If a callable is passed, it must have the signature::

                                  def tr_function(p, T, time, env_temp):
                                      # 'p' is tank pressure (Pa)
                                      # 'T' is tank temperature (K)
                                      # 'time' is the time elapsed within the simulation (s)
                                      # 'env_temp' is the temperature of surroundings (K)
                                      ....
                                      # Returned is the thermal resistance (K/W)
                                      return tr_value
   :type thermal_resistance: Callable or float, optional
   :param surface_area: The surface area of the tank that is in contact with the
                        environment (m^2). The default is 0.
   :type surface_area: float, optional
   :param heat_transfer_coefficient: A function which returns the heat transfer coefficient of the tank
                                     walls (W/(m^2 K)) as a function of tank pressure (Pa), tank
                                     temperature (K), time (s), and temperature of surroundings (K). The
                                     default is 0. If a float is  provided, it will be converted to a
                                     function which returns that value everywhere.

                                     If a callable is passed, it must have the signature::

                                         def htc_function(p, T, time, env_temp):
                                             # 'p' is tank pressure (Pa)
                                             # 'T' is tank temperature (K)
                                             # 'time' is the time elapsed within the simulation (s)
                                             # 'env_temp' is the temperature of surroundings (K)
                                             ....
                                             # Returned is the heat transfer coefficient (W/(m^2 K))
                                             return heat_transfer_coef
   :type heat_transfer_coefficient: Callable or float, optional
   :param volume: Internal volume of the storage tank (m^3). The default is None.
                  This value is required unless the set capacity and operating
                  conditions are defined, in which case the volume is calculated from
                  the capacity and operating conditions.
   :type volume: float, optional
   :param set_capacity: Set internal capacity of the storage tank (mol). The default is
                        None. If specified, this will override the user-specified tank
                        volume.
   :type set_capacity: float, optional
   :param full_pressure: Pressure (Pa) of the tank when it is considered full. The default
                         is None.
   :type full_pressure: float, optional
   :param empty_pressure: Pressure (Pa) of the tank when it is considered empty. The default
                          is None.
   :type empty_pressure: float, optional
   :param full_temperature: Temperature (K) of the tank when it is considered full. The
                            default is None.
   :type full_temperature: float, optional
   :param empty_temperature: Temperature (K) of the tank when it is considered empty. The
                             default is None.
   :type empty_temperature: float, optional
   :param full_quality: Vapor quality of the tank when it is considered full. The default
                        is 1 (Gas).
   :type full_quality: float, optional
   :param empty_quality: Vapor quality of the tank when it is considered empty. The default
                         is 1 (Gas).
   :type empty_quality: float, optional

   :raises ValueError: If any of the mass values provided are less than 0.
   :raises ValueError: If the vent pressure set is higher than what can be calculated by
       'CoolProp'.
   :raises ValueError: If neither the volume nor the complete capacity and the pressure
       and temperature swing conditions were provided.

   :returns: A storage tank object which can be passed as arguments to dynamic
             simulations and can calculate certain properties on its own.
   :rtype: StorageTank


   .. py:method:: capacity(p, T, q = 0, unit = 'mol')

      Return the amount of fluid stored in the tank at given conditions.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. Can vary between 0 and 1.
                The default is 0.
      :type q: float, optional
      :param unit: Unit of the capacity to be returned. Valid units are "mol" and
                   "kg". The default is "mol".
      :type unit: str, optional

      :returns: Amount of fluid stored.
      :rtype: float



   .. py:method:: capacity_bulk(p, T, q = 0, unit = 'mol')

      Calculate the amount of bulk fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. Can vary between 0 and 1.
                The default is 0.
      :type q: float, optional
      :param unit: Unit of the capacity to be returned. Valid units are "mol" and
                   "kg". The default is "mol".
      :type unit: str, optional

      :returns: Amount of bulk fluid stored.
      :rtype: float



   .. py:method:: find_quality_at_saturation_capacity(T, capacity)

      Find vapor quality at the given temperature and capacity.

      :param T: Temperature (K)
      :type T: float
      :param capacity: Amount of fluid in the tank (moles).
      :type capacity: float

      :returns: Vapor quality of the fluid being stored. This is assuming that the
                fluid is on the saturation line.
      :rtype: float



   .. py:method:: internal_energy(p, T, q = 1)

      Calculate the internal energy of the fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the fluid being stored (J).
      :rtype: float



   .. py:method:: conditions_at_capacity_temperature(cap, T, p_guess, q_guess)

      Find conditions corresponding to a given capacity and temperature.

      :param cap: Amount of fluid inside the tank (moles).
      :type cap: float
      :param T: Temperature (K).
      :type T: float
      :param p_guess: Initial guess for pressure value (Pa) to be optimized.
      :type p_guess: float
      :param q_guess: Initial guess for vaport quality value to be optimized.
      :type q_guess: float

      :returns: The optimization result represented as a OptimizeResult object.
                The relevant attribute for this method is x, the solution array.
                x[0] contains the pressure value and x[1] contains the vapor
                quality value.
      :rtype: OptimizeResult



   .. py:method:: conditions_at_capacity_pressure(cap, p, T_guess, q_guess)

      Find conditions corresponding to a given capacity and temperature.

      :param cap: Amount of fluid inside the tank (moles).
      :type cap: float
      :param P: Pressure (Pa).
      :type P: float
      :param T_guess: Initial guess for temperature value (K) to be optimized.
      :type T_guess: float
      :param q_guess: Initial guess for vaport quality value to be optimized.
      :type q_guess: float

      :returns: The optimization result represented as a OptimizeResult object.
                The relevant attribute for this package is x, the solution array.
                x[0] contains the temperature value and x[1] contains the vapor
                quality value.
      :rtype: scipy.optimize.OptimizeResult



   .. py:method:: calculate_dormancy(p, T, heating_power, q = 0)

      Calculate dormancy time given a constant heating rate.

      :param p: Initial tank pressure (Pa).
      :type p: float
      :param T: Initial tank temperature (K).
      :type T: float
      :param heating_power: The heating power going into the tank during parking (W).
      :type heating_power: float
      :param q: Initial vapor quality of the tank. The default is 0 (pure liquid).
      :type q: float, optional

      :returns: Pandas dataframe containing calculation conditions and results.
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
      :rtype: pd.DataFrame



   .. py:method:: thermal_res(p, T, time, env_temp)

      Calculate the thermal resistance of the tank.

      :param p: Pressure (Pa) of fluid inside tank.
      :type p: float
      :param T: Temperature (K) of fluid inside tank
      :type T: float
      :param time: Time elapsed in simulation (s).
      :type time: float
      :param env_temp: Temperature (K) of environment surrounding tank.
      :type env_temp: float

      :returns: Thermal resistance of the tank (K/W).
      :rtype: float



.. py:class:: SorbentTank(sorbent_material, aluminum_mass = 0, carbon_fiber_mass = 0, steel_mass = 0, vent_pressure = None, min_supply_pressure = 100000.0, thermal_resistance = 0, surface_area = 0, heat_transfer_coefficient = 0, volume = None, set_capacity = None, full_pressure = None, empty_pressure = None, full_temperature = None, empty_temperature = None, full_quality = 1, empty_quality = 1, set_sorbent_fill = 1)

   Bases: :py:obj:`StorageTank`


   Stores properties of a fluid storage tank filled with sorbents.

   .. attribute:: volume

      Internal volume of the storage tank (m^3).

      :type: float

   .. attribute:: sorbent_material

      An object storing the properties of the sorbent material used in
      the tank.

      :type: SorbentMaterial

   .. attribute:: aluminum_mass

      The mass of aluminum making up the tank walls (kg). The default is
      0.

      :type: float, optional

   .. attribute:: carbon_fiber_mass

      The mass of carbon fiber making up the tank walls (kg). The default
      is 0.

      :type: float, optional

   .. attribute:: steel_mass

      The mass of steel making up the tank walls (kg). The default is 0.

      :type: float, optional

   .. attribute:: vent_pressure

      Maximum pressure at which the tank has to be vented (Pa). The
      default is None.

      :type: float, optional

   .. attribute:: min_supply_pressure

      The minimum supply pressure (Pa) for discharging simulations. The
      default is 1E5.

      :type: float, optional

   .. attribute:: thermal_resistance

      The thermal resistance of the tank walls (K/W). The default is 0.

      :type: Callable, optional

   .. attribute:: surface_area

      Outer surface area of the tank in contact with the environment
      (m^2). The default is 0.

      :type: float, optional

   .. attribute:: heat_transfer_coefficient

      The heat transfer coefficient of the tank surface (W/(m^2 K)).
      The default is 0.

      :type: Callable, optional

   Initialize a SorbentTank object.

   :param sorbent_material: An object storing the properties of the sorbent material used in
                            the tank.
   :type sorbent_material: SorbentMaterial
   :param aluminum_mass: The mass of aluminum making up the tank walls (kg). The default is
                         0.
   :type aluminum_mass: float, optional
   :param carbon_fiber_mass: The mass of carbon fiber making up the tank walls (kg). The default
                             is 0.
   :type carbon_fiber_mass: float, optional
   :param steel_mass: The mass of steel making up the tank walls (kg). The default is 0.
   :type steel_mass: float, optional
   :param vent_pressure: Maximum pressure at which the tank has to be vented (Pa). The
                         default is None.
   :type vent_pressure: float, optional
   :param min_supply_pressure: The minimum supply pressure (Pa) for discharging simulations. The
                               default is 1E5.
   :type min_supply_pressure: float, optional
   :param thermal_resistance: A function which returns the thermal resistance of the tank
                              walls (K/W) as a function of tank pressure (Pa), tank
                              temperature (K), time (s), and temperature of surroundings (K). The
                              default is 0. If a float is provided, it will be converted to a
                              function which returns that value everywhere. If both this and
                              the arguments 'surface_area' and 'heat_transfer_coefficient' are
                              passed, two values of thermal resistance will be calculated and
                              the highest value between the two will be taken at each time step.
                              Thus, to avoid confusion, one should either: (a) use the other two
                              arguments together, or (b) use this one, but not both at the same
                              time.

                              If a callable is passed, it must have the signature::

                                  def tr_function(p, T, time, env_temp):
                                      # 'p' is tank pressure (Pa)
                                      # 'T' is tank temperature (K)
                                      # 'time' is the time elapsed within the simulation (s)
                                      # 'env_temp' is the temperature of surroundings (K)
                                      ....
                                      # Returned is the thermal resistance (K/W)
                                      return tr_value
   :type thermal_resistance: Callable or float, optional
   :param surface_area: Outer surface area of the tank in contact with the environment
                        (m^2). The default is 0.
   :type surface_area: float, optional
   :param heat_transfer_coefficient: A function which returns the heat transfer coefficient of the tank
                                     walls (W/(m^2 K)) as a function of tank pressure (Pa), tank
                                     temperature (K), time (s), and temperature of surroundings (K). The
                                     default is 0. If a float is  provided, it will be converted to a
                                     function which returns that value everywhere.

                                     If a callable is passed, it must have the signature::

                                         def htc_function(p, T, time, env_temp):
                                             # 'p' is tank pressure (Pa)
                                             # 'T' is tank temperature (K)
                                             # 'time' is the time elapsed within the simulation (s)
                                             # 'env_temp' is the temperature of surroundings (K)
                                             ....
                                             # Returned is the heat transfer coefficient (W/(m^2 K))
                                             return heat_transfer_coef
   :type heat_transfer_coefficient: Callable or float, optional
   :param volume: Internal volume of the storage tank (m^3). The default is None.
                  This value is required unless the set capacity and operating
                  conditions are defined, in which case the volume is calculated from
                  the capacity and operating conditions.
   :type volume: float, optional
   :param set_capacity: Set internal capacity of the storage tank (mol). The default is
                        None. If specified, this will override the user-specified tank
                        volume.
   :type set_capacity: float, optional
   :param full_pressure: Pressure (Pa) of the tank when it is considered full. The default
                         is None.
   :type full_pressure: float, optional
   :param empty_pressure: Pressure (Pa) of the tank when it is considered empty. The default
                          is None.
   :type empty_pressure: float, optional
   :param full_temperature: Temperature (K) of the tank when it is considered full. The
                            default is None.
   :type full_temperature: float, optional
   :param empty_temperature: Temperature (K) of the tank when it is considered empty. The
                             default is None.
   :type empty_temperature: float, optional
   :param full_quality: Vapor quality of the tank when it is considered full. The default
                        is 1 (Gas).
   :type full_quality: float, optional
   :param empty_quality: Vapor quality of the tank when it is considered empty. The default
                         is 1 (Gas).
   :type empty_quality: float, optional
   :param set_sorbent_fill: Ratio of tank volume filled with sorbent. The default is 1
                            (completely filled with sorbent).
   :type set_sorbent_fill: float, optional

   :returns: Object which stores various properties of a storage tank containing
             sorbents. It also has some useful methods related to the tank, most
             notably dormancy calculation.
   :rtype: SorbentTank


   .. py:method:: bulk_fluid_volume(p, T)

      Calculate the volume of bulk fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature(K).
      :type T: float

      :returns: Bulk fluid volume within the tank (m^3).
      :rtype: float



   .. py:method:: capacity(p, T, q = 0)

      Return the amount of fluid stored in the tank at given conditions.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. Can vary between 0 and 1.
                The default is 0.
      :type q: float, optional

      :returns: Amount of fluid stored (moles).
      :rtype: float



   .. py:method:: capacity_bulk(p, T, q = 0)

      Calculate the amount of bulk fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. Can vary between 0 and 1.
                The default is 0.
      :type q: float, optional

      :returns: Amount of bulk fluid stored (moles).
      :rtype: float



   .. py:method:: internal_energy(p, T, q = 1)

      Calculate the internal energy of the fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the fluid being stored (J).
      :rtype: float



   .. py:method:: internal_energy_sorbent(p, T, q = 1)

      Calculate the internal energy of the adsorbed fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the adsorbed fluid in the tank (J).
      :rtype: float



   .. py:method:: internal_energy_bulk(p, T, q = 1)

      Calculate the internal energy of the bulk fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the bulk fluid in the tank (J).
      :rtype: float



   .. py:method:: find_quality_at_saturation_capacity(T, capacity)

      Find vapor quality at the given temperature and capacity.

      :param T: Temperature (K)
      :type T: float
      :param capacity: Amount of fluid in the tank (moles).
      :type capacity: float

      :returns: Vapor quality of the fluid being stored. This is assuming that the
                fluid is on the saturation line.
      :rtype: float



   .. py:method:: find_temperature_at_saturation_quality(q, cap)

      Find temperature at a given capacity and vapor quality value.

      :param q: Vapor quality. Can vary between 0 and 1.
      :type q: float
      :param cap: Amount of fluid stored in the tank (moles).
      :type cap: float

      :returns: The optimization result represented as a OptimizeResult object.
                The relevant attribute for this function is x, the optimized
                temperature value.
      :rtype: scipy.optimize.OptimizeResult



   .. py:method:: calculate_dormancy(p, T, heating_power, q = 0)

      Calculate dormancy time given a constant heating rate.

      :param p: Initial tank pressure (Pa).
      :type p: float
      :param T: Initial tank temperature (K).
      :type T: float
      :param heating_power: The heating power going into the tank during parking (W).
      :type heating_power: float
      :param q: Initial vapor quality of the tank. The default is 0 (pure liquid).
      :type q: float, optional

      :returns: Pandas dataframe containing calculation conditions and results.
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
      :rtype: pd.DataFrame



.. py:class:: SimResults(pressure, temperature, time, moles_adsorbed, moles_gas, moles_liquid, moles_supercritical, tank_params, sim_params, stop_reason, sim_type = None, inserted_amount = 0, flow_energy_in = 0, cooling_required = 0, heating_required = 0, cooling_additional = 0, heating_additional = 0, heat_leak_in = 0, vented_amount = 0, vented_energy = 0)

   Class for storing the results of dynamic simulations.

   It comes with methods for exporting the results to csv, plotting the
   results, and for combining the results of multiple simulations.

   .. attribute:: df

      A dataframe containing the results of dynamic simulations. See notes
      for the column names and the variables each column has.

      :type: pd.DataFrame

   .. rubric:: Notes

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

   Initialize a SimResults object.

   :param pressure: A list or numpy array containing the pressure values inside of the
                    tank (Pa) as it changes over time.
   :type pressure: Union[List[float], np.ndarray]
   :param temperature: A list or numpy array containing the temperature values inside of
                       the tank (K) as it changes over time.
   :type temperature: Union[List[float], np.ndarray]
   :param time: A list or numpy array containing the simulation time points (s) at
                which results are reported.
   :type time: Union[List[float], np.ndarray]
   :param moles_adsorbed: A list or numpy array containing the amount of fluid that is
                          adsorbed (moles) at given points in time.
   :type moles_adsorbed: Union[List[float], np.ndarray]
   :param moles_gas: A list or numpy array containing the amount of fluid stored in
                     gaseous form (moles) at given points in time.
   :type moles_gas: Union[List[float], np.ndarray]
   :param moles_liquid: A list or numpy array containing the amount of fluid stored in
                        liquid form (moles) at given points in time.
   :type moles_liquid: Union[List[float], np.ndarray]
   :param moles_supercritical: A list or numpy array containing the amount of supercritical fluid
                               in the tank (moles) at given points in time.
   :type moles_supercritical: Union[List[float], np.ndarray]
   :param tank_params: An object containing the parameters of the storage tank used for
                       the dynamic simulation.
   :type tank_params: Union[StorageTank, SorbentTank]
   :param sim_type: A string describing the type of simulation that was conducted.
   :type sim_type: str
   :param sim_params: An object containing the parameters used for the simulation.
   :type sim_params: SimParams
   :param stop_reason: A string describing why the simulation was terminated.
   :type stop_reason: str
   :param inserted_amount: The cumulative amount of fluid inserted into the tank (moles)
                           throughout the dynamic simulation. The default is 0.
   :type inserted_amount: Union[List[float], np.ndarray], optional
   :param flow_energy_in: The cumulative amount of enthalpy brought by fluid flowing into the
                          tank (J) throughout the dynamic simulation. The default is 0.
   :type flow_energy_in: Union[List[float], np.ndarray], optional
   :param cooling_required: The cumulative amount of cooling required (J) to maintain a
                            constant pressure during refueling. The default is 0.
   :type cooling_required: Union[List[float], np.ndarray], optional
   :param heating_required: The cumulative amount of heating required (J) to maintain a
                            constant pressure during discharging. The default is 0.
   :type heating_required: Union[List[float], np.ndarray], optional
   :param cooling_additional: The cumulative amount of additional cooling (J) inputted to the
                              simulation via a user-defined function. The default is 0.
   :type cooling_additional: Union[List[float], np.ndarray], optional
   :param heating_additional: The cumulative amount of additional heating (J) inputted to the
                              simulation via a user-defined function. The default is 0.
   :type heating_additional: Union[List[float], np.ndarray], optional
   :param heat_leak_in: The cumulative amount of heat (J) which has leaked into the tank
                        from the environment. The default is 0.
   :type heat_leak_in: Union[List[float], np.ndarray], optional
   :param vented_amount: The cumulative amount of fluid vented (moles) throughout the
                         dynamic simulation. The default is 0.
   :type vented_amount: Union[List[float], np.ndarray], optional
   :param vented_energy: The cumulative amount of enthalpy taken by fluid flowing out of the
                         tank (J) throughout the dynamic simulation. The default is 0.
   :type vented_energy: Union[List[float], np.ndarray], optional

   :returns: An object containing the results of a dynamic simulation run by
             pytanksim. It has functions for exporting and plotting.
   :rtype: SimResults


   .. py:method:: get_final_conditions(idx = -1)

      Output final tank conditions at the end of the simulation.

      :param idx: The index of the simulation results array from which the values are
                  to be taken. The default is -1 (the last time point in the
                  simulation).
      :type idx: int, optional

      :returns: A dictionary containing tank conditions at'idx'.
      :rtype: dict



   .. py:method:: to_csv(filename, verbose = True)

      Export simulation results to a csv file.

      :param filename: The desired filepath for the csv file to be created.
      :type filename: str
      :param verbose: Whether or nor to report the completion of the export. The default
                      value is True.
      :type verbose: bool, optional



   .. py:method:: from_csv(filename, import_components = False)
      :classmethod:


      Import simulation results from a csv file.

      :param filename: Path to a csv file which was exported by pytanksim.
      :type filename: str
      :param import_components: If True, this function will return a tuple with contents as
                                follows: SimResults, StorageTank, SimParams.
                                If False, this function will only return the SimResults object.
                                The default option is False.
      :type import_components: bool

      :returns: A single object containing the simulation results, or a tuple
                with SimResults, StorageTank, and SimParams objects.
      :rtype: SimResults|Tuple



   .. py:method:: interpolate(x_var = 't')

      Interpolate simulation results between points.

      :param x_var: Variable to be used as a basis/input for interpolation.The default
                    is "t".
      :type x_var: str, optional

      :returns: A dictionary containing functions which interpolate each variable
                in the SimResults object w.r.t. the variable chosen in x_var.
      :rtype: "dict[Callable[[float], float]]"



   .. py:method:: plot(x_axis, y_axes, colors = ['r', 'b', 'g'])

      Plot the results of the simulation.

      :param x_axis: A string specifying what variable should be on the x-axis.
                     See notes for valid inputs.
      :type x_axis: str
      :param y_axes: A string or a list of strings specifying what is to be plotted on
                     the y-axis. See notes for valid inputs
      :type y_axes: Union[str, List[str]]
      :param colors: A string or a list of strings specifying colors for the lines in
                     the plot. The default is ["r", "b", "g"].
      :type colors: Union[str, List[str]], optional

      :raises ValueError: If more than 3 y-variables are specified to be plotted.

      :returns: A matplolib axis or a numpy array of several axes.
      :rtype: Union[np.ndarray, plt.Axes]

      .. rubric:: Notes

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



   .. py:method:: combine(sim_results_list)
      :classmethod:


      Combine the results of several simulations into a single object.

      :param sim_results_list: A list of SimResults objects from several different simulations.
      :type sim_results_list: "List[SimResults]"

      :returns: A single object containing the combined simulation results.
      :rtype: SimResults



.. py:class:: SimParams

   A class to store simulation parameters.

   This data class stores the parameters of the tank at the start of the
   simulation as well as the conditions specified to stop the simulation.
   Additionally, it also stores the setting for the number of data points
   to be reported at the end of the simulation.

   .. attribute:: init_temperature

      The temperature (K) of the tank being simulated at the beginning of
      the simulation.

      :type: float

   .. attribute:: init_pressure

      The pressure of the tank being simulated (Pa) at the beginning of the
      simulation. The default value is 1E5. This parameter was made optional
      as the two-phase simulations did not require it to be filled, rather
      pytanksim will automatically calculate the saturation pressure given
      a starting temperature.

      :type: float, optional

   .. attribute:: final_time

      The time (seconds) at which the simulation is to be stopped.

      :type: float

   .. attribute:: init_time

      The time (seconds) at which the beginning of the simulation is set to.
      The default value is set to 0 seconds.

      :type: float, optional

   .. attribute:: displayed_points

      The number of data points to be reported at the end of the simulation.
      The default is 200.

      :type: int, optional

   .. attribute:: target_temp

      The target temperature (K) at which the simulation is to be stopped.
      The default value is 0, which effectively means the simulation
      does not have a set temperature at which the simulation is stopped.

      :type: float, optional

   .. attribute:: target_pres

      The target pressure (Pa) at which the simulation is to be stopped.
      The default value is 0, which effectively means the simulation does
      not have a set pressure at which the simulation is stopped.

      :type: float, optional

   .. attribute:: stop_at_target_pressure

      If True, it will stop the simulation when the target pressure is met.
      The default is False.

      :type: bool, optional

   .. attribute:: stop_at_target_temp

      If True, it will stop the simulation when the target temperature is
      met. The default is False.

      :type: bool, optional

   .. attribute:: target_capacity

      The amount of fluid (moles) stored in the tank at which the simulation
      is to be stopped. The default is 0.

      :type: float, optional

   .. attribute:: init_ng

      The initial amount of gas (moles) stored in the tank at the beginning
      of the simulation. The default value is 0.

      :type: float, optional

   .. attribute:: init_nl

      The initial amount of liquid (moles) stored in the tank at the
      beginning of the simulation. The default value is 0.

      :type: float, optional

   .. attribute:: init_q

      The initial quality of the fluid being stored. It can vary between 0
      and 1. The default is None.

      :type: float, optional

   :param inserted_amount: The amount of fluid which has been previously inserted into the tank
                           (moles) at the beginning of the simulation. Used to track refueling
                           processes across multiple simulations. The default value is 0.
   :type inserted_amount: float, optional
   :param vented_amount: The amount of fluid which has been previously vented from the tank
                         (moles) at the beginning of the simulation. Used to track discharging
                         and boil-off processes across multiple simulations. The default value
                         is 0.
   :type vented_amount: float, optional
   :param cooling_required: The cumulative amount of required cooling (J) to maintain a constant
                            pressure prior to the start of a simulation. The default value is 0.
                            Useful when restarting a stopped cooled refuel simulation.
   :type cooling_required: float, optional
   :param heating_required: The cumulative amount of required heating (J) to maintain a constant
                            pressure prior to the start of a simulation. The default value is 0.
                            Useful when restarting a stopped heated discharge simulation.
   :type heating_required: float, optional
   :param vented_energy: Cumulative amount of enthalpy (J) contained in the fluid vented prior
                         to the start of the simulation. The default is 0. Useful when
                         stopping and restarting discharge simulations.
   :type vented_energy: float, optional
   :param flow_energy_in: Cumulative amount of enthalpy (J) contained in the fluid inserted
                          prior to the start of the simulation. The default is 0. Useful when
                          stopping and restarting refueling simulations.
   :type flow_energy_in: float, optional
   :param cooling_additional: The cumulative amount of user-specified cooling (J) prior to the start
                              of a simulation. The default value is 0. Useful when stopping and
                              restarting simulations with user-specified cooling.
   :type cooling_additional: float, optional
   :param heating_additional: The cumulative amount of user-specified cooling (J) prior to the start
                              of a simulation. The default value is 0. Useful when stopping and
                              restarting simulations with user-specified heating.
   :type heating_additional: float, optional
   :param heat_leak_in: The cumulative amount of heat (J) which has leaked into the tank prior
                        to the start of a simulation. The default value is 0. Useful when
                        stopping and restarting simulations involving heat leakage.
   :type heat_leak_in: float, optional
   :param verbose: Whether or not the simulation will print out its progress bars and
                   give a notification once it has finished. The default value is True.
   :type verbose: bool, optional


   .. py:method:: from_SimResults(sim_results, displayed_points = None, init_time = None, final_time = None, target_pres = None, target_temp = None, stop_at_target_pressure = None, stop_at_target_temp = None, target_capacity = None, verbose = None)
      :classmethod:


      Take final conditions from a previous simulation as new parameters.

      :param sim_results: An object containing previous simulation results.
      :type sim_results: SimResults
      :param displayed_points: The number of data points to be reported at the end of the
                               simulation. The default is 200.
      :type displayed_points: float, optional
      :param init_time: The time (seconds) at which the beginning of the simulation is set.
                        The default value is None.
      :type init_time: float, optional
      :param final_time: The time (seconds) at which the simulation is to be stopped. If
                         None, then the final_time setting from the previous simulation is
                         used. The default is None.
      :type final_time: float, optional
      :param target_pres: The target pressure (Pa) at which the simulation is to be stopped.
                          If None, then the target_pres setting from the previous simulation
                          is used. The default is None.
      :type target_pres: float, optional
      :param target_temp: The target temperature (K) at which the simulation is to be
                          stopped. If None, then the target_temp setting from the previous
                          simulation is used. The default is None.
      :type target_temp: float, optional
      :param stop_at_target_pressure: If True, it will stop the simulation when the target pressure is
                                      met. If None, then the stop_at_target_pressure setting from the
                                      previous simulation is used. The default is None.
      :type stop_at_target_pressure: bool, optional
      :param stop_at_target_temp: If True, it will stop the simulation when the target temperature
                                  is  met. If None, then the stop_at_target_temp setting from the
                                  previous simulation is used. The default is None.
      :type stop_at_target_temp: bool, optional
      :param target_capacity: The amount of fluid (moles) stored in the tank at which the
                              simulation is to be stopped. If None, then the target_capacity
                              value from the previous simulation is used. The default is None.
      :type target_capacity: float, optional

      :returns: A SimParams object containing the final conditions taken from
                sim_results set as the new starting parameters.
      :rtype: SimParams



.. py:class:: BoundaryFlux(mass_flow_in = 0.0, mass_flow_out = 0.0, heating_power = 0.0, cooling_power = 0.0, pressure_in = None, temperature_in = None, environment_temp = 0, enthalpy_in = None, enthalpy_out = None, heat_leak_in = None)

   Stores information of the mass and energy fluxes on the tank boundaries.

   .. attribute:: mass_flow_in

      A function which returns mass flow into the tank (kg/s) as a function
      of tank pressure (Pa), tank temperature (K), and time (s). The default
      is a function which returns 0 everywhere.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: mass_flow_out

      A function which returns mass flow exiting the tank (kg/s) as a
      function of tank pressure (Pa), tank temperature (K), and time (s).
      The default is a function which returns 0 everywhere.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: heating_power

      A function which returns heating power added to the tank (W) as a
      function of tank pressure (Pa), tank temperature (K), and time (s).
      The default is a function which returns 0 everywhere.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: cooling_power

      A function which returns cooling power added to the tank (W) as a
      function of tank pressure (Pa), tank temperature (K), and time (s).
      The default is a function which returns 0 everywhere.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: pressure_in

      A function which returns the pressure (Pa) of the fluid being inserted
      into the tank as a  function of tank pressure (Pa), tank temperature
      (K), and time (s). The default is None.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: temperature_in

      A function which returns the temperature (K) of the fluid being
      inserted into the tank as a  function of tank pressure (Pa), tank
      temperature (K), and time (s). The default is None.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: environment_temp

      The temperature (K) of the environment surrounding the tank.
      This value is used in the dynamic simulation to calculate heat leakage
      into the tank. The default is 0, in which case heat leakage into the
      tank is not considered.

      :type: float, optional

   .. attribute:: enthalpy_in

      A function which returns the enthalpy (J/mol) of the fluid being
      inserted into the tank as a  function of tank pressure (Pa), tank
      temperature (K), and time (s). The default is None.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: enthalpy_out

      A function which returns the enthalpy (J/mol) of the fluid exiting
      the tank as a  function of tank pressure (Pa), tank temperature (K),
      and time (s). The default is None.

      :type: Callable[[float, float, float], float], optional

   .. attribute:: heat_leak_in

      A function which returns the heat (J/mol) leaking into the tank as a
      function of tank pressure (Pa), tank temperature (K), time (s), and
      temperature of tank surroundings (K). The default is None.

      :type: Callable[[float, float, float,float], float], optional

   Initialize a BoundaryFlux object.

   :param mass_flow_in: A function which returns mass flow into the tank (kg/s) as a
                        function of tank pressure (Pa), tank temperature (K), and time (s).
                        The default is a function which returns 0 everywhere. If a float is
                        provided, it will be converted to a function which returns that
                        value everywhere.

                        If a callable is passed, it must have the signature::

                            def mass_flow_in_function(p, T, time):
                                # 'p' is tank pressure (Pa)
                                # 'T' is tank temperature (K)
                                # 'time' is the time elapsed within the simulation (s)
                                ....
                                #Returned is the mass flow going into the tank (kg/s)
                                return mass_flow_in
   :type mass_flow_in: Callable or float, optional
   :param mass_flow_out: A function which returns mass flow exiting the tank (kg/s) as a
                         function of tank pressure (Pa), tank temperature (K), and time (s).
                         The default is a function which returns 0 everywhere. If a float is
                         provided it will be converted to a function which returns that
                         value everywhere.

                         If a callable is passed, it must have the signature::

                             def mass_flow_out_function(p, T, time):
                                 # 'p' is tank pressure (Pa)
                                 # 'T' is tank temperature (K)
                                 # 'time' is the time elapsed within the simulation (s)
                                 ....
                                 # Returned is the mass flow going out of the tank (kg/s)
                                 return mass_flow_out
   :type mass_flow_out: Callable or float, optional
   :param heating_power: A function which returns heating power added to the tank (W) as a
                         function of tank pressure (Pa), tank temperature (K), and time (s).
                         The default is a function which returns 0 everywhere. If a float is
                         provided, it will be converted to a function which returns that
                         value everywhere.

                         If a callable is passed, it must have the signature::

                             def heating_power_function(p, T, time):
                                 # 'p' is tank pressure (Pa)
                                 # 'T' is tank temperature (K)
                                 # 'time' is the time elapsed within the simulation (s)
                                 ....
                                 # Returned is the heat put into the tank (W)
                                 return heating_power
   :type heating_power: Callable or float, optional
   :param cooling_power: A function which returns cooling power added to the tank (W) as a
                         function of tank pressure (Pa), tank temperature (K), and time (s).
                         The default is a function which returns 0 everywhere. If a float is
                         provided,it will be converted to a function which returns that
                         value everywhere.

                         If a callable is passed, it must have the signature::

                             def cooling_power_function(p, T, time):
                                 # 'p' is tank pressure (Pa)
                                 # 'T' is tank temperature (K)
                                 # 'time' is the time elapsed within the simulation (s)
                                 ....
                                 # Returned is the heat taken out of the tank (W)
                                 return cooling_power
   :type cooling_power: Callable or float, optional
   :param pressure_in: A function which returns the pressure (Pa) of the fluid being
                       inserted into the tank as a  function of tank pressure (Pa), tank
                       temperature (K), and time (s). The default is None. If a float is
                       provided,it will be converted to a function which returns that
                       value everywhere.

                       If a callable is passed, it must have the signature::

                           def pressure_in_function(p, T, time):
                               # 'p' is tank pressure (Pa)
                               # 'T' is tank temperature (K)
                               # 'time' is the time elapsed within the simulation (s)
                               ....
                               # Returned is the pressure (Pa) of the fluid going into
                               # the tank.
                               return pressure_in
   :type pressure_in: Callable or float, optional
   :param temperature_in: A function which returns the temperature (K) of the fluid being
                          inserted into the tank as a  function of tank pressure (Pa), tank
                          temperature (K), and time (s). The default is None. If a float is
                          provided,it will be converted to a function which returns that
                          value everywhere.

                          If a callable is passed, it must have the signature::

                              def temperature_in_function(p, T, time):
                                  # 'p' is tank pressure (Pa)
                                  # 'T' is tank temperature (K)
                                  # 'time' is the time elapsed within the simulation (s)
                                  ....
                                  # Returned is the temperature (K) of the fluid going into
                                  # the tank.
                                  return temperature_in
   :type temperature_in: Callable or float, optional
   :param environment_temp: The temperature (K) of the environment surrounding the tank. This
                            value is used in the dynamic simulation to calculate heat leakage
                            into the tank. It can be provided either as a float or as a
                            function of tank pressure (Pa), tank temperature (K). The default
                            is 0, in which case heat leakage into the tank is not considered.

                            If a callable is passed, it must have the signature::

                                def env_temp_function(p, T, time):
                                    # 'p' is tank pressure (Pa)
                                    # 'T' is tank temperature (K)
                                    # 'time' is the time elapsed within the simulation (s)
                                    ....
                                    # Returned is the temperature of the surroundings in the
                                    # unit of K.
                                    return enthalpy_in
   :type environment_temp: Callable or float, optional
   :param enthalpy_in: A function which returns the enthalpy (J/mol) of the fluid being
                       inserted into the tank as a  function of tank pressure (Pa), tank
                       temperature (K), and time (s). The default is None. If a float is
                       provided,it will be converted to a function which returns that
                       value everywhere.

                       If a callable is passed, it must have the signature::

                           def enthalpy_in_function(p, T, time):
                               # 'p' is tank pressure (Pa)
                               # 'T' is tank temperature (K)
                               # 'time' is the time elapsed within the simulation (s)
                               ....
                               # Returned is the enthalpy (J/mol) of the fluid going into
                               # the tank.
                               return enthalpy_in
   :type enthalpy_in: Callable or float, optional
   :param enthalpy_out: A function which returns the enthalpy (J/mol) of the fluid exiting
                        the tank as a  function of tank pressure (Pa), tank temperature
                        (K), and time (s). The default is None. If a float is provided, it
                        will be converted to a function which returns that value
                        everywhere.

                        If a callable is passed, it must have the signature::

                            def enthalpy_out_function(p, T, time):
                                # 'p' is tank pressure (Pa)
                                # 'T' is tank temperature (K)
                                # 'time' is the time elapsed within the simulation (s)
                                ....
                                # Returned is the enthalpy (J/mol) of the fluid going out
                                # of the tank.
                                return enthalpy_out
   :type enthalpy_out: Callable or float, optional
   :param heat_leak_in: A function which returns the amount of heat leakage into the tank
                        (W) as a  function of tank pressure (Pa), tank temperature
                        (K), time (s), and temperature of surroundings (K). The default is
                        None, which will use the thermal resistance calculation from the
                        storage tank. Otherwise, it will override that calculation. If a
                        float is  provided, it will be converted to a function which
                        returns that value everywhere.

                        If a callable is passed, it must have the signature::

                            def enthalpy_out_function(p, T, time, env_temp):
                                # 'p' is tank pressure (Pa)
                                # 'T' is tank temperature (K)
                                # 'time' is the time elapsed within the simulation (s)
                                # 'env_temp' is the temperature of surroundings (K)
                                ....
                                # Returned is the enthalpy (J/mol) of the fluid going out
                                # of the tank.
                                return enthalpy_out
   :type heat_leak_in: Callable or float, optional

   :raises ValueError: If the mass flow going in is specified but the parameters that
       specify its enthalpy (i.e., either pressure and temperature or
       its enthalpy value) are not specified.

   :returns: An object which stores information of the mass and energy fluxes on
             the tank boundaries.
   :rtype: BoundaryFlux


.. py:class:: BaseSimulation(simulation_params, storage_tank, boundary_flux)

   An abstract base class for dynamic simulations.

   Other simulation classes inherit some attributes and methods from this
   class.

   .. attribute:: sim_type

      Type of simulation (default, heated discharge, cooled refuel, etc.)

      :type: str

   .. attribute:: sim_phase

      1 or 2 phases.

      :type: int

   .. attribute:: simulation_params

      Object which stores simulation parameters.

      :type: SimParams

   .. attribute:: storage_tank

      Object which stores the properties of the tank being simulated.

      :type: StorageTank

   .. attribute:: boundary_flux

      Object which stores the amount of energy entering and exiting the tank.

      :type: BoundaryFlux

   .. attribute:: stop_reason

      A string stating the reason for the simulation to have stopped.
      It will be passed to the SimResults object once the simulation
      finishes.

      :type: str

   .. attribute:: Initialize the BaseSimulation class.

      

   .. attribute:: 

      

      :type: param simulation_params: Object containing simulation-specific parameters.

   .. attribute:: 

      

      :type: type simulation_params: SimParams

   .. attribute:: 

      storage tank being simulated.

      :type: param storage_tank: Object containing attributes and methods specific to the

   .. attribute:: 

      

      :type: type storage_tank: StorageTank

   .. attribute:: 

      and out of the tank during the simulation.

      :type: param boundary_flux: Object containing information on the mass and energy going in

   .. attribute:: 

      

      :type: type boundary_flux: BoundaryFlux

   .. attribute:: 

      initial values for liquid and gas in the tank, or, alternatively,
      the initial vapor quality, was not specified.

      :type: raises ValueError: If the simulation is set to begin on the saturation line but the

   .. attribute:: 

      specified as well as the initial vapor quality, but the values
      don't match each other.

      :type: raises ValueError: If both the initial values for liquid and gas in the tank is

   .. attribute:: 

      

      :type: returns: A simulation object which can be run to get results.

   .. attribute:: 

      

      :type: rtype: BaseSimulation


   .. py:method:: heat_leak_in(p, T, time)

      Calculate the heat leakage rate from the environment into the tank.

      :param p: Pressure (Pa) of the storage tank.
      :type p: float
      :param T: Temperature (K) of the storage tank.
      :type T: float
      :param time: Simulation time (s)
      :type time: float

      :returns: The rate of heat leakage into the tank from the environment (W).
      :rtype: float



   .. py:method:: run()
      :abstractmethod:


      Abstract function which will be defined in the child classes.

      :raises NotImplementedError: Raises an error since it is not implemented in this abstract base
          class.

      :rtype: None.



   .. py:method:: enthalpy_in_calc(p, T, time)

      Calculate the enthalpy (J/mol) of fluid going into the tank.

      :param p: Pressure inside of the tank (Pa)
      :type p: float
      :param T: Temperature inside of the tank (K)
      :type T: float
      :param time: Time (s) in the simulation.
      :type time: float

      :returns: Enthalpy of the fluid going into the tank (J/mol).
      :rtype: float



   .. py:method:: enthalpy_out_calc(fluid_property_dict, p, T, time)

      Calculate the enthalpy (J/mol) of fluid going out of the tank.

      :param fluid_property_dict: A dictionary of properties of the fluid being stored inside of the
                                  tank. In the case of the two phase simulation, it is the properties
                                  of the gas and not the liquid. For this function, this dictionary
                                  must return an enthalpy (J/mol) value given the key "hf".
      :type fluid_property_dict: Dict[str,float]
      :param p: Pressure inside of the tank (Pa)
      :type p: float
      :param T: Temperature inside of the tank (K)
      :type T: float
      :param time: Time (s) in the simulation.
      :type time: float

      :returns: Enthalpy of the fluid going out of the tank (J/mol).
      :rtype: float



.. py:class:: OnePhaseSorbentSim(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`pytanksim.classes.basesimclass.BaseSimulation`


   Base class for simulation of sorbent tanks in the one phase region.

   It includes functions to calculate the governing ODE

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


.. py:class:: OnePhaseSorbentDefault(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseSorbentSim`


   Simulates sorbent tanks in the one phase region without constraints.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(p, T, time)

      Find the right hand side of the governing ODE at a given time step.

      :param p: Current pressure (Pa).
      :type p: float
      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseSorbentVenting(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseSorbentSim`


   Sorbent tank venting at constant pressure in the one phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(T, time)

      Find the right hand side of the governing ODE at a given time step.

      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseSorbentCooled(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseSorbentSim`


   Sorbent tank cooled at constant pressure in the one phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(T, time)

      Find the right hand side of the governing ODE at a given time step.

      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseSorbentHeatedDischarge(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseSorbentSim`


   Sorbent tank heated at constant pressure in the one phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(T, time)

      Find the right hand side of the governing ODE at a given time step.

      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseSorbentSim(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`pytanksim.classes.basesimclass.BaseSimulation`


   Base class for sorbent tanks in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


.. py:class:: TwoPhaseSorbentDefault(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseSorbentSim`


   Simulate sorbent tanks in the two phase region without constraints.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(ng, nl, T, time)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float
      :param ng: Current amount of gas in the tank (moles).
      :type ng: float
      :param nl: Current amount of liquid in the tank (moles).
      :type nl: float
      :param T: Current temperature (K).
      :type T: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseSorbentVenting(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseSorbentSim`


   Sorbent tank venting at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(ng, nl, time)

      Find the right hand side of the governing ODE at a given time step.

      :param ng: Current amount of gas in the tank (moles).
      :type ng: float
      :param nl: Current amount of liquid in the tank (moles).
      :type nl: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseSorbentCooled(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseSorbentSim`


   Sorbent tank cooled at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, ng, nl)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float
      :param ng: Current amount of gas in the tank (moles).
      :type ng: float
      :param nl: Current amount of liquid in the tank (moles).
      :type nl: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseSorbentHeatedDischarge(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseSorbentSim`


   Sorbent tank heated at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, ng, nl)

      Find the right hand side of the governing ODE at a given time step.

      :param ng: Current amount of gas in the tank (moles).
      :type ng: float
      :param nl: Current amount of liquid in the tank (moles).
      :type nl: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseFluidSim(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`pytanksim.classes.basesimclass.BaseSimulation`


   Base class for one phase fluid simulations.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


.. py:class:: OnePhaseFluidDefault(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseFluidSim`


   Class for simulating fluid storage dynamics in the one phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, p, T)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float
      :param p: Current pressure (Pa).
      :type p: float
      :param T: Current temperature (K).
      :type T: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseFluidVenting(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseFluidSim`


   Simulate the dynamics of a fluid tank venting at constant pressure.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, T)

      Solve for the right hand side of the governing ODE.

      :param time: Current time step in the simulation (s).
      :type time: float
      :param T: Current temperature (K).
      :type T: float

      :returns: Numpy array containing values for the RHS of the governing ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseFluidCooled(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseFluidSim`


   Simulates a tank being cooled to maintain constant pressure.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, T)

      Solve for the right hand side of the governing ODE.

      :param time: Current time step in the simulation (s).
      :type time: float
      :param T: Current temperature (K).
      :type T: float

      :returns: Numpy array containing values for the RHS of the governing ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseFluidHeatedDischarge(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`OnePhaseFluidSim`


   Simulates a tank being heated to discharge at a constant pressure.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, T)

      Solve for the right hand side of the governing ODE.

      :param time: Current time step in the simulation (s).
      :type time: float
      :param T: Current temperature (K).
      :type T: float

      :returns: Numpy array containing values for the RHS of the governing ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseFluidSim(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`pytanksim.classes.basesimclass.BaseSimulation`


   Base class for the simulation of fluid tanks in the two-phase region.

   Contains functions for calculating the governing ODEs.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


.. py:class:: TwoPhaseFluidDefault(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseFluidSim`


   Simulation of fluid tanks in the two-phase region w/o constraints.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time, ng, nl, T)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float
      :param ng: Current amount of gas in the tank (moles).
      :type ng: float
      :param nl: Current amount of liquid in the tank (moles).
      :type nl: float
      :param T: Current temperature (K).
      :type T: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseFluidVenting(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseFluidSim`


   Fluid tank venting at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseFluidCooled(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseFluidSim`


   Fluid tank being cooled at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: TwoPhaseFluidHeatedDischarge(simulation_params, storage_tank, boundary_flux)

   Bases: :py:obj:`TwoPhaseFluidSim`


   Fluid tank being heated at constant pressure in the two-phase region.

   Initialize the BaseSimulation class.

   :param simulation_params: Object containing simulation-specific parameters.
   :type simulation_params: SimParams
   :param storage_tank: Object containing attributes and methods specific to the
                        storage tank being simulated.
   :type storage_tank: StorageTank
   :param boundary_flux: Object containing information on the mass and energy going in
                         and out of the tank during the simulation.
   :type boundary_flux: BoundaryFlux

   :raises ValueError: If the simulation is set to begin on the saturation line but the
       initial values for liquid and gas in the tank, or, alternatively,
       the initial vapor quality, was not specified.
   :raises ValueError: If both the initial values for liquid and gas in the tank is
       specified as well as the initial vapor quality, but the values
       don't match each other.

   :returns: A simulation object which can be run to get results.
   :rtype: BaseSimulation


   .. py:method:: solve_differentials(time)

      Find the right hand side of the governing ODE at a given time step.

      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run()

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as the end of
          the phase change, or if the simulation hits the maximum pressure of
          the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



