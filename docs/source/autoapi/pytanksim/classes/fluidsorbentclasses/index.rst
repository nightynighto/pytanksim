pytanksim.classes.fluidsorbentclasses
=====================================

.. py:module:: pytanksim.classes.fluidsorbentclasses

.. autoapi-nested-parse::

   Contains classes related to the fluids and sorbents to be simulated.

   More specifically, contains the StoredFluid, SorbentMaterial, ModelIsotherm,
   and its derivatives.



Classes
-------

.. autoapisummary::

   pytanksim.classes.fluidsorbentclasses.StoredFluid
   pytanksim.classes.fluidsorbentclasses.ModelIsotherm
   pytanksim.classes.fluidsorbentclasses.DAModel
   pytanksim.classes.fluidsorbentclasses.MDAModel
   pytanksim.classes.fluidsorbentclasses.SorbentMaterial


Module Contents
---------------

.. py:class:: StoredFluid(fluid_name: str, EOS: str = 'HEOS', mole_fractions: List = None)

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


   .. py:method:: fluid_property_dict(p: float, T: float) -> Dict[str, float]

      Generate a dictionary of fluid properties using CoolProp.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K)
      :type T: float

      :returns: Dictionary containing several fluid properties needed for various
                calculations in pytanksim. "hf" is the enthalpy (J/mol). "drho_dp"
                is the first partial derivative of density (mol/m^3) w.r.t.
                pressure (Pa). "drho_dT" is the first partial derivative of density
                (mol/m^3) w.r.t. temperature (K). "rhof" is density (mol/m^3).
                "dh_dp" is the first partial derivative of enthalpy (J/mol) w.r.t.
                pressure (Pa). "dh_dT" is the first partial derivative of enthalpy
                (J/mol) w.r.t. temperature (K). "uf" is the internal energy
                (J/mol). "du_dp" is the first partial derivative of internal energy
                (J/mol) w.r.t. pressure (Pa). "du_dT" is the first partial
                derivative of internal energy (J/mol) w.r.t. temperature (K). "MW"
                is molar mass (kg/mol).
      :rtype: Dict[str, float]



   .. py:method:: saturation_property_dict(T: float, Q: int = 0) -> Dict[str, float]

      Generate a dictionary of fluid properties at saturation.

      :param T: Temperature in K.
      :type T: float
      :param Q: Vapor quality of the fluid being stored.
      :type Q: float

      :returns: A dictionary containing the fluid properties at saturation
                at a given temperature. "psat" is the saturation vapor pressure
                (Pa). "dps_dT" is the first derivative of the saturation vapor
                pressure (Pa) w.r.t. temperature (K). "hf" is the enthalpy (J/mol).
                "drho_dp" is the first partial derivative of density (mol/m^3)
                w.r.t. pressure (Pa). "drho_dT" is the first partial derivative of
                density (mol/m^3) w.r.t. temperature (K). "rhof" is density
                (mol/m^3). "dh_dp" is the first partial derivative of enthalpy
                (J/mol) w.r.t. pressure (Pa). "dh_dT" is the first partial
                derivative of enthalpy (J/mol) w.r.t. temperature (K). "uf" is the
                internal energy (J/mol). "du_dp" is the first partial derivative of
                internal energy (J/mol) w.r.t. pressure (Pa). "du_dT" is the first
                partial derivative of internal energy (J/mol) w.r.t. temperature
                (K). "MW" is molar mass (kg/mol).
      :rtype: Dict[str, float]



   .. py:method:: determine_phase(p: float, T: float) -> str

      Determine the phase of the fluid being stored.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: String that could either be "Supercritical", "Gas", "Liquid",
                or "Saturated" depending on the bulk fluid phase.
      :rtype: str



.. py:class:: ModelIsotherm

   A base class for model isotherm objects.

   Contains methods to calculate various thermodynamic properties of
   the adsorbed phase.



   .. py:method:: pressure_from_absolute_adsorption(n_abs: float, T: float, p_max_guess: float = 35000000.0) -> float

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



   .. py:method:: isosteric_enthalpy(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: isosteric_internal_energy(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: _derivfunc(func: Callable, var: int, point: float, qinit: float, stepsize: float) -> float

      Calculate the first partial derivative.

      It automatically decides the direction of the derivative so that the
      evaluations are done for fluids at the same phases. Otherwise, there
      will be discontinuities in the fluid properties at different phases
      which causes the resulting derivative values to be invalid.




   .. py:method:: _derivfunc_second(func: Callable, point: float, qinit: float, stepsize: float) -> float

      Calculate the second partial derivative.

      It automatically decides the direction of the derivative so that the
      evaluations are done for fluids at the same phases. Otherwise, there
      will be discontinuities in the fluid properties at different phases
      which causes the resulting derivative values to be invalid.




   .. py:method:: isosteric_energy_temperature_deriv(p: float, T: float, q: float = 1, stepsize: float = 0.001) -> float

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



   .. py:method:: differential_energy(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: differential_heat(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: internal_energy_adsorbed(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: areal_immersion_energy(T: float) -> float

      Calculate the areal energy of immersion (J/m^2).

      The calculation is based on the one written in Rouquerol et al. [1]_.

      :param T: Temperature (K).
      :type T: float

      :returns: Areal energy of immersion (J/m^2)
      :rtype: float



.. py:class:: DAModel(sorbent: str, stored_fluid: StoredFluid, w0: float, f0: float, eps: float, m: float = 2, k: float = 2, rhoa: float = None, va: float = None, va_mode: str = 'Constant', rhoa_mode: str = 'Constant', f0_mode: str = 'Dubinin')

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


   .. py:method:: f0_calc(T: float) -> float

      Calculate the fugacity at saturation (Pa) at a given temperature.

      :param T: Temperature (K).
      :type T: float

      :returns: Fugacity at saturation (Pa).
      :rtype: float



   .. py:method:: rhoa_calc(T: float) -> float

      Calculate the density of the adsorbed phase at a given temperature.

      :param T: Temperature (K).
      :type T: float

      :returns: The density of the adsorbed phase (mol/m^3).
      :rtype: float



   .. py:method:: v_ads(p: float, T: float) -> float

      Calculate the volume of the adsorbed phase (m^3/kg).

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: Volume of the adsorbed phase (m^3/kg).
      :rtype: float



   .. py:method:: n_absolute(p: float, T: float) -> float

      Calculate the absolute adsorbed amount at a given condition.

      :param p: Pressure(Pa).
      :type p: float
      :param T: Temperature(K).
      :type T: float

      :returns: Absolute adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: n_excess(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: differential_energy(p, T, q)

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



   .. py:method:: internal_energy_adsorbed(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: from_ExcessIsotherms(ExcessIsotherms: List[pytanksim.classes.excessisothermclass.ExcessIsotherm], stored_fluid: StoredFluid = None, sorbent: str = None, w0guess: float = 0.001, f0guess: float = 1470000000.0, epsguess: float = 3000, vaguess: float = 0.001, rhoaguess: float = None, mguess: float = 2.0, kguess: float = 2.0, rhoa_mode: str = 'Fit', f0_mode: str = 'Fit', m_mode: str = 'Fit', k_mode: str = 'Fit', va_mode: str = 'Excess', pore_volume: float = 0.003, verbose: bool = True) -> DAModel
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



.. py:class:: MDAModel(sorbent: str, stored_fluid: StoredFluid, nmax: float, f0: float, alpha: float, beta: float, va: float, m: float = 2, k: float = 2, va_mode: str = 'Constant', f0_mode: str = 'Constant')

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


   .. py:method:: n_absolute(p: float, T: float) -> float

      Calculate the absolute adsorbed amount at given conditions.

      :param p: Pressure (Pa)
      :type p: float
      :param T: Temperature (K)
      :type T: float

      :returns: Absolute adsorbed amount (mol/kg).
      :rtype: float



   .. py:method:: v_ads(p: float, T: float) -> float

      Calculate the adsorbed phase volume at the given condtions.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float

      :returns: Adsorbed phase volume (m^3/kg)
      :rtype: float



   .. py:method:: n_excess(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: internal_energy_adsorbed(p: float, T: float, q: float = 1) -> float

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



   .. py:method:: differential_energy(p, T, q=1)

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



   .. py:method:: from_ExcessIsotherms(ExcessIsotherms: List[pytanksim.classes.excessisothermclass.ExcessIsotherm], stored_fluid: StoredFluid = None, sorbent: str = None, nmaxguess: float = 71.6, f0guess: float = 1470000000.0, alphaguess: float = 3080, betaguess: float = 18.9, vaguess: float = 0.00143, mguess: float = 2.0, kguess: float = 2.0, va_mode: str = 'Fit', f0_mode: str = 'Fit', m_mode: str = 'Fit', k_mode: str = 'Fit', beta_mode: str = 'Fit', pore_volume: float = 0.003, verbose: bool = True) -> MDAModel
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



.. py:class:: SorbentMaterial(skeletal_density: float, bulk_density: float, specific_surface_area: float, model_isotherm: ModelIsotherm, mass: float = 0, molar_mass: float = 0.01201, Debye_temperature: float = 1500, heat_capacity_function: Callable[[float], float] = None)

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


