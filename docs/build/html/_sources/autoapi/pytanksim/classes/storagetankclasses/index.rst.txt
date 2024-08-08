pytanksim.classes.storagetankclasses
====================================

.. py:module:: pytanksim.classes.storagetankclasses

.. autoapi-nested-parse::

   Contains classes which store the properties of the storage tanks.

   The StorageTank and SorbentTank classes are part of this module.



Classes
-------

.. autoapisummary::

   pytanksim.classes.storagetankclasses.StorageTank
   pytanksim.classes.storagetankclasses.SorbentTank


Module Contents
---------------

.. py:class:: StorageTank(stored_fluid: pytanksim.classes.fluidsorbentclasses.StoredFluid, aluminum_mass: float = 0, carbon_fiber_mass: float = 0, steel_mass: float = 0, vent_pressure: float = None, min_supply_pressure: float = 100000.0, thermal_resistance: float = 0, surface_area: float = 0, heat_transfer_coefficient: float = 0, volume: float = None, set_capacity: float = None, full_pressure: float = None, empty_pressure: float = None, full_temperature: float = None, empty_temperature: float = None, full_quality: float = 1, empty_quality: float = 1)

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
      If 0, the value will not be considered in simulations. If the
      arguments 'surface_area' and 'heat_transfer' are passed,
      'thermal_resistance' will be calculated based on those two arguments
      as long as the user does not pass a value to 'thermal_resistance'.

      :type: float, optional

   .. attribute:: surface_area

      The surface area of the tank that is in contact with the
      environment (m^2). The default is 0.

      :type: float, optional

   .. attribute:: heat_transfer_coefficient

      The heat transfer coefficient of the tank surface (W/(m^2 K)).
      The default is 0.

      :type: float, optional

   .. attribute:: Initialize a StorageTank object.

      

   .. attribute:: 

      being stored.

      :type: param stored_fluid: Object to calculate the thermophysical properties of the fluid

   .. attribute:: 

      

      :type: type stored_fluid: StoredFluid

   .. attribute:: 

      
      
      0.

      :type: param aluminum_mass: The mass of aluminum making up the tank walls (kg). The default is

   .. attribute:: 

      

      :type: type aluminum_mass: float, optional

   .. attribute:: 

      is 0.

      :type: param carbon_fiber_mass: The mass of carbon fiber making up the tank walls (kg). The default

   .. attribute:: 

      

      :type: type carbon_fiber_mass: float, optional

   .. attribute:: 

      

      :type: param steel_mass: The mass of steel making up the tank walls (kg). The default is 0.

   .. attribute:: 

      

      :type: type steel_mass: float, optional

   .. attribute:: 

      The default is None. If None, the value will be taken as the
      maximum value where the CoolProp backend can calculate the
      properties of the fluid being stored.

      :type: param vent_pressure: The pressure (Pa) at which the fluid being stored must be vented.

   .. attribute:: 

      

      :type: type vent_pressure: float, optional

   .. attribute:: 

      default is 1E5.

      :type: param min_supply_pressure: The minimum supply pressure (Pa) for discharging simulations.The

   .. attribute:: 

      

      :type: type min_supply_pressure: float, optional

   .. attribute:: 

      If 0, the value will not be considered in simulations. If the
      arguments 'surface_area' and 'heat_transfer' are passed,
      'thermal_resistance' will be calculated based on those two
      arguments as long as the user does not pass a value to
      'thermal_resistance'.

      :type: param thermal_resistance: The thermal resistance of the tank walls (K/W). The default is 0.

   .. attribute:: 

      

      :type: type thermal_resistance: float, optional

   .. attribute:: 

      environment (m^2). The default is 0.

      :type: param surface_area: The surface area of the tank that is in contact with the

   .. attribute:: 

      

      :type: type surface_area: float, optional

   .. attribute:: 

      The default is 0.

      :type: param heat_transfer_coefficient: The heat transfer coefficient of the tank surface (W/(m^2 K)).

   .. attribute:: 

      

      :type: type heat_transfer_coefficient: float, optional

   .. attribute:: 

      This value is required unless the set capacity and operating
      conditions are defined, in which case the volume is calculated from
      the capacity and operating conditions.

      :type: param volume: Internal volume of the storage tank (m^3). The default is None.

   .. attribute:: 

      

      :type: type volume: float, optional

   .. attribute:: 

      None. If specified, this will override the user-specified tank
      volume.

      :type: param set_capacity: Set internal capacity of the storage tank (mol). The default is

   .. attribute:: 

      

      :type: type set_capacity: float, optional

   .. attribute:: 

      is None.

      :type: param full_pressure: Pressure (Pa) of the tank when it is considered full. The default

   .. attribute:: 

      

      :type: type full_pressure: float, optional

   .. attribute:: 

      is None.

      :type: param empty_pressure: Pressure (Pa) of the tank when it is considered empty. The default

   .. attribute:: 

      

      :type: type empty_pressure: float, optional

   .. attribute:: 

      default is None.

      :type: param full_temperature: Temperature (K) of the tank when it is considered full. The

   .. attribute:: 

      

      :type: type full_temperature: float, optional

   .. attribute:: 

      default is None.

      :type: param empty_temperature: Temperature (K) of the tank when it is considered empty. The

   .. attribute:: 

      

      :type: type empty_temperature: float, optional

   .. attribute:: 

      is 1 (Gas).

      :type: param full_quality: Vapor quality of the tank when it is considered full. The default

   .. attribute:: 

      

      :type: type full_quality: float, optional

   .. attribute:: 

      is 1 (Gas).

      :type: param empty_quality: Vapor quality of the tank when it is considered empty. The default

   .. attribute:: 

      

      :type: type empty_quality: float, optional

   .. attribute:: 

      

      :type: raises ValueError: If any of the mass values provided are less than 0.

   .. attribute:: 

      'CoolProp'.

      :type: raises ValueError: If the vent pressure set is higher than what can be calculated by

   .. attribute:: 

      and temperature swing conditions were provided.

      :type: raises ValueError: If neither the volume nor the complete capacity and the pressure

   .. attribute:: 

      simulations and can calculate certain properties on its own.

      :type: returns: A storage tank object which can be passed as arguments to dynamic

   .. attribute:: 

      

      :type: rtype: StorageTank


   .. py:method:: capacity(p: float, T: float, q: float = 0, unit: str = 'mol') -> float

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



   .. py:method:: capacity_bulk(p: float, T: float, q: float = 0, unit: str = 'mol') -> float

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



   .. py:method:: find_quality_at_saturation_capacity(T: float, capacity: float) -> float

      Find vapor quality at the given temperature and capacity.

      :param T: Temperature (K)
      :type T: float
      :param capacity: Amount of fluid in the tank (moles).
      :type capacity: float

      :returns: Vapor quality of the fluid being stored. This is assuming that the
                fluid is on the saturation line.
      :rtype: float



   .. py:method:: internal_energy(p: float, T: float, q: float = 1) -> float

      Calculate the internal energy of the fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the fluid being stored (J).
      :rtype: float



   .. py:method:: conditions_at_capacity_temperature(cap: float, T: float, p_guess: float, q_guess: float) -> scipy.optimize.OptimizeResult

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



   .. py:method:: conditions_at_capacity_pressure(cap: float, p: float, T_guess: float, q_guess: float) -> scipy.optimize.OptimizeResult

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



   .. py:method:: calculate_dormancy(p: float, T: float, heating_power: float, q: float = 0) -> pandas.DataFrame

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



.. py:class:: SorbentTank(sorbent_material: pytanksim.classes.fluidsorbentclasses.SorbentMaterial, aluminum_mass: float = 0, carbon_fiber_mass: float = 0, steel_mass: float = 0, vent_pressure: float = None, min_supply_pressure: float = 100000.0, thermal_resistance: float = 0, surface_area: float = 0, heat_transfer_coefficient: float = 0, volume: float = None, set_capacity: float = None, full_pressure: float = None, empty_pressure: float = None, full_temperature: float = None, empty_temperature: float = None, full_quality: float = 1, empty_quality: float = 1, set_sorbent_fill: float = 1)

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
      If 0, the value will not be considered in simulations. If the
      arguments 'surface_area' and 'heat_transfer' are passed,
      'thermal_resistance' will be calculated based on those two
      arguments as long as the user does not pass a value to
      'thermal_resistance'.

      :type: float, optional

   .. attribute:: surface_area

      Outer surface area of the tank in contact with the environment
      (m^2). The default is 0.

      :type: float, optional

   .. attribute:: heat_transfer_coefficient

      The heat transfer coefficient of the tank surface (W/(m^2 K)).
      The default is 0.

      :type: float, optional

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
   :param thermal_resistance: The thermal resistance of the tank walls (K/W). The default is 0.
                              If 0, the value will not be considered in simulations. If the
                              arguments 'surface_area' and 'heat_transfer' are passed,
                              'thermal_resistance' will be calculated based on those two
                              arguments as long as the user does not pass a value to
                              'thermal_resistance'.
   :type thermal_resistance: float, optional
   :param surface_area: Outer surface area of the tank in contact with the environment
                        (m^2). The default is 0.
   :type surface_area: float, optional
   :param heat_transfer_coefficient: The heat transfer coefficient of the tank surface (W/(m^2 K)).
                                     The default is 0.
   :type heat_transfer_coefficient: float, optional
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


   .. py:method:: bulk_fluid_volume(p: float, T: float) -> float

      Calculate the volume of bulk fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature(K).
      :type T: float

      :returns: Bulk fluid volume within the tank (m^3).
      :rtype: float



   .. py:method:: capacity(p: float, T: float, q: float = 0) -> float

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



   .. py:method:: capacity_bulk(p: float, T: float, q: float = 0) -> float

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



   .. py:method:: internal_energy(p: float, T: float, q: float = 1) -> float

      Calculate the internal energy of the fluid inside of the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the fluid being stored (J).
      :rtype: float



   .. py:method:: internal_energy_sorbent(p: float, T: float, q: float = 1) -> float

      Calculate the internal energy of the adsorbed fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the adsorbed fluid in the tank (J).
      :rtype: float



   .. py:method:: internal_energy_bulk(p: float, T: float, q: float = 1) -> float

      Calculate the internal energy of the bulk fluid in the tank.

      :param p: Pressure (Pa).
      :type p: float
      :param T: Temperature (K).
      :type T: float
      :param q: Vapor quality of the fluid being stored. The default is 1.
      :type q: float, optional

      :returns: Internal energy of the bulk fluid in the tank (J).
      :rtype: float



   .. py:method:: find_quality_at_saturation_capacity(T: float, capacity: float) -> float

      Find vapor quality at the given temperature and capacity.

      :param T: Temperature (K)
      :type T: float
      :param capacity: Amount of fluid in the tank (moles).
      :type capacity: float

      :returns: Vapor quality of the fluid being stored. This is assuming that the
                fluid is on the saturation line.
      :rtype: float



   .. py:method:: find_temperature_at_saturation_quality(q: float, cap: float) -> scipy.optimize.OptimizeResult

      Find temperature at a given capacity and vapor quality value.

      :param q: Vapor quality. Can vary between 0 and 1.
      :type q: float
      :param cap: Amount of fluid stored in the tank (moles).
      :type cap: float

      :returns: The optimization result represented as a OptimizeResult object.
                The relevant attribute for this function is x, the optimized
                temperature value.
      :rtype: scipy.optimize.OptimizeResult



   .. py:method:: calculate_dormancy(p: float, T: float, heating_power: float, q: float = 0) -> pandas.DataFrame

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



