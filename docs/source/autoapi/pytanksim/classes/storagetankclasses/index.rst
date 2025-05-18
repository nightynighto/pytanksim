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



