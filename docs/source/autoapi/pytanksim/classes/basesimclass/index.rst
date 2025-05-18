pytanksim.classes.basesimclass
==============================

.. py:module:: pytanksim.classes.basesimclass

.. autoapi-nested-parse::

   Contains classes related exclusively to the dynamic simulations.

   This includes SimParams, BoundaryFlux, and BaseSimulation.



Classes
-------

.. autoapisummary::

   pytanksim.classes.basesimclass.SimParams
   pytanksim.classes.basesimclass.BoundaryFlux
   pytanksim.classes.basesimclass.BaseSimulation


Module Contents
---------------

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
   :param vented_energy: Cumulative amount of enthaloy (J) contained in the fluid vented prior
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



