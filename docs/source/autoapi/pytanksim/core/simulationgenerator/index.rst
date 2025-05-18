pytanksim.core.simulationgenerator
==================================

.. py:module:: pytanksim.core.simulationgenerator

.. autoapi-nested-parse::

   Main module of pytanksim, used to generate simulations.



Functions
---------

.. autoapisummary::

   pytanksim.core.simulationgenerator.generate_simulation
   pytanksim.core.simulationgenerator.automatic_simulation


Module Contents
---------------

.. py:function:: generate_simulation(storage_tank, boundary_flux, simulation_params, simulation_type = 'Default', phase = 1)

   Generate a dynamic simulation object.

   :param storage_tank: An object with the properties of the storage tank. Can either be of the
                        class StorageTank or its child class SorbentTank.
   :type storage_tank: Union[StorageTank, SorbentTank]
   :param boundary_flux: An object containing information about the mass and energy entering and
                         leaving the control volume of the tank.
   :type boundary_flux: BoundaryFlux
   :param simulation_params: An object containing various parameters for the dynamic simulation.
   :type simulation_params: SimParams
   :param simulation_type: A string describing the type of the simulation to be run. The default
                           is "Default". The valid types are:

                               - ``Default`` : A regular dynamic simulation with no constraints.
                               - ``Cooled`` : A simulation where the tank is cooled to maintain a
                                 constant pressure. Here, the cooling power becomes one of the
                                 output variables. Typically used for simulating refueling after
                                 the tank has reached maximum allowable working pressure, or for
                                 simulating zero boil-off systems which are actively cooled.
                               - ``Heated``: A simulation where the tank is heated to maintain a
                                 constant pressure. Here, the heating power becomes one of the
                                 output variables. Typically used for simulating discharging when
                                 the tank has reached the minimum supply pressure of the fuel cell
                                 system.
                               - ``Venting`` : A simulation where the tank vents the fluid stored
                                 inside to maintain a constant pressure. Here, the amount vented
                                 becomes an output variable. Typically used to simulate boil-off
                                 or refueling with a feed-and-bleed scheme.
   :type simulation_type: str, optional
   :param phase: Specifies whether the fluid being stored is a single phase (1) or a
                 two-phase (2) liquid and gas mixture. The default is 1 for single
                 phase.
   :type phase: int, optional

   :returns: A simulation object which can be ``run()`` to output a SimResults
             object. Which class will be generated depends on the parameters
             provided to this function.
   :rtype: A child class of BaseSimulation


.. py:function:: automatic_simulation(storage_tank, boundary_flux, simulation_params, stop_at_max_pres = False, stop_at_min_pres = False, handle_max_pres = 'Cooled', handle_min_pres = 'Heated')

   Automatically run and restart simulations until a target is reached.

   :param storage_tank: An object with the properties of the storage tank. Can either be of the
                        class StorageTank or its child class SorbentTank.
   :type storage_tank: Union[StorageTank, SorbentTank]
   :param boundary_flux: An object containing information about the mass and energy entering and
                         leaving the control volume of the tank.
   :type boundary_flux: BoundaryFlux
   :param simulation_params: An object containing various parameters for the dynamic simulation.
   :type simulation_params: SimParams
   :param stop_at_max_pres: Whether or not the simulation is to be stopped when the tank hits its
                            maximum allowable working pressure. The default is False.
   :type stop_at_max_pres: bool, optional
   :param stop_at_min_pres: Whether or not the simulation is to be stopped when the tank hits its
                            minimum supply pressure. The default is False.
   :type stop_at_min_pres: bool, optional
   :param handle_max_pres: A string indicating how the simulation is to continue if the tank has
                           reached its maximum allowable working pressure. "Cooled" means that the
                           tank will not vent any gas, but will be actively cooled down. "Venting"
                           means that the tank will begin to vent the exact amount of fluid inside
                           to maintain the maximum pressure. The default is "Cooled".
   :type handle_max_pres: str, optional
   :param handle_min_pres: A string indicating how the simulation is to continue if the tank has
                           reached its minimum supply pressure. "Heated" means exactly enough heat
                           will be provided to the tank to maintain the minimum supply pressure.
                           "Continue" means the simulation will restart without changing any
                           parameters. The default is "Heated".
   :type handle_min_pres: str, optional

   :returns: An object for storing and manipulating the results of the dynamic
             simulations.
   :rtype: SimResults


