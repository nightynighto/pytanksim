pytanksim.classes.onephasefluidsimclasses
=========================================

.. py:module:: pytanksim.classes.onephasefluidsimclasses

.. autoapi-nested-parse::

   Module for simulating one phase fluid storage tanks without sorbents.



Classes
-------

.. autoapisummary::

   pytanksim.classes.onephasefluidsimclasses.OnePhaseFluidSim
   pytanksim.classes.onephasefluidsimclasses.OnePhaseFluidDefault
   pytanksim.classes.onephasefluidsimclasses.OnePhaseFluidVenting
   pytanksim.classes.onephasefluidsimclasses.OnePhaseFluidCooled
   pytanksim.classes.onephasefluidsimclasses.OnePhaseFluidHeatedDischarge


Module Contents
---------------

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



