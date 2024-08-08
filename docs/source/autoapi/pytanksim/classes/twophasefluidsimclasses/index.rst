pytanksim.classes.twophasefluidsimclasses
=========================================

.. py:module:: pytanksim.classes.twophasefluidsimclasses

.. autoapi-nested-parse::

   Module for simulating fluid tanks in the two-phase region.



Classes
-------

.. autoapisummary::

   pytanksim.classes.twophasefluidsimclasses.TwoPhaseFluidSim
   pytanksim.classes.twophasefluidsimclasses.TwoPhaseFluidDefault
   pytanksim.classes.twophasefluidsimclasses.TwoPhaseFluidVenting
   pytanksim.classes.twophasefluidsimclasses.TwoPhaseFluidCooled
   pytanksim.classes.twophasefluidsimclasses.TwoPhaseFluidHeatedDischarge


Module Contents
---------------

.. py:class:: TwoPhaseFluidSim(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


.. py:class:: TwoPhaseFluidDefault(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(time: float, ng: float, nl: float, T: float) -> numpy.ndarray

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



.. py:class:: TwoPhaseFluidVenting(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(time: float) -> numpy.ndarray

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



.. py:class:: TwoPhaseFluidCooled(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(time: float) -> numpy.ndarray

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



.. py:class:: TwoPhaseFluidHeatedDischarge(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(time: float) -> numpy.ndarray

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



