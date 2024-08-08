pytanksim.classes.onephasesorbentsimclasses
===========================================

.. py:module:: pytanksim.classes.onephasesorbentsimclasses

.. autoapi-nested-parse::

   Module for the simulation of sorbent tanks in the one phase region.



Classes
-------

.. autoapisummary::

   pytanksim.classes.onephasesorbentsimclasses.OnePhaseSorbentSim
   pytanksim.classes.onephasesorbentsimclasses.OnePhaseSorbentDefault
   pytanksim.classes.onephasesorbentsimclasses.OnePhaseSorbentVenting
   pytanksim.classes.onephasesorbentsimclasses.OnePhaseSorbentCooled
   pytanksim.classes.onephasesorbentsimclasses.OnePhaseSorbentHeatedDischarge


Module Contents
---------------

.. py:class:: OnePhaseSorbentSim(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


.. py:class:: OnePhaseSorbentDefault(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(p: float, T: float, time: float) -> numpy.ndarray

      Find the right hand side of the governing ODE at a given time step.

      :param p: Current pressure (Pa).
      :type p: float
      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run() -> pytanksim.classes.simresultsclass.SimResults

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseSorbentVenting(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(T: float, time: float) -> numpy.ndarray

      Find the right hand side of the governing ODE at a given time step.

      :param T: Current temperature (K).
      :type T: float
      :param time: Current time step (in s).
      :type time: float

      :returns: An array containing the right hand side of the ODE.
      :rtype: np.ndarray



   .. py:method:: run() -> pytanksim.classes.simresultsclass.SimResults

      Run the dynamic simulation.

      :raises TerminateSimulation: Stops the simulation when it detects an event such as hitting the
          saturation line, or hitting the maximum pressure limit of the tank.

      :returns: An object for storing and manipulating the results of the dynamic
                simulation.
      :rtype: SimResults



.. py:class:: OnePhaseSorbentCooled(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(T: float, time: float) -> numpy.ndarray

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



.. py:class:: OnePhaseSorbentHeatedDischarge(simulation_params: SimParams, storage_tank: pytanksim.classes.storagetankclasses.StorageTank, boundary_flux: BoundaryFlux)

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


   .. py:method:: solve_differentials(T: float, time: float) -> numpy.ndarray

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



