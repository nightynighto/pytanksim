pytanksim.classes.twophasesorbentsimclasses
===========================================

.. py:module:: pytanksim.classes.twophasesorbentsimclasses

.. autoapi-nested-parse::

   Module for simulating sorbent tanks in the two-phase region.



Classes
-------

.. autoapisummary::

   pytanksim.classes.twophasesorbentsimclasses.TwoPhaseSorbentSim
   pytanksim.classes.twophasesorbentsimclasses.TwoPhaseSorbentDefault
   pytanksim.classes.twophasesorbentsimclasses.TwoPhaseSorbentCooled
   pytanksim.classes.twophasesorbentsimclasses.TwoPhaseSorbentVenting
   pytanksim.classes.twophasesorbentsimclasses.TwoPhaseSorbentHeatedDischarge


Module Contents
---------------

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



