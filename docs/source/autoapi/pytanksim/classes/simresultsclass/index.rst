pytanksim.classes.simresultsclass
=================================

.. py:module:: pytanksim.classes.simresultsclass

.. autoapi-nested-parse::

   Contains the SimResults class.

   It is used for storing and post-processing the results of dynamic simulations.



Classes
-------

.. autoapisummary::

   pytanksim.classes.simresultsclass.SimResults


Module Contents
---------------

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



