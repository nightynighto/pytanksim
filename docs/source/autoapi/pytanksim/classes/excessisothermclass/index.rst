pytanksim.classes.excessisothermclass
=====================================

.. py:module:: pytanksim.classes.excessisothermclass

.. autoapi-nested-parse::

   Contains the ExcessIsotherm class.



Classes
-------

.. autoapisummary::

   pytanksim.classes.excessisothermclass.ExcessIsotherm


Module Contents
---------------

.. py:class:: ExcessIsotherm(adsorbate: str, sorbent: str, temperature: float, loading: List[float], pressure: List[float])

   Stores experimental excess isotherm measurement results.

   This class can be provided values directly in Python or it can import
   the values from a csv file.

   .. attribute:: adsorbate

      Name of the adsorbate gas.

      :type: str

   .. attribute:: sorbent

      Name of the sorbent material.

      :type: str

   .. attribute:: temperature

      Temperature (K) at which the isotherm was measured.

      :type: float

   .. attribute:: loading

      A list of excess adsorption values (mol/kg).

      :type: List[float]

   .. attribute:: pressure

      A list of pressures (Pa) corresponding to points at which the excess
      adsorption values were measured.

      :type: list[float]

   Initialize the ExcessIsotherm class.

   :param adsorbate: Name of the adsorbate gas.
   :type adsorbate: str
   :param sorbent: Name of the sorbent material.
   :type sorbent: str
   :param temperature: Temperature (K) at which the isotherm was measured.
   :type temperature: float
   :param loading: A list of excess adsorption values (mol/kg).
   :type loading: List[float]
   :param pressure: A list of pressures (Pa) corresponding to points at which the
                    excess adsorption values were measured.
   :type pressure: list[float]

   :raises ValueError: If the lengths of the loading and pressure data don't match.

   :returns: A class which stores experimental excess adsorption data.
   :rtype: ExcessIsotherm


   .. py:method:: from_csv(filename: str, adsorbate: str, sorbent: str, temperature: float) -> ExcessIsotherm
      :classmethod:


      Import loading and pressure data from a csv file.

      :param filename: Path leading to the file from which the data is to be imported.
      :type filename: str
      :param adsorbate: Name of adsorbate gas.
      :type adsorbate: str
      :param sorbent: Name of sorbent material.
      :type sorbent: str
      :param temperature: Temperature (K) at which the data was measured.
      :type temperature: float

      :returns: A class which stores experimental excess adsorption data.
      :rtype: ExcessIsotherm



