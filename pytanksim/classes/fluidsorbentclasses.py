# -*- coding: utf-8 -*-
"""Contains classes related to the fluids and sorbents to be simulated.

More specifically, contains the StoredFluid, SorbentMaterial, and the various
ModelIsotherm classes.
"""

__all__ = ["StoredFluid", "SorbentMaterial", "MDAModel", "DAModel"]

import numpy as np
import CoolProp as CP
import lmfit
from pytanksim.classes.excessisothermclass import ExcessIsotherm
from copy import deepcopy
import pytanksim.utils.finitedifferences as fd
from typing import List, Dict, Callable
import scipy as sp


class StoredFluid:
    """A class to calculate the properties of the fluid being stored.

    Attributes
    ----------
    fluid_name : str
        The name of the fluid being stored which corresponds to fluid names
        in the package CoolProp.

    EOS : str
        The name of the equation of state to be used for the calculations
        of fluid properties by the package CoolProp.

    backend : CoolProp.AbstractState
        The CoolProp backend used for calculation of fluid properties at
        various conditions.

    """

    def __init__(self,
                 fluid_name: str,
                 EOS: str) -> "StoredFluid":
        """Initialize a StoredFluid object.

        Parameters
        ----------
        fluid_name : str, optional
            Name of the fluid. Valid fluid names that work with CoolProp can be
            found 'here
            <http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids>'_.

        EOS : str, optional
            Name of the equation of state to be used for calculations.

        Returns
        -------
        StoredFluid
            A class to calculate the properties of the fluid being stored.

        """
        self.fluid_name = fluid_name
        self.EOS = EOS
        self.backend = CP.AbstractState(EOS, fluid_name)

    def fluid_property_dict(self, p: float, T: float) -> Dict[str, float]:
        """Generate a dictionary of fluid properties using CoolProp.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K)

        Returns
        -------
        Dict[str, float]
            Dictionary containing several fluid properties needed for various
            calculations in pytanksim. "hf" is the enthalpy (J/mol). "drho_dp"
            is the first partial derivative of density (mol/m^3) w.r.t.
            pressure (Pa). "drho_dT" is the first partial derivative of density
            (mol/m^3) w.r.t. temperature (K). "rhof" is density (mol/m^3).
            "dh_dp" is the first partial derivative of enthalpy (J/mol) w.r.t.
            pressure (Pa). "dh_dT" is the first partial derivative of enthalpy
            (J/mol) w.r.t. temperature (K). "uf" is the internal energy
            (J/mol). "du_dp" is the first partial derivative of internal energy
            (J/mol) w.r.t. pressure (Pa). "du_dT" is the first partial
            derivative of internal energy (J/mol) w.r.t. temperature (K). "MW"
            is molar mass (kg/mol).

        """
        backend = self.backend
        backend.update(CP.PT_INPUTS, p, T)
        return {
            "hf": backend.hmolar(),
            "drho_dp": backend.first_partial_deriv(CP.iDmolar, CP.iP, CP.iT),
            "drho_dT":  backend.first_partial_deriv(CP.iDmolar, CP.iT, CP.iP),
            "rhof": backend.rhomolar(),
            "dh_dp": backend.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT),
            "dh_dT": backend.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP),
            "uf": backend.umolar(),
            "du_dp": backend.first_partial_deriv(CP.iUmolar, CP.iP, CP.iT),
            "du_dT": backend.first_partial_deriv(CP.iUmolar, CP.iT, CP.iP),
            "MW": backend.molar_mass()
            }

    def saturation_property_dict(self,
                                 T: float,
                                 Q: int = 0) -> Dict[str, float]:
        """Generate a dictionary of fluid properties at saturation.

        Parameters
        ----------
        T : float
            Temperature in K.

        Q : float
            Vapor quality of the fluid being stored.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the fluid properties at saturation
            at a given temperature. "psat" is the saturation vapor pressure
            (Pa). "dps_dT" is the first derivative of the saturation vapor
            pressure (Pa) w.r.t. temperature (K). "hf" is the enthalpy (J/mol).
            "drho_dp" is the first partial derivative of density (mol/m^3)
            w.r.t. pressure (Pa). "drho_dT" is the first partial derivative of
            density (mol/m^3) w.r.t. temperature (K). "rhof" is density
            (mol/m^3). "dh_dp" is the first partial derivative of enthalpy
            (J/mol) w.r.t. pressure (Pa). "dh_dT" is the first partial
            derivative of enthalpy (J/mol) w.r.t. temperature (K). "uf" is the
            internal energy (J/mol). "du_dp" is the first partial derivative of
            internal energy (J/mol) w.r.t. pressure (Pa). "du_dT" is the first
            partial derivative of internal energy (J/mol) w.r.t. temperature
            (K). "MW" is molar mass (kg/mol).

        """
        backend = self.backend
        backend.update(CP.QT_INPUTS, Q, T)
        return {
            "psat": backend.p(),
            "dps_dT": backend.first_saturation_deriv(CP.iP, CP.iT),
            "hf": backend.hmolar(),
            "drho_dp": backend.first_partial_deriv(CP.iDmolar, CP.iP, CP.iT),
            "drho_dT":  backend.first_partial_deriv(CP.iDmolar, CP.iT, CP.iP),
            "rhof": backend.rhomolar(),
            "dh_dp": backend.first_partial_deriv(CP.iHmolar, CP.iP, CP.iT),
            "dh_dT": backend.first_partial_deriv(CP.iHmolar, CP.iT, CP.iP),
            "uf": backend.umolar(),
            "du_dp": backend.first_partial_deriv(CP.iUmolar, CP.iP, CP.iT),
            "du_dT": backend.first_partial_deriv(CP.iUmolar, CP.iT, CP.iP),
            "MW": backend.molar_mass()
            }

    def determine_phase(self,
                        p: float,
                        T: float) -> str:
        """Determine the phase of the fluid being stored.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        Returns
        -------
        str
            String that could either be "Supercritical", "Gas", "Liquid",
            or "Saturated" depending on the bulk fluid phase.

        """
        fluid = self.backend
        Tcrit = fluid.T_critical()
        pcrit = fluid.p_critical()
        if T > Tcrit:
            if p > pcrit:
                return "Supercritical"
            else:
                return "Gas"
        else:
            fluid.update(CP.QT_INPUTS, 0, T)
            psat = fluid.p()
            if np.abs(p-psat) <= (psat * 1E-5):
                return "Saturated"
            elif p < psat:
                return "Gas"
            elif p > psat and p > pcrit:
                return "Supercritical"
            elif p > psat and p < pcrit:
                return "Liquid"


class ModelIsotherm:
    """A base class for model isotherm objects.

    Contains methods to calculate various thermodynamic properties of
    the adsorbed phase.

    """

    def pressure_from_absolute_adsorption(self, n_abs: float, T: float,
                                          p_max_guess: float = 20E6) -> float:
        """Calculate a pressure value corresponding to an adsorbed amount.

        Parameters
        ----------
        n_abs : float
            Amount adsorbed (mol/kg).

        T : float
            Temperature (K).

        p_max_guess : float, optional
            Maximum pressure (Pa) for the optimization. The default is 20E6.
            If the value provided is larger than the maximum that can be
            handled by the CoolProp backend, it will take the maximum that
            can be handled by the CoolProp backend.

        Returns
        -------
        float
            Pressure (Pa) corresponding to the specified adsorbed amount
            and temperature value.

        """
        p_max_guess = min(p_max_guess, self.stored_fluid.backend.pmax()/10)
        if n_abs == 0:
            return 0

        def optimum_pressure(p):
            return (self.n_absolute(p, T) - n_abs)**2

        root = sp.optimize.minimize_scalar(
                                fun=optimum_pressure,
                                bounds=(1, p_max_guess),
                                method="bounded")
        return root.x

    def isosteric_enthalpy(self, p: float, T: float,
                           q: float = 1) -> float:
        """Calculate isosteric adsorbed enthalpy (J/mol).

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        Returns
        -------
        float
            Isosteric enthalpy of adsorption (J/mol).

        """
        nabs = self.n_absolute(p, T)
        fluid = self.stored_fluid.backend

        def diff_function(x):
            pres = self.pressure_from_absolute_adsorption(nabs, 1/x)
            phase = self.stored_fluid.determine_phase(pres, 1/x)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, 1/x)
            else:
                fluid.update(CP.QT_INPUTS, q, 1/x)
            return fluid.chemical_potential(0) * x

        phase = self.stored_fluid.determine_phase(p, T)
        x_loc = 1/T
        step = x_loc * 1E-2
        temp2 = 1/(x_loc + step)
        phase2 = self.stored_fluid.determine_phase(p, temp2)
        temp3 = 1/(x_loc - step)
        phase3 = self.stored_fluid.determine_phase(p, temp3)

        if phase == phase2 == phase3 != "Saturated":
            hadsorbed = fd.pardev(diff_function, x_loc, step)
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            if q == 1:
                hadsorbed = fd.backdev(diff_function, x_loc, step)
            else:
                hadsorbed = fd.fordev(diff_function, x_loc, step)
            if phase == "Saturated":
                fluid.update(CP.QT_INPUTS, q, T)
            else:
                fluid.update(CP.PT_INPUTS, p, T)
        hfluid = fluid.hmolar()
        return hfluid - hadsorbed

    def isosteric_internal_energy(self, p: float,
                                  T: float,
                                  q: float = 1) -> float:
        """Calculate the isosteric internal energy of the adsorbed phase.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        Returns
        -------
        float
            Isosteric internal energy of the adsorbed phase (J/mol).

        """
        nabs = self.n_absolute(p, T)
        fluid = self.stored_fluid.backend

        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)

        phase = self.stored_fluid.determine_phase(p, T)
        x_loc = T
        step = 1E-2
        temp2 = x_loc + step
        phase2 = self.stored_fluid.determine_phase(p, temp2)
        temp3 = x_loc - step
        phase3 = self.stored_fluid.determine_phase(p, temp3)
        if phase == phase2 == phase3 != "Saturated":
            hadsorbed = fd.pardev(diff_function, x_loc, step)
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            if q == 0:
                hadsorbed = fd.backdev(diff_function, x_loc, step)
            else:
                hadsorbed = fd.fordev(diff_function, x_loc, step)
            if phase == "Saturated":
                fluid.update(CP.QT_INPUTS, q, T)
            else:
                fluid.update(CP.PT_INPUTS, p, T)
        chempot = fluid.chemical_potential(0)
        uadsorbed = chempot - T * hadsorbed
        ufluid = fluid.umolar()
        return ufluid - uadsorbed

    def _derivfunc(self, func: Callable, var: int, point: float, qinit: float,
                   stepsize: float) -> float:
        """Calculate the first partial derivative.

        It automatically decides the direction of the derivative so that the
        evaluations are done for fluids at the same phases. Otherwise, there
        will be discontinuities in the fluid properties at different phases
        which causes the resulting derivative values to be invalid.

        """
        pT = point[:2]

        def phase_func(x):
            pT[var] = x
            return self.stored_fluid.determine_phase(pT[0], pT[1])

        x0 = point[var]
        x1 = x0 + stepsize
        x2 = x0 - stepsize
        phase1 = phase_func(x0)
        phase2 = phase_func(x1)
        phase3 = phase_func(x2)
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.partial_derivative(func, var, point, stepsize)
        elif phase1 == "Saturated":
            if (qinit == 0 and var == 1) or (qinit == 1 and var == 0):
                return fd.backward_partial_derivative(func, var, point,
                                                      stepsize)
            else:
                return fd.forward_partial_derivative(func, var, point,
                                                     stepsize)
        else:
            if phase1 == phase3:
                return fd.backward_partial_derivative(func, var, point,
                                                      stepsize)
            elif phase1 == phase2:
                return fd.forward_partial_derivative(func, var, point,
                                                     stepsize)

    def _derivfunc_second(self, func: Callable, point: float, qinit: float,
                          stepsize: float) -> float:
        """Calculate the second partial derivative.

        It automatically decides the direction of the derivative so that the
        evaluations are done for fluids at the same phases. Otherwise, there
        will be discontinuities in the fluid properties at different phases
        which causes the resulting derivative values to be invalid.

        """
        pT = point

        def phase_func(x):
            pT[1] = x
            return self.stored_fluid.determine_phase(pT[0], pT[1])

        x0 = point[1]
        x1 = x0 + stepsize
        x2 = x0 - stepsize
        phase1 = phase_func(x0)
        phase2 = phase_func(x1)
        phase3 = phase_func(x2)
        if phase1 == phase2 == phase3 != "Saturated":
            return fd.secder(func, x0, stepsize)
        elif phase1 == "Saturated":
            if qinit == 0:
                return fd.secbackder(func, x0, stepsize)
            else:
                return fd.secforder(func, x0, stepsize)
        else:
            if phase1 == phase3:
                return fd.secbackder(func, x0, stepsize)
            elif phase1 == phase2:
                return fd.secforder(func, x0, stepsize)

    def isosteric_energy_temperature_deriv(self, p: float, T: float,
                                           q: float = 1,
                                           stepsize: float = 1E-3) -> float:
        """Calculate first derivative of isosteric internal energy w.r.t. T.

        This function calculates the first partial derivative of the isosteric
        internal energy of the adsorbed phase (J/mol) w.r.t. temperature (K).

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        stepsize : float, optional
            Stepsize for numerical derivative. The default is 1E-3.

        Returns
        -------
        float
            The first partial derivative of the isosteric internal energy
            of the adsorbed phase (J/mol) w.r.t. temperature (K).

        """
        nabs = self.n_absolute(p, T)
        vads = self.v_ads(p, T)
        fluid = self.stored_fluid.backend

        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)

        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        du_dT = fluid.first_partial_deriv(CP.iUmolar, CP.iT, CP.iP)
        dhads_dT = - T * self._derivfunc_second(diff_function, [p, T],
                                                q, stepsize)
        dnabs_dT = self._derivfunc(self.n_absolute, 1, [p, T], q, stepsize)
        dvads_dT = self._derivfunc(self.v_ads, 1, [p, T], q, stepsize)
        return du_dT - (dhads_dT - (p / (nabs ** 2)) * (nabs * dvads_dT -
                                                        dnabs_dT * vads))

    def differential_energy(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the differential energy of adsorption (J/mol).

        The calculation is based on Myers & Monson [1]_.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        Returns
        -------
        float
            The differential energy of adsorption (J/mol).

        Notes
        -----
        .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
           the case for absolute adsorption as the basis for thermodynamic
           analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
           doi: 10.1007/s10450-014-9604-1.

        """
        nabs = self.n_absolute(p, T)
        fluid = self.stored_fluid.backend

        def diff_function(Temper):
            pres = self.pressure_from_absolute_adsorption(nabs, Temper)
            phase = self.stored_fluid.determine_phase(pres, Temper)
            if phase != "Saturated":
                fluid.update(CP.PT_INPUTS, pres, Temper)
            else:
                fluid.update(CP.QT_INPUTS, q, Temper)
            return fluid.chemical_potential(0)

        x_loc = T
        step = 1E-2
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            fluid.update(CP.QT_INPUTS, q, T)
        chempot = fluid.chemical_potential(0)
        hadsorbed = fd.pardev(diff_function, x_loc, step)
        uadsorbed = chempot - T * hadsorbed
        return uadsorbed

    def differential_heat(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the differential heat of adsorption (J/mol).

        The calculation is based on Myers & Monson [1]_.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        Returns
        -------
        float
            The differential energy of adsorption (J/mol).

        Notes
        -----
        .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
           the case for absolute adsorption as the basis for thermodynamic
           analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
           doi: 10.1007/s10450-014-9604-1.

        """
        if p == 0:
            return 0
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase == "Saturated":
            fluid.update(CP.QT_INPUTS, q, T)
        else:
            fluid.update(CP.PT_INPUTS, p, T)
        u_molar = fluid.umolar()
        return u_molar - self.differential_energy(p, T)

    def internal_energy_adsorbed(self, p: float, T: float,
                                 q: float = 1) -> float:
        """Calculate the molar integral internal energy of adsorption (J/mol).

        The calculation is based on Myers & Monson [1]_.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 to 1.
            The default is 1.

        Returns
        -------
        float
            The differential energy of adsorption (J/mol).

        Notes
        -----
        .. [1] A. L. Myers and P. A. Monson, ‘Physical adsorption of gases:
           the case for absolute adsorption as the basis for thermodynamic
           analysis’, Adsorption, vol. 20, no. 4, pp. 591–622, May 2014,
           doi: 10.1007/s10450-014-9604-1.

        """
        n_abs = self.n_absolute(p, T)
        n_grid = np.linspace(n_abs/50, n_abs, 50)
        p_grid = np.array([self.pressure_from_absolute_adsorption(n, T)
                           if n != 0 else 0 for n in n_grid])
        heat_grid = np.array([self.differential_energy(pres, T, q)
                              for pres in p_grid])
        return sp.integrate.simps(heat_grid, n_grid) / n_abs

    def areal_immersion_energy(self, T: float) -> float:
        """Calculate the areal energy of immersion (J/m^2).

        The calculation is based on the one written in Rouquerol et al. [1]_.

        Parameters
        ----------
        T : float
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        fluid = self.stored_fluid.backend

        def sur_tension(T):
            fluid.update(CP.QT_INPUTS, 0, T)
            sur_ten = self.stored_fluid.backend.surface_tension()
            return sur_ten

        diff = fd.partial_derivative(sur_tension, 0, [T], 0.001) \
            if T < fluid.T_critical() - 0.001  \
            else fd.backward_partial_derivative(sur_tension, 0, [T], 0.001)
        return T * diff - sur_tension(T)


class DAModel(ModelIsotherm):
    """A class for the Dubinin-Astakhov model for adsorption in micropores.

    Attributes
    ----------
    sorbent : str
        Name of sorbent material.

    stored_fluid : StoredFluid
        Object containing properties of the adsorbate.

    w0 : float
        The volume of the adsorbed phase at saturation (m^3/kg).

    f0 : float
        The fugacity at adsorption saturation (Pa).

    eps : float
        Characteristic energy of adsorption (J/mol).

    m : float, optional
        The empirical heterogeneity parameter for the Dubinin-Astakhov
        model. The default is 2.

    k : float, optional
        The empirical heterogeneity parameter for Dubinin's approximation
        of the saturation fugacity above critical temperatures. The default
        is 2.

    rhoa : float, optional
        The density of the adsorbed phase (mol/m^3). The default is None.
        If None, the value will be taken as the liquid density at 1 bar.

    va : float, optional
        The volume of the adsorbed phase (m^3/kg). The default is None.
        If None and va_mode is "Constant", the va_mode will be switched to
        "Excess" and the va will be assumed to be 0.

    va_mode : str, optional
        Determines how the adsorbed phase volume is calculated. "Excess"
        assumes that the adsorbed phase volume is 0, so the model
        calculates excess adsorption instead of absolute adsorption.
        "Constant" assumes a constant adsorbed phase volume. "Vary" will
        assume that the adsorbed phase volume varies according to the pore
        filling mechanism posited by the Dubinin-Astakhov equation. The
        default is "Constant", but if the parameter va is not specified it
        will switch to "Excess".

    rhoa_mode : str, optional
        Determines how the adsorbed phase density is calculated. "Ozawa"
        uses Ozawa's approximation to calculate the adsorbed phase density.
        "Constant" assumes a constant adsorbed phase volume. The default is
        "Constant".

    f0_mode : str, optional
        Determines how the fugacity at saturation is calculated. "Dubinin"
        uses Dubinin's approximation. "Constant" assumes a constant value
        for the fugacity at saturation. The default is "Dubinin".

    """

    def __init__(self,
                 sorbent: str,
                 stored_fluid: StoredFluid,
                 w0: float,
                 f0: float,
                 eps: float,
                 m: float = 2,
                 k: float = 2,
                 rhoa: float = None,
                 va: float = None,
                 va_mode: str = "Constant",
                 rhoa_mode: str = "Constant",
                 f0_mode: str = "Dubinin") -> "DAModel":
        """Initialize the DAModel class.

        Parameters
        ----------
        sorbent : str
            Name of sorbent material.

        stored_fluid : StoredFluid
            Object containing properties of the adsorbate.

        w0 : float
            The volume of the adsorbed phase at saturation (m^3/kg).

        f0 : float
            The fugacity at adsorption saturation (Pa).

        eps : float
            Characteristic energy of adsorption (J/mol).

        m : float, optional
            The empirical heterogeneity parameter for the Dubinin-Astakhov
            model. The default is 2.

        k : float, optional
            The empirical heterogeneity parameter for Dubinin's approximation
            of the saturation fugacity above critical temperatures. The default
            is 2.

        va : float, optional
            The volume of the adsorbed phase (m^3/kg). The default is None.

        rhoa : float, optional
            The density of the adsorbed phase (mol/m^3). The default is None.
            If None, the value will be taken as the liquid density at 1 bar.

        va_mode : str, optional
            Determines how the adsorbed phase volume is calculated. "Excess"
            assumes that the adsorbed phase volume is 0, so the model
            calculates excess adsorption instead of absolute adsorption.
            "Constant" assumes a constant adsorbed phase volume. "Vary" will
            assume that the adsorbed phase volume varies according to the pore
            filling mechanism posited by the Dubinin-Astakhov equation. The
            default is "Constant", but if the parameter va is not specified it
            will switch to "Excess".

        rhoa_mode : str, optional
            Determines how the adsorbed phase density is calculated. "Ozawa"
            uses Ozawa's approximation to calculate the adsorbed phase density.
            "Constant" assumes a constant adsorbed phase volume. The default is
            "Constant".

        f0_mode : str, optional
            Determines how the fugacity at saturation is calculated. "Dubinin"
            uses Dubinin's approximation. "Constant" assumes a constant value
            for the fugacity at saturation. The default is "Dubinin".

        Returns
        -------
        DAModel
            A DAModel object which can calculate excess and absolute adsorption
            at various conditions as well as the thermophysical properties of
            the adsorbed phase.

        """
        if (rhoa is None) and rhoa_mode == "Constant":
            stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            rhoa = stored_fluid.backend.rhomolar()
        if (va is None) and va_mode == "Constant":
            va_mode = "Excess"
        self.sorbent = sorbent
        self.stored_fluid = stored_fluid
        self.w0 = w0
        self.f0 = f0
        self.eps = eps
        self.m = m
        self.va = va
        self.rhoa = rhoa
        self.rhoa_mode = rhoa_mode
        self.va_mode = va_mode
        self.f0_mode = f0_mode
        self.k = k

    def f0_calc(self, T: float) -> float:
        """Calculate the fugacity at saturation (Pa) at a given temperature.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Fugacity at saturation (Pa).

        """
        if self.f0_mode == "Constant":
            f0 = self.f0
        if self.f0_mode == "Dubinin":
            pc = self.stored_fluid.backend.p_critical()
            Tc = self.stored_fluid.backend.T_critical()
            if T < Tc:
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            else:
                p0 = ((T/Tc)**self.k) * pc
                self.stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
        return f0

    def rhoa_calc(self, T: float) -> float:
        """Calculate the density of the adsorbed phase at a given temperature.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            The density of the adsorbed phase (mol/m^3).

        """
        if self.rhoa_mode == "Constant":
            rhoa = self.rhoa
        if self.rhoa_mode == "Ozawa":
            self.stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            Tb = self.stored_fluid.backend.T()
            vb = 1/self.stored_fluid.backend.rhomolar()
            ads_specific_volume = vb * np.exp((T-Tb)/T)
            rhoa = 1/ads_specific_volume
        return rhoa

    def v_ads(self, p: float, T: float) -> float:
        """Calculate the volume of the adsorbed phase (m^3/kg).

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Volume of the adsorbed phase (m^3/kg).

        """
        if self.va_mode == "Excess":
            return 0
        if self.va_mode == "Constant":
            return self.va
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)
        f0 = self.f0_calc(T)
        vfilled = self.w0 * np.exp(-((sp.constants.R * T /
                                     (self.eps))**self.m)
                                   * ((np.log(f0/fug))**self.m))
        if self.va_mode == "Vary":
            return vfilled

    def n_absolute(self, p: float, T: float) -> float:
        """Calculate the absolute adsorbed amount at a given condition.

        Parameters
        ----------
        p : float
            Pressure(Pa).
        T : float
            Temperature(K).

        Returns
        -------
        float
            Absolute adsorbed amount (mol/kg).

        """
        rhoa = self.rhoa_calc(T)
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)
        f0 = self.f0_calc(T)
        return rhoa * self.w0 * np.exp(-((sp.constants.R * T /
                                         (self.eps))**self.m)
                                       * ((np.log(f0/fug))**self.m))

    def n_excess(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the excess adsorbed amount at a given condition.

        Parameters
        ----------
        p : float
            Pressure (Pa)
        T : float
            Temperature (K)
        q : float, optional
            The vapor quality of the bulk adsorbate. Can vary between 0 and 1.
            The default is 1.

        Returns
        -------
        float
            Excess adsorbed amount (mol/kg).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, q, T)
        rhomolar = fluid.rhomolar()
        return self.n_absolute(p, T) - rhomolar * self.v_ads(p, T)

    @classmethod
    def from_ExcessIsotherms(cls,
                             ExcessIsotherms: List[ExcessIsotherm],
                             stored_fluid: StoredFluid = None,
                             sorbent: str = None,
                             w0guess: float = 0.001,
                             f0guess: float = 1470E6,
                             epsguess: float = 3000,
                             vaguess: float = 0.001,
                             rhoaguess: float = None,
                             mguess: float = 2.0,
                             kguess: float = 2.0,
                             rhoa_mode: str = "Fit",
                             f0_mode: str = "Fit",
                             m_mode: str = "Fit",
                             k_mode: str = "Fit",
                             va_mode: str = "Excess",
                             pore_volume: float = 0.003) -> "DAModel":
        """Fit the DA model to a list of ExcessIsotherm data.

        Parameters
        ----------
        ExcessIsotherms : List[ExcessIsotherm]
            A list containing ExcessIsotherm objects which contain measurement
            data at various temperatures.

        stored_fluid : StoredFluid, optional
            Object for calculating the properties of the adsorbate. The default
            is None. If None, the StoredFluid object inside of one of the
            ExcessIsotherm objects passed will be used.

        sorbent : str, optional
            Name of sorbent material. The default is None. If None, name will
            be taken from one of the ExcessIsotherm objects passed.

        w0guess : float, optional
            The initial guess for the adsorbed phase volume at saturation
            (m^3/kg). The default is 0.001.

        f0guess : float, optional
            The initial guess for the fugacity at saturation (Pa). The default
            is 1470E6.

        epsguess : float, optional
            The initial guess for the characteristic energy of adsorption
            (J/mol). The default is 3000.

        vaguess : float, optional
            The initial guess for the volume of the adsorbed phase (m^3/kg).
            The default is 0.001.

        rhoaguess : float, optional
            The initial guess for the adsorbed phase density (mol/m^3).
            The default is None. If None, it will be taken as the liquid
            density at 1 bar.

        mguess : float, optional
            The initial guess for the heterogeneity parameter of the
            Dubinin-Astakhov equation. The default is 2.0.

        kguess : float, optional
            The initial guess for the heterogeneity parameter of Dubinin's
            approximation method for saturation fugacity. The default is 2.0.

        rhoa_mode : str, optional
            Determines how the density of the adsorbed phase (rhoa) is
            calculated. If "Fit", rhoa is a constant to be fitted
            statistically. If "Ozawa", Ozawa's approximation is used to
            calculate rhoa and rhoa is not a fitting parameter. If "Constant",
            the user supplied value for rhoaguess is taken as the density.
            The default is "Fit".

        f0_mode : str, optional
            Determines how the fugacity at saturation (f0) is calculated. If
            "Fit" then f0 is a constant to be statistically fitted to the data.
            If "Dubinin" then Dubinin's approximation is used. If "Constant"
            then the user supplied value for f0guess is used. The default is
            "Fit".

        m_mode : str, optional
            Determines whether the heterogeneity parameter of the Dubinin-
            Astakhov equation is taken as a user-supplied constant (if
            "Constant") or a fitted parameter (if "Fit"). The default is "Fit".

        k_mode : str, optional
            Determines whether the heterogeneity parameter of Dubinin's
            approximation for the fugacity above the critical temperature is
            taken as a user-supplied constant value (if "Constant") or as a
            statistically fitted parameter (if "Fit"). The default is "Fit".

        va_mode : str, optional
            Determines how the volume of the adsorbed phase is calculated. If
            "Fit", the value is a statistically fitted constant. If "Constant",
            the value is the user defined value vaguess. If "Vary", the value
            varies w.r.t. pressure according to the micropore filling
            mechanism posited by the Dubinin-Astakhov model. The default is
            "Excess".

        pore_volume : float, optional
            The experimentally measured pore volume of the sorbent material
            (m^3/kg). It serves as the maximum possible physical value for the
            parameters w0 and va. The default is 0.003.

        Returns
        -------
        DAModel
            A DAModel object which can calculate excess and absolute adsorption
            at various conditions as well as the thermophysical properties of
            the adsorbed phase.

        """
        # Make a deepcopy of the ExcessIsotherms
        excess_isotherms = deepcopy(ExcessIsotherms)
        # If some defaults are not supplied, take values from ExcessIsotherms
        if sorbent is None:
            sorbent = excess_isotherms[0].sorbent
        if stored_fluid is None:
            stored_fluid = StoredFluid(
                fluid_name=excess_isotherms[0].adsorbate, EOS="HEOS")
        if rhoaguess is None and \
                (rhoa_mode == "Constant" or rhoa_mode == "Fit"):
            stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            rhoaguess = stored_fluid.backend.rhomolar()

        # Switcher functions depending on whether or not a variable is to be
        # fitted.

        def rhoa_switch(paramsvar, p, T, stored_fluid):
            if rhoa_mode == "Fit":
                return paramsvar["rhoa"]
            if rhoa_mode == "Constant":
                return rhoaguess
            elif rhoa_mode == "Ozawa":
                stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
                Tb = stored_fluid.backend.T()
                vb = 1/stored_fluid.backend.rhomolar()
                ads_specific_volume = vb * np.exp((T-Tb)/T)
                return 1/ads_specific_volume

        def m_switch(paramsvar):
            if m_mode == "Constant":
                return mguess
            elif m_mode == "Fit":
                return paramsvar["m"]

        def k_switch(paramsvar):
            if k_mode == "Constant":
                return kguess
            elif k_mode == "Fit" and f0_mode == "Dubinin":
                return paramsvar["k"]
            else:
                return kguess

        def f0_switch(paramsvar, T, stored_fluid, k):
            if f0_mode == "Fit":
                return paramsvar["f0"]
            if f0_mode == "Constant":
                return f0guess
            if f0_mode == "Dubinin":
                pc = stored_fluid.backend.p_critical()
                Tc = stored_fluid.backend.T_critical()
                if T < Tc:
                    stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                else:
                    p0 = ((T/Tc)**k) * pc
                    stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                return f0

        def va_switch(paramsvar, vfill):
            if va_mode == "Fit":
                return paramsvar["va"]
            if va_mode == "Excess":
                return 0
            if va_mode == "Constant":
                return vaguess
            if va_mode == "Vary":
                return vfill
        # Combine data from multiple isotherms into a single list.
        loading_combined = []
        temperature_combined = []
        pressure_combined = []
        for i, isotherm in enumerate(excess_isotherms):
            pressure_data = isotherm.pressure
            loading_data = isotherm.loading
            temperature = isotherm.temperature
            loading_combined = np.append(loading_combined, loading_data)
            temperature_combined = np.append(temperature_combined,
                                             np.repeat(temperature,
                                                       len(pressure_data)))
            pressure_combined = np.append(pressure_combined, pressure_data)
        # Set up parameters to be fitted.
        params = lmfit.Parameters()
        params.add("w0", w0guess, True, min=0, max=pore_volume)
        params.add("eps", epsguess, True, min=300, max=80000)
        if f0_mode == "Fit":
            params.add("f0", f0guess, True, min=1E5)
        if rhoa_mode == "Fit":
            params.add("rhoa", rhoaguess, min=0)
        if m_mode == "Fit":
            params.add("m", mguess, min=1, max=6)
        if k_mode == "Fit" and f0_mode == "Dubinin":
            params.add("k", kguess, min=0, max=6)
        if va_mode == "Fit":
            params.add("va", vaguess, min=0, max=pore_volume)

        # Define the isotherm model and the loss function.

        def n_excess(p, T, params, stored_fluid):
            phase = stored_fluid.determine_phase(p, T)
            if phase != "Saturated":
                stored_fluid.backend.update(CP.PT_INPUTS, p, T)
            else:
                stored_fluid.backend.update(CP.QT_INPUTS, 1, T)
            fug = stored_fluid.backend.fugacity(0)
            rhof = stored_fluid.backend.rhomolar()
            k = k_switch(params)
            f0 = f0_switch(params, T, stored_fluid, k)
            m = m_switch(params)
            vads = params["w0"] * \
                np.exp(-((sp.constants.R * T /
                          (params["eps"]))**m) * ((np.log(f0/fug))**m))
            rhoa = rhoa_switch(params, p, T, stored_fluid)
            va = va_switch(params, vads)
            return vads * rhoa - va * rhof

        def fit_penalty(params, dataP, dataAd, dataT, stored_fluid):
            value = params.valuesdict()
            difference = []
            for i in range(0, len(dataP)):
                difference.append(n_excess(dataP[i], dataT[i], value,
                                           stored_fluid) - dataAd[i])
            return difference

        fitting = lmfit.minimize(fit_penalty, params,
                                 args=(pressure_combined,
                                       loading_combined,
                                       temperature_combined,
                                       stored_fluid))

        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()

        # For the "Fit" results, it means the mode in the actual model object
        # should be constant.
        # Also, the paramsdict indices would not exist unless the "Fit" mode
        # is chosen, so need to add conditionals.
        f0_res = paramsdict["f0"] if f0_mode == "Fit" else f0guess
        k_res = paramsdict["k"] if k_mode == "Fit" and\
            f0_mode == "Dubinin" else kguess
        m_res = paramsdict["m"] if m_mode == "Fit" else mguess
        rhoa_res = paramsdict["rhoa"] if rhoa_mode == "Fit" else rhoaguess
        va_res = paramsdict["va"] if va_mode == "Fit" else vaguess
        vamode = "Constant" if va_mode == "Fit" else va_mode
        f0mode = "Constant" if f0_mode == "Fit" else f0_mode
        rhoamode = "Constant" if rhoa_mode == "Fit" else rhoa_mode

        return cls(sorbent=sorbent,
                   stored_fluid=stored_fluid,
                   w0=paramsdict["w0"],
                   f0=f0_res,
                   eps=paramsdict["eps"],
                   m=m_res,
                   k=k_res,
                   rhoa=rhoa_res,
                   va=va_res,
                   rhoa_mode=rhoamode,
                   va_mode=vamode,
                   f0_mode=f0mode)


class MDAModel(ModelIsotherm):
    """A class for the Modified Dubinin-Astakhov model for adsorption.

    A key modification compared to the DA model is the use of the enthalpic and
    entropic factors to calculate the adsorption energy as a function of
    temperature instead of treating it as a constant.
    """

    def __init__(self,
                 sorbent: str,
                 stored_fluid: StoredFluid,
                 nmax: float,
                 f0: float,
                 alpha: float,
                 beta: float,
                 va: float,
                 m: float = 2,
                 k: float = 2,
                 va_mode: str = "Constant",
                 f0_mode: str = "Constant") -> "MDAModel":
        """Initialize the MDAModel class.

        Parameters
        ----------
        sorbent : str
            Name of the sorbent material.

        stored_fluid : StoredFluid
            Object to calculate the thermophysical properties of the adsorbate.

        nmax : float
            Maximum adsorbed amount (mol/kg) at saturation.

        f0 : float
            Fugacity at saturation (Pa).

        alpha : float
            The empirical enthalpic factor for determining the characteristic
            energy of adsorption.

        beta : float
            The empirical entropic factor for determining the characteristic
            energy of adsorption.

        va : float
            The volume of the adsorbed phase (m^3/kg).

        m : float, optional
            The empirical heterogeneity parameter for the Dubinin-Astakhov
            model. The default is 2.

        k : float, optional
            The empirical heterogeneity parameter for Dubinin's approximation
            of the saturation fugacity above critical temperatures. The default
            is 2.

        va_mode : str, optional
            Determines how the adsorbed phase density is calculated. "Ozawa"
            uses Ozawa's approximation to calculate the adsorbed phase density.
            "Constant" assumes a constant adsorbed phase volume. The default is
            "Constant".

        f0_mode : str, optional
            Determines how the fugacity at saturation is calculated. "Dubinin"
            uses Dubinin's approximation. "Constant" assumes a constant value
            for the fugacity at saturation. The default is "Constant".

        Returns
        -------
        MDAModel
            An MDAModel object. It can calculate the excess and absolute
            adsorbed amounts at various pressures and temperatures, and it can
            provide thermophysical properties of the adsorbed phase.

        """
        self.sorbent = sorbent
        self.stored_fluid = stored_fluid
        self.f0 = f0
        self.alpha = alpha
        self.beta = beta
        self.va = va
        self.nmax = nmax
        self.m = m
        self.k = k
        self.va_mode = va_mode
        self.f0_mode = f0_mode

    def n_absolute(self, p: float, T: float) -> float:
        """Calculate the absolute adsorbed amount at given conditions.

        Parameters
        ----------
        p : float
            Pressure (Pa)

        T : float
            Temperature (K)

        Returns
        -------
        float
            Absolute adsorbed amount (mol/kg).

        """
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            self.stored_fluid.backend.update(CP.PT_INPUTS, p, T)
        else:
            self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
        fug = self.stored_fluid.backend.fugacity(0)

        if self.f0_mode == "Constant":
            f0 = self.f0
        if self.f0_mode == "Dubinin":
            pc = self.stored_fluid.backend.p_critical()
            Tc = self.stored_fluid.backend.T_critical()
            if T < Tc:
                self.stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                f0 = self.stored_fluid.backend.fugacity(0)
            else:
                p0 = ((T/Tc)**self.k) * pc
                self.stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                f0 = self.stored_fluid.backend.fugacity(0)

        return self.nmax * np.exp(-((sp.constants.R * T /
                                     (self.alpha + self.beta * T))**self.m)
                                  * ((np.log(f0/fug))**self.m))

    def v_ads(self, p: float, T: float) -> float:
        """Calculate the adsorbed phase volume at the given condtions.

        Parameters
        ----------
        p : float
            Pressure (Pa).

        T : float
            Temperature (K).

        Returns
        -------
        float
            Adsorbed phase volume (m^3/kg)

        """
        if self.va_mode == "Constant":
            return self.va
        if self.va_mode == "Ozawa":
            self.stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
            Tb = self.stored_fluid.backend.T()
            vb = 1/self.stored_fluid.backend.rhomolar()
            ads_specific_volume = vb * np.exp((T-Tb)/T)
            ads_density = 1/ads_specific_volume
            na = self.n_absolute(p, T)
            return na / ads_density

    def n_excess(self, p: float, T: float, q: float = 1) -> float:
        """Calculate the excess adsorbed amount at the given conditions.

        Parameters
        ----------
        p : float
            Pressure (Pa)

        T : float
            Temperature (K).

        q : float, optional
            Vapor quality of the bulk fluid. Can vary between 0 and 1. The
            default is 1.

        Returns
        -------
        float
            Excess adsorbed amount (mol/kg).

        """
        fluid = self.stored_fluid.backend
        phase = self.stored_fluid.determine_phase(p, T)
        if phase != "Saturated":
            fluid.update(CP.PT_INPUTS, p, T)
        else:
            fluid.update(CP.QT_INPUTS, q, T)
        rhomolar = fluid.rhomolar()
        return self.n_absolute(p, T) - rhomolar * self.v_ads(p, T)

    @classmethod
    def from_ExcessIsotherms(cls,
                             ExcessIsotherms: List[ExcessIsotherm],
                             stored_fluid: StoredFluid = None,
                             sorbent: str = None,
                             nmaxguess: float = 71.6,
                             f0guess: float = 1470E6,
                             alphaguess: float = 3080,
                             betaguess: float = 18.9,
                             vaguess: float = 0.00143,
                             mguess: float = 2.0,
                             kguess: float = 2.0,
                             va_mode: str = "Fit",
                             f0_mode: str = "Fit",
                             m_mode: str = "Fit",
                             k_mode: str = "Fit",
                             beta_mode: str = "Fit",
                             pore_volume: float = 0.003) -> "MDAModel":
        """Fit the MDA model from a list of excess adsorption data.

        Parameters
        ----------
        ExcessIsotherms : List[ExcessIsotherm]
            A list of ExcessIsotherm objects which contain measurement
            data at various temperatures.

        stored_fluid : StoredFluid, optional
            Object for calculating the properties of the adsorbate. The default
            is None. If None, the StoredFluid object inside of one of the
            ExcessIsotherm objects passed will be used.


        sorbent : str, optional
            Name of sorbent material. The default is None. If None, name will
            be taken from one of the ExcessIsotherm objects passed.

        nmaxguess : float, optional
            The initial guess for the maximum adsorbed amount (mol/kg). The
            default is 71.6.

        f0guess : float, optional
            The initial guess for the fugacity at saturation (Pa). The default
            is 1470E6.

        alphaguess : float, optional
            The initial guess for the enthalpy factor determining the
            characteristic energy of adsorption. The default is 3080.

        betaguess : float, optional
            The initial guess for the entropy factor determining the
            characteristic energy of adsorption. The default is 18.9.

        vaguess : float, optional
            Initial guess for the adsorbed phase volume (m^3/kg). The default
            is 0.00143.

        mguess : float, optional
            The initial guess for the heterogeneity parameter of the
            Dubinin-Astakhov equation. The default is 2.0.

        kguess : float, optional
            The initial guess for the heterogeneity parameter of Dubinin's
            approximation method for saturation fugacity. The default is 2.0.


        va_mode : str, optional
            Determines how the volume of the adsorbed phase (va) is
            calculated. If "Fit", va is a constant to be fitted
            statistically. If "Ozawa", Ozawa's approximation is used to
            calculate va and va is not a fitting parameter. If "Constant",
            the user supplied value for vaguess is taken as the volume.
            The default is "Fit".

        f0_mode : str, optional
            Determines how the fugacity at saturation (f0) is calculated. If
            "Fit" then f0 is a constant to be statistically fitted to the data.
            If "Dubinin" then Dubinin's approximation is used. If "Constant"
            then the user supplied value for f0guess is used. The default is
            "Fit".

        m_mode : str, optional
            Determines whether the heterogeneity parameter of the Dubinin-
            Astakhov equation is taken as a user-supplied constant (if
            "Constant") or a fitted parameter (if "Fit"). The default is "Fit".

        k_mode : str, optional
            Determines whether the heterogeneity parameter of Dubinin's
            approximation for the fugacity above the critical temperature is
            taken as a user-supplied constant value (if "Constant") or as a
            statistically fitted parameter (if "Fit"). The default is "Fit".

        beta_mode : str, optional
            Determines whether the entropic factor determining the
            characteristic energy of adsorption is taken as a user-supplied
            constant (if "Constant") or as a fitted parameter (if "Fit"). The
            default is "Fit".

        pore_volume : float, optional
            The experimentally measured pore volume of the sorbent material
            (m^3/kg). It serves as the maximum possible physical value for the
            parameters w0 and va. The default is 0.003.

        Returns
        -------
        MDAModel
            An MDAModel object. It can calculate the excess and absolute
            adsorbed amounts at various pressures and temperatures, and it can
            provide thermophysical properties of the adsorbed phase.

        """
        excess_isotherms = deepcopy(ExcessIsotherms)

        # Take values from excess isotherm if not supplied in argument
        if sorbent is None:
            sorbent = excess_isotherms[0].sorbent
        if stored_fluid is None:
            stored_fluid = StoredFluid(
                fluid_name=excess_isotherms[0].adsorbate, EOS="HEOS")

        loading_combined = []
        temperature_combined = []
        pressure_combined = []

        def va_switch(paramsvar, p, T, stored_fluid, nabs):
            if va_mode == "Fit":
                return paramsvar["va"]
            elif va_mode == "Constant":
                return vaguess
            elif va_mode == "Ozawa":
                stored_fluid.backend.update(CP.PQ_INPUTS, 1E5, 0)
                Tb = stored_fluid.backend.T()
                vb = 1/stored_fluid.backend.rhomolar()
                ads_specific_volume = vb * np.exp((T-Tb)/T)
                ads_density = 1/ads_specific_volume
                return nabs / ads_density

        def m_switch(paramsvar):
            if m_mode == "Constant":
                return mguess
            elif m_mode == "Fit":
                return paramsvar["m"]

        def k_switch(paramsvar):
            if k_mode == "Constant":
                return kguess
            elif k_mode == "Fit" and f0_mode == "Dubinin":
                return paramsvar["k"]
            else:
                return kguess

        def f0_switch(paramsvar, T, stored_fluid, k):
            if f0_mode == "Fit":
                return paramsvar["f0"]
            elif f0_mode == "Constant":
                return f0guess
            elif f0_mode == "Dubinin":
                pc = stored_fluid.backend.p_critical()
                Tc = stored_fluid.backend.T_critical()
                if T < Tc:
                    stored_fluid.backend.update(CP.QT_INPUTS, 0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                else:
                    p0 = ((T/Tc)**k) * pc
                    stored_fluid.backend.update(CP.PT_INPUTS, p0, T)
                    f0 = stored_fluid.backend.fugacity(0)
                return f0

        min_nmax = 1
        for i, isotherm in enumerate(excess_isotherms):
            pressure_data = isotherm.pressure
            loading_data = isotherm.loading
            temperature = isotherm.temperature
            loading_combined = np.append(loading_combined, loading_data)
            temperature_combined = np.append(temperature_combined,
                                             np.repeat(temperature,
                                                       len(pressure_data)))
            pressure_combined = np.append(pressure_combined, pressure_data)
            min_nmax = max(loading_data) if max(loading_data) > min_nmax else \
                min_nmax
        params = lmfit.Parameters()
        params.add("nmax", nmaxguess, True, min_nmax, 300)
        if f0_mode == "Fit":
            params.add("f0", f0guess, True, 1E5)
        params.add("alpha", alphaguess, True, 500, 80000)
        params.add("beta", betaguess, beta_mode == "Fit", 0, 100)
        if va_mode == "Fit":
            params.add("va", vaguess, min=0, max=pore_volume)
        if m_mode == "Fit":
            params.add("m", mguess, min=1, max=6)
        if k_mode == "Fit" and f0_mode == "Dubinin":
            params.add("k", kguess, min=0, max=6)

        def n_excess(p, T, params, stored_fluid):
            phase = stored_fluid.determine_phase(p, T)
            if phase != "Saturated":
                stored_fluid.backend.update(CP.PT_INPUTS, p, T)
            else:
                stored_fluid.backend.update(CP.QT_INPUTS, 1, T)
            fug = stored_fluid.backend.fugacity(0)
            rhof = stored_fluid.backend.rhomolar()
            k = k_switch(params)
            f0 = f0_switch(params, T, stored_fluid, k)
            m = m_switch(params)
            nabs = params["nmax"] * \
                np.exp(-((sp.constants.R * T /
                          (params["alpha"] + params["beta"] * T))**m)
                       * ((np.log(f0/fug))**m))
            va = va_switch(params, p, T, stored_fluid, nabs)
            return nabs - rhof * va

        def fit_penalty(params, dataP, dataAd, dataT, stored_fluid):
            value = params.valuesdict()
            difference = []
            for i in range(0, len(dataP)):
                difference.append(n_excess(dataP[i], dataT[i], value,
                                           stored_fluid) - dataAd[i])
            return difference

        fitting = lmfit.minimize(fit_penalty, params,
                                 args=(pressure_combined,
                                       loading_combined,
                                       temperature_combined,
                                       stored_fluid))

        print(lmfit.fit_report(fitting))
        paramsdict = fitting.params.valuesdict()

        f0_res = paramsdict["f0"] if f0_mode == "Fit" else f0guess
        va_res = paramsdict["va"] if va_mode == "Fit" else vaguess
        m_res = paramsdict["m"] if m_mode == "Fit" else mguess
        k_res = paramsdict["k"] if k_mode == "Fit" \
            and f0_mode == "Dubinin" else kguess
        vamode = "Constant" if va_mode == "Fit" else va_mode
        f0mode = "Constant" if f0_mode == "Fit" else f0_mode

        return cls(sorbent=sorbent,
                   stored_fluid=stored_fluid,
                   nmax=paramsdict["nmax"],
                   f0=f0_res,
                   alpha=paramsdict["alpha"],
                   beta=paramsdict["beta"],
                   va=va_res,
                   m=m_res,
                   k=k_res,
                   va_mode=vamode,
                   f0_mode=f0mode)


class SorbentMaterial:
    """Class containing the various properties of a sorbent material.

    Attributes
    ----------
    mass : float
        Mass of sorbent (kg).

    skeletal_density : float
        Skeletal density of the sorbent (kg/m^3).

    bulk_density : float
        Tapped/compacted bulk density of the sorbent (kg/m^3).

    specific_surface_area : float
        Specific surface area of the sorbent (m^2/g).

    model_isotherm : ModelIsotherm
        Model of fluid adsorption on the sorbent.

    molar_mass : float, optional
        Molar mass of the sorbent material. The default is 12.01E-3 which
        corresponds to carbon materials.

    Debye_temperature : float, optional
        The Debye temperature determining the specific heat of the sorbent
        at various temperatures. The default is 1500, the value for carbon.

    """

    def __init__(self,
                 mass: float,
                 skeletal_density: float,
                 bulk_density: float,
                 specific_surface_area: float,
                 model_isotherm: ModelIsotherm,
                 molar_mass: float = 12.01E-3,
                 Debye_temperature: float = 1500) -> "SorbentMaterial":
        """Initialize the SorbentMaterial class.

        Parameters
        ----------
        mass : float
            Mass of sorbent (kg).

        skeletal_density : float
            Skeletal density of the sorbent (kg/m^3).

        bulk_density : float
            Tapped/compacted bulk density of the sorbent (kg/m^3).

        specific_surface_area : float
            Specific surface area of the sorbent (m^2/g).

        model_isotherm : ModelIsotherm
            Model of fluid adsorption on the sorbent.

        molar_mass : float, optional
            Molar mass of the sorbent material. The default is 12.01E-3 which
            corresponds to carbon materials.

        Debye_temperature : float, optional
            The Debye temperature determining the specific heat of the sorbent
            at various temperatures. The default is 1500, the value for carbon.

        Returns
        -------
        SorbentMaterial
            Class containing the properties of a sorbent material.

        """
        self.mass = mass
        self.skeletal_density = skeletal_density
        self.bulk_density = bulk_density
        self.model_isotherm = model_isotherm
        self.specific_surface_area = specific_surface_area
        self.molar_mass = molar_mass
        self.Debye_temperature = Debye_temperature
