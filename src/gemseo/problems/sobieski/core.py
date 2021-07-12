# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation
#               and/or initial documentation
#        :author: Sobieski, Agte, and Sandusky
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Damien Guenot
#        :author: Francois Gallard
# From NASA/TM-1998-208715
# Bi-Level Integrated System Synthesis (BLISS)
# Sobieski, Agte, and Sandusky
"""
SSBJ core computations
**********************
"""
from __future__ import division, unicode_literals

import cmath
import logging
import math
import random
from os.path import dirname, join
from random import uniform

from numpy import array, complex128, concatenate, float64, ones
from six import string_types

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.base import SobieskiBase
from gemseo.problems.sobieski.core_aerodynamics import SobieskiAerodynamics
from gemseo.problems.sobieski.core_mission import SobieskiMission
from gemseo.problems.sobieski.core_propulsion import SobieskiPropulsion
from gemseo.problems.sobieski.core_structure import SobieskiStructure

LOGGER = logging.getLogger(__name__)
DIRNAME = dirname(__file__)

DEG_TO_RAD = math.pi / 180.0


class SobieskiProblem(object):

    """Class defining Sobieski problem and related method to the problem such as
    disciplines computation, constraints, reference optimum."""

    CONTRAINTS_NAMES_INEQUALITY = (
        "c_Stress_x1",
        "c_Stress_x2",
        "c_Stress_x3",
        "c_Stress_x4",
        "c_Stress_x5",
        "c_Twist_upper",
        "c_Twist_lower",
        "c_Pgrad",
        "c_ESF_upper",
        "c_ESF_lower",
        "c_Throttle",
        "c_Temperature",
    )
    CONTRAINTS_NAMES = (
        "Stress_x1",
        "Stress_x2",
        "Stress_x3",
        "Stress_x4",
        "Stress_x5",
        "Twist",
        "Pgrad",
        "ESF",
        "Temperature",
        "Throttle",
    )

    DV_NAMES_NORMALIZED = (
        "x_TaperRatio",
        "x_SectionalArea",
        "x_Cf",
        "x_Throttle_setting",
        "x_eta",
        "x_h",
        "x_Mach",
        "x_AR",
        "x_Phi",
        "x_sref",
    )

    DV_NAMES = (
        "TaperRatio",
        "SectionalArea",
        "Cf",
        "Throttle_setting",
        "eta",
        "h",
        "Mach",
        "AR",
        "Phi",
        "sref",
    )

    COUPLING_VARIABLES_NAMES = (
        "Total weight",
        "Fuel weight",
        "Wing twist",
        "Lift",
        "Drag",
        "Lift/Drag",
        "SFC",
        "Engine weight",
    )

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"

    STRESS_LIMIT = SobieskiStructure.STRESS_LIMIT
    TWIST_UPPER_LIMIT = SobieskiStructure.TWIST_UPPER_LIMIT
    TWIST_LOWER_LIMIT = SobieskiStructure.TWIST_LOWER_LIMIT
    PRESSURE_GRADIENT_LIMIT = SobieskiAerodynamics.PRESSURE_GRADIENT_LIMIT
    ESF_UPPER_LIMIT = SobieskiPropulsion.ESF_UPPER_LIMIT
    ESF_LOWER_LIMIT = SobieskiPropulsion.ESF_LOWER_LIMIT
    TEMPERATURE_LIMIT = SobieskiPropulsion.TEMPERATURE_LIMIT

    def __init__(self, dtype=DTYPE_DOUBLE):
        """Constructor.

        :param dtype: data type of problem, either "float64" or "complex128".
        :type dtype: str
        """
        if dtype == self.DTYPE_COMPLEX:
            self.dtype = complex128
            self.math = cmath
        elif dtype == self.DTYPE_DOUBLE:
            self.math = math
            self.dtype = float64
        else:
            raise ValueError("Unknown dtype : " + str(dtype))

        self.base = SobieskiBase(dtype=self.dtype)

        self.sobieski_structure = SobieskiStructure(self.base)
        self.sobieski_aerodynamics = SobieskiAerodynamics(self.base)
        self.sobieski_propulsion = SobieskiPropulsion(self.base)
        self.sobieski_mission = SobieskiMission(self.base)

        self.constants = self.base.default_constants()
        self.i_0 = self.base.get_default_x0()
        (
            self.x_initial,
            self.tc_initial,
            self.half_span_initial,
            self.aero_center_initial,
            self.cf_initial,
            self.mach_initial,
            self.h_initial,
            self.throttle_initial,
            self.lift_initial,
            self.twist_initial,
            self.esf_initial,
        ) = self.base.get_initial_values()
        self.u_b, self.l_b = self.base.get_sobieski_bounds()

    def get_default_x0(self):
        """Function that returns a default initial value for design variables.

        :returns: i_0 , initial design variables
        :rtype: ndarray
        """
        # :warning: DO NOT CHANGE VALUE: THERE ARE
        #    USED FOR POLYNOMIAL APPROXIMATION
        return self.base.get_default_x0()

    def default_constants(self):
        """Definition of constants vector C for Sobieski problem.

        :returns: constant vector
        :rtype: ndarray
        """
        return self.base.default_constants()

    def get_bounds_by_name(self, variables_names):
        """Class method that return bounds of design variables and coupling variables.

        :param variables_names: name of variable
        :returns: lower bound and upper bound
        """
        return self.base.get_bounds_by_name(variables_names)

    def __set_indata(self, input_vect, names):
        """Returns a set of default inputs for the different disciplines.

        :param input_vect: design vector used to fill the dictionary
        :param names: specific data names, if None, returns all inputs
        :type names: str or list(str)
        :returns: indata , a dict of input data
        :rtype: dict
        """

        indata = {
            "x_shared": input_vect[4:10],
            "y_12": array([50606.9742, 0.95], dtype=self.dtype),
            "y_14": array((50606.9741711, 7306.20262124), dtype=self.dtype),
            "y_21": array([50606.9741711], dtype=self.dtype),
            "y_23": array([12562.01206488], dtype=self.dtype),
            "y_24": array([4.15006276], dtype=self.dtype),
            "y_31": array([6354.32430691], dtype=self.dtype),
            "y_32": array([0.50279625], dtype=self.dtype),
            "y_34": array([1.10754577], dtype=self.dtype),
            "y_1": ones(3, dtype=self.dtype),
            "y_2": ones(3, dtype=self.dtype),
            "y_3": ones(3, dtype=self.dtype),
            "g_1": ones(6, dtype=self.dtype),
            "g_2": ones(1, dtype=self.dtype),
            "g_3": ones(3, dtype=self.dtype),
            "x_1": input_vect[:2],
            "x_2": array([input_vect[2]], dtype=self.dtype),
            "x_3": array([input_vect[3]], dtype=self.dtype),
        }
        if isinstance(names, string_types):
            names = [names]
        if names is not None:
            return {k: indata[k] for k in names if k in indata}
        return indata

    def get_default_inputs(self, names=None):
        """Returns a set of default inputs for the different disciplines.

        :param names: specific data names, if None, returns all inputs
            (Default value = None)
        :type names: str or list(str)
        :returns: indata , a dict of input data
        :rtype: dict
        """
        input_vect = self.get_default_x0()
        return self.__set_indata(input_vect, names)

    def get_default_inputs_feasible(self, names=None):
        """Returns a set of default inputs for the different disciplines.

        :param names: specific data names, if None, returns all inputs
            (Default value = None)
        :type names: str or list(str)
        :returns: indata , a dict of input data
        :rtype: dict
        """
        input_vect = self.get_x0_feasible()
        return self.__set_indata(input_vect, names)

    def get_default_inputs_equilibrium(self, names=None):
        """Returns a set of default inputs, where coupling variables are at the
        equilibrium (MDA) for X0.

        :param names: specific data names, if None, returns all inputs
            (Default value = None)
        :type names: str or list(str)
        :returns: indata , a dict of input data
        :rtype: dict
        """
        input_vect = self.get_default_x0()
        indata = {
            "x_shared": array(input_vect[4:], dtype=self.dtype),
            "y_34": array([1.10754577], dtype=self.dtype),
            "y_32": array([0.50279625], dtype=self.dtype),
            "y_21": array([50606.97417114], dtype=self.dtype),
            "y_31": array([6354.32430691], dtype=self.dtype),
            "y_23": array([12194.26719338], dtype=self.dtype),
            "y_24": array([4.15006276], dtype=self.dtype),
            "g_3": array(
                [-0.99720375, -0.00279625, 0.16206032, -0.02], dtype=self.dtype
            ),
            "g_2": array([-0.04], dtype=self.dtype),
            "g_1": array(
                [0.035, -0.00666667, -0.0275, -0.04, -0.04833333, -0.09, -0.15],
                dtype=self.dtype,
            ),
            "y_11": array([0.15591894], dtype=self.dtype),
            "y_14": array([50606.97417114, 7306.20262124], dtype=self.dtype),
            "y_3": array(
                [1.10754577e00, 6.35432431e03, 5.02796251e-01], dtype=self.dtype
            ),
            "y_12": array([5.06069742e04, 9.5e-01], dtype=self.dtype),
            "y_4": array([535.78844818], dtype=self.dtype),
            "y_2": array(
                [5.06069742e04, 1.21942672e04, 4.15006276e00], dtype=self.dtype
            ),
            "x_1": array(input_vect[:2], dtype=self.dtype),
            "x_2": array([input_vect[2]], dtype=self.dtype),
            "x_3": array([input_vect[3]], dtype=self.dtype),
        }

        if names is not None:
            return {k: indata[k] for k in names}
        return indata

    def get_random_input(self, names=None, seed=None):
        """Get a randomized starting point with specified variables names.

        :param names: specific data names, if None, returns all inputs
            (Default value = None)
        :type names: str or list(str)
        :param seed: the seed for random number generation
            (Default value = None)
        :returns: values of specified design variable name
        :rtype: ndarray
        """
        if seed is not None:
            random.seed(seed)
        upper_bound, lower_bound = self.get_sobieski_bounds()
        indata = {
            "x_shared": array(
                uniform(lower_bound[4:], upper_bound[4:]), dtype=self.dtype
            ),
            "y_34": array([uniform(0.8, 1.2)], dtype=self.dtype),
            "y_21": array((uniform(1.0, 6e4),), dtype=self.dtype),
            "y_31": array((uniform(1.0, 1e4),), dtype=self.dtype),
            "y_23": array([uniform(1.0, 2e4)], dtype=self.dtype),
            "y_24": array([uniform(0.7, 6.0)], dtype=self.dtype),
            "y_14": array([uniform(1.0, 6e4), uniform(1.0, 1e4)], dtype=self.dtype),
            "y_12": array([uniform(1.0, 6e4), uniform(0.6, 1.1)], dtype=self.dtype),
            "y_32": array([uniform(0.6, 1.1)], dtype=self.dtype),
            "x_1": array(uniform(lower_bound[:2], upper_bound[:2]), dtype=self.dtype),
            "x_2": array([uniform(lower_bound[2], upper_bound[2])], dtype=self.dtype),
            "x_3": array([uniform(lower_bound[3], upper_bound[3])], dtype=self.dtype),
        }

        if names is not None:
            return {k: indata[k] for k in names}

        return indata

    def get_x0_feasible(self, names=None):
        """Gets a feasible starting point with specified variables names.

        :param names: specific data names, if None, returns all inputs
            (Default value = None)
        :type names: str or list(str)
        :returns: values of specified design variable name
        :rtype: ndarray
        """
        if isinstance(names, string_types):
            names = [names]
        if names is None:
            names = ["x_1", "x_2", "x_3", "x_shared"]
        opts = {
            "x_1": array([0.14951, 7.5e-01], dtype=self.dtype),
            "x_2": array([7.5e-01], dtype=self.dtype),
            "x_3": array([0.1675], dtype=self.dtype),
            "x_shared": array(
                [6.0e-02, 5.4e04, 1.4e00, 4.4e00, 6.6e01, 1.2e03], dtype=self.dtype
            ),
        }

        return concatenate([opts[zname] for zname in names])

    def get_sobieski_bounds(self):
        """Set the input design bounds and return them as 2 ndarrays.

        :returns: ub,lb: upper and lower bounds
        :rtype: ndarray,ndarray
        """
        return self.base.get_sobieski_bounds()

    def get_sobieski_optimum(self):
        """Optimum by Sobieski with BLISS.

        :returns: array of x optimum x_1, x_2, x_3, x_shared concatenated
        :rtype: ndarray
        """
        return array(
            (0.38757, 0.75, 0.75, 0.15624, 0.06, 60000.0, 1.4, 2.5, 70.0, 1500.0),
            dtype=self.dtype,
        )

    def get_sobieski_optimum_range(self):
        """Return range value by Sobieski with BLISS.

        :returns: optimal range value
        :rtype: ndarray
        """
        return array([3963.98], dtype=self.dtype)

    def get_sobieski_constraints(self, g_1, g_2, g_3, true_cstr=False):
        """Compare the constraints to their limits for Sobieski problem.

        :param g_1: vector of constraints for weight analysis:

            - g_1[0] to g_1[4]: stress on wing
            - g_1[5]: wing twist as constraint

        :type g_1: ndarray
        :param g_2: vector of constraints for aerodynamics analysis:

            - g_2[0]: pressure gradient

        :type g_2: ndarray
        :param g_3: vector of constraints for propulsion analysis:

            - g_3[0]: engine scale factor constraint
            - g_3[1]: engine temperature
            - g_3[2]: throttle setting constraint: must be, at least,
              requested throttle

        :type g_3: ndarray
        :param true_cstr: choice for true value of constraints or
            comparison to bounds (Default value = False)
        :type true_cstr: bool
        :returns: constraints_values or comparison to bounds
        :rtype: ndarray
        """
        if true_cstr:
            constraints_values = concatenate(
                (g_1[0:5], array((g_1[5], g_2[0], g_3[0], g_3[2], g_3[1])))
            )
        else:
            constraints_values = concatenate(
                (
                    g_1[0:5] - self.STRESS_LIMIT,
                    array(
                        (
                            g_1[5] - self.TWIST_UPPER_LIMIT,
                            self.TWIST_LOWER_LIMIT - g_1[5],
                            g_2[0] - self.PRESSURE_GRADIENT_LIMIT,
                            g_3[0] - self.ESF_UPPER_LIMIT,
                            self.ESF_LOWER_LIMIT - g_3[0],
                            g_3[2],
                            g_3[1] - self.TEMPERATURE_LIMIT,
                        )
                    ),
                )
            )

        return constraints_values

    def normalize_inputs(self, input_vector):
        """This function normalizes design variables w.r.t lower and upper bounds of
        these design variables They will be defined in [0,1]

        :param input_vector: real design variables vector
        :type input_vector: ndarray
        :returns: normalized vector of design variables
        :rtype: ndarray
        """
        upper_bound, lower_bound = self.base.get_sobieski_bounds()
        return (input_vector - lower_bound) / (upper_bound - lower_bound)

    def unnormalize_inputs(self, input_vector):
        """This function unnormalizes design variables.

        :param input_vector: normalized design variables vector
        :type input_vector: ndarray
        :returns: real vector of design variables
        :rtype: ndarray
        """
        upper_bound, lower_bound = self.get_sobieski_bounds()
        return input_vector * (upper_bound - lower_bound) + lower_bound

    def blackbox_structure(self, x_shared, y_21, y_31, x_1, true_cstr=False):
        """This function calculates the weight of the aircraft by structure and adds
        them to obtain a total aircraft weight.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_21: lift
        :type y_21: ndarray
        :param y_31: engine weight
        :type y_31: ndarray
        :param x_1: weight design variables:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param true_cstr: Default value = False)
        :returns: g_1,y_1, y_12:

            - g_1 : vector of constraints for weight analysis
            - g_1[0] to g_1[4]: stress on wing
            - g_1[5]: wing twist as constraint
            - y_1: weight analysis outputs
            - y_1[0]: total aircraft weight
            - y_1[1]: fuel weight
            - y_1[2]: wing twist
            - y_12: shared variables used for aero. computations
              (blackbox_aerodynamics)
            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist
            - y_14: shared variables used for range computation
              (blackbox_mission)
            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :rtype: ndarray, ndarray, ndarray, ndarray
        """
        return self.sobieski_structure.blackbox_structure(
            x_shared, y_21, y_31, x_1, true_cstr
        )

    def derive_blackbox_structure(self, x_shared, y_21, y_31, x_1, true_cstr=False):
        """Compute jacobian matrix of structural analysis y_1 is the vector of
        structural outputs and g_1 are the structural constraints.

        - y_1[0]: total aircraft weight
        - y_1[1]: fuel weight
        - y_1[2]: wing twist

        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param x_1: structure design variable vector:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics (lift)
        :type y_21: ndarray
        :param y_31: coupling variable from propulsion (Engine weight)
        :type y_31: ndarray
        :param true_cstr: Default value = False)
        :returns: J : Jacobian matrix
        :rtype: dict(ndarray)
        """

        return self.sobieski_structure.derive_blackbox_structure(
            x_shared, y_21, y_31, x_1, true_cstr=true_cstr
        )

    def blackbox_aerodynamics(self, x_shared, y_12, y_32, x_2, true_cstr=False):
        """This function calculates drag and lift to drag ratio of A/C.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_12: shared variables coming from blackbox_structure:

            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist

        :type y_12: ndarray
        :param y_32: shared variables coming from blackbox_propulsion:

            - y_32[0]: engine scale factor

        :type y_32: ndarray
        :param x_2: aero. design variable:

            - x_2[0]: friction coeff

        :type x_2: ndarray
        :param true_cstr: Default value = False)
        :returns: y_2, y_21, y_23, y_24, g_2:

            - y_2: aero. analysis outputs
            - y_2[0]: lift
            - y_2[1]: drag
            - y_2[2]: lift/drag ratio
            - y_21: shared variable for blackbox_structure (lift)
            - y_23:  shared variable for blackbox_propulsion (drag)
            - y_24: shared variable for blackbox_mission (lift/drag ratio)
            - g_2: aero constraint (pressure gradient)

        :rtype: ndarray, ndarray, ndarray, ndarray, ndarray
        """

        return self.sobieski_aerodynamics.blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2, true_cstr=true_cstr
        )

    def derive_blackbox_aerodynamics(self, x_shared, y_12, y_32, x_2):
        """This function calculates drag and lift to drag ratio of A/C.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_12: shared variables coming from blackbox_structure:

            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist

        :type y_12: ndarray
        :param y_32: shared variables coming from blackbox_propulsion:

            - y_32[0]: engine scale factor

        :type y_32: ndarray
        :param x_2: aero. design variable:

            - x_2[0]: friction coeff

        :type x_2: ndarray
        :returns: J : Jacobian matrix
        :rtype: dict(ndarray)
        """

        return self.sobieski_aerodynamics.derive_blackbox_aerodynamics(
            x_shared, y_12, y_32, x_2
        )

    def blackbox_propulsion(self, x_shared, y_23, x_3, true_cstr=False):
        """This function calculates fuel comsumption, engine weight and engine scale
        factor.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_23: shared variables coming from blackbox_aerodynamics (drag)
        :type y_23: ndarray
        :param x_3: power/propulsion design variable (throttle setting)
        :type x_3: ndarray
        :param true_cstr: analysis returns constraint absolute value or
            relative value to bounds (Default value = False)
        :type true_cstr: bool
        :returns: y_3, y_34, y_31, y_32, g_3:

            - y_3: output variables for propulsion analysis
            - y_3[0]: SFC
            - y_3[1]: engine weight
            - y_3[2]: engine scale factor
            - y_34: shared variable for blackbox_mission (SFC)
            - y_31: shared variable for blackbox_structure (engine weight)
            - y_32: shared variable for blackbox_aerodynamics (ESF)
            - g_3:  propulsion constraints
            - g_3[0]: engine scale factor constraint
            - g_3[1]: engine temperature
            - g_3[2]: throttle setting constraint

        :rtype: ndarray, ndarray, ndarray, ndarray, ndarray
        """
        return self.sobieski_propulsion.blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=true_cstr
        )

    def derive_blackbox_propulsion(self, x_shared, y_23, x_3, true_cstr=False):
        """This function calculates the Jacobian matrix of propulsion.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_23: shared variables coming from blackbox_aerodynamics (drag)
        :type y_23: ndarray
        :param x_3: power/propulsion design variable (throttle setting)
        :type x_3: ndarray
        :param true_cstr: analysis returns constraint absolute value or
            relative value to bounds (Default value = False)
        :type true_cstr: bool
        :returns: J : Jacobian matrix
        :rtype: dict(ndarray)
        """

        return self.sobieski_propulsion.derive_blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=true_cstr
        )

    def blackbox_mission(self, x_shared, y_14, y_24, y_34):
        """THIS SECTION COMPUTES THE A/C RANGE from Breguet's law.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: ndarray
        :param y_24: shared variables coming from
            blackbox_aerodynamics (lift/drag ratio)
        :param y_34: shared variables coming from
            blackbox_propulsion (SFC)
        :type y_34: ndarray
        :returns: y_4: range value
        :rtype: ndarray
        """

        return self.sobieski_mission.blackbox_mission(x_shared, y_14, y_24, y_34)

    def derive_blackbox_mission(self, x_shared, y_14, y_24, y_34):
        """

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: ndarray
        :param y_24: shared variables coming from
            blackbox_aerodynamics (lift/drag ratio)
        :param y_34: shared variables coming from
            blackbox_propulsion (SFC)
        :type y_34: ndarray
        :returns: J : Jacobian matrix
        :rtype: dict(ndarray)
        """
        return self.sobieski_mission.derive_blackbox_mission(x_shared, y_14, y_24, y_34)

    def systemanalysis_gauss_seidel(
        self, design_vector, true_cstr=False, accuracy=1e-3
    ):
        """This subfunction uses Gauss-Seidel iterations on the A/C range optimization
        model to compute behavior variables, given a set of design variables. Black
        boxes WEIGHT, DRAGPOLAR, and POWER are called.

        :param design_vector: design variable vector
        :type design_vector: ndarray
        :param true_cstr: return constraint value
                or compare it to bounds (Default value = False)
        :type true_cstr: bool
        :param accuracy: system resolution accuracy (Default value = 1e-3)
        :type accuracy: float
        :returns: y_1, y_2, y_3, y_4, y_12, y_14, y_21, y_23, y_24,
            y_31, y_32, y_34, g_1, g_2, g_3:

            - y_1: weight analysis outputs
            - y_1[0]: total aircraft weight
            - y_1[1]: fuel weight
            - y_1[2]: wing twist
            - y_2: aero. analysis outputs
            - y_2[0]: lift
            - y_2[1]: drag
            - y_2[2]: lift/drag ratio
            - y_3: output variables for propulsion analysis
            - y_3[0]: SFC
            - y_3[1]: engine weight
            - y_4: range computation output
            - y_12: shared variables from blackbox_structure
              for blackbox_aerodynamics
            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist
            - y_14: shared variables coming from blackbox_structure
              for blackbox_mission
            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight
            - y_21: lift from blackbox_aerodynamics for blackbox_structure
            - y_23: drag from blackbox_aerodynamics for blackbox_propulsion
            - y_24: lift/drag ratio coming from blackbox_aerodynamics
              for blackbox_mission
            - y_31: engine weight coming from blackbox_propulsion
              for blackbox_structure
            - y_32: engine scale factor coming from BlackBoxPower
              for blackbox_aerodynamics
            - y_34:SFC coming from blackbox_propulsion for blackbox_mission
            - g_1: vector of constraints for weight analysis
            - g_1[0] to g_1[4]: stress on wing
            - g_1[5]: wing twist as constraint
            - g_2: aero constraint (pressure gradient)
            - g_3: propulsion constraints
            - g_3[0]: engine scale factor constraint
            - g_3[1]: engine temperature
            - g_3[2]: throttle setting constraint

        :rtype: ndarray
        """

        x_1 = design_vector[:2]
        x_2 = array([design_vector[2]], dtype=self.dtype)
        x_3 = array([design_vector[3]], dtype=self.dtype)
        x_shared = design_vector[4:]

        y_12 = ones(2, dtype=self.dtype)
        y_21 = ones(1, dtype=self.dtype)
        y_31 = ones(1, dtype=self.dtype)
        y_32 = ones(1, dtype=self.dtype)

        # Execute Gauss Seidel iteration on system to find Y variables
        lift_convergence = y_21[0] + 10.0
        eng_weight_conv = y_31[0] + 10.0
        esf_convergence = y_32[0] + 10.0
        # loop_index = 1
        while (abs(lift_convergence - y_21[0]) > (y_21[0] * accuracy)) or (
            (abs(eng_weight_conv - y_31[0]) > (y_31[0] * accuracy))
            or (abs(esf_convergence - y_32[0]) > (y_32[0] * accuracy))
        ):
            lift_convergence = y_21[0]
            eng_weight_conv = y_31[0]
            esf_convergence = y_32[0]
            # Call Black Boxes
            y_1, _, y_12, y_14, g_1 = self.blackbox_structure(
                x_shared, y_21, y_31, x_1, true_cstr=true_cstr
            )
            y_2, y_21, y_23, y_24, g_2 = self.blackbox_aerodynamics(
                x_shared, y_12, y_32, x_2, true_cstr=true_cstr
            )
            y_3, y_34, y_31, y_32, g_3 = self.blackbox_propulsion(
                x_shared, y_23, x_3, true_cstr=true_cstr
            )
            y_4 = self.blackbox_mission(x_shared, y_14, y_24, y_34)

        return (
            y_1,
            y_2,
            y_3,
            y_4,
            y_12,
            y_14,
            y_21,
            y_23,
            y_24,
            y_31,
            y_32,
            y_34,
            g_1,
            g_2,
            g_3,
        )

    def get_constraints(self, design_vector, true_cstr=False):
        """Compute all constraints of Sobieski problem.

        :param design_vector: design variable vector
        :type design_vector: ndarray
        :param true_cstr: indicates if user
            wants absolute value or relative to limits (Default value = False)
        :type true_cstr: bool
        :returns: outputs : g_1,g_2,g_3:constraint values
        :rtype: ndarray
        """
        outputs = self.systemanalysis_gauss_seidel(design_vector, true_cstr=true_cstr)
        return outputs[-3], outputs[-2], outputs[-1]

    def read_design_space(self):
        """Reads the sobieski design space file and creates a DesignSpace instance."""
        input_file = join(DIRNAME, "sobieski_design_space.txt")
        design_space = DesignSpace.read_from_txt(input_file)
        if self.dtype == complex128:
            x_dict = design_space.get_current_x_dict()
            for var_name, value in x_dict.items():
                x_dict[var_name] = array(value, dtype=complex128)
            design_space.set_current_x(x_dict)

        return design_space
