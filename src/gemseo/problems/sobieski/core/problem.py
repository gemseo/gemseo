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
"""The Sobieski's SSBJ problem."""
from __future__ import annotations

import cmath
import logging
import math
import random
from collections import namedtuple
from pathlib import Path
from random import uniform
from typing import ClassVar
from typing import Iterable
from typing import Sequence

from numpy import array
from numpy import complex128
from numpy import concatenate
from numpy import float64
from numpy import ndarray
from numpy import ones

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.aerodynamics import SobieskiAerodynamics
from gemseo.problems.sobieski.core.mission import SobieskiMission
from gemseo.problems.sobieski.core.propulsion import SobieskiPropulsion
from gemseo.problems.sobieski.core.structure import SobieskiStructure
from gemseo.problems.sobieski.core.utils import SobieskiBase
from gemseo.utils.python_compatibility import Final

LOGGER = logging.getLogger(__name__)


_Disciplines = namedtuple(
    "Disciplines",
    ("aerodynamics", "structure", "propulsion", "mission"),
)


class SobieskiProblem:
    r"""The Sobieski's SSBJ problem.

    This problem seeks to maximize the range of a supersonic business jet (SSBJ)
    over a bounded design space whilst satisfying inequality constraints.

    The objective and constraint functions come from a system of four disciplines:

    1. the structure,
       computing the total aircraft mass :math:`y_{1,4,0}` and :math:`y_{1,2,0}`,
       the fuel mass :math:`y_{1,4,1}`,
       the wing twist :math:`y_{1,2,1}`
       and the five stress constraints :math:`g_{1,0},\ldots,g_{4,0}`.
    2. the aerodynamics discipline,
       computing the lift :math:`y_{2,1}`,
       drag :math:`y_{2,3}`,
       lift-to-drag ratio
       and pressure gradient constraint :math:`g_2`,
    3. the propulsion,
       computing the engine mass :math:`y_{3,1,0}`,
       the engine scale factor :math:`y_{3,2,0}`,
       which is also the constraint :math:`g_{3,0}`,
       the specific fuel consumption :math:`y_{3,4,0}`,
       the engine temperature constraint :math:`g_{3,1}`,
       and the throttle setting constraint :math:`g_{3,2}`
    4. the mission,
       computing the range :math:`y_{4,0}`.

    Notes:
        - The structure, aerodynamics, propulsion and mission disciplines
          are numbered from 1 to 4.
        - The variable :math:`y_{i,j,k}` is a coupling variable
          from the discipline :math:`i` to the discipline :math:`j`.
        - The aerodynamics, structure and propulsion disciplines are strongly coupled,
          i.e. each of them depends directly or indirectly on the others,
          and provide inputs to the mission discipline.

    The design variables can be classified into four groups:

    - the design variables which are inputs to at least two disciplines,

        - :math:`x_{0,0}`: the thickness-to-chord ratio,
        - :math:`x_{0,1}`: the altitude (ft),
        - :math:`x_{0,2}`: the Mach number,
        - :math:`x_{0,3}`: the aspect ratio,
        - :math:`x_{0,4}`: the wing sweep (deg),
        - :math:`x_{0,5}`: the wing surface area (ft :sup:`2`\),

    - the design variables which are inputs of the structure discipline only:

        - :math:`x_{1,0}`: the wing taper ratio,
        - :math:`x_{1,1}`: the wingbox x-sectional area

    - the design variables which are inputs of the aerodynamics discipline only:

        - :math:`x_{2,0}`: the skin friction coefficient,

    - the design variables which are inputs of the propulsion discipline only:

        - :math:`x_{3,0}`: the throttle setting (engin mass flow).

    Lastly,
    this problem is based on five constants:

    - :math:`c_0`: the minimum fuel weight,
    - :math:`c_1`: the miscellaneous weight,
    - :math:`c_2`: the maximum load factor,
    - :math:`c_3`: the reference engine weight,
    - :math:`c_4`: the minimum drag coefficient.
    """

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

    __DESIGN_VARIABLE_NAMES: Final[tuple[str]] = ("x_1", "x_2", "x_3", "x_shared")

    USE_ORIGINAL_DESIGN_VARIABLES_ORDER: ClassVar[bool] = False
    """Whether to sort the :attr:`.DesignSpace` as in :cite:`SobieskiBLISS98`.

    If so,
    the order of the design variables will be
    ``"x_1"``, ``"x_2"``, ``"x_3"`` and ``"x_shared"``.
    Otherwise,
    ``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``.
    """

    STRESS_LIMIT = SobieskiStructure.STRESS_LIMIT
    TWIST_UPPER_LIMIT = SobieskiStructure.TWIST_UPPER_LIMIT
    TWIST_LOWER_LIMIT = SobieskiStructure.TWIST_LOWER_LIMIT
    PRESSURE_GRADIENT_LIMIT = SobieskiAerodynamics.PRESSURE_GRADIENT_LIMIT
    ESF_UPPER_LIMIT = SobieskiPropulsion.ESF_UPPER_LIMIT
    ESF_LOWER_LIMIT = SobieskiPropulsion.ESF_LOWER_LIMIT
    TEMPERATURE_LIMIT = SobieskiPropulsion.TEMPERATURE_LIMIT

    def __init__(
        self,
        dtype: str = SobieskiBase.DTYPE_DOUBLE,
    ) -> None:
        """
        Args:
            dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        """
        if dtype == SobieskiBase.DTYPE_COMPLEX:
            self.__dtype = complex128
            self.__math = cmath
        elif dtype == SobieskiBase.DTYPE_DOUBLE:
            self.__math = math
            self.__dtype = float64
        else:
            raise ValueError(f"Unknown dtype: {dtype}.")

        self.__base = SobieskiBase(dtype=self.__dtype)

        self.disciplines = _Disciplines(
            aerodynamics=SobieskiAerodynamics(self.__base),
            mission=SobieskiMission(self.__base),
            propulsion=SobieskiPropulsion(self.__base),
            structure=SobieskiStructure(self.__base),
        )

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
        ) = self.__base.get_initial_values()
        self.__l_b, self.__u_b = self.__base.design_bounds
        self.__design_space = None
        self.__names_to_feasible_values = {
            "x_1": array([0.14951, 7.5e-01], dtype=self.__dtype),
            "x_2": array([7.5e-01], dtype=self.__dtype),
            "x_3": array([0.1675], dtype=self.__dtype),
            "x_shared": array(
                [6.0e-02, 5.4e04, 1.4e00, 4.4e00, 6.6e01, 1.2e03], dtype=self.__dtype
            ),
        }

    @property
    def initial_design(self) -> ndarray:
        """The initial design :math:`x`."""
        return self.__base.initial_design.copy()

    @property
    def constants(self) -> ndarray:
        """The constant vector."""
        return self.__base.constants

    @property
    def aerodynamics(self) -> SobieskiAerodynamics:
        """The aerodynamics discipline."""
        return self.disciplines.aerodynamics

    @property
    def mission(self) -> SobieskiMission:
        """The mission discipline."""
        return self.disciplines.mission

    @property
    def propulsion(self) -> SobieskiPropulsion:
        """The propulsion discipline."""
        return self.disciplines.propulsion

    @property
    def structure(self) -> SobieskiStructure:
        """The structure discipline."""
        return self.disciplines.structure

    def get_bounds_by_name(
        self,
        variables_names: Sequence[str],
    ) -> tuple[ndarray, ndarray]:
        """Return the lower and upper bounds of variables.

        Args:
            variables_names: The names of the variables.

        Returns:
            The lower and upper bounds of the variables;
            the array components keep the order of the variables.
        """
        return self.__base.get_bounds_by_name(variables_names)

    def __set_indata(
        self,
        design_vector: ndarray,
        names: str | Iterable[str] | None,
    ) -> dict[str, ndarray]:
        """Return the default values of some variables of the problem.

        Args:
            design_vector: The design vector to be used.
            names: The names of the variables of interest.
                If ``None``, use all the variables of the problem.

        Returns:
            The default values of some variables of the problem.
        """
        dtype = self.__dtype
        constants = self.__base.constants
        names_to_default_values = {
            "x_shared": design_vector[4:10],
            "y_12": array([50606.9742, 0.95], dtype=dtype),
            "y_14": array((50606.9741711, 7306.20262124), dtype=dtype),
            "y_21": array([50606.9741711], dtype=dtype),
            "y_23": array([12562.01206488], dtype=dtype),
            "y_24": array([4.15006276], dtype=dtype),
            "y_31": array([6354.32430691], dtype=dtype),
            "y_32": array([0.50279625], dtype=dtype),
            "y_34": array([1.10754577], dtype=dtype),
            "y_1": ones(3, dtype=dtype),
            "y_2": ones(3, dtype=dtype),
            "y_3": ones(3, dtype=dtype),
            "g_1": ones(6, dtype=dtype),
            "g_2": ones(1, dtype=dtype),
            "g_3": ones(3, dtype=dtype),
            "x_1": design_vector[:2],
            "x_2": array([design_vector[2]], dtype=dtype),
            "x_3": array([design_vector[3]], dtype=dtype),
            "c_0": array([constants[0]], dtype=dtype),
            "c_1": array([constants[1]], dtype=dtype),
            "c_2": array([constants[2]], dtype=dtype),
            "c_3": array([constants[3]], dtype=dtype),
            "c_4": array([constants[4]], dtype=dtype),
        }

        if names is None:
            return names_to_default_values

        if isinstance(names, str):
            names = [names]

        return {
            names: names_to_default_values[names]
            for names in names
            if names in names_to_default_values
        }

    def get_default_inputs(
        self,
        names: str | Iterable[str] | None = None,
    ) -> dict[str, ndarray]:
        """Return the default variable values at the default initial point.

        Args:
            names: The names of the variables of interest.
                If ``None``, use all the variables of the problem.

        Returns:
            The default values of some variables at the default initial point.
        """
        return self.__set_indata(self.initial_design, names)

    def get_default_inputs_feasible(
        self,
        names: str | Iterable[str] | None = None,
    ) -> dict[str, ndarray]:
        """Return the default variable values at the default initial feasible point.

        Args:
            names: The names of the variables of interest.
                If ``None``, use all the variables of the problem.

        Returns:
            The default values of some variables at the default initial feasible point.
        """
        return self.__set_indata(self.get_x0_feasible(), names)

    def get_default_inputs_equilibrium(
        self,
        names: str | Iterable[str] | None = None,
    ) -> dict[str, ndarray]:
        """Return the default variable values at a multidisciplinary feasible point.

        The coupling variables are at the equilibrium,
        in the sense of the multidisciplinary analysis (MDA).

        Args:
            names: The names of the variables of interest.
                If ``None``, use all the variables of the problem.

        Returns:
            The default values of some variables at a multidisciplinary feasible point.
        """
        design_vector = self.initial_design
        dtype = self.__dtype
        constants = self.__base.constants
        names_to_default_values = {
            "x_shared": array(design_vector[4:], dtype=dtype),
            "y_34": array([1.10754577], dtype=dtype),
            "y_32": array([0.50279625], dtype=dtype),
            "y_21": array([50606.97417114], dtype=dtype),
            "y_31": array([6354.32430691], dtype=dtype),
            "y_23": array([12194.26719338], dtype=dtype),
            "y_24": array([4.15006276], dtype=dtype),
            "g_3": array([-0.99720375, -0.00279625, 0.16206032, -0.02], dtype=dtype),
            "g_2": array([-0.04], dtype=dtype),
            "g_1": array(
                [0.035, -0.00666667, -0.0275, -0.04, -0.04833333, -0.09, -0.15],
                dtype=dtype,
            ),
            "y_11": array([0.15591894], dtype=dtype),
            "y_14": array([50606.97417114, 7306.20262124], dtype=dtype),
            "y_3": array([1.10754577e00, 6.35432431e03, 5.02796251e-01], dtype=dtype),
            "y_12": array([5.06069742e04, 9.5e-01], dtype=dtype),
            "y_4": array([535.78844818], dtype=dtype),
            "y_2": array([5.06069742e04, 1.21942672e04, 4.15006276e00], dtype=dtype),
            "x_1": array(design_vector[:2], dtype=dtype),
            "x_2": array([design_vector[2]], dtype=dtype),
            "x_3": array([design_vector[3]], dtype=dtype),
            "c_0": array([constants[0]], dtype=dtype),
            "c_1": array([constants[1]], dtype=dtype),
            "c_2": array([constants[2]], dtype=dtype),
            "c_3": array([constants[3]], dtype=dtype),
            "c_4": array([constants[4]], dtype=self.__dtype),
        }

        if names is not None:
            return {name: names_to_default_values[name] for name in names}

        return names_to_default_values

    def get_random_input(
        self,
        names: str | Iterable[str] | None = None,
        seed: int | None = None,
    ) -> ndarray:
        """Return a randomized starting point related to some input variables.

        Args:
            names: The names of the variables.
                If ``None``, use all the input variables.
            seed: The seed for the random number generation.
                If ``None``, do not set the seed.

        Returns:
            The randomized starting point.
        """
        if seed is not None:
            random.seed(seed)

        lower_bound, upper_bound = self.design_bounds
        dtype = self.__dtype
        names_to_random_values = {
            "x_shared": array(uniform(lower_bound[4:], upper_bound[4:]), dtype=dtype),
            "y_34": array([uniform(0.8, 1.2)], dtype=dtype),
            "y_21": array((uniform(1.0, 6e4),), dtype=dtype),
            "y_31": array((uniform(1.0, 1e4),), dtype=dtype),
            "y_23": array([uniform(1.0, 2e4)], dtype=dtype),
            "y_24": array([uniform(0.7, 6.0)], dtype=dtype),
            "y_14": array([uniform(1.0, 6e4), uniform(1.0, 1e4)], dtype=dtype),
            "y_12": array([uniform(1.0, 6e4), uniform(0.6, 1.1)], dtype=dtype),
            "y_32": array([uniform(0.6, 1.1)], dtype=dtype),
            "x_1": array(uniform(lower_bound[:2], upper_bound[:2]), dtype=dtype),
            "x_2": array([uniform(lower_bound[2], upper_bound[2])], dtype=dtype),
            "x_3": array([uniform(lower_bound[3], upper_bound[3])], dtype=dtype),
            "c_0": array([self.__base.constants[0]], dtype=dtype),
            "c_1": array([self.__base.constants[1]], dtype=dtype),
            "c_2": array([self.__base.constants[2]], dtype=dtype),
            "c_3": array([self.__base.constants[3]], dtype=dtype),
            "c_4": array([self.__base.constants[4]], dtype=dtype),
        }

        if names is not None:
            return {name: names_to_random_values[name] for name in names}

        return names_to_random_values

    def get_x0_feasible(self, names: str | Iterable[str] | None = None) -> ndarray:
        """Return a feasible starting point related to some input variables.

        Args:
            names: The names of the variables.
                If ``None``, use all the input variables.

        Returns:
            The feasible starting point.
        """
        if isinstance(names, str):
            names = [names]

        if names is None:
            names = self.__DESIGN_VARIABLE_NAMES

        return concatenate([self.__names_to_feasible_values[name] for name in names])

    @property
    def design_bounds(self) -> tuple[ndarray, ndarray]:
        """The lower and upper bounds of the design variables."""
        return self.__base.design_bounds

    @property
    def optimum_design(self) -> ndarray:
        """The optimal design vector found by Sobieski with BLISS."""
        return array(
            (0.38757, 0.75, 0.75, 0.15624, 0.06, 60000.0, 1.4, 2.5, 70.0, 1500.0),
            dtype=self.__dtype,
        )

    @property
    def optimum_range(self) -> ndarray:
        """The optimal range found by Sobieski with BLISS."""
        return array([3963.98], dtype=self.__dtype)

    def get_sobieski_constraints(
        self,
        g_1: ndarray,
        g_2: ndarray,
        g_3: ndarray,
        true_cstr: ndarray = False,
    ) -> ndarray:
        """Return either the value of the constraints or the distance to the thresholds.

        Args:
            g_1: The constraints from the structure discipline:
                ``g_1[0]`` to ``g_1[4]`` are the stresses on wing
                and ``g_1[5]`` is the wing twist.
            g_2: The constraint (pressure gradient) from the aerodynamics discipline.
            g_3: The constraints from the propulsion discipline:
                ``g_3[0]`` is the engine scale factor,
                 ``g_3[1]`` is the engine temperature
                 and ``g_3[2]`` is the throttle setting constraint.
            true_cstr: If ``True``,
                return the value of the outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.

        Returns:
            The constraints or the distance to the thresholds,
            according to ``true_cstr``.
        """
        if true_cstr:
            return concatenate(
                (g_1[0:5], array((g_1[5], g_2[0], g_3[0], g_3[2], g_3[1])))
            )

        return concatenate(
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

    def normalize_inputs(
        self,
        input_vector: ndarray,
    ) -> ndarray:
        """Normalize an input vector with respect to the variable bounds.

        Args:
            input_vector: The input vector.

        Returns:
            The normalized input vector with components in :math:`[0,1]`.
        """
        lower_bound, upper_bound = self.design_bounds
        return (input_vector - lower_bound) / (upper_bound - lower_bound)

    def unnormalize_inputs(
        self,
        input_vector: ndarray,
    ) -> ndarray:
        """Unnormalize an input vector with respect to the variable bounds.

        Args:
            input_vector: The normalized input vector.

        Returns:
            The input vector in the variable space.
        """
        lower_bound, upper_bound = self.design_bounds
        return input_vector * (upper_bound - lower_bound) + lower_bound

    def __compute_mda(
        self,
        design_vector: tuple[ndarray],
        true_cstr: bool = False,
        accuracy: float = 1e-3,
    ) -> tuple[ndarray]:
        """Compute the output variables at equilibrium with the Gauss-Seidel algorithm.

        Args:
            design_vector: The design vector.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            accuracy: The system resolution accuracy.

        Returns:
            The output variables at equilibrium:
            - ``y_1``: The weight analysis outputs:
                - ``y_1[0]``: The total aircraft weight,
                - ``y_1[1]``: The fuel weight,
                - ``y_1[2]``: The wing twist,
            - ``y_2``: The outputs of the aerodynamics analysis:
                - ``y_2[0]``: the lift,
                - ``y_2[1]``: the drag,
                - ``y_2[2]``: the lift/drag ratio,
            - ``y_3``: The outputs of the propulsion analysis, namely
                - ``y_3[0]``: the specific fuel consumption (SFC),
                - ``y_3[1]``: the engine weight (EW),
                - ``y_3[2]``: the engine scale factor (ESF),
            - ``y_4``: The A/C range,
            - ``y_12``: The coupling variables for the aerodynamics discipline:
                - ``y_12[0]``: The total aircraft weight,
                - ``y_12[1]``: The wing twist,
            - ``y_14``: The coupling variables for the mission discipline:
                - ``y_14[0]``: The total aircraft weight
                - ``y_14[1]``: The fuel weight
            - ``y_21``: The coupling variable (lift) for the structure discipline,
            - ``y_23``: The coupling variable (drag) for the propulsion discipline,
            - ``y_24``: The coupling variable (lift/drag ratio)
               for the mission discipline,
            - ``y_34``: The couping variable SFC for the mission discipline,
            - ``y_31``: The coupling variable (EW) for the structure discipline,
            - ``y_32``: The coupling variable (ESG) for the aerodynamics discipline,
            - ``g_1``: the outputs to be constrained,
                where ``g_1[0]`` to ``g_1[4]`` are stresses on the wing
                and ``g_1[5]`` is the wing twist,
            - ``g_2``: The pressure gradient to be constrained.
            - ``g_3``: The propulsion constraints,
                where ``g_3[0]`` is the engine scale factor constraint,
                `g_3[1]``is the engine temperature,
                and ``g_3[2]`` is the throttle setting constraint.
        """
        dtype = self.__dtype
        x_1 = design_vector[:2]
        x_2 = array([design_vector[2]], dtype=dtype)
        x_3 = array([design_vector[3]], dtype=dtype)
        x_shared = design_vector[4:]

        y_12 = ones(2, dtype=dtype)
        y_21 = ones(1, dtype=dtype)
        y_31 = ones(1, dtype=dtype)
        y_32 = ones(1, dtype=dtype)

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
            y_1, _, y_12, y_14, g_1 = self.structure.execute(
                x_shared, y_21, y_31, x_1, true_cstr=true_cstr
            )
            y_2, y_21, y_23, y_24, g_2 = self.aerodynamics.execute(
                x_shared, y_12, y_32, x_2, true_cstr=true_cstr
            )
            y_3, y_34, y_31, y_32, g_3 = self.propulsion.execute(
                x_shared, y_23, x_3, true_cstr=true_cstr
            )
            y_4 = self.mission.execute(x_shared, y_14, y_24, y_34)

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

    def get_constraints(
        self,
        design_vector: ndarray,
        true_cstr: bool = False,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Compute all the constraints.

        Args:
            design_vector: The design vector.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.

        Returns:
            The value of the constraints :math:`g_1`, :math:`g_2` and :math:`g_3`.
        """
        outputs = self.__compute_mda(design_vector, true_cstr=true_cstr)
        return outputs[-3], outputs[-2], outputs[-1]

    def __read_design_space(self, suffix: str = "") -> DesignSpace:
        """Create a design space from a file.

        Args:
            suffix: The suffix used in the file name.

        Returns:
            The design space.
        """
        if self.__design_space is None:
            if self.USE_ORIGINAL_DESIGN_VARIABLES_ORDER:
                file_name = f"sobieski_original_design_space{suffix}.txt"
            else:
                file_name = f"sobieski_design_space{suffix}.txt"

            self.__design_space = DesignSpace.read_from_txt(
                Path(__file__).parent / file_name
            )
            if self.__dtype == complex128:
                current_x = self.__design_space.get_current_value(as_dict=True)
                for variable_name, current_value in current_x.items():
                    current_x[variable_name] = array(current_value, dtype=complex128)
                self.__design_space.set_current_value(current_x)

        return self.__design_space

    @property
    def design_space(self) -> DesignSpace:
        """The design space."""
        return self.__read_design_space()

    @property
    def design_space_with_physical_naming(self) -> DesignSpace:
        """The design space with physical naming."""
        return self.__read_design_space("_pn")
