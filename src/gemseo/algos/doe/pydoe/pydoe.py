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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""PyDOE algorithms wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy.random import RandomState
from pyDOE3.doe_box_behnken import bbdesign
from pyDOE3.doe_composite import ccdesign
from pyDOE3.doe_factorial import ff2n
from pyDOE3.doe_lhs import lhs
from pyDOE3.doe_plackett_burman import pbdesign

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.pydoe.pydoe_full_factorial_doe import PyDOEFullFactorialDOE
from gemseo.algos.doe.pydoe.settings.base_pydoe_settings import BasePyDOESettings
from gemseo.algos.doe.pydoe.settings.pydoe_bbdesign import PYDOE_BBDESIGN_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_ccdesign import PYDOE_CCDESIGN_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_ff2n import PYDOE_FF2N_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_pbdesign import PYDOE_PBDESIGN_Settings
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem

OptionType = (
    str
    | int
    | float
    | bool
    | Sequence[int]
    | tuple[int, int]
    | RealArray
    | RandomState
    | None
)


class PyDOEAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a pyDOE algorithm."""

    library_name = "PyDOE"
    """The library name."""


class PyDOELibrary(BaseDOELibrary[BasePyDOESettings]):
    """The PyDOE DOE algorithms library."""

    __NAMES_TO_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "PYDOE_BBDESIGN": bbdesign,
        "PYDOE_CCDESIGN": ccdesign,
        "PYDOE_FF2N": ff2n,
        "PYDOE_LHS": lhs,
        "PYDOE_PBDESIGN": pbdesign,
    }
    """The algorithm names bound to the corresponding pyDOE function."""

    __DOC: Final[str] = "https://pythonhosted.org/pyDOE/"

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "PYDOE_BBDESIGN": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_BBDESIGN",
            description="Box-Behnken design",
            internal_algorithm_name="bbdesign",
            website=f"{__DOC}rsm.html#box-behnken",
            settings_class=PYDOE_BBDESIGN_Settings,
            minimum_dimension=3,
        ),
        "PYDOE_CCDESIGN": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_CCDESIGN",
            description="Central Composite",
            internal_algorithm_name="ccdesign",
            website=f"{__DOC}rsm.html#central-composite",
            settings_class=PYDOE_CCDESIGN_Settings,
            minimum_dimension=2,
        ),
        "PYDOE_FF2N": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_FF2N",
            description="2-Level Full-Factorial",
            internal_algorithm_name="ff2n",
            website=f"{__DOC}factorial.html#level-full-factorial",
            settings_class=PYDOE_FF2N_Settings,
        ),
        "PYDOE_FULLFACT": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_FULLFACT",
            description="Full-Factorial",
            internal_algorithm_name="fullfact",
            website=f"{__DOC}factorial.html#general-full-factorial",
            settings_class=PYDOE_FULLFACT_Settings,
        ),
        "PYDOE_LHS": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_LHS",
            description="Latin Hypercube Sampling",
            internal_algorithm_name="lhs",
            website=f"{__DOC}randomized.html#latin-hypercube",
            settings_class=PYDOE_LHS_Settings,
        ),
        "PYDOE_PBDESIGN": PyDOEAlgorithmDescription(
            algorithm_name="PYDOE_PBDESIGN",
            description="Plackett-Burman design",
            internal_algorithm_name="pbdesign",
            website=f"{__DOC}factorial.html#plackett-burman",
            settings_class=PYDOE_PBDESIGN_Settings,
        ),
    }

    _SETTINGS_CLASS_TO_EXCLUDE: ClassVar[type[BasePyDOESettings]] = BasePyDOESettings

    def _generate_unit_samples(self, design_space: DesignSpace) -> RealArray:
        if self._algo_name == "PYDOE_FULLFACT":
            return PyDOEFullFactorialDOE().generate_samples(
                design_space.dimension, self._settings
            )

        filtered_settings = self._filter_settings()
        doe_algorithm = self.__NAMES_TO_FUNCTIONS[self._algo_name]
        if self._algo_name == "PYDOE_LHS":
            filtered_settings["random_state"] = RandomState(
                self._seeder.get_seed(self._settings.random_state)
            )
            filtered_settings["samples"] = filtered_settings["n_samples"]
            del filtered_settings["n_samples"]
            return doe_algorithm(design_space.dimension, **filtered_settings)

        data = doe_algorithm(design_space.dimension, **filtered_settings)
        # Scale data from [-1,1] to [0,1]
        return (data + 1.0) / 2.0

    @classmethod
    def _get_unsuitability_reason(
        cls,
        algorithm_description: DOEAlgorithmDescription,
        problem: OptimizationProblem,
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if (
            reason
            or problem.design_space.dimension >= algorithm_description.minimum_dimension
        ):
            return reason

        return _UnsuitabilityReason.SMALL_DIMENSION
