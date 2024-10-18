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

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Optional
from typing import Union

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
from gemseo.algos.doe.pydoe.settings.bbdesign import BBDesignSettings
from gemseo.algos.doe.pydoe.settings.ccdesign import CCDesignSettings
from gemseo.algos.doe.pydoe.settings.ff2n import FF2NSettings
from gemseo.algos.doe.pydoe.settings.fullfact import FullFactSettings
from gemseo.algos.doe.pydoe.settings.lhs import LHSSettings
from gemseo.algos.doe.pydoe.settings.pbdesign import PBDesignSettings
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem

OptionType = Optional[
    Union[str, int, float, bool, Sequence[int], tuple[int, int], RealArray, RandomState]
]


class PyDOEAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a pyDOE algorithm."""

    library_name = "PyDOE"
    """The library name."""


class PyDOELibrary(BaseDOELibrary):
    """The PyDOE DOE algorithms library."""

    __NAMES_TO_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "bbdesign": bbdesign,
        "ccdesign": ccdesign,
        "ff2n": ff2n,
        "lhs": lhs,
        "pbdesign": pbdesign,
    }
    """The algorithm names bound to the corresponding pyDOE function."""

    __DOC: Final[str] = "https://pythonhosted.org/pyDOE/"

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "bbdesign": PyDOEAlgorithmDescription(
            algorithm_name="bbdesign",
            description="Box-Behnken design",
            internal_algorithm_name="bbdesign",
            website=f"{__DOC}rsm.html#box-behnken",
            Settings=BBDesignSettings,
            minimum_dimension=3,
        ),
        "ccdesign": PyDOEAlgorithmDescription(
            algorithm_name="ccdesign",
            description="Central Composite",
            internal_algorithm_name="ccdesign",
            website=f"{__DOC}rsm.html#central-composite",
            Settings=CCDesignSettings,
            minimum_dimension=2,
        ),
        "ff2n": PyDOEAlgorithmDescription(
            algorithm_name="ff2n",
            description="2-Level Full-Factorial",
            internal_algorithm_name="ff2n",
            website=f"{__DOC}factorial.html#level-full-factorial",
            Settings=FF2NSettings,
        ),
        "fullfact": PyDOEAlgorithmDescription(
            algorithm_name="fullfact",
            description="Full-Factorial",
            internal_algorithm_name="fullfact",
            website=f"{__DOC}factorial.html#general-full-factorial",
            Settings=FullFactSettings,
        ),
        "lhs": PyDOEAlgorithmDescription(
            algorithm_name="lhs",
            description="Latin Hypercube Sampling",
            internal_algorithm_name="lhs",
            website=f"{__DOC}randomized.html#latin-hypercube",
            Settings=LHSSettings,
        ),
        "pbdesign": PyDOEAlgorithmDescription(
            algorithm_name="pbdesign",
            description="Plackett-Burman design",
            internal_algorithm_name="pbdesign",
            website=f"{__DOC}factorial.html#plackett-burman",
            Settings=PBDesignSettings,
        ),
    }

    def _generate_unit_samples(
        self, design_space: DesignSpace, **settings: OptionType
    ) -> RealArray:
        n = design_space.dimension
        if self._algo_name == "fullfact":
            return PyDOEFullFactorialDOE().generate_samples(dimension=n, **settings)

        doe_algorithm = self.__NAMES_TO_FUNCTIONS[self._algo_name]
        if self._algo_name == "lhs":
            settings["random_state"] = RandomState(
                self._seeder.get_seed(settings["random_state"])
            )
            settings["samples"] = settings["n_samples"]
            del settings["n_samples"]
            return doe_algorithm(n, **settings)

        return self.__scale(doe_algorithm(n, **settings))

    @staticmethod
    def __scale(result: RealArray) -> RealArray:
        """Scale the DOE design variables to [0, 1].

        Args:
            result: The design variables to be scaled.

        Returns:
            The scaled design variables.
        """
        return (result + 1.0) * 0.5

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
