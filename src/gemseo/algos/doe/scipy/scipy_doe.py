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
"""Design of experiments based on SciPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Union

from packaging.version import parse as parse_version
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import PoissonDisk
from scipy.stats.qmc import QMCEngine
from scipy.stats.qmc import Sobol
from strenum import StrEnum

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.scipy.settings.halton import Halton_Settings
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.doe.scipy.settings.poisson_disk import PoissonDisk_Settings
from gemseo.algos.doe.scipy.settings.sobol import Sobol_Settings
from gemseo.typing import RealArray
from gemseo.utils.compatibility.scipy import SCIPY_VERSION
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy import integer
    from numpy.random import Generator
    from numpy.random import RandomState

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import (
        BaseSciPyDOESettings,
    )

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]

LOGGER = logging.getLogger(__name__)


class _AlgoData(NamedTuple):
    cls: type
    settings: type[BaseSciPyDOESettings]


@dataclass
class SciPyDOEAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a DOE algorithm from the SciPy library."""

    library_name: str = "SciPy DOE"
    """The library name."""


class _MonteCarlo(QMCEngine):
    """Monte Carlo sampling."""

    def __init__(
        self, d: int, seed: int | integer | Generator | RandomState | None = SEED
    ) -> None:
        super().__init__(d=d, seed=seed)

    if parse_version("1.10") > SCIPY_VERSION:

        def random(self, n: int = 1) -> RealArray:
            self.num_generated += n
            return self.rng.random((n, self.d))

    else:

        def _random(self, n: int = 1, *, workers: int = 1) -> RealArray:
            return self.rng.random((n, self.d))


class SciPyDOE(BaseDOELibrary):
    """The SciPy DOE algorithms library."""

    # Algorithm names within GEMSEO
    __HALTON: Final[str] = "Halton"
    __LHS: Final[str] = "LHS"
    __MONTE_CARLO: Final[str] = "MC"
    __POISSON_DISK: Final[str] = "PoissonDisk"
    __SOBOL: Final[str] = "Sobol"

    __NAMES_TO_CLASSES: Final[Mapping[str, type[QMCEngine]]] = {
        __HALTON: Halton,
        __LHS: LatinHypercube,
        __MONTE_CARLO: _MonteCarlo,
        __POISSON_DISK: PoissonDisk,
        __SOBOL: Sobol,
    }
    """The algorithm names bound to the SciPy classes."""

    __SCIPY_OPTION_NAMES: Final[list[str]] = [
        "bits",
        "centered",
        "hypersphere",
        "ncandidates",
        "optimization",
        "radius",
        "scramble",
        "strength",
    ]
    """The names of the SciPy options for the quasi Monte Carlo engines."""

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        __HALTON: SciPyDOEAlgorithmDescription(
            algorithm_name=__HALTON,
            description=__NAMES_TO_CLASSES[__HALTON].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__NAMES_TO_CLASSES[__HALTON].__name__,
            Settings=Halton_Settings,
        ),
        __LHS: SciPyDOEAlgorithmDescription(
            algorithm_name=__LHS,
            description=__NAMES_TO_CLASSES[__LHS].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__NAMES_TO_CLASSES[__LHS].__name__,
            Settings=LHS_Settings,
        ),
        __MONTE_CARLO: SciPyDOEAlgorithmDescription(
            algorithm_name=__MONTE_CARLO,
            description=__NAMES_TO_CLASSES[__MONTE_CARLO].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__NAMES_TO_CLASSES[__MONTE_CARLO].__name__,
            Settings=MC_Settings,
        ),
        __POISSON_DISK: SciPyDOEAlgorithmDescription(
            algorithm_name=__POISSON_DISK,
            description=__NAMES_TO_CLASSES[__POISSON_DISK].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__NAMES_TO_CLASSES[__POISSON_DISK].__name__,
            Settings=PoissonDisk_Settings,
        ),
        __SOBOL: SciPyDOEAlgorithmDescription(
            algorithm_name=__SOBOL,
            description=__NAMES_TO_CLASSES[__SOBOL].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__NAMES_TO_CLASSES[__SOBOL].__name__,
            Settings=Sobol_Settings,
        ),
    }

    class Hypersphere(StrEnum):
        """The sampling strategy for the poisson disk algorithm."""

        VOLUME = "volume"
        SURFACE = "surface"

    class Optimizer(StrEnum):
        """The optimization scheme to improve the quality of the DOE after sampling."""

        RANDOM_CD = "random-cd"
        LLOYD = "lloyd"
        NONE = ""

    def _generate_unit_samples(
        self, design_space: DesignSpace, **settings: OptionType
    ) -> RealArray:
        option_names = self.__SCIPY_OPTION_NAMES.copy()
        if self._algo_name == self.__SOBOL:
            self.__remove_recent_scipy_options(option_names, "bits", "1.9")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self._algo_name == self.__HALTON:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self._algo_name == self.__LHS:
            self.__remove_recent_scipy_options(option_names, "scramble", "1.10")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.8")
            self.__remove_recent_scipy_options(option_names, "strength", "1.8")
            if parse_version("1.10") <= SCIPY_VERSION:
                option_names.remove("centered")
        elif self._algo_name == self.__POISSON_DISK:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")

        algo = self.__NAMES_TO_CLASSES[self._algo_name](
            design_space.dimension,
            seed=self._seeder.get_seed(settings[self._SEED]),
            **{k: v for k, v in settings.items() if k in option_names},
        )
        return algo.random(settings[self._N_SAMPLES])

    @staticmethod
    def __remove_recent_scipy_options(
        scipy_option_names: list[str], option_name: str, version_name: str
    ) -> None:
        """Remove the SciPy options not yet available in the current SciPy version.

        Args:
            scipy_option_names: The names of the SciPy options.
            option_name: The name of the option.
            version_name: The version of SciPy which introduced this option.
        """
        if parse_version(version_name) > SCIPY_VERSION:
            scipy_option_names.remove(option_name)
            LOGGER.warning(
                "Removed the option %s which is only available from SciPy %s.",
                option_name,
                version_name,
            )
