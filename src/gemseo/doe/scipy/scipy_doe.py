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
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import TextIO

from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import PoissonDisk
from scipy.stats.qmc import QMCEngine
from scipy.stats.qmc import Sobol

from gemseo.doe.core.base_doe_library import BaseDOELibrary
from gemseo.doe.core.base_doe_library import DOEAlgorithmDescription
from gemseo.doe.scipy.settings.base_scipy_doe_settings import BaseSciPyDOESettings
from gemseo.doe.scipy.settings.base_scipy_doe_settings import Hypersphere
from gemseo.doe.scipy.settings.base_scipy_doe_settings import Optimizer
from gemseo.doe.scipy.settings.halton import Halton_Settings
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.doe.scipy.settings.poisson_disk import PoissonDisk_Settings
from gemseo.doe.scipy.settings.sobol import Sobol_Settings
from gemseo.util._compatibility.scipy import scipy_version  # noqa: F401
from gemseo.util.seeder import seed
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy import integer
    from numpy.random import Generator
    from numpy.random import RandomState

    from gemseo.space.design import DesignSpace

OptionType = str | int | float | bool | list[str] | Path | TextIO | RealArray | None

logger = logging.getLogger(__name__)


@dataclass
class SciPyDOEAlgorithmDescription(DOEAlgorithmDescription):
    """The description of a DOE algorithm from the SciPy library."""

    library_name: str = "SciPy DOE"
    """The library name."""


class _MonteCarlo(QMCEngine):
    """Monte Carlo sampling."""

    def __init__(
        self, d: int, seed: int | integer | Generator | RandomState | None = seed
    ) -> None:
        super().__init__(d=d, seed=seed)

    def _random(self, n: int = 1, *, workers: int = 1) -> RealArray:
        return self.rng.random((n, self.d))


class SciPyDOE(BaseDOELibrary[BaseSciPyDOESettings]):
    """The SciPy DOE algorithms library."""

    # Algorithm names within GEMSEO
    __halton: Final[str] = "Halton"
    __lhs: Final[str] = "LHS"
    __monte_carlo: Final[str] = "MC"
    __poisson_disk: Final[str] = "PoissonDisk"
    __sobol: Final[str] = "Sobol"

    __names_to_classes: Final[Mapping[str, type[QMCEngine]]] = MappingProxyType({
        __halton: Halton,
        __lhs: LatinHypercube,
        __monte_carlo: _MonteCarlo,
        __poisson_disk: PoissonDisk,
        __sobol: Sobol,
    })
    """The algorithm names bound to the SciPy classes."""

    __scipy_option_names: Final[tuple[str, ...]] = (
        "bits",
        "centered",
        "hypersphere",
        "ncandidates",
        "optimization",
        "radius",
        "scramble",
        "strength",
    )
    """The names of the SciPy options for the quasi Monte Carlo engines."""

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        __halton: SciPyDOEAlgorithmDescription(
            algorithm_name=__halton,
            description=__names_to_classes[__halton].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__names_to_classes[__halton].__name__,
            settings_class=Halton_Settings,
        ),
        __lhs: SciPyDOEAlgorithmDescription(
            algorithm_name=__lhs,
            description=__names_to_classes[__lhs].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__names_to_classes[__lhs].__name__,
            settings_class=LHS_Settings,
        ),
        __monte_carlo: SciPyDOEAlgorithmDescription(
            algorithm_name=__monte_carlo,
            description=__names_to_classes[__monte_carlo].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__names_to_classes[__monte_carlo].__name__,
            settings_class=MC_Settings,
        ),
        __poisson_disk: SciPyDOEAlgorithmDescription(
            algorithm_name=__poisson_disk,
            description=__names_to_classes[__poisson_disk].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__names_to_classes[__poisson_disk].__name__,
            settings_class=PoissonDisk_Settings,
        ),
        __sobol: SciPyDOEAlgorithmDescription(
            algorithm_name=__sobol,
            description=__names_to_classes[__sobol].__doc__.split("\n")[0][:-1],
            internal_algorithm_name=__names_to_classes[__sobol].__name__,
            settings_class=Sobol_Settings,
        ),
    }

    Hypersphere = Hypersphere
    """The sampling strategy for the poisson disk algorithm."""

    Optimizer = Optimizer
    """The optimization scheme to improve the quality of the DOE after sampling."""

    def _generate_unit_samples(self, design_space: DesignSpace) -> RealArray:
        algo = self.__names_to_classes[self._algo_name](
            design_space.dimension,
            seed=self._seeder.get_seed(self._settings.seed),
            **self._settings.model_dump(include=self.__scipy_option_names),
        )
        return algo.random(self._settings.n_samples)
