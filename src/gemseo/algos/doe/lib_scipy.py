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
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import Optional
from typing import TextIO
from typing import Union

import scipy
from numpy import integer
from numpy import ndarray
from packaging import version
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import PoissonDisk
from scipy.stats.qmc import QMCEngine
from scipy.stats.qmc import Sobol
from strenum import StrEnum

from gemseo import SEED
from gemseo.algos.doe.doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.doe_library import DOELibrary

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.random import RandomState
    from packaging.version import Version

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, ndarray]]

LOGGER = logging.getLogger(__name__)


class _MonteCarlo(QMCEngine):
    """Monte Carlo sampling."""

    def __init__(
        self, d: int, seed: int | integer | Generator | RandomState | None = SEED
    ) -> None:
        super().__init__(d=d, seed=seed)

    if version.parse(scipy.__version__) < version.parse("1.10"):

        def random(self, n: int = 1) -> ndarray:
            self.num_generated += n
            return self.rng.random((n, self.d))
    else:

        def _random(self, n: int = 1, *, workers: int = 1) -> ndarray:
            return self.rng.random((n, self.d))


class SciPyDOE(DOELibrary):
    """A library of designs of experiments based on SciPy."""

    LIBRARY_NAME: ClassVar[str] = "SciPy"
    OPTIONS_DIR: ClassVar[Path] = Path("options") / "scipy"

    __HALTON_ALGO_NAME: Final[str] = "Halton"
    __LHS_ALGO_NAME: Final[str] = "LHS"
    __MC_ALGO_NAME: Final[str] = "MC"
    __POISSON_DISK_ALGO_NAME: Final[str] = "PoissonDisk"
    __SOBOL_ALGO_NAME: Final[str] = "Sobol"

    __NAMES_TO_CLASSES: Final[str, type] = {
        __HALTON_ALGO_NAME: Halton,
        __LHS_ALGO_NAME: LatinHypercube,
        __MC_ALGO_NAME: _MonteCarlo,
        __POISSON_DISK_ALGO_NAME: PoissonDisk,
        __SOBOL_ALGO_NAME: Sobol,
    }
    """The algorithm names bound to the SciPy classes."""

    __SCIPY_VERSION: Final[Version] = version.parse(scipy.__version__)
    """The version of SciPy."""

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

    class Hypersphere(StrEnum):
        """The sampling strategy for the poisson disk algorithm."""

        VOLUME = "volume"
        SURFACE = "surface"

    class Optimizer(StrEnum):
        """The optimization scheme to improve the quality of the DOE after sampling."""

        RANDOM_CD = "random-cd"
        LLOYD = "lloyd"
        NONE = ""

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for name, cls in self.__NAMES_TO_CLASSES.items():
            self.descriptions[name] = DOEAlgorithmDescription(
                algorithm_name=name,
                description=cls.__doc__.split("\n")[0][:-1],
                internal_algorithm_name=cls.__name__,
                library_name=self.algo_name,
            )

    def _get_options(
        self,
        max_time: float = 0,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        seed: int | None = None,
        n_samples: int = 1,
        centered: bool = False,
        scramble: bool = True,
        radius: float = 0.05,
        hypersphere: Hypersphere = Hypersphere.VOLUME,
        ncandidates: int = 30,
        bits: int | None = None,
        optimization: Optimizer = Optimizer.NONE,
        strength: Literal[1, 2] = 1,
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        """Set the options.

        Args:
            max_time: The maximum runtime in seconds, disabled if 0.
            eval_jac: Whether to evaluate the jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            seed: The seed value.
                If ``None``,
                use the seed of the library,
                namely :attr:`.SciPyDOE.seed`.
            n_samples: The number of samples.
            centered: Whether to center the samples
                within the cells of a multi-dimensional grid.
                If SciPy >= 1.10.0, use ``scramble`` instead.
            scramble: Whether to use scrambling (Owen type).
                Only available with SciPy >= 1.10.0.
            radius: The minimal distance to keep between points
                when sampling new candidates.
            hypersphere: The sampling strategy
                to generate potential candidates to be added in the final sample.
            ncandidates: The number of candidates to sample per iteration.
            bits: The number of bits of the generator.
                New in SciPy 1.9.0.
            optimization: The name of an optimization scheme
                to improve the quality of the DOE.
                If ``None``, use the DOE as is.
                New in SciPy 1.10.0.
            strength: The strength of the LHS.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        if optimization == self.Optimizer.NONE:
            optimization = None

        return self._process_options(
            max_time=max_time,
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            n_samples=n_samples,
            seed=seed,
            centered=centered,
            scramble=scramble,
            radius=radius,
            hypersphere=hypersphere,
            ncandidates=ncandidates,
            optimization=optimization,
            bits=bits,
            strength=strength,
            **kwargs,
        )

    def _generate_samples(self, **options: OptionType) -> ndarray:
        seed = options[self.SEED]
        option_names = self.__SCIPY_OPTION_NAMES.copy()
        if self.algo_name == self.__SOBOL_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "bits", "1.9")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self.algo_name == self.__HALTON_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self.algo_name == self.__LHS_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "scramble", "1.10")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.8")
            self.__remove_recent_scipy_options(option_names, "strength", "1.8")
        elif self.algo_name == self.__POISSON_DISK_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")

        scipy_options = {k: v for k, v in options.items() if k in option_names}
        algo = self.__NAMES_TO_CLASSES[self.algo_name](
            options[self.DIMENSION],
            seed=self.seed if seed is None else seed,
            **scipy_options,
        )
        return algo.random(options[self.N_SAMPLES])

    def __remove_recent_scipy_options(
        self, scipy_option_names: list[str], option_name: str, version_name: str
    ) -> None:
        """Remove the SciPy options not yet available in the current SciPy version.

        Args:
            scipy_option_names: The names of the SciPy options.
            option_name: The name of the option.
            version_name: The version of SciPy which introduced this option.
        """
        if version.parse(version_name) > self.__SCIPY_VERSION:
            scipy_option_names.remove(option_name)
            LOGGER.warning(
                "Removed the option %s which is only available from SciPy %s.",
                option_name,
                version_name,
            )
