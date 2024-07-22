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

from packaging.version import parse as parse_version
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import PoissonDisk
from scipy.stats.qmc import QMCEngine
from scipy.stats.qmc import Sobol
from strenum import StrEnum

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.typing import RealArray
from gemseo.utils.compatibility.scipy import SCIPY_VERSION
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import integer
    from numpy.random import Generator
    from numpy.random import RandomState

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]

LOGGER = logging.getLogger(__name__)


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
    """A library of designs of experiments based on SciPy."""

    _OPTIONS_DIR: ClassVar[Path] = Path("options") / "scipy"

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

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        name: DOEAlgorithmDescription(
            algorithm_name=name,
            description=cls.__doc__.split("\n")[0][:-1],
            internal_algorithm_name=cls.__name__,
            library_name="SciPy",
        )
        for name, cls in __NAMES_TO_CLASSES.items()
    }

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
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        """Set the options.

        Args:
            max_time: The maximum runtime in seconds, disabled if 0.
            eval_jac: Whether to evaluate the jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            seed: The seed used for reproducibility reasons.
                If ``None``, use :attr:`.seed`.
            n_samples: The number of samples.
            centered: Whether to center the samples
                within the cells of a multi-dimensional grid.
                If SciPy >= 1.10.0,
                this argument is ignore;
                use ``scramble`` instead.
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
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
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
            callbacks=callbacks,
            **kwargs,
        )

    def _generate_unit_samples(
        self, design_space: DesignSpace, **options: OptionType
    ) -> RealArray:
        option_names = self.__SCIPY_OPTION_NAMES.copy()
        if self._algo_name == self.__SOBOL_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "bits", "1.9")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self._algo_name == self.__HALTON_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")
        elif self._algo_name == self.__LHS_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "scramble", "1.10")
            self.__remove_recent_scipy_options(option_names, "optimization", "1.8")
            self.__remove_recent_scipy_options(option_names, "strength", "1.8")
            if parse_version("1.10") <= SCIPY_VERSION:
                option_names.remove("centered")
        elif self._algo_name == self.__POISSON_DISK_ALGO_NAME:
            self.__remove_recent_scipy_options(option_names, "optimization", "1.10")

        algo = self.__NAMES_TO_CLASSES[self._algo_name](
            design_space.dimension,
            seed=self._seeder.get_seed(options[self._SEED]),
            **{k: v for k, v in options.items() if k in option_names},
        )
        return algo.random(options[self._N_SAMPLES])

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
