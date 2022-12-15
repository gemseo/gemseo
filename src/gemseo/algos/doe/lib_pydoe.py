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

from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import pyDOE2 as pyDOE
from numpy import array
from numpy import ndarray
from numpy.random import RandomState

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.doe.doe_lib import DOEAlgorithmDescription
from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.algos.opt_problem import OptimizationProblem

OptionType = Optional[
    Union[str, int, float, bool, Sequence[int], Tuple[int, int], ndarray]
]


class PyDOE(DOELibrary):
    """PyDOE optimization library interface See DOELibrary."""

    # Available designs
    PYDOE_DOC = "https://pythonhosted.org/pyDOE/"
    PYDOE_LHS = "lhs"
    PYDOE_LHS_DESC = "Latin Hypercube Sampling implemented in pyDOE"
    PYDOE_LHS_WEB = PYDOE_DOC + "randomized.html#latin-hypercube"
    PYDOE_2LEVELFACT = "ff2n"
    PYDOE_2LEVELFACT_DESC = "2-Level Full-Factorial implemented in pyDOE"
    PYDOE_2LEVELFACT_WEB = PYDOE_DOC + "factorial.html#level-full-factorial"
    PYDOE_FULLFACT = "fullfact"
    PYDOE_FULLFACT_DESC = "Full-Factorial implemented in pyDOE"
    PYDOE_FULLFACT_WEB = PYDOE_DOC + "factorial.html#general-full-factorial"
    PYDOE_PBDESIGN = "pbdesign"
    PYDOE_PBDESIGN_DESC = "Plackett-Burman design implemented in pyDOE"
    PYDOE_PBDESIGN_WEB = PYDOE_DOC + "factorial.html#plackett-burman"
    PYDOE_BBDESIGN = "bbdesign"
    PYDOE_BBDESIGN_DESC = "Box-Behnken design implemented in pyDOE"
    PYDOE_BBDESIGN_WEB = PYDOE_DOC + "rsm.html#box-behnken"
    PYDOE_CCDESIGN = "ccdesign"
    PYDOE_CCDESIGN_DESC = "Central Composite implemented in pyDOE"
    PYDOE_CCDESIGN_WEB = PYDOE_DOC + "rsm.html#central-composite"
    ALGO_LIST = [
        PYDOE_FULLFACT,
        PYDOE_2LEVELFACT,
        PYDOE_PBDESIGN,
        PYDOE_BBDESIGN,
        PYDOE_CCDESIGN,
        PYDOE_LHS,
    ]
    DESC_LIST = [
        PYDOE_FULLFACT_DESC,
        PYDOE_2LEVELFACT_DESC,
        PYDOE_PBDESIGN_DESC,
        PYDOE_BBDESIGN_DESC,
        PYDOE_CCDESIGN_DESC,
        PYDOE_LHS_DESC,
    ]
    WEB_LIST = [
        PYDOE_FULLFACT_WEB,
        PYDOE_2LEVELFACT_WEB,
        PYDOE_PBDESIGN_WEB,
        PYDOE_BBDESIGN_WEB,
        PYDOE_CCDESIGN_WEB,
        PYDOE_LHS_WEB,
    ]
    CRITERION_KEYWORD = "criterion"
    ITERATION_KEYWORD = "iterations"
    ALPHA_KEYWORD = "alpha"
    FACE_KEYWORD = "face"
    CENTER_BB_KEYWORD = "center_bb"
    CENTER_CC_KEYWORD = "center_cc"
    LIBRARY_NAME = "PyDOE"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for idx, algo in enumerate(self.ALGO_LIST):
            self.descriptions[algo] = DOEAlgorithmDescription(
                algorithm_name=algo,
                description=self.DESC_LIST[idx],
                internal_algorithm_name=algo,
                library_name=self.__class__.__name__,
                website=self.WEB_LIST[idx],
            )

        self.descriptions["bbdesign"].minimum_dimension = 3
        self.descriptions["ccdesign"].minimum_dimension = 2

    def _get_options(
        self,
        alpha: str = "orthogonal",
        face: str = "faced",
        criterion: str | None = None,
        iterations: int = 5,
        eval_jac: bool = False,
        center_bb: int | None = None,
        center_cc: tuple[int, int] | None = None,
        n_samples: int | None = None,
        levels: Sequence[int] | None = None,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        seed: int | None = None,
        max_time: float = 0,
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:  # pylint: disable=W0221
        """Set the options.

        Args:
            alpha: A parameter to describe how the variance is distributed.
                Either "orthogonal" or "rotatable".
            face: The relation between the start points and the corner
            (factorial) points. Either "circumscribed", "inscribed" or "faced".
            criterion: The criterion to use when sampling the points. If None,
                randomize the points within the intervals.
            iterations: The number of iterations in the `correlation` and
                `maximin` algorithms.
            eval_jac: Whether to evaluate the jacobian.
            center_bb: The number of center points for the Box-Behnken design.
                If None, use a pre-determined number of points.
            center_cc: The 2-tuple of center points for the central composite
                design. If None, use (4, 4).
            n_samples: The number of samples. If None, then use the number of
                levels per input dimension provided by the argument `levels`.
            levels: The level in each direction for the full-factorial design.
                If `None`, then the number of samples provided by the argument
                `n_samples` is used in order to deduce the levels.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            seed: The seed value.
                If ``None``,
                use the seed of the library,
                namely :attr:`.PyDOE.seed`.
            max_time: The maximum runtime in seconds, disabled if 0.
            **kwargs: The additional arguments.

        Returns:
            The options for the DOE.
        """
        if center_cc is None:
            center_cc = [4, 4]
        wtbs = wait_time_between_samples
        popts = self._process_options(
            alpha=alpha,
            face=face,
            criterion=criterion,
            iterations=iterations,
            center_cc=center_cc,
            center_bb=center_bb,
            eval_jac=eval_jac,
            n_samples=n_samples,
            n_processes=n_processes,
            levels=levels,
            wait_time_between_samples=wtbs,
            seed=seed,
            max_time=max_time,
            **kwargs,
        )

        return popts

    @staticmethod
    def __translate(
        result: ndarray,
    ) -> ndarray:
        """Translate the DOE design variables to [0,1].

        Args:
            result: The design variables to be translated.

        Returns:
            The translated design variables.
        """
        return (result + 1.0) * 0.5

    def _generate_samples(self, **options: OptionType) -> ndarray:
        """Generate the samples for the DOE.

        Args:
            **options: The options for the algorithm,
                see the associated JSON file.

        Returns:
            The samples for the DOE.
        """
        self.seed += 1
        if self.algo_name == self.PYDOE_LHS:
            return pyDOE.lhs(
                options[self.DIMENSION],
                random_state=RandomState(options[self.SEED] or self.seed),
                samples=options["n_samples"],
                criterion=options.get(self.CRITERION_KEYWORD),
                iterations=options.get(self.ITERATION_KEYWORD),
            )

        if self.algo_name == self.PYDOE_CCDESIGN:
            return self.__translate(
                pyDOE.ccdesign(
                    options[self.DIMENSION],
                    center=options[self.CENTER_CC_KEYWORD],
                    alpha=options[self.ALPHA_KEYWORD],
                    face=options[self.FACE_KEYWORD],
                )
            )

        if self.algo_name == self.PYDOE_BBDESIGN:
            # Initially designed for quadratic model fitting
            # center point is can be run several times to allow for a more
            # uniform estimate of the prediction variance over the
            # entire design space. Default value of center depends on dv_size
            return self.__translate(
                pyDOE.bbdesign(
                    options[self.DIMENSION], center=options.get(self.CENTER_BB_KEYWORD)
                )
            )

        if self.algo_name == self.PYDOE_FULLFACT:
            return self._generate_fullfact(
                options[self.DIMENSION],
                levels=options.get(self.LEVEL_KEYWORD),
                n_samples=options.get(self.N_SAMPLES),
            )

        if self.algo_name == self.PYDOE_2LEVELFACT:
            return self.__translate(pyDOE.ff2n(options[self.DIMENSION]))

        if self.algo_name == self.PYDOE_PBDESIGN:
            return self.__translate(pyDOE.pbdesign(options[self.DIMENSION]))

    def _generate_fullfact_from_levels(self, levels: int | Sequence[int]) -> ndarray:
        doe = pyDOE.fullfact(levels)

        # Because pyDOE return the DOE where the values of levels are integers from 0 to
        # the maximum level number,
        # we need to divide by levels - 1.
        # To not divide by zero,
        # we first find the null denominators,
        # we replace them by one,
        # then we change the final values of the DOE by 0.5.
        divide_factor = array(levels) - 1
        null_indices = divide_factor == 0
        divide_factor[null_indices] = 1
        doe /= divide_factor
        doe[:, null_indices] = 0.5
        return doe

    @classmethod
    def _get_unsuitability_reason(
        cls,
        algorithm_description: DOEAlgorithmDescription,
        problem: OptimizationProblem,
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason or problem.dimension >= algorithm_description.minimum_dimension:
            return reason

        return _UnsuitabilityReason.SMALL_DIMENSION
