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

from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
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
from gemseo.algos.doe.pydoe_full_factorial_doe import PyDOEFullFactorialDOE
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType

OptionType = Optional[
    Union[str, int, float, bool, Sequence[int], tuple[int, int], RealArray]
]


class PyDOE(BaseDOELibrary):
    """PyDOE optimization library interface See BaseDOELibrary."""

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "fullfact": DOEAlgorithmDescription(
            algorithm_name="fullfact",
            description="Full-Factorial",
            internal_algorithm_name="fullfact",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/factorial.html#general-full-factorial",
        ),
        "ff2n": DOEAlgorithmDescription(
            algorithm_name="ff2n",
            description="2-Level Full-Factorial",
            internal_algorithm_name="ff2n",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/factorial.html#level-full-factorial",
        ),
        "pbdesign": DOEAlgorithmDescription(
            algorithm_name="pbdesign",
            description="Plackett-Burman design",
            internal_algorithm_name="pbdesign",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/factorial.html#plackett-burman",
        ),
        "bbdesign": DOEAlgorithmDescription(
            algorithm_name="bbdesign",
            description="Box-Behnken design",
            internal_algorithm_name="bbdesign",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/rsm.html#box-behnken",
        ),
        "ccdesign": DOEAlgorithmDescription(
            algorithm_name="ccdesign",
            description="Central Composite",
            internal_algorithm_name="ccdesign",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/rsm.html#central-composite",
        ),
        "lhs": DOEAlgorithmDescription(
            algorithm_name="lhs",
            description="Latin Hypercube Sampling",
            internal_algorithm_name="lhs",
            library_name="PyDOE",
            website="https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube",
        ),
    }
    ALGORITHM_INFOS["bbdesign"].minimum_dimension = 3
    ALGORITHM_INFOS["ccdesign"].minimum_dimension = 2

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
        callbacks: Iterable[CallbackType] = (),
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:  # pylint: disable=W0221
        """Set the options.

        Args:
            alpha: A parameter to describe how the variance is distributed.
                Either "orthogonal" or "rotatable".
            face: The relation between the start points and the corner
            (factorial) points. Either "circumscribed", "inscribed" or "faced".
            criterion: The criterion to use when sampling the points. If ``None``,
                randomize the points within the intervals.
            iterations: The number of iterations in the `correlation` and
                `maximin` algorithms.
            eval_jac: Whether to evaluate the jacobian.
            center_bb: The number of center points for the Box-Behnken design.
                If ``None``, use a pre-determined number of points.
            center_cc: The 2-tuple of center points for the central composite
                design. If ``None``, use (4, 4).
            n_samples: The number of samples. If there is a parameter ``levels``,
                the latter can be specified
                and the former set to its default value ``None``.
            levels: The levels. If there is a parameter ``n_samples``,
                the latter can be specified
                and the former set to its default value ``None``.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            seed: The seed used for reproducibility reasons.
                If ``None``, use :attr:`.seed`.
            max_time: The maximum runtime in seconds, disabled if 0.
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
            **kwargs: The additional arguments.

        Returns:
            The options for the DOE.
        """
        if center_cc is None:
            center_cc = [4, 4]
        return self._process_options(
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
            wait_time_between_samples=wait_time_between_samples,
            seed=seed,
            max_time=max_time,
            callbacks=callbacks,
            **kwargs,
        )

    @staticmethod
    def __translate(
        result: RealArray,
    ) -> RealArray:
        """Translate the DOE design variables to [0,1].

        Args:
            result: The design variables to be translated.

        Returns:
            The translated design variables.
        """
        return (result + 1.0) * 0.5

    def _generate_unit_samples(
        self, design_space: DesignSpace, **options: OptionType
    ) -> RealArray:
        if self._algo_name == "lhs":
            return lhs(
                design_space.dimension,
                random_state=RandomState(self._seeder.get_seed(options[self._SEED])),
                samples=options[self._N_SAMPLES],
                criterion=options.get("criterion"),
                iterations=options.get("iterations"),
            )

        if self._algo_name == "ccdesign":
            return self.__translate(
                ccdesign(
                    design_space.dimension,
                    center=options["center_cc"],
                    alpha=options["alpha"],
                    face=options["face"],
                )
            )

        if self._algo_name == "bbdesign":
            # Initially designed for quadratic model fitting
            # center point is can be run several times to allow for a more
            # uniform estimate of the prediction variance over the
            # entire design space. Default value of center depends on dv_size
            return self.__translate(
                bbdesign(
                    design_space.dimension,
                    center=options.get("center_bb"),
                )
            )

        if self._algo_name == "fullfact":
            return PyDOEFullFactorialDOE().generate_samples(
                options.pop(self._N_SAMPLES),
                design_space.dimension,
                **options,
            )

        if self._algo_name == "ff2n":
            return self.__translate(ff2n(design_space.dimension))

        if self._algo_name == "pbdesign":
            return self.__translate(pbdesign(design_space.dimension))

        msg = f"Bad algo_name: {self._algo_name}"
        raise ValueError(msg)

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
