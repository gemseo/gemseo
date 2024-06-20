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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for libraries of linear solvers."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from uuid import uuid4

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spilu

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_algorithm_library import AlgorithmDescription
from gemseo.algos.base_algorithm_library import BaseAlgorithmLibrary

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.linear_solvers.linear_problem import LinearProblem

LOGGER = logging.getLogger(__name__)


@dataclass
class LinearSolverDescription(AlgorithmDescription):
    """The description of a linear solver."""

    lhs_must_be_symmetric: bool = False
    """Whether the left-hand side matrix must be symmetric."""

    lhs_must_be_positive_definite: bool = False
    """Whether the left-hand side matrix must be positive definite."""

    lhs_must_be_linear_operator: bool = False
    """Whether the left-hand side matrix must be a linear operator."""


class BaseLinearSolverLibrary(BaseAlgorithmLibrary):
    """Base class for libraries of linear solvers."""

    file_path: Path
    """The file path to save the linear problem after an execution."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self.file_path = Path("linear_system.pck")

    @staticmethod
    def _build_ilu_preconditioner(
        lhs: ndarray,
        dtype: str | None = None,
    ) -> LinearOperator:
        """Construct a preconditioner using an incomplete LU factorization.

        Args:
            lhs: The linear system matrix.
            dtype: The numpy dtype of the resulting linear operator.
                If ``None``, XXX.

        Returns:
            The preconditioner operator.
        """
        ilu = spilu(csc_matrix(lhs))
        return LinearOperator(lhs.shape, ilu.solve, dtype=dtype)

    @classmethod
    def _get_unsuitability_reason(
        cls,
        algorithm_description: LinearSolverDescription,
        problem: LinearProblem,
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason:
            return reason

        if not problem.is_symmetric and algorithm_description.lhs_must_be_symmetric:
            return _UnsuitabilityReason.NOT_SYMMETRIC

        if (
            not problem.is_positive_def
            and algorithm_description.lhs_must_be_positive_definite
        ):
            return _UnsuitabilityReason.NOT_POSITIVE_DEFINITE

        if (
            problem.is_lhs_linear_operator
            and not algorithm_description.lhs_must_be_linear_operator
        ):
            return _UnsuitabilityReason.NOT_LINEAR_OPERATOR

        return reason

    def _pre_run(
        self,
        problem: LinearProblem,
        **options: Any,
    ) -> None:
        problem.solver_options = options
        problem.solver_name = self._algo_name

    def _post_run(
        self,
        problem: LinearProblem,
        result: ndarray,
        **options: Any,
    ) -> None:
        # If the save_when_fail option is True, save the LinearProblem to the disk when
        # the system failed and print the file name in the warnings.
        if not problem.is_converged:
            LOGGER.warning(
                "The linear solver %s did not converge.", problem.solver_name
            )

        if options.get("save_when_fail", False) and not problem.is_converged:
            file_path = Path(f"linear_system_{uuid4()}.pck")
            with file_path.open("wb") as stream:
                pickle.dump(problem, stream)

            LOGGER.warning(
                "Linear solver failed, saving problem to file: %s", file_path
            )
            self.file_path = file_path
