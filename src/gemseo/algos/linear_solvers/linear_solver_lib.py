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
"""Base wrapper for all linear solvers."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from numpy import ndarray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spilu

from gemseo.algos.algo_lib import AlgoLib
from gemseo.algos.algo_lib import AlgorithmDescription
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


class LinearSolverLib(AlgoLib):
    """Abstract class for libraries of linear solvers."""

    SAVE_WHEN_FAIL = "save_when_fail"

    save_fpath: str | None
    """The file path to save the linear problem."""

    def __init__(self) -> None:
        super().__init__()
        self.save_fpath = None

    def solve(
        self,
        linear_problem: LinearProblem,
        algo_name: str,
        **options: Any,
    ) -> Any:
        """Solve the linear system.

        Args:
            linear_problem: The problem to solve.
            algo_name: The name of the algorithm.
            **options: The algorithm options.

        Returns:
            The execution result.
        """
        return self.execute(linear_problem, algo_name=algo_name, **options)

    def _build_ilu_preconditioner(
        self,
        lhs: ndarray,
        dtype: str | None = None,
    ) -> LinearOperator:
        """Construct a preconditioner using an incomplete LU factorization.

        Args:
            lhs: The linear system matrix.
            dtype: The numpy dtype of the resulting linear operator.
                If None, XXX.

        Returns:
            The preconditioner operator.
        """
        ilu = spilu(csc_matrix(lhs))
        return LinearOperator(lhs.shape, ilu.solve, dtype=dtype)

    @property
    def solution(self) -> ndarray:
        """The solution of the problem."""
        return self.problem.solution

    @staticmethod
    def is_algorithm_suited(
        algorithm_description: LinearSolverDescription,
        problem: LinearProblem,
    ) -> bool:
        """Check if the algorithm is suited to the problem according to algo_dict.

        Args:
            algorithm_description: The description of the algorithm.
            problem: The problem to be solved.

        Returns:
            Whether the algorithm suits.
        """
        if not problem.is_symmetric and algorithm_description.lhs_must_be_symmetric:
            return False

        if (
            not problem.is_positive_def
            and algorithm_description.lhs_must_be_positive_definite
        ):
            return False

        if (
            problem.is_lhs_linear_operator
            and not algorithm_description.lhs_must_be_linear_operator
        ):
            return False

        return True

    def _pre_run(
        self,
        problem: LinearProblem,
        algo_name: str,
        **options: Any,
    ) -> None:
        """Set the solver options and name in the problem attributes.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            **options: The options for the algorithm, see associated JSON file.
        """
        problem.solver_options = options
        problem.solver_name = algo_name

    def _post_run(
        self,
        problem: LinearProblem,
        algo_name: str,
        result: ndarray,
        **options: Any,
    ) -> None:  # noqa: D107
        """Save the LinearProblem to the disk when required.

        If the save_when_fail option is True, save the LinearProblem to the disk when
        the system failed and print the file name in the warnings.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The result of the run, i.e. the solution.
            **options: The options for the algorithm, see associated JSON file.
        """
        if options.get(self.SAVE_WHEN_FAIL, False):
            if not self.problem.is_converged:
                f_path = f"linear_system_{uuid4()}.pck"
                pickle.dump(self.problem, open(f_path, "wb"))
                LOGGER.warning(
                    "Linear solver failed, saving problem to file: %s", f_path
                )
                self.save_fpath = f_path
