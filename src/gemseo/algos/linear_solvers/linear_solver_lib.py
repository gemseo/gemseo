# -*- coding: utf-8 -*-
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

import logging
import pickle
from typing import Any, Mapping, Optional
from uuid import uuid4

from numpy import ndarray
from scipy.sparse.linalg import LinearOperator, spilu

from gemseo.algos.algo_lib import AlgoLib
from gemseo.algos.linear_solvers.linear_problem import LinearProblem

LOGGER = logging.getLogger(__name__)


class LinearSolverLib(AlgoLib):
    """Abstract class for libraries of linear solvers.

    Attributes:
        XXX how about backup_path or file_path?
        save_fpath (str): The path to the file to save the problem when
            it is not converged and the attribute save_when_fail is True.
    """

    LHS_MUST_BE_SYMMETRIC = "LHS_symmetric"
    LHS_MUST_BE_POSITIVE_DEFINITE = "LHS_positive_definite"
    LHS_CAN_BE_LINEAR_OPERATOR = "LHS_linear_operator"

    SAVE_WHEN_FAIL = "save_when_fail"

    def __init__(self):  # type: (...) -> None
        super(LinearSolverLib, self).__init__()
        self.save_fpath = None

    def solve(
        self,
        linear_problem,  # type: LinearProblem
        algo_name,  # type: str
        **options  # type: Any
    ):  # type: (...) -> Any
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
        lhs,  # type: ndarray
        dtype=None,  # type: Optional[str]
    ):  # type: (...) -> LinearOperator
        """Construct a preconditioner using an incomplete LU factorization.

        Args:
            lhs: The linear system matrix.
            dtype: The numpy dtype of the resulting linear operator.
                If None, XXX.

        Returns:
            The preconditioner operator.
        """
        ilu = spilu(lhs)
        return LinearOperator(lhs.shape, ilu.solve, dtype=dtype)

    @property
    def solution(self):  # type: (...) -> ndarray
        """The solution of the problem."""
        return self.problem.solution

    @staticmethod
    def is_algorithm_suited(
        algo_dict,  # type: Mapping[str, bool]
        problem,  # type: LinearProblem
    ):  # type: (...) -> bool
        """Check if the algorithm is suited to the problem according to algo_dict.

        Args:
            algo_dict: The algorithm characteristics.
            problem: The problem to be solved.

        Returns:
            Whether the algorithm suits.
        """
        if not problem.is_symmetric and algo_dict.get(
            LinearSolverLib.LHS_MUST_BE_SYMMETRIC, False
        ):
            return False

        if not problem.is_positive_def and algo_dict.get(
            LinearSolverLib.LHS_MUST_BE_POSITIVE_DEFINITE, False
        ):
            return False

        if problem.is_lhs_linear_operator and not algo_dict.get(
            LinearSolverLib.LHS_CAN_BE_LINEAR_OPERATOR, False
        ):
            return False

        return True

    def _pre_run(
        self,
        problem,  # type: LinearProblem
        algo_name,  # type: str
        **options  # type: Any
    ):  # type: (...) -> None
        """Set the solver options and name in the problem attributes.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            options: The options for the algorithm, see associated JSON file.
        """
        problem.solver_options = options
        problem.solver_name = algo_name

    def _post_run(
        self,
        problem,  # type: LinearProblem
        algo_name,  # type: str
        result,  # type: ndarray
        **options  # type: Any
    ):  # type: (...) -> None # noqa: D107
        """Save the LinearProblem to the disk when required.

        If the save_when_fail option is True, save the LinearProblem to the disk when
        the system failed and print the file name in the warnings.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The result of the run, i.e. the solution.
            options: The options for the algorithm, see associated JSON file.
        """
        if options.get(self.SAVE_WHEN_FAIL, False):
            if not self.problem.is_converged:
                f_path = "linear_system_{}.pck".format(uuid4())
                pickle.dump(self.problem, open(f_path, "wb"))
                LOGGER.warning(
                    "Linear solver failed, saving problem to file: %s", f_path
                )
                self.save_fpath = f_path
