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

"""Jacobian checker for array functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_2d
from numpy import load as np_load
from numpy import save as np_save

from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.constants import N_CPUS
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.approximators.factory import GradientApproximatorFactory
from gemseo.utils.derivatives.check.base import BaseJacobianChecker
from gemseo.utils.derivatives.derivatives_approx import compare_jacobian_matrices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.core.functions.array_function import ArrayFunction
    from gemseo.typing import StrPath


class FunctionJacobianChecker(BaseJacobianChecker[int]):
    """Checks the Jacobian of an `ArrayFunction` by numerical approximation."""

    __function: ArrayFunction
    """The function whose Jacobian is to be checked."""

    def __init__(self, function: ArrayFunction) -> None:
        """
        Args:
            function: The function whose Jacobian is to be checked.
        """  # noqa: D205, D212
        super().__init__()
        self.__function = function

    def check(
        self,
        input_value: ndarray,
        atol: float = 1e-8,
        rtol: float = 1e-8,
        inputs: Sequence[int] = (),
        outputs: Sequence[int] = (),
        reference_jacobian_path: StrPath = "",
        save_reference_jacobian: bool = False,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float | None = 1e-7,
        n_processes: int = 1,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
    ) -> bool:
        """Check the Jacobian at the given input vector.

        Args:
            inputs: The indices of the input vector components
                to include in the Jacobian check.
                If empty, check all components.
            outputs: The indices of the output vector components
                to include in the Jacobian check.
                If empty, check all components.

        Returns:
            Whether the analytical Jacobian is correct.

        Raises:
            ValueError: If the shapes of the analytical and approximated Jacobian
                matrices are inconsistent.
            NotImplementedError: If ``step`` is ``None``.
        """
        if step is None:
            msg = "FunctionJacobianChecker does not support step=None."
            raise NotImplementedError(msg)

        gradient_approximator = GradientApproximatorFactory().create(
            approximation_mode,
            self.__function.evaluate,
            step=step,
            parallel=n_processes != 1,
            n_processes=N_CPUS if n_processes == 0 else n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
        )

        jacobian = self.__function.jac(input_value).real

        if isinstance(jacobian, sparse_classes):
            jacobian = jacobian.toarray()

        if not reference_jacobian_path or save_reference_jacobian:
            approximated_jacobian = gradient_approximator.f_gradient(input_value).real
        else:
            approximated_jacobian = np_load(reference_jacobian_path)

        if save_reference_jacobian:
            np_save(reference_jacobian_path, approximated_jacobian)

        if inputs:
            approximated_jacobian = approximated_jacobian[..., inputs]
            jacobian = jacobian[..., inputs]

        if outputs:
            approximated_jacobian = approximated_jacobian[outputs, :]
            jacobian = jacobian[outputs, :]

        name = self.__function.name
        if atleast_2d(approximated_jacobian).shape != atleast_2d(jacobian).shape:
            label = f" computed by {name}" if name else ""
            msg = (
                f"The Jacobian matrix{label} has a wrong shape; "
                f"expected {approximated_jacobian.shape}, got {jacobian.shape}."
            )
            raise ValueError(msg)

        return compare_jacobian_matrices(
            jacobian, approximated_jacobian, atol=atol, rtol=rtol, name=name
        )
