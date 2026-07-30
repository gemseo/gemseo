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

"""Abstract base class for Jacobian correctness checkers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import StrPath

T = TypeVar("T", bound=int | str)


class BaseJacobianChecker(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for Jacobian correctness checkers."""

    @abstractmethod
    def check(
        self,
        input_value: Any,
        atol: float = 1e-8,
        rtol: float = 1e-8,
        inputs: Iterable[T] = (),
        outputs: Iterable[T] = (),
        reference_jacobian_path: StrPath = "",
        save_reference_jacobian: bool = False,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float | None = 1e-7,
        n_processes: int = 1,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
    ) -> bool:
        """Check the Jacobian.

        Args:
            input_value: The input at which to check the Jacobian.
            atol: The absolute tolerance.
            rtol: The relative tolerance.
            inputs: The inputs wrt which to differentiate the outputs.
            outputs: The outputs to be differentiated.
            reference_jacobian_path: The path of the reference Jacobian file.
                If empty, compute the reference Jacobian numerically.
            save_reference_jacobian: Whether to save the reference Jacobian
                to `reference_jacobian_path`.
            approximation_mode: The numerical differentiation method.
            step: The step of the numerical differentiation method.
                If `None`, an optimal step will be used.
                The latter is not compatible with
                `approximation_mode=ApproximationMode.COMPLEX_STEP`.
            n_processes: The maximum number of threads to run simultaneously
                if `use_threading` is `True`,
                or processes otherwise,
                used to parallelize the execution.
                If `0`, use the number of CPUs available on the system.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the data,
                while threading will share all the memory.
            wait_time_between_fork: The time waited between two forks
                of the process or thread.

        Returns:
            Whether the analytical Jacobian is correct.
        """
