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

"""Jacobian checker for disciplines."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from gemseo.core.derivative.derivation_mode import DerivationMode
from gemseo.util.constant import N_CPUS
from gemseo.util.constant import READ_ONLY_EMPTY_DICT
from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.derivative.check.base import BaseJacobianChecker
from gemseo.util.derivative.derivatives_approx import DisciplineJacApprox

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.util.typing import StrKeyMapping
    from gemseo.util.typing import StrPath


@contextmanager
def _restore_discipline_state(discipline: Discipline) -> Iterator[None]:
    """Snapshot the data of a discipline on entry and restore it on exit.

    The numerical approximation of the Jacobian executes the discipline
    at perturbed points, which mutates its input and output data
    and its Jacobian matrices.
    This context manager snapshots them on entry and restores them on exit,
    even if an exception is raised,
    so that the analytic Jacobian and the data computed before entering
    the context (by the linearization) are preserved.

    Args:
        discipline: The discipline whose data is to be restored.

    Yields:
        Nothing.
    """
    # The Jacobian is saved by reference because linearize reassigns it
    # rather than mutating it in place.
    jac = discipline.jac
    input_data = discipline.io.input_data.copy()
    output_data = discipline.io.output_data.copy()
    try:
        yield
    finally:
        discipline.jac = jac
        discipline.io.input_data = input_data
        discipline.io.output_data = output_data


class DisciplineJacobianChecker(BaseJacobianChecker[str]):
    """Checks the Jacobian of a `Discipline` by numerical approximation."""

    _discipline: Discipline
    """The discipline whose Jacobian is to be checked."""

    def __init__(self, discipline: Discipline) -> None:
        """
        Args:
            discipline: The discipline whose Jacobian is to be checked.
        """  # noqa: D205, D212
        super().__init__()
        self._discipline = discipline

    def _prepare_io(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        """Return the input and output names to use for checking, with defaults applied.

        Args:
            input_names: The requested input names (empty means all).
            output_names: The requested output names (empty means all).

        Returns:
            The resolved input and output names.
        """
        if not input_names:
            input_names = self._discipline.io.input_grammar
        if not output_names:
            output_names = self._discipline.io.output_grammar
        return list(input_names), list(output_names)

    def check(
        self,
        input_value: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        atol: float = 1e-8,
        rtol: float = 1e-8,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        reference_jacobian_path: StrPath = "",
        save_reference_jacobian: bool = False,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float | None = 1e-7,
        n_processes: int = 1,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        linearization_mode: DerivationMode = DerivationMode.AUTO,
        plot_result: bool = False,
        file_path: StrPath = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10,
        fig_size_y: float = 10,
        indices: Mapping[
            str, int | Sequence[int] | Ellipsis | slice
        ] = READ_ONLY_EMPTY_DICT,
    ) -> bool:
        """Check the Jacobian.

        Args:
            input_value: The input at which to check the Jacobian.
                If empty, use the default input data of the discipline.
            inputs: The names of the inputs wrt which to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.
            linearization_mode: The mode of linearization: direct, adjoint
                or automated switch depending on dimensions of inputs and outputs.
            plot_result: Whether to plot the result of the validation
                (computed vs approximated Jacobians).
            file_path: The path to the output file if `plot_result` is `True`.
            show: Whether to open the figure.
            fig_size_x: The x-size of the figure in inches.
            fig_size_y: The y-size of the figure in inches.
            indices: The indices of the inputs and outputs
                for the different sub-Jacobian matrices,
                formatted as `{variable_name: variable_components}`
                where `variable_components` can be either
                an integer, e.g. `2`,
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0, 3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
                If empty, consider all the components of all the `inputs` and `outputs`.

        Returns:
            Whether the analytical Jacobian is correct
            with respect to the reference one.
        """
        if approximation_mode == ApproximationMode.COMPLEX_STEP and step is None:
            msg = (
                "The complex step approximation technique does not support "
                "the value None for the step parameter."
            )
            raise ValueError(msg)

        discipline = self._discipline
        inputs, outputs = self._prepare_io(inputs, outputs)

        discipline.add_differentiated_inputs(inputs)
        discipline.add_differentiated_outputs(outputs)
        discipline.linearization_mode = linearization_mode

        approx = DisciplineJacApprox(
            discipline,
            approx_method=approximation_mode,
            parallel=n_processes != 1,
            n_processes=N_CPUS if n_processes == 0 else n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            **({} if step is None else {"step": step}),
        )
        if step is None:
            approx.auto_set_step(outputs, inputs)

        discipline.linearize(input_value)

        # Revert the perturbations of the numerical approximation
        with _restore_discipline_state(discipline):
            return approx.check_jacobian(
                outputs,
                inputs,
                atol=atol,
                rtol=rtol,
                plot_result=plot_result,
                file_path=file_path,
                show=show,
                fig_size_x=fig_size_x,
                fig_size_y=fig_size_y,
                reference_jacobian_path=reference_jacobian_path,
                save_reference_jacobian=save_reference_jacobian,
                indices=indices,
                input_data=input_value,
            )
