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
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Finite differences approximation."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Sized
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import atleast_1d
from scipy.sparse import hstack as sparse_hstack

from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.constants import N_CPUS
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.error_estimators import EPSILON
from gemseo.utils.derivatives.factory import GradientApproximatorFactory
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from numbers import Number

    from matplotlib.pyplot import Figure

    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import JacobianData
    from gemseo.utils.derivatives.base_gradient_approximator import (
        BaseGradientApproximator,
    )

from matplotlib import pyplot as plt
from numpy import absolute
from numpy import allclose
from numpy import amax
from numpy import arange
from numpy import atleast_2d
from numpy import concatenate
from numpy import divide
from numpy import ndarray
from numpy import zeros

LOGGER = logging.getLogger(__name__)


# TODO: API: rename to JacobianApproximator?
class DisciplineJacApprox:
    """Approximates a discipline Jacobian using finite differences or Complex step."""

    approximator: BaseGradientApproximator | None
    """The gradient approximation method."""

    generator_class: ClassVar[type[DisciplineAdapterGenerator]] = (
        DisciplineAdapterGenerator
    )
    """The generator class used to create ``MDOFunction`` from an ``Discipline``."""

    def __init__(
        self,
        discipline: BaseDiscipline,
        approx_method: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: Number | Iterable[Number] = 1e-7,
        parallel: bool = False,
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0,
    ) -> None:
        """
        Args:
            discipline: The discipline
                for which the Jacobian approximation shall be made.
            approx_method: The approximation method,
                either ``complex_step`` or ``finite_differences``.
            step: The differentiation step. The ``finite_differences`` takes either
                a float or an iterable of floats with the same length as the inputs.
                The ``complex_step`` method takes either a complex or a float as input.
            parallel: Whether to differentiate the discipline in parallel.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            wait_time_between_fork: The time waited between two forks
                of the process / thread.
        """  # noqa:D205 D212 D415
        self.discipline = discipline
        self.approx_method = approx_method
        self.step = step
        self.generator = self.generator_class(discipline)
        self.func = None
        self.approximator = None
        self.auto_steps = {}
        self.__par_args = {
            "n_processes": n_processes,
            "use_threading": use_threading,
            "wait_time_between_fork": wait_time_between_fork,
        }
        self.__parallel = parallel

    def _create_approximator(
        self,
        output_names: Sequence[str],
        input_names: Sequence[str],
        input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """Create the Jacobian approximation class.

        Args:
            input_names: The names of the inputs used to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.
            input_data: The input data for the Jacobian approximation.

        Raises:
            ValueError: If the Jacobian approximation method is unknown.
        """
        self.func = self.generator.get_function(
            input_names,
            output_names,
            input_data or None,
        )
        self.approximator = GradientApproximatorFactory().create(
            self.approx_method,
            self.func.evaluate,
            step=atleast_1d(self.step),
            parallel=self.__parallel,
            **self.__par_args,
        )

    def auto_set_step(
        self,
        output_names: Sequence[str],
        input_names: Sequence[str],
        print_errors: bool = True,
        numerical_error: float = EPSILON,
    ) -> tuple[ndarray, dict[str, ndarray]]:
        r"""Compute the optimal step.

        Require a first evaluation of the perturbed functions values.

        The optimal step is reached when the truncation error
        (cut in the Taylor development),
        and the numerical cancellation errors
        (round-off when doing :math:`f(x+step)-f(x))` are equal.

        Args:
            input_names: The names of the inputs used to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.
            print_errors: Whether to log the cancellation
                and truncation error estimates.
            numerical_error: The numerical error
                associated to the calculation of :math:`f`.
                By default, Machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.

        See Also:
            https://en.wikipedia.org/wiki/Numerical_differentiation
            and *Numerical Algorithms and Digital Representation*,
            Knut Morken, Chapter 11, "Numerical Differentiation"

        Returns:
            The Jacobian of the function.
        """
        self._create_approximator(output_names, input_names)

        with self.__set_zero_cache_tol():
            compute_opt_step = self.approximator.compute_optimal_step

        x_vect = self._prepare_xvect(
            input_names, self.discipline.io.input_grammar.defaults
        )
        steps_opt, errors = compute_opt_step(x_vect, numerical_error=numerical_error)

        if print_errors:
            LOGGER.info(
                "Set optimal step for finite differences. "
                "Estimated approximation errors ="
            )
            LOGGER.info(errors)

        data = self.discipline.io.input_grammar.defaults or self.discipline.io.data
        names_to_slices = (
            self.discipline.io.input_grammar.data_converter.compute_names_to_slices(
                input_names,
                data,
            )[0]
        )

        self.auto_steps = (
            self.discipline.io.input_grammar.data_converter.convert_array_to_data(
                steps_opt, names_to_slices
            )
        )

        return errors, self.auto_steps

    @contextmanager
    def __set_zero_cache_tol(self) -> None:
        """A context manager to temporary set the discipline cache tolerance to zero."""
        if self.discipline.cache is not None:
            old_cache_tol = self.discipline.cache.tolerance
            self.discipline.cache.tolerance = 0.0
            yield
            self.discipline.cache.tolerance = old_cache_tol
        else:
            yield

    def _prepare_xvect(
        self,
        input_names: Iterable[str],
        data: DisciplineData = READ_ONLY_EMPTY_DICT,
    ) -> ndarray:
        """Convert an input data mapping into an input array.

        Args:
            input_names: The names of the inputs to be used for the differentiation.
            data: The input data mapping.
                If empty, use the local data of the discipline.

        Returns:
            The input array.
        """
        if not data:
            data = self.discipline.io.data

        return self.discipline.io.input_grammar.data_converter.convert_data_to_array(
            input_names,
            data,
        )

    def compute_approx_jac(
        self,
        output_names: Iterable[str],
        input_names: Iterable[str],
        x_indices: Sequence[int] = (),
        input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
    ) -> dict[str, dict[str, ndarray]]:
        """Approximate the Jacobian.

        Args:
            output_names: The names of the outputs to be differentiated.
            input_names: The names of the inputs used to differentiate the outputs.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If empty, use all the components.
            input_data: The input data needed to approximate the Jacobian
                according to the discipline input grammar.
                If empty, use the :attr:`.Discipline.default_input_data`.

        Returns:
            The approximated Jacobian.
        """
        self._create_approximator(output_names, input_names, input_data)

        if self.auto_steps and all(key in self.auto_steps for key in input_names):
            step = (
                self.discipline.io.input_grammar.data_converter.convert_data_to_array(
                    input_names,
                    self.auto_steps,
                )
            )
        else:
            step = self.step

        x_vect = self._prepare_xvect(input_names, self.discipline.io.data)

        if isinstance(step, Sized) and 1 < len(step) != len(x_vect):
            msg = f"Inconsistent step size, expected {x_vect.size} got {len(step)}."
            raise ValueError(msg)

        with self.__set_zero_cache_tol():
            flat_jac = atleast_2d(
                self.approximator.f_gradient(x_vect, x_indices=x_indices, step=step)
            )

        data_names_to_sizes = (
            self.discipline.io.output_grammar.data_converter.compute_names_to_sizes(
                output_names,
                self.discipline.io.data,
            )
        )
        input_names_to_sizes = (
            self.discipline.io.input_grammar.data_converter.compute_names_to_sizes(
                input_names,
                self.discipline.io.data,
            )
        )

        if not x_indices:
            flat_jac_complete = flat_jac
        else:
            flat_jac_complete = zeros([
                sum(data_names_to_sizes.values()),
                sum(input_names_to_sizes.values()),
            ])
            flat_jac_complete[:, x_indices] = flat_jac

        data_names_to_sizes.update(input_names_to_sizes)

        return split_array_to_dict_of_arrays(
            flat_jac_complete, data_names_to_sizes, output_names, input_names
        )

    def check_jacobian(
        self,
        output_names: Iterable[str],
        input_names: Iterable[str],
        analytic_jacobian: JacobianData = READ_ONLY_EMPTY_DICT,
        threshold: float = 1e-8,
        plot_result: bool = False,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10.0,
        fig_size_y: float = 10.0,
        reference_jacobian_path: str | Path = "",
        save_reference_jacobian: bool = False,
        indices: Mapping[
            str, int | Sequence[int] | Ellipsis | slice
        ] = READ_ONLY_EMPTY_DICT,
        input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
    ) -> bool:
        """Check if the analytical Jacobian is correct with respect to a reference one.

        If `reference_jacobian_path` is not empty
        and `save_reference_jacobian` is `True`,
        compute the reference Jacobian with the approximation method
        and save it in `reference_jacobian_path`.

        If `reference_jacobian_path` is not empty
        and `save_reference_jacobian` is `False`,
        do not compute the reference Jacobian
        but read it from `reference_jacobian_path`.

        If `reference_jacobian_path` is `None`,
        compute the reference Jacobian without saving it.

        Args:
            input_names: The names of the inputs used to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.
            analytic_jacobian: The Jacobian to validate.
            threshold: The acceptance threshold for the Jacobian error.
            plot_result: Whether to plot the result of the validation
                (computed vs approximated Jacobians).
            file_path: The path to the output file if ``plot_result`` is ``True``.
            show: Whether to open the figure.
            fig_size_x: The x-size of the figure in inches.
            fig_size_y: The y-size of the figure in inches.
            reference_jacobian_path: The path of the reference Jacobian file.
            save_reference_jacobian: Whether to save the reference Jacobian.
            indices: The indices of the inputs and outputs
                for the different sub-Jacobian matrices,
                formatted as ``{variable_name: variable_components}``
                where ``variable_components`` can be either
                an integer, e.g. `2`
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0,3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
                If ``None``,
                consider all the components of all the ``inputs`` and ``outputs``.
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
                If empty, use the :attr:`.Discipline.default_input_data`.

        Returns:
            Whether the analytical Jacobian is correct.
        """
        input_names_to_indices = {}
        input_indices = {}

        if indices:
            input_indices, input_names_to_indices = self._compute_variable_indices(
                indices,
                input_names,
                self.discipline.io.input_grammar.data_converter.compute_names_to_sizes(
                    input_names,
                    self.discipline.io.input_grammar.defaults,
                ),
            )

        # TODO: fix or make it work without side effects.
        # This statement must be done before calling compute_approx_jac because
        # it will overwrite the discipline jac.
        analytic_jacobian = analytic_jacobian or self.discipline.jac

        if not reference_jacobian_path or save_reference_jacobian:
            for input_name in tuple(input_data):
                if input_name not in self.discipline.input_grammar.names:
                    del input_data[input_name]
            approximated_jacobian = self.compute_approx_jac(
                output_names, input_names, input_indices, input_data
            )
        else:
            with Path(reference_jacobian_path).open("rb") as infile:
                approximated_jacobian = pickle.load(infile)

        if save_reference_jacobian:
            with Path(reference_jacobian_path).open("wb") as outfile:
                pickle.dump(approximated_jacobian, outfile)

        output_names_to_indices = {}

        if indices:
            output_sizes = {
                output_name: output_jacobian[next(iter(output_jacobian))].shape[0]
                for output_name, output_jacobian in approximated_jacobian.items()
            }
            _, output_names_to_indices = self._compute_variable_indices(
                indices, output_names, output_sizes
            )

        if not input_names_to_indices:
            input_names_to_indices = Ellipsis

        if not output_names_to_indices:
            output_names_to_indices = Ellipsis

        succeed = True

        for output_name, output_jacobian in approximated_jacobian.items():
            for input_name, approx_jac in output_jacobian.items():
                computed_jac = analytic_jacobian[output_name][input_name]
                if indices:
                    row_idx = atleast_2d(output_names_to_indices[output_name]).T
                    col_idx = input_names_to_indices[input_name]
                    computed_jac = computed_jac[row_idx, col_idx]
                    approx_jac = approx_jac[row_idx, col_idx]

                if approx_jac.shape != computed_jac.shape:
                    succeed = False
                    msg = (
                        f"{self.discipline.name} Jacobian: dp {output_name}/dp "
                        f"{input_name} is of wrong shape; "
                        f"got: {computed_jac.shape} while expected: {approx_jac.shape}."
                    )
                    LOGGER.error(msg)
                else:
                    if isinstance(computed_jac, sparse_classes):
                        computed_jac = computed_jac.toarray()

                    success_loc = allclose(
                        computed_jac, approx_jac, atol=threshold, rtol=threshold
                    )
                    if not success_loc:
                        err = amax(
                            divide(
                                absolute(computed_jac - approx_jac),
                                absolute(approx_jac) + 1.0,
                            )
                        )
                        LOGGER.error(
                            "%s Jacobian: dp %s/d %s is wrong by %s%%.",
                            self.discipline.name,
                            output_name,
                            input_name,
                            err * 100.0,
                        )
                        LOGGER.info("Approximate jacobian = \n%s", approx_jac)
                        LOGGER.info("Provided by linearize method = \n%s", computed_jac)
                        LOGGER.info(
                            "Difference of jacobians = \n%s", approx_jac - computed_jac
                        )
                        succeed = succeed and success_loc
                    else:
                        LOGGER.info(
                            "Jacobian: dp %s/dp %s succeeded.", output_name, input_name
                        )

        LOGGER.info(
            "Linearization of Discipline: %s is %s.",
            self.discipline.name,
            "correct" if succeed else "wrong",
        )

        if plot_result:
            self.plot_jac_errors(
                analytic_jacobian,
                approximated_jacobian,
                file_path,
                show,
                fig_size_x,
                fig_size_y,
            )
        return succeed

    @staticmethod
    def _compute_variable_indices(
        indices: Mapping[str, int | Sequence[int] | Ellipsis | slice],
        variable_names: Iterable[str],
        variable_sizes: Mapping[str, int],
    ) -> tuple[list[int], dict[str, int]]:
        """Return indices.

        Args:
            indices: The indices for variables
                formatted as ``{variable_name: variable_components}``
                where ``variable_components`` can be either
                an integer, e.g. `2`
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0,3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
            variable_names: The names of the variables.

        Returns:
            The indices of the variables.
        """
        indices_sequence = []
        names_to_indices = {}
        variable_position = 0
        for variable_name in variable_names:
            variable_size = variable_sizes[variable_name]
            variable_indices = list(range(variable_size))
            indices_sequence.append(indices.get(variable_name, variable_indices))

            if isinstance(indices_sequence[-1], int):
                indices_sequence[-1] = [indices_sequence[-1]]

            if isinstance(indices_sequence[-1], slice):
                indices_sequence[-1] = variable_indices[indices_sequence[-1]]

            if indices_sequence[-1] in (Ellipsis, None):
                indices_sequence[-1] = variable_indices

            names_to_indices[variable_name] = indices_sequence[-1]
            indices_sequence[-1] = [
                variable_index + variable_position
                for variable_index in indices_sequence[-1]
            ]
            variable_position += variable_size

        indices_sequence = [item for sublist in indices_sequence for item in sublist]
        return indices_sequence, names_to_indices

    @staticmethod
    def __concatenate_jacobian_per_output_names(
        analytic_jacobian: JacobianData,
        approximated_jacobian: JacobianData,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray], list[str]]:
        """Concatenate the Jacobian matrices per output name.

        Args:
            analytic_jacobian: The reference Jacobian
                of the form ``{output_name: {input_name: sub_jacobian}}``.
            approximated_jacobian: The approximated Jacobian
                of the form ``{output_name: {input_name: sub_jacobian}}``.

        Returns:
            The analytic Jacobian of the form ``{output_name: sub_jacobian}``,
            the approximated Jacobian of the form ``{output_name: sub_jacobian}``
            and the names of the output components
            corresponding to the columns of ``sub_jacobian``.
        """
        approx_jacobian = {}
        analytic_jacobian_ = {}
        jacobian = analytic_jacobian[next(iter(analytic_jacobian))]
        input_names = list(jacobian.keys())
        input_component_names = [
            f"{input_name}_{i + 1}"
            for input_name in input_names
            for i in range(jacobian[input_name].shape[1])
        ]
        for output_name, output_approximated_jacobian in approximated_jacobian.items():
            output_approx_jacobian = concatenate(
                [
                    output_approximated_jacobian[input_name]
                    for input_name in input_names
                ],
                axis=1,
            )

            analytic_jacobian_out = analytic_jacobian[output_name]
            contains_sparse = any(
                isinstance(analytic_jacobian_out[input_name], sparse_classes)
                for input_name in input_names
            )

            if contains_sparse:
                output_analytic_jacobian = sparse_hstack(
                    [analytic_jacobian_out[input_name] for input_name in input_names],
                ).tocsr()
            else:
                output_analytic_jacobian = concatenate(
                    [analytic_jacobian_out[input_name] for input_name in input_names],
                    axis=1,
                )

            n_f = len(output_approx_jacobian)

            if n_f == 1:
                approx_jacobian[output_name] = output_approx_jacobian.flatten()

                if contains_sparse:
                    output_analytic_jacobian = output_analytic_jacobian.toarray()

                analytic_jacobian_[output_name] = output_analytic_jacobian.flatten()
            else:
                for i in range(n_f):
                    output_name = f"{output_name}_{i}"
                    approx_jacobian[output_name] = output_approx_jacobian[i, :]
                    analytic_jacobian_[output_name] = output_analytic_jacobian[i, :]

        return analytic_jacobian_, approx_jacobian, input_component_names

    def plot_jac_errors(
        self,
        computed_jac: JacobianData,
        approx_jac: JacobianData,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10.0,
        fig_size_y: float = 10.0,
    ) -> Figure:
        """Generate a plot of the exact vs approximated Jacobian.

        Args:
            computed_jac: The Jacobian to validate.
            approx_jac: The approximated Jacobian.
            file_path: The path to the output file if ``plot_result`` is ``True``.
            show: Whether to open the figure.
            fig_size_x: The x-size of the figure in inches.
            fig_size_y: The y-size of the figure in inches.
        """
        comp_grad, app_grad, x_labels = self.__concatenate_jacobian_per_output_names(
            computed_jac, approx_jac
        )
        n_funcs = len(app_grad)
        if n_funcs == 0:
            msg = "No gradients to plot!"
            raise ValueError(msg)
        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1
        ncols = 2
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=2,
            sharex=True,
            figsize=(fig_size_x, fig_size_y),
            squeeze=False,
        )
        i = 0
        j = -1

        n_subplots = 2 * nrows
        abscissa = arange(len(x_labels))
        for func, grad in sorted(comp_grad.items()):
            if isinstance(grad, sparse_classes):
                grad = grad.toarray().flatten()
            j += 1
            if j == ncols:
                j = 0
                i += 1
            ax = axs[i][j]
            ax.plot(abscissa, grad, "bo")
            ax.plot(abscissa, app_grad[func], "ro")
            ax.set_title(func)
            ax.set_xticklabels(x_labels, fontsize=14)
            ax.set_xticks(abscissa)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

            # Update y labels spacing
            vis_labels = [
                label for label in ax.get_yticklabels() if label.get_visible() is True
            ]
            plt.setp(vis_labels[::2], visible=False)
        #             plt.xticks(rotation=90)

        if len(comp_grad.items()) < n_subplots:
            # xlabel must be written with the same fontsize on the 2 columns
            j += 1
            ax = axs[i][j]
            ax.set_xticklabels(x_labels, fontsize=14)
            ax.set_xticks(abscissa)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

        fig.suptitle(
            "Computed and approximate derivatives. "
            " blue = computed, red = approximated derivatives",
            fontsize=14,
        )
        save_show_figure(fig, show, file_path)
        return fig
