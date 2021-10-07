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
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Finite differences approximation."""
from __future__ import division, unicode_literals

import logging
import pickle
from itertools import chain
from multiprocessing import cpu_count
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from gemseo.utils.derivatives.gradient_approximator import GradientApproximationFactory

if TYPE_CHECKING:
    from gemseo.core.discipline import MDODiscipline

from matplotlib import pyplot
from matplotlib.pyplot import Figure
from numpy import (
    absolute,
    allclose,
    amax,
    arange,
    array,
    atleast_2d,
    concatenate,
    divide,
    finfo,
    ndarray,
)

from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import Path, xrange

EPSILON = finfo(float).eps
LOGGER = logging.getLogger(__name__)


class DisciplineJacApprox(object):
    """Approximates a discipline Jacobian using finite differences or Complex step."""

    COMPLEX_STEP = "complex_step"
    FINITE_DIFFERENCES = "finite_differences"

    N_CPUS = cpu_count()

    def __init__(
        self,
        discipline,  # type: MDODiscipline
        approx_method=FINITE_DIFFERENCES,  # type: str
        step=1e-7,  # type: float
        parallel=False,  # type: bool
        n_processes=N_CPUS,  # type: int
        use_threading=False,  # type: bool
        wait_time_between_fork=0,  # type: float
    ):
        """
        Args:
            discipline: The discipline
                for which the Jacobian approximation shall be made.
            approx_method: The approximation method,
                either "complex_step" or "finite_differences".
            step: The differentiation step.
            parallel: Whether to differentiate the discipline in parallel.
            n_processes: The maximum number of processors on which to run.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            wait_time_between_fork: The time waited between two forks
                of the process / thread.
        """
        from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator

        self.discipline = discipline
        self.approx_method = approx_method
        self.step = step
        self.generator = MDOFunctionGenerator(discipline)
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
        outputs,  # type: Sequence[str]
        inputs,  # type: Sequence[str]
    ):
        """Create the Jacobian approximation class.

        Args:
            inputs: The names of the inputs used to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.

        Raises:
            ValueError: If the Jacobian approximation method is unknown.
        """
        self.func = self.generator.get_function(
            input_names_list=inputs, output_names_list=outputs
        )
        if self.approx_method not in [self.FINITE_DIFFERENCES, self.COMPLEX_STEP]:
            raise ValueError(
                "Unknown Jacobian approximation method {}.".format(self.approx_method)
            )
        factory = GradientApproximationFactory()
        self.approximator = factory.create(
            self.approx_method,
            self.func,
            step=self.step,
            parallel=self.__parallel,
            **self.__par_args
        )

    def auto_set_step(
        self,
        outputs,  # type: Sequence[str]
        inputs,  # type: Sequence[str]
        print_errors=True,  # type: bool
        numerical_error=EPSILON,  # type: float
    ):  # type: (...) -> ndarray
        r"""Compute the optimal step.

        Require a first evaluation of the perturbed functions values.

        The optimal step is reached when the truncation error
        (cut in the Taylor development),
        and the numerical cancellation errors
        (round-off when doing :math:`f(x+step)-f(x))` are equal.

        See:
        - https://en.wikipedia.org/wiki/Numerical_differentiation
        - *Numerical Algorithms and Digital Representation*,
          Knut Morken, Chapter 11, "Numerical Differenciation"

        Args:
            inputs: The names of the inputs used to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.
            print_errors: Whether to log the cancellation
                and truncation error estimates.
            numerical_error: The numerical error
                associated to the calculation of :math:`f`.
                By default Machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.

        Returns:
            The Jacobian of the function.
        """
        self._create_approximator(outputs, inputs)
        old_cache_tol = self.discipline.cache_tol
        self.discipline.cache_tol = 0.0
        x_vect = self._prepare_xvect(inputs, self.discipline.default_inputs)
        compute_opt_step = self.approximator.compute_optimal_step
        steps_opt, errors = compute_opt_step(x_vect, numerical_error=numerical_error)
        if print_errors:
            LOGGER.info(
                "Set optimal step for finite differences. "
                "Estimated approximation errors ="
            )
            LOGGER.info(errors)
        self.discipline.cache_tol = old_cache_tol
        data = self.discipline.local_data
        data_sizes = {key: val.size for key, val in data.items()}
        self.auto_steps = DataConversion.array_to_dict(steps_opt, inputs, data_sizes)
        return errors, self.auto_steps

    def _prepare_xvect(
        self,
        inputs,  # type: Iterable[str]
        data=None,  # type: Optional[Dict[str,ndarray]]
    ):  # type: (...) -> ndarray
        """Convert a input data mapping into an input array.

        Args:
            inputs: The names of the inputs to be used for the differentiation.
            data: The input data mapping.
                If None, use the local data of the discipline.

        Returns:
            The input array.
        """
        if data is None:
            data = self.discipline.local_data
        x_vect = DataConversion.dict_to_array(data, inputs)
        return x_vect

    def compute_approx_jac(
        self,
        outputs,  # type:Iterable[str]
        inputs,  # type:Iterable[str]
        x_indices=None,  # type: Optional[Sequence[int]]
    ):  # type: (...) -> Dict[str,Dict[str,ndarray]]
        """Approximate the Jacobian.

        Args:
            inputs: The names of the inputs used to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If None, use all the components.

        Returns:
            The approximated Jacobian.
        """
        self._create_approximator(outputs, inputs)
        old_cache_tol = self.discipline.cache_tol
        self.discipline.cache_tol = 0.0
        local_data = self.discipline.local_data
        x_vect = self._prepare_xvect(inputs)
        if (
            self.auto_steps is not None
            and array([key in self.auto_steps for key in inputs]).all()
        ):
            step = concatenate([self.auto_steps[key] for key in inputs])
        else:
            step = self.step

        if hasattr(step, "len") and 1 < len(step) != len(x_vect):
            raise ValueError(
                "Inconsistent step size, "
                "expected {} got {}.".format(x_vect.size, len(step))
            )
        flat_jac = self.approximator.f_gradient(x_vect, x_indices=x_indices, step=step)
        flat_jac = atleast_2d(flat_jac)
        data_sizes = {key: len(local_data[key]) for key in chain(inputs, outputs)}
        self.discipline.cache_tol = old_cache_tol
        return DataConversion.jac_2dmat_to_dict(flat_jac, outputs, inputs, data_sizes)

    def check_jacobian(
        self,
        analytic_jacobian,  # type: Dict[str,Dict[str,ndarray]]
        outputs,  # type: Iterable[str]
        inputs,  # type: Iterable[str]
        discipline,  # type: MDODiscipline
        threshold=1e-8,  # type: float
        plot_result=False,  # type: bool
        file_path="jacobian_errors.pdf",  # type: Union[str,Path]
        show=False,  # type: bool
        figsize_x=10,  # type: int
        figsize_y=10,  # type: int
        reference_jacobian_path=None,  # type: Optional[Union[str,Path]]
        save_reference_jacobian=False,  # type: bool
        indices=None,  # type: Optional[Union[int,Sequence[int],slice,Ellipsis]]
    ):  # type: (...) -> bool
        """Check if the analytical Jacobian is correct with respect to a reference one.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `True`,
        compute the reference Jacobian with the approximation method
        and save it in `reference_jacobian_path`.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `False`,
        do not compute the reference Jacobian
        but read it from `reference_jacobian_path`.

        If `reference_jacobian_path` is `None`,
        compute the reference Jacobian without saving it.

        Args:
            analytic_jacobian: The Jacobian to validate.
            inputs: The names of the inputs used to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.
            threshold: The acceptance threshold for the Jacobian error.
            plot_result: Whether to plot the result of the validation
                (computed vs approximated Jacobians).
            file_path: The path to the output file if ``plot_result`` is ``True``.
            show: Whether to open the figure.
            figsize_x: The x-size of the figure in inches.
            figsize_y: The y-size of the figure in inches.
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
                If None,
                consider all the components of all the ``inputs`` and ``outputs``.

        Returns:
            Whether the analytical Jacobian is correct.
        """
        inputs_indices = input_indices = outputs_indices = output_indices = None
        if indices is not None:
            input_indices, inputs_indices = self._compute_variables_indices(
                indices,
                inputs,
                {name: len(self.discipline.default_inputs[name]) for name in inputs},
            )

        if reference_jacobian_path is None or save_reference_jacobian:
            approx_jac_complete = self.compute_approx_jac(
                outputs, inputs, input_indices
            )
        else:
            with Path(reference_jacobian_path).open("rb") as infile:
                approx_jac_complete = pickle.load(infile)

        if save_reference_jacobian:
            with Path(reference_jacobian_path).open("wb") as outfile:
                pickle.dump(approx_jac_complete, outfile)

        name = discipline.name
        succeed = True

        if indices is not None:
            outputs_sizes = {
                output_name: apprx_jac_dict[next(iter(apprx_jac_dict))].shape[0]
                for output_name, apprx_jac_dict in approx_jac_complete.items()
            }
            output_indices, outputs_indices = self._compute_variables_indices(
                indices, outputs, outputs_sizes
            )

        if inputs_indices is None:
            inputs_indices = Ellipsis

        if outputs_indices is None:
            outputs_indices = Ellipsis

        for out_data, apprx_jac_dict in approx_jac_complete.items():
            for in_data, approx_jac in apprx_jac_dict.items():
                computed_jac = analytic_jacobian[out_data][in_data]
                if indices is not None:
                    computed_jac = computed_jac[
                        outputs_indices[out_data], inputs_indices[in_data]
                    ]
                    approx_jac = approx_jac[
                        outputs_indices[out_data], inputs_indices[in_data]
                    ]

                if approx_jac.shape != computed_jac.shape:
                    succeed = False
                    msg = (
                        "{} Jacobian: dp {}/dp {} is of wrong shape; "
                        "got: {} while expected: {}.".format(
                            name,
                            out_data,
                            in_data,
                            computed_jac.shape,
                            approx_jac.shape,
                        )
                    )
                    LOGGER.error(msg)
                else:
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
                        msg = "{} Jacobian: dp {}/d {} is wrong by {}%.".format(
                            name, out_data, in_data, err * 100.0
                        )
                        LOGGER.error(msg)
                        LOGGER.info("Approximate jacobian = \n%s", approx_jac)
                        LOGGER.info(
                            "Provided by linearize method = \n{}%s", computed_jac
                        )
                        LOGGER.info(
                            "Difference of jacobians = \n%s", approx_jac - computed_jac
                        )
                        succeed = succeed and success_loc
                    else:
                        LOGGER.info(
                            "Jacobian:  dp %s/dp %s succeeded!", out_data, in_data
                        )
        if succeed:
            LOGGER.info("Linearization of MDODiscipline: %s is correct.", name)
        else:
            LOGGER.info("Linearization of MDODiscipline: %s is wrong.", name)

        if plot_result:
            self.plot_jac_errors(
                analytic_jacobian,
                approx_jac_complete,
                file_path,
                show,
                figsize_x,
                figsize_y,
            )
        return succeed

    @staticmethod
    def _compute_variables_indices(
        indices,  # type: Mapping[str,Union[int,Sequence[int],Ellipsis,slice]]
        variables_names,  # type: Iterable[str]
        variables_sizes,  # type: Mapping[str,int]
    ):  # type: (...) -> List[int]
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
            variables_names: The names of the variables.

        Returns:
            The indices of the variables.
        """
        indices_sequence = []
        variables_indices = {}
        variable_position = 0
        for variable_name in variables_names:
            variable_size = variables_sizes[variable_name]
            variable_indices = list(range(variable_size))
            indices_sequence.append(indices.get(variable_name, variable_indices))

            if isinstance(indices_sequence[-1], int):
                indices_sequence[-1] = [indices_sequence[-1]]

            if isinstance(indices_sequence[-1], slice):
                indices_sequence[-1] = variable_indices[indices_sequence[-1]]

            if indices_sequence[-1] in [Ellipsis, None]:
                indices_sequence[-1] = variable_indices

            variables_indices[variable_name] = indices_sequence[-1]
            indices_sequence[-1] = [
                variable_index + variable_position
                for variable_index in indices_sequence[-1]
            ]
            variable_position += variable_size

        indices_sequence = [item for sublist in indices_sequence for item in sublist]
        return indices_sequence, variables_indices

    @staticmethod
    def __format_jac_as_grad_dict(
        computed_jac,  # type: Dict[str,Dict[str,ndarray]]
        approx_jac,  # type: Dict[str,Dict[str,ndarray]]
    ):  # type: (...) -> Tuple[Dict[str,ndarray],Dict[str,ndarray],List[str]]
        """Format the approximate Jacobian dictionaries as a dictionary of gradients.

        Args:
            computed_jac: The reference computed Jacobian dictionary of dictionaries.
            approx_jac: The dictionary of of gradients.

        Returns:
            grad dict, approx dict, and design var names
        """
        approx_grad_dict = {}
        computed_grad_dict = {}
        in_names = None
        for out_data, apprx_jac_dict in approx_jac.items():
            com_jac_dict = computed_jac[out_data]
            approx_grad = []
            computed_grad = []

            if in_names is None:
                in_names = list(iter(computed_jac[out_data].keys()))
                x_names = [
                    inp + "_" + str(i + 1)
                    for inp in in_names
                    for i in xrange(apprx_jac_dict[inp].shape[1])
                ]

            for in_data in in_names:
                approx_grad.append(apprx_jac_dict[in_data])
                computed_grad.append(com_jac_dict[in_data])
            approx_grad = concatenate(approx_grad, axis=1)
            computed_grad = concatenate(computed_grad, axis=1)
            n_f, _ = approx_grad.shape
            if n_f == 1:
                approx_grad_dict[out_data] = approx_grad.flatten()
                computed_grad_dict[out_data] = computed_grad.flatten()
            else:
                for i in xrange(n_f):
                    out_name = out_data + "_" + str(i)
                    approx_grad_dict[out_name] = approx_grad[i, :]
                    computed_grad_dict[out_name] = computed_grad[i, :]
        return computed_grad_dict, approx_grad_dict, x_names

    def plot_jac_errors(
        self,
        computed_jac,  # type: ndarray
        approx_jac,  # type: ndarray
        file_path="jacobian_errors.pdf",  # type: Union[str,Path]
        show=False,  # type: bool
        figsize_x=10,  # type: int
        figsize_y=10,  # type: int
    ):  # type: (...) -> Figure
        """Generate a plot of the exact vs approximated Jacobian.

        Args:
            computed_jac: The Jacobian to validate.
            approx_jac: The approximated Jacobian.
            file_path: The path to the output file if ``plot_result`` is ``True``.
            show: Whether to open the figure.
            figsize_x: The x-size of the figure in inches.
            figsize_y: The y-size of the figure in inches.
        """
        comp_grad, app_grad, x_labels = self.__format_jac_as_grad_dict(
            computed_jac, approx_jac
        )
        n_funcs = len(app_grad)
        if n_funcs == 0:
            raise ValueError("No gradients to plot!")
        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1
        ncols = 2
        fig, axes = pyplot.subplots(
            nrows=nrows,
            ncols=2,
            sharex=True,
            sharey=False,
            figsize=(figsize_x, figsize_y),
        )
        i = 0
        j = -1

        axes = atleast_2d(axes)
        n_subplots = len(axes) * len(axes[0])
        abscissa = arange(len(x_labels))
        for func, grad in sorted(comp_grad.items()):
            j += 1
            if j == ncols:
                j = 0
                i += 1
            axe = axes[i][j]
            axe.plot(abscissa, grad, "bo")
            axe.plot(abscissa, app_grad[func], "ro")
            axe.set_title(func)
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)
            for tick in axe.get_xticklabels():
                tick.set_rotation(90)

            # Update y labels spacing
            vis_labels = [
                label for label in axe.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels[::2], visible=False)
        #             pyplot.xticks(rotation=90)

        if len(comp_grad.items()) < n_subplots:
            # xlabel must be written with the same fontsize on the 2 columns
            j += 1
            axe = axes[i][j]
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)
            for tick in axe.get_xticklabels():
                tick.set_rotation(90)

        fig.suptitle(
            "Computed and approximate derivatives. "
            + " blue = computed, red = approximated derivatives",
            fontsize=14,
        )
        if file_path is not None:
            pyplot.savefig(file_path)
        if show:
            pyplot.show()
        return fig


def comp_best_step(
    f_p,  # type: ndarray
    f_x,  # type: ndarray
    f_m,  # type: ndarray
    step,  # type: float
    epsilon_mach=EPSILON,  # type: float
):  # type: (...) -> Tuple[Optional[ndarray],Optional[ndarray],float]
    r"""Compute the optimal step for finite differentiation.

    Applied to a forward first order finite differences gradient approximation.

    Require a first evaluation of the perturbed functions values.

    The optimal step is reached when the truncation error
    (cut in the Taylor development),
    and the numerical cancellation errors
    (round-off when doing :math:`f(x+step)-f(x))` are equal.

    See:
    - https://en.wikipedia.org/wiki/Numerical_differentiation
    - *Numerical Algorithms and Digital Representation*,
      Knut Morken, Chapter 11, "Numerical Differenciation"

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        * The estimation of the truncation error.
          None if the Hessian approximation is too small to compute the optimal step.
        * The estimation of the cancellation error.
          None if the Hessian approximation is too small to compute the optimal step.
        * The optimal step.
    """
    hess = approx_hess(f_p, f_x, f_m, step)

    if abs(hess) < 1e-10:
        LOGGER.debug("Hessian approximation is too small, can't compute optimal step.")
        return None, None, step

    opt_step = 2 * (epsilon_mach * abs(f_x) / abs(hess)) ** 0.5
    trunc_error = compute_truncature_error(hess, step)
    cancel_error = compute_cancellation_error(f_x, opt_step)
    return trunc_error, cancel_error, opt_step


def compute_truncature_error(
    hess,  # type: ndarray
    step,  # type: float
):  # type: (...) -> ndarray
    r"""Estimate the truncation error.

    Defined for a first order finite differences scheme.

    Args:
        hess: The second-order derivative :math:`d^2f/dx^2`.
        step: The differentiation step.

    Returns:
        The truncation error.
    """
    trunc_error = abs(hess) * step / 2
    return trunc_error


def compute_cancellation_error(
    f_x,  # type: ndarray
    step,  # type: float
    epsilon_mach=EPSILON,
):  # type: (...) -> ndarray
    r"""Estimate the cancellation error.

    This is the round-off when doing :math:`f(x+\\delta_x)-f(x)`.

    Args:
        f_x: The value of the function at the current step :math:`x`.
        step: The step used for the calculations of the perturbed functions values.
        epsilon_mach: The machine epsilon.

    Returns:
        The cancellation error.
    """
    epsa = epsilon_mach * abs(f_x)
    cancel_error = 2 * epsa / step
    return cancel_error


def approx_hess(
    f_p,  # type:ndarray
    f_x,  # type:ndarray
    f_m,  # type:ndarray
    step,  # type: float
):  # type: (...) -> ndarray
    r"""Compute the second-order approximation of the Hessian matrix :math:`d^2f/dx^2`.

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        The approximation of the Hessian matrix at the current step :math:`x`.
    """
    hess = (f_p - 2 * f_x + f_m) / (step ** 2)
    return hess
