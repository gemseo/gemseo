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
"""
Finite differences & complex step approximations
************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import chain
from multiprocessing import cpu_count

from future import standard_library
from matplotlib import pyplot
from numpy import (
    absolute,
    allclose,
    amax,
    arange,
    argmax,
    array,
    atleast_2d,
    complex128,
    concatenate,
    divide,
    finfo,
    float64,
    ndarray,
    ones,
    tile,
    where,
    zeros,
)
from numpy.linalg import norm

from gemseo.core.parallel_execution import ParallelExecution
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import xrange

standard_library.install_aliases()
EPSILON = finfo(float).eps
from gemseo import LOGGER


class ComplexStep(object):
    """Complex step, second order gradient calculation.
    Enables a much lower step than real finite differences,
    typically fd_step=1e-30 since there is no
    cancellation error due to a difference calculation

    grad = Imaginary part(f(x+j*fd_step)/(fd_step))
    """

    def __init__(self, f_pointer, step=1e-20, parallel=False, **parallel_args):
        """
        Constructor

        :param f_pointer: pointer on function to derive
        :param step: differentiation step
        :param parallel: if True, executes in parallel
        :param parallel_args: arguments passed to the parallel execution,
            see gemseo.core.parallel_execution
        """
        self.f_pointer = f_pointer
        self.__par_args = parallel_args
        self.__parallel = parallel
        if step.imag != 0:
            self.step = step.imag
        else:
            self.step = step

    def f_gradient(self, x_vect, step=None, **kwargs):
        """Compute gradient by complex step

        :param x_vect: design vector
        :type x_vect: numpy array
        :param kwargs: optional arguments for the function
        :returns: function gradient
        :rtype: numpy array
        """
        if norm(x_vect.imag) != 0.0:
            raise ValueError(
                "Impossible to check the gradient at a complex "
                + "point using the complex step method !"
            )
        n_dim = len(x_vect)
        grad = []
        if step is None:
            step = self.step
        x_p_arr = self.generate_perturbations(n_dim, x_vect, step)

        if self.__parallel:

            def func_noargs(xval):
                """Function calling the f_pointer
                without explicitly passed arguments"""
                return self.f_pointer(xval, **kwargs)

            function_list = [func_noargs] * (n_dim)
            parallel_execution = ParallelExecution(function_list, **self.__par_args)

            all_x = [x_vect + x_p_arr[:, i] for i in xrange(n_dim)]
            output_list = parallel_execution.execute(all_x)

            for i in xrange(n_dim):
                f_p = output_list[i]
                grad.append(f_p.imag / (x_p_arr[i, i].imag))
        else:

            for i in xrange(n_dim):
                x_p = x_vect + x_p_arr[:, i]
                f_p = self.f_pointer(x_p, **kwargs)
                grad.append(f_p.imag / (x_p_arr[i, i].imag))
        return array(grad, dtype=float64).T

    def generate_perturbations(self, n_dim, x_vect, step=None):
        """Generates the perturbations x_perturb which will be used
        to compute f(x_vect+x_perturb)

        :param n_dim: dimension
        :type n_dim: integer
        :param x_vect: design vector
        :type x_vect: numpy array
        :returns: perturbations
        :rtype: numpy array
        """
        if step is None:
            step = self.step
        x_perturb = zeros((n_dim, n_dim), dtype=complex128)
        x_nnz = where(x_vect == 0.0, 1.0, x_vect)
        x_perturb[range(n_dim), range(n_dim)] = 1j * x_nnz * step
        return x_perturb


class FirstOrderFD(object):
    """Finites differences at first order, first order gradient calculation.

    grad =(f(x+fd_step)-f(x))/fd_step
    """

    def __init__(self, f_pointer, step=1e-6, parallel=False, **parallel_args):
        """
        Constructor

        :param f_pointer: pointer on function to derive
        :param step: differentiation step
        :param parallel: if True, executes in parallel
        :param parallel_args: arguments passed to the parallel execution,
            see gemseo.core.parallel_execution
        """
        self.f_pointer = f_pointer
        self.step = step
        self.__par_args = parallel_args
        self.__parallel = parallel

    def f_gradient(self, x_vect, step=None, **kwargs):
        """Compute gradient by real step

        :param x_vect: design vector
        :type x_vect: numpy array
        :param kwargs: additional arguments passed to the function
        :returns: function gradient
        :rtype: numpy array
        """
        n_dim = len(x_vect)
        x_p_arr = self.generate_perturbations(n_dim, x_vect, step)
        grad = []
        if step is None:
            step = self.step

        if not isinstance(step, ndarray):
            step = step * ones(n_dim)

        if self.__parallel:

            def func_noargs(xval):
                """Function calling the f_pointer
                without explicitly passed arguments"""
                return self.f_pointer(xval, **kwargs)

            function_list = [func_noargs] * (n_dim + 1)
            parallel_execution = ParallelExecution(function_list, **self.__par_args)

            all_x = [x_vect] + [x_p_arr[:, i] for i in xrange(n_dim)]
            output_list = parallel_execution.execute(all_x)

            f_0 = output_list[0]
            for i in xrange(n_dim):
                f_p = output_list[i + 1]
                g_approx = ((f_p - f_0) / step[i]).real
                grad.append(g_approx)

        else:
            f_0 = self.f_pointer(x_vect, **kwargs)
            for i in xrange(n_dim):
                f_p = self.f_pointer(x_p_arr[:, i], **kwargs)
                g_approx = ((f_p - f_0) / step[i]).real
                grad.append(g_approx)
        return array(grad, dtype=float64).T

    def _get_opt_step(self, f_p, f_0, f_m, numerical_error=EPSILON):
        """
        Compute the optimal step, f may be a vector function
        In this case, take the worst case


        :param f_0: f(x)
        :param f_m: f(x-e)
        :param f_p: f(x+e)
        :param numerical_error: numerical error associated to the calculation
            of f. By default Machine epsilon (appx 1e-16),
            but can be higher when
            the calculation of f requires a numerical resolution
        """
        n_out = f_p.size
        if n_out == 1:
            t_e, c_e, opt_step = comp_best_step(
                f_p, f_0, f_m, self.step, epsilon_mach=numerical_error
            )
            if t_e is None:
                error = 0.0
            else:
                error = t_e + c_e
        else:
            errors = zeros(n_out)
            opt_steps = zeros(n_out)
            for i in xrange(n_out):
                t_e, c_e, opt_steps[i] = comp_best_step(
                    f_p[i], f_0[i], f_m[i], self.step, epsilon_mach=numerical_error
                )
                if t_e is None:
                    errors[i] = 0.0
                else:
                    errors[i] = t_e + c_e
            max_i = argmax(errors)
            error = errors[max_i]
            opt_step = opt_steps[max_i]
        return error, opt_step

    def compute_optimal_step(self, x_vect, numerical_error=EPSILON, **kwargs):
        """Compute gradient by real step

        :param x_vect: design vector
        :type x_vect: numpy array
        :param kwargs: additional arguments passed to the function
        :param numerical_error: numerical error associated to the calculation
            of f. By default Machine epsilon$
            (appx 1e-16), but can be higher when
            the calculation of f requires a numerical resolution
        :returns: function gradient
        :rtype: numpy array
        """
        n_dim = len(x_vect)
        x_p_arr = self.generate_perturbations(n_dim, x_vect)
        x_m_arr = self.generate_perturbations(n_dim, x_vect, -self.step)
        opt_steps = self.step * ones(n_dim)
        errors = zeros(n_dim)
        comp_step = self._get_opt_step
        if self.__parallel:

            def func_noargs(xval):
                """Function calling the f_pointer
                without explicitly passed arguments"""
                return self.f_pointer(xval, **kwargs)

            function_list = [func_noargs] * (n_dim + 1)
            parallel_execution = ParallelExecution(function_list, **self.__par_args)

            all_x = [x_vect] + [x_p_arr[:, i] for i in xrange(n_dim)]
            all_x += [x_m_arr[:, i] for i in xrange(n_dim)]
            output_list = parallel_execution.execute(all_x)

            f_0 = output_list[0]
            for i in xrange(n_dim):
                f_p = output_list[i + 1]
                f_m = output_list[n_dim + i + 1]
                errs, opt_step = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        else:
            f_0 = self.f_pointer(x_vect, **kwargs)
            for i in xrange(n_dim):
                f_p = self.f_pointer(x_p_arr[:, i], **kwargs)
                f_m = self.f_pointer(x_m_arr[:, i], **kwargs)
                errs, opt_step = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        self.step = opt_steps
        return opt_steps, errors

    def generate_perturbations(self, n_dim, x_vect, step=None):
        """Generates the perturbations x_perturb which will be used
        to compute f(x_vect+x_perturb)
        Generates the perturbations x_perturb which will be used
        to compute f(x_vect+x_perturb)

        :param n_dim: dimension
        :type n_dim: integer
        :param x_vect: design vector
        :type x_vect: numpy array
        :param step: step for the finite differences
        :returns: perturbations x_perturb
        :rtype: numpy array
        """
        if step is None:
            loc_step = self.step
        else:
            loc_step = step
        x_perturb = tile(x_vect, n_dim).reshape((n_dim, n_dim)).T
        x_perturb[xrange(n_dim), xrange(n_dim)] += loc_step
        return x_perturb


class DisciplineJacApprox(object):
    """
    Approximates a discipline Jacobian using finite differences
    or Complex step
    """

    COMPLEX_STEP = "complex_step"
    FINITE_DIFFERENCES = "finite_differences"
    N_CPUS = cpu_count()

    def __init__(
        self,
        discipline,
        approx_method=FINITE_DIFFERENCES,
        step=1e-7,
        parallel=False,
        n_processes=N_CPUS,
        use_threading=False,
        wait_time_between_fork=0,
    ):
        """
        Constructor:

        :param discipline: the discipline for which the jacobian
            approximation shall be made
        :param approx_method: "complex_step" or "finite_differences"
        :param step: the step for finite differences or complex step
        :param parallel: if True, executes in parallel
        :param n_processes: maximum number of processors on which to run
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note if you want to execute the same
            discipline multiple times, you shall use multiprocessing
        :param wait_time_between_fork: time waited between two forks of the
         process /Thread
        """
        from gemseo.core.function import MDOFunctionGenerator

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

    def _create_approximator(self, outputs, inputs):
        """
        Creates the approximation class for the function jacobian

        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
        """
        self.func = self.generator.get_function(
            input_names_list=inputs, output_names_list=outputs
        )
        if self.approx_method == self.FINITE_DIFFERENCES:
            self.approximator = FirstOrderFD(
                self.func, self.step, self.__parallel, **self.__par_args
            )
        elif self.approx_method == self.COMPLEX_STEP:
            self.approximator = ComplexStep(
                self.func, self.step, self.__parallel, **self.__par_args
            )
        else:
            raise Exception(
                "Unknown jacobian approximation method " + str(self.approx_method)
            )

    def auto_set_step(
        self, outputs, inputs, print_errors=True, numerical_error=EPSILON
    ):
        """
        Compute optimal step for a forward first order finite differences
        gradient approximation Requires a first evaluation of perturbed
        functions values.  The optimal step is reached when the truncation
        error (cut in the Taylor development), and the numerical cancellation
        errors (roundoff when doing f(x+step)-f(x)) are equal.

        :param outputs: the list of outputs to compute the derivative
        :param inputs: this list of outputs to derive wrt
        :param print_errors: if True logs the cancellation and truncation
            error estimates
        :param numerical_error: numerical error associated to the calculation
            of f. By default Machine epsilon (appx 1e-16),
            but can be higher when
            the calculation of f requires a numerical resolution
        :returns: function gradient

        See:
        https://en.wikipedia.org/wiki/Numerical_differentiation
        and
        "Numerical Algorithms and Digital Representation", Knut Morken ,
        Chapter 11, "Numerical Differenciation"
        """
        self._create_approximator(outputs, inputs)
        old_cache_tol = self.discipline.cache_tol
        self.discipline.cache_tol = 0.0
        x_vect = self._prepare_xvect(inputs, self.discipline.default_inputs)
        compute_opt_step = self.approximator.compute_optimal_step
        steps_opt, errors = compute_opt_step(x_vect, numerical_error=numerical_error)
        if print_errors:
            LOGGER.info(
                "Set optimal step for finite differences. Estimated"
                " approximation errors ="
            )
            LOGGER.info(errors)
        self.discipline.cache_tol = old_cache_tol
        data = self.discipline.local_data
        data_sizes = {key: val.size for key, val in data.items()}
        self.auto_steps = DataConversion.array_to_dict(steps_opt, inputs, data_sizes)
        return errors, self.auto_steps

    def _prepare_xvect(self, inputs, data=None):
        """
        :param inputs: derive outputs wrt inputs
        """
        if data is None:
            data = self.discipline.local_data
        x_vect = DataConversion.dict_to_array(data, inputs)
        return x_vect

    def compute_approx_jac(self, outputs, inputs):
        """
        Computes the approximation

        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
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

        if hasattr(step, "len") and len(step) > 1 and len(x_vect) != len(step):
            raise ValueError(
                "Inconsistent step size, expected "
                + str(len(x_vect))
                + " got "
                + str(len(step))
            )
        flat_jac = self.approximator.f_gradient(x_vect, step)
        flat_jac = atleast_2d(flat_jac)
        data_sizes = {key: len(local_data[key]) for key in chain(inputs, outputs)}
        self.discipline.cache_tol = old_cache_tol
        return DataConversion.jac_2dmat_to_dict(flat_jac, outputs, inputs, data_sizes)

    def check_jacobian(
        self,
        analytic_jacobian,
        outputs,
        inputs,
        discipline,
        threshold=1e-8,
        plot_result=False,
        file_path="jacobian_errors.pdf",
        show=False,
        figsize_x=10,
        figsize_y=10,
    ):
        """Checks if the jacobian provided by the linearize() method is correct

        :param analytic_jacobian: jacobian to validate
        :param inputs: list of inputs wrt which to differentiate
        :param outputs: list of outputs to differentiate
        :param threshold: acceptance threshold for the jacobian error
            (Default value = 1e-8)
        :param inputs: list of inputs wrt which to differentiate
            (Default value = None)
        :param plot_result: plot the result of the validation (computed
            and approximate jacobians)
        :param file_path: path to the output file if plot_result is True
        :param show: if True, open the figure
        :param figsize_x: x size of the figure in inches
        :param figsize_y: y size of the figure in inches
        :returns: True if the check is accepted, False otherwise
        """
        approx_jac_complete = self.compute_approx_jac(outputs, inputs)
        name = discipline.name
        succeed = True
        for out_data, apprx_jac_dict in approx_jac_complete.items():
            for in_data, approx_jac in apprx_jac_dict.items():
                computed_jac = analytic_jacobian[out_data][in_data]
                if approx_jac.shape != computed_jac.shape:
                    succeed = False
                    msg = name + " Jacobian:  dp " + str(out_data) + "/dp "
                    msg += str(in_data) + " is of wrong shape ! "
                    msg += "Got:" + str(computed_jac.shape)
                    msg += " while expected: " + str(approx_jac.shape)
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
                        msg = name + " Jacobian:  dp " + str(out_data)
                        msg += "/d " + str(in_data) + " is wrong by "
                        msg += str(err * 100.0) + "%"
                        LOGGER.error(msg)
                        msg = "Approximate jacobian = \n" + str(approx_jac)
                        LOGGER.info(msg)
                        msg = "Provided by linearize method = \n"
                        msg += str(computed_jac)
                        LOGGER.info(msg)
                        msg = "Difference of jacobians = \n"
                        msg += str(approx_jac - computed_jac)
                        LOGGER.info(msg)
                        succeed = succeed and success_loc
                    else:
                        LOGGER.info(
                            "Jacobian:  dp %s/dp %s succeeded!",
                            str(out_data),
                            str(in_data),
                        )
        if succeed:
            LOGGER.info("Linearization of MDODiscipline: %s" " is correct !", str(name))
        else:
            LOGGER.info("Linearization of MDODiscipline: %s" " is wrong !", str(name))

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
    def __format_jac_as_grad_dict(computed_jac, approx_jac):
        """
        Formats the approximate jacobian dict as a dict of
        gradients

        :param computed_jac: reference computed jac dict of dicts
        :param approx_jac: dict of gradients

        :returns grad dict, approx dict, and design var names
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
        computed_jac,
        approx_jac,
        file_path="jacobian_errors.pdf",
        show=False,
        figsize_x=10,
        figsize_y=10,
    ):
        """
        Generate a plot of the exact vs approximate jacobian

        :param computed_jac: computed jacobianfrom linearize method
        :param approx_jac: finite differences approximate jacobian
        :param file_path: path to the output file if plot_result is True
        :param show: if True, open the figure
        :param figsize_x: x size of the figure in inches
        :param figsize_y: y size of the figure in inches
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


def comp_best_step(f_p, f_x, f_m, step, epsilon_mach=EPSILON):
    """
    Compute optimal step for a forward first order finite differences gradient
    approximation Requires a first evaluation of perturbed functions values.
    The optimal step is reached when the truncation error (cut in the Taylor
    development), and the numerical cancellation errors (roundoff when doing
    f(x+step)-f(x)) are equal.

    See:
    https://en.wikipedia.org/wiki/Numerical_differentiation
    and
    "Numerical Algorithms and Digital Representation", Knut Morken ,
    Chapter 11, "Numerical Differenciation"

    :param f_p: f(x+step)
    :param f_x: f(x)
    :param f_m: f(x-step)
    :param step: step used for the calculations of perturbed functions values
    :returns: trunc_error, cancel_error, optimal step
    """
    hess = approx_hess(f_p, f_x, f_m, step)

    if abs(hess) < 1e-10:
        LOGGER.debug("Hessian approximation is too small, can't compute optimal step !")
        return None, None, step

    opt_step = 2 * (epsilon_mach * abs(f_x) / abs(hess)) ** 0.5
    trunc_error = compute_truncature_error(hess, step)
    cancel_error = compute_cancellation_error(f_x, opt_step)
    return trunc_error, cancel_error, opt_step


def compute_truncature_error(hess, step):
    """
    Computes the truncation error estimation for a first order finite
    differences scheme

    :param hess: second order derivative (d²f/dx²)
    :param step: step of the finite differences used for the derivatives
        approximation

    :returns: trunc_error the trucation error
    """
    trunc_error = abs(hess) * step / 2
    return trunc_error


def compute_cancellation_error(f_x, step, epsilon_mach=EPSILON):
    """
    Compute the cancellation error, ie roundoff when doing f(x+step)-f(x)

    :param f_x: value of the function at current point
    :param step: step used for the calculations of perturbed functions values
    :param epsilon_mach: machine epsilon

    :returns: the cancellation error
    """
    epsa = epsilon_mach * abs(f_x)
    cancel_error = 2 * epsa / step
    return cancel_error


def approx_hess(f_p, f_x, f_m, step):
    """
    Second order approximation of the hessian (d²f/dx²)

    :param f_p: f(x+step)
    :param f_x: f(x)
    :param f_m: f(x-step)
    :param step: step used for the calculations of perturbed functions values
    :returns: hessian approximation
    """
    hess = (f_p - 2 * f_x + f_m) / (step ** 2)
    return hess
