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
#    INITIAL AUTHORS - initial API and implementation and/or
#                  initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalable diagonal model
=======================

This module implements the concept of scalable diagonal model,
which is a particular scalable model built from an input-output
dataset relying on a diagonal design of experiments (DOE)
where inputs vary proportionally from their lower bounds
to their upper bounds, following the diagonal of the input space.

So for every output, the dataset catches its evolution
with respect to this proportion, which makes it a mono dimensional behavior.
Then, for a new user-defined problem dimension,
the scalable model extrapolates this mono dimensional behavior
to the different input directions.

The concept of scalable diagonal model is implemented through
the :class:`.ScalableDiagonalModel` class
which is composed of a :class:`.ScalableDiagonalApproximation`.
With regard to the diagonal DOE, |g| proposes the
:class:`.DiagonalDOE` class.
"""
from __future__ import annotations

from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import arange
from numpy import argsort
from numpy import array
from numpy import array_equal
from numpy import atleast_1d
from numpy import hstack
from numpy import max as np_max
from numpy import mean
from numpy import median
from numpy import min as np_min
from numpy import nan_to_num
from numpy import sqrt
from numpy import vstack
from numpy import where
from numpy import zeros
from numpy.random import choice
from numpy.random import rand
from numpy.random import randint
from numpy.random import seed as npseed
from scipy.interpolate import InterpolatedUnivariateSpline

from gemseo.problems.scalable.data_driven.model import ScalableModel
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.matplotlib_figure import save_show_figure


class ScalableDiagonalModel(ScalableModel):
    """Scalable diagonal model."""

    ABBR = "sdm"

    def __init__(
        self,
        data,
        sizes=None,
        fill_factor=-1,
        comp_dep=None,
        inpt_dep=None,
        force_input_dependency=False,
        allow_unused_inputs=True,
        seed=1,
        group_dep=None,
    ):
        """Constructor.

        :param Dataset data: learning dataset.
        :param dict sizes: sizes of input and output variables.
            If None, use the original sizes.
            Default: None.
        :param fill_factor: degree of sparsity of the dependency matrix.
            Default: -1.
        :param comp_dep: matrix that establishes the selection
            of a single original component for each scalable component
        :param inpt_dep: dependency matrix that establishes the
            dependency of outputs wrt inputs
        :param bool force_input_dependency: for any output, force dependency
            with at least on input.
        :param bool allow_unused_inputs: possibility to have an input
            with no dependence with any output
        :param int seed: seed
        :param dict(list(str)) group_dep: dependency
            between inputs and outputs
        """
        if isinstance(fill_factor, Number):
            fill_factor = {
                function_name: fill_factor
                for function_name in data.get_names(data.OUTPUT_GROUP)
            }
        elif not isinstance(fill_factor, dict):
            raise TypeError(
                "Fill factor must be either a number between 0 and 1, "
                "a number equal to -1 or a dictionary."
            )
        parameters = {
            "fill_factor": fill_factor,
            "comp_dep": comp_dep,
            "inpt_dep": inpt_dep,
            "force_input_dependency": force_input_dependency,
            "allow_unused_inputs": allow_unused_inputs,
            "seed": seed,
            "group_dep": group_dep,
        }
        super().__init__(data, sizes, **parameters)
        self.t_scaled, self.f_scaled = self.__build_scalable_functions()

    def __build_dependencies(self):
        """Build dependencies.

        :return: matrix that establishes the selection of a single original
            component for each scalable component, dependency matrix that
            establishes the dependency of outputs wrt inputs.
        :rtype: ndarray, ndarray
        """
        comp_dep = self.parameters["comp_dep"]
        inpt_dep = self.parameters["inpt_dep"]
        if comp_dep is None or inpt_dep is None:
            comp_dep, inpt_dep = self.generate_random_dependency()
        return comp_dep, inpt_dep

    def scalable_function(self, input_value=None):
        """Evaluate the scalable functions.

        :param dict input_value: input values.
            If None, use default inputs.
        :return: evaluation of the scalable functions.
        :rtype: dict
        """
        input_value = input_value or self.default_inputs
        input_value = concatenate_dict_of_arrays_to_array(
            input_value, self.inputs_names
        )
        scal_func = self.model.get_scalable_function
        return {fname: scal_func(fname)(input_value) for fname in self.outputs_names}

    def scalable_derivatives(self, input_value=None):
        """Evaluate the scalable derivatives.

        :param dict input_value: input values.
            If None, use default inputs.
        :return: evaluation of the scalable derivatives.
        :rtype: dict
        """
        input_value = input_value or self.default_inputs
        input_value = concatenate_dict_of_arrays_to_array(
            input_value, self.inputs_names
        )
        scal_der = self.model.get_scalable_derivative
        return {fname: scal_der(fname)(input_value) for fname in self.outputs_names}

    def build_model(self):
        """Build model with original sizes for input and output variables.

        :return: scalable approximation.
        :rtype: ScalableDiagonalApproximation
        """
        comp_dep, inpt_dep = self.__build_dependencies()
        seed = self.parameters["seed"]
        scalable_approximation = ScalableDiagonalApproximation(
            self.sizes, comp_dep, inpt_dep, seed
        )
        return scalable_approximation

    def __build_scalable_functions(self):
        """Builds all the required functions from the original dataset."""
        t_scaled = {}
        f_scaled = {}
        for function_name in self.outputs_names:
            t_sc, f_sc = self.model.build_scalable_function(
                function_name, self.data, self.inputs_names
            )
            t_scaled[function_name] = t_sc
            f_scaled[function_name] = f_sc
        return t_scaled, f_scaled

    def __get_variables_locations(self, names):
        """Get the locations of first component of each variable.

        :param names: list of variables names.
        :type names: list(str)
        :return: list of locations.
        :rtype: list(int)
        """
        positions = []
        current_position = 0
        for name in names:
            positions.append(current_position)
            current_position += self.sizes[name]
        return positions

    def __convert_dependency_to_array(self, dependency):
        """Convert a dependency object of type dictionary into a dependency object of
        type array.

        :param dependency: input-output dependency structure.
        :type dependency: dict
        :return: dependency matrix.
        :rtype: ndarray
        """
        matrix = None
        for output_name in self.outputs_names:
            row = hstack([dependency[output_name][inpt] for inpt in self.inputs_names])
            matrix = row if matrix is None else vstack((matrix, row))
        return matrix.T

    def plot_dependency(
        self, add_levels=True, save=True, show=False, directory=".", png=False
    ):
        """This method plots the dependency matrix of a discipline in the form of a
        chessboard, where rows represent inputs, columns represent output and gray scale
        represent the dependency level between inputs and outputs.

        :param bool add_levels: add values of dependency levels in percentage.
            Default: True.
        :param bool save: if True, export the plot into a file.
            Default: True.
        :param bool show: if True, display the plot.
            Default: False.
        :param str directory: directory path. Default: '.'.
        :param bool png: if True, the file format is PNG. Otherwise, use PDF.
            Default: False.
        """
        inputs_positions = self.__get_variables_locations(self.inputs_names)
        outputs_positions = self.__get_variables_locations(self.outputs_names)
        dependency = self.model.io_dependency
        dependency_matrix = self.__convert_dependency_to_array(dependency)
        is_binary_matrix = array_equal(
            dependency_matrix, dependency_matrix.astype(bool)
        )
        if not is_binary_matrix:
            dp_sum = dependency_matrix.sum(0)
            dependency_matrix = dependency_matrix / dp_sum

        fig, axes = plt.subplots()
        axes.matshow(dependency_matrix, cmap="Greys", vmin=0)
        axes.set_yticks(inputs_positions)
        axes.set_yticklabels(self.inputs_names)
        axes.set_xticks(outputs_positions)
        axes.set_xticklabels(self.outputs_names, ha="left", rotation="vertical")
        axes.tick_params(axis="both", which="both", length=0)
        for pos in inputs_positions[1:]:
            axes.axhline(y=pos - 0.5, color="r", linewidth=0.5)
        for pos in outputs_positions[1:]:
            axes.axvline(x=pos - 0.5, color="r", linewidth=0.5)
        if add_levels and not is_binary_matrix:
            for i in range(dependency_matrix.shape[0]):
                for j in range(dependency_matrix.shape[1]):
                    val = int(round(dependency_matrix[i, j] * 100))
                    med = median(dependency_matrix[:, j] * 100)
                    col = "white" if val > med else "black"
                    axes.text(j, i, val, ha="center", va="center", color=col)
        if save:
            extension = "png" if png else "pdf"
            file_path = Path(directory) / "{}_dependency.{}".format(
                self.name, extension
            )
        else:
            file_path = None
        save_show_figure(fig, show, file_path)
        return str(file_path)

    def plot_1d_interpolations(
        self, save=False, show=False, step=0.01, varnames=None, directory=".", png=False
    ):
        r"""Plot the scaled 1D interpolations, a.k.a. the basis functions.

        A basis function is a mono dimensional function
        interpolating the samples of a given output component
        over the input sampling line
        :math:`t\in[0,1]\mapsto \\underline{x}+t(\overline{x}-\\underline{x})`.

        There are as many basis functions
        as there are output components from the discipline.
        Thus, for a discipline with a single output in dimension 1,
        there is 1 basis function.
        For a discipline with a single output in dimension 2,
        there are 2 basis functions.
        For a discipline with an output in dimension 2
        and an output in dimension 13,
        there are 15 basis functions. And so on.
        This method allows to plot the basis functions associated
        with all outputs or only part of them,
        either on screen (:code:`show=True`), in a file (:code:`save=True`)
        or both.
        We can also specify the discretization :code:`step`
        whose default value is :code:`0.01`.

        :param bool save: if True, export the plot as a PDF file
            (Default value = False)
        :param bool show: if True, display the plot (Default value = False)
        :param bool step: Step to evaluate the 1d interpolation function
            (Default value = 0.01)
        :param list(str) varnames: names of the variable to plot;
            if None, all variables are plotted (Default value = None)
        :param str directory: directory path. Default: '.'.
        :param bool png: if True, the file format is PNG. Otherwise, use PDF.
            Default: False.
        """
        function_names = varnames or self.outputs_names
        x_vals = arange(0.0, 1.0 + step, step)
        fnames = []
        for func in function_names:
            components = self.model.interpolation_dict[func]
            for index, function in enumerate(components):
                plt.figure()
                doe_t = self.t_scaled[func]
                doe_f = [output[index] for output in self.f_scaled[func]]
                y_vals = function(x_vals)
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.1, 1.1)
                plt.plot(
                    x_vals,
                    (y_vals - min(y_vals)) / (max(y_vals) - min(y_vals)),
                    label=func + str(index),
                )
                plt.plot(
                    doe_t,
                    (doe_f - min(y_vals)) / (max(y_vals) - min(y_vals)),
                    "or",
                )
                plt.xlabel("Scaled abscissa", fontsize=18)
                plt.ylabel("Interpolation value", fontsize=18)
                plt.title("1D interpolation of " + self.name + "." + func)
                fig = plt.gcf()
                if save:
                    extension = "png" if png else "pdf"
                    file_path = Path(directory) / "{}_{}_1D_interpolation_{}.{}".format(
                        self.name, func, index, extension
                    )
                    fnames.append(str(file_path))
                else:
                    file_path = None
                save_show_figure(fig, show, file_path)
        return fnames

    def generate_random_dependency(self):
        """Generates a random dependency structure for use in scalable discipline.

        :return: output component dependency and input-output dependency
        :rtype: dict(int), dict(dict(ndarray))
        """
        npseed(self.parameters["seed"])
        io_dependency = self.parameters["group_dep"] or {}
        for function_name in self.outputs_names:
            input_names = io_dependency.get(function_name, self.inputs_names)
            io_dependency[function_name] = input_names

        if self.parameters.get("inpt_dep") is None:
            io_dep = self.__generate_random_io_dep(io_dependency)

        if self.parameters.get("comp_dep") is None:
            out_map = self.__generate_random_output_map()

        # If an output function does not have any dependency with inputs,
        # add a random dependency if independent outputs are forbidden
        if self.parameters["force_input_dependency"]:
            for function_name in self.outputs_names:
                for function_component in range(self.sizes.get(function_name)):
                    self.__complete_random_dep(
                        io_dep, function_name, function_component, io_dependency
                    )

        # If an input parameter does not have any dependency with output
        # functions, add a random dependency if unused inputs are not allowed
        if not self.parameters["allow_unused_inputs"]:
            for input_name in self.inputs_names:
                for input_component in range(self.sizes.get(input_name)):
                    self.__complete_random_dep(
                        io_dep, input_name, input_component, io_dependency
                    )
        return out_map, io_dep

    def __generate_random_io_dep(self, io_dependency):
        """Generate the dependency between the new inputs and the new outputs.

        :param io_dependency: input-output dependency structure. If None,
            all output components can depend on all input components.
            Default: None.
        :type io_dependency: dict(list(str))
        :return: random input-output dependencies
        :rtype: dict(dict(ndarray))
        """
        error_msg = (
            "Fill factor must be a number, "
            "either -1 or a real number between 0 and 1."
        )
        r_io_dependency = {}
        for function_name in self.outputs_names:
            fill_factor = self.parameters["fill_factor"].get(function_name, 0.7)
            if fill_factor != -1.0 and (fill_factor < 0.0 or fill_factor > 1):
                raise TypeError(error_msg)
            function_size = self.sizes.get(function_name)
            r_io_dependency[function_name] = {}
            for input_name in self.inputs_names:
                input_size = self.sizes.get(input_name)
                if input_name in io_dependency[function_name]:
                    if 0 <= fill_factor <= 1:
                        rand_dep = choice(
                            2,
                            (function_size, input_size),
                            p=[1.0 - fill_factor, fill_factor],
                        )
                    else:
                        rand_dep = rand(function_size, input_size)
                    r_io_dependency[function_name][input_name] = rand_dep
                else:
                    zeros_dep = zeros((function_size, input_size))
                    r_io_dependency[function_name][input_name] = zeros_dep
        return r_io_dependency

    def __generate_random_output_map(self):
        """Generate the dependency between the original and new output components for the
        different outputs.

        :return: component dependencies
        :rtype: dict(int)
        """
        out_map = {}
        for function_name in self.outputs_names:
            original_function_size = self.original_sizes.get(function_name)
            function_size = self.sizes.get(function_name)
            out_map[function_name] = randint(original_function_size, size=function_size)
        return out_map

    def __complete_random_dep(self, r_io_dep, dataname, index, io_dep):
        """Complete random dependency if row (input name) or column (function name) of
        the random dependency matrix is empty.

        :param ndarray r_io_dep: input-output dependency.
        :param str dataname: name of the variable to check
            if component is empty.
        :param int index: component index of the variable.
        :param io_dep: input-output dependency structure. If None,
            all output components can depend on all input components.
            Default: None.
        :type io_dep: dict(list(str))
        """
        is_input = dataname in self.inputs_names
        if is_input:
            varnames = []
            for function_name, inputs in io_dep.items():
                if dataname in inputs:
                    varnames.append(function_name)
            inpt_dep_mat = hstack(
                [r_io_dep[varname][dataname].T for varname in varnames]
            )
        else:
            varnames = io_dep[dataname]
            inpt_dep_mat = hstack([r_io_dep[dataname][varname] for varname in varnames])
        if sum(inpt_dep_mat[index, :]) == 0:
            prob = [self.sizes.get(varname, 1) for varname in varnames]
            prob = [float(x) / sum(prob) for x in prob]
            id_var = choice(len(varnames), p=prob)
            id_comp = randint(0, self.sizes.get(varnames[id_var], 1))
            if is_input:
                varname = varnames[id_var]
                r_io_dep[varname][dataname][id_comp, index] = 1
            else:
                varname = varnames[id_var]
                r_io_dep[dataname][varname][index, id_comp] = 1


class ScalableDiagonalApproximation:
    """Methodology that captures the trends of a physical problem, and extends it into a
    problem that has scalable input and outputs dimensions The original and the resulting
    scalable problem have the same interface:

    all inputs and outputs have the same names; only their dimensions vary.
    """

    def __init__(self, sizes, output_dependency, io_dependency, seed=0):
        """
        Constructor:

        :param sizes: sizes of both input and output variables.
        :type sizes: dict
        :param output_dependency: dependency between old and new outputs.
        :type output_dependency: dict
        :param io_dependency: dependency between new inputs and new outputs.
        :type io_dependency: dict
        """
        super().__init__()
        self.sizes = sizes
        # dependency matrices
        self.output_dependency = output_dependency
        self.io_dependency = io_dependency
        # dictionaries of interpolations and extrapolations
        self.interpolation_dict = {}
        self.d_interpolation_dict = {}
        self.interpolators_dict = {}
        self.scalable_functions = {}
        self.scalable_dfunctions = {}
        # seed for random generator
        npseed(seed)

    def build_scalable_function(self, function_name, dataset, input_names, degree=3):
        """Build interpolation from a 1D input and output function. Add the model to the
        local dictionary.

        :param str function_name: name of the output function
        :param Dataset dataset: the input-output dataset
        :param list(str) input_names: names of the input variables
        :param int degree: degree of interpolation (Default value = 3)
        """
        x2_scaled = nan_to_num(dataset.get_data_by_group(dataset.INPUT_GROUP) ** 2)
        t_scaled = [atleast_1d(sqrt(mean(val.real))) for val in x2_scaled]
        f_scaled = dataset[function_name].real

        # sort t_scaled and f_scaled following the t_scaled ascending order
        indices = argsort([val[0] for val in t_scaled])
        t_scaled = [t_scaled[index] for index in indices]
        f_scaled = [f_scaled[index] for index in indices]

        # scale the output samples: [a1, b1] x ... x [am, bm] -> [0, 1]^m
        f_scaled = self.scale_samples(f_scaled)

        # interpolate the (t_scaled, f_scaled) data
        self._interpolate(function_name, t_scaled, f_scaled, degree)

        # compute the total input and output sizes
        (input_size, output_size) = self._compute_sizes(function_name, input_names)

        # extrapolation
        self._extrapolate(function_name, input_names, input_size, output_size)

        return t_scaled, f_scaled

    def get_scalable_function(self, output_function):
        """Retrieve the scalable function generated from the original discipline.

        :param str output_function: name of the output function
        """
        return self.scalable_functions[output_function]

    def get_scalable_derivative(self, output_function):
        """Retrieve the (scalable) gradient of the scalable function generated from the
        original discipline.

        :param str output_function: name of the output function
        """
        return self.scalable_dfunctions[output_function]

    @staticmethod
    def scale_samples(samples):
        """Scale samples of array into [0, 1]

        :param samples: samples of multivariate array
        :type samples: list(ndarray)
        :return: samples of multivariate array
        :rtype: ndarray
        """
        samples = array(samples)
        col_min = np_min(samples, 0)
        col_max = np_max(samples, 0)
        scaled_samples = samples - col_min
        range_col = col_max - col_min
        scaling = where(abs(range_col) > 1e-6, range_col, 1)
        scaled_samples /= scaling
        return scaled_samples

    def _interpolate(self, function_name, t_scaled, f_scaled, degree=3):
        """Interpolate a set of samples (t, y(t)) with a polynomial spline.

        :param str function_name: name of the interpolated function
        :param list(list(float)) t_scaled: set of points
        :param list(list(float)) f_scaled: set of images
        :param int degree: degree of the polynomial interpolation
        """
        nb_components = f_scaled[0].size
        list_interpolations = []
        list_derivatives = []
        for component in range(nb_components):
            f_scaled_component = [output[component] for output in f_scaled]
            # compute interpolation
            interpolation = InterpolatedUnivariateSpline(
                t_scaled, f_scaled_component, k=degree
            )
            # store spline and derivative
            list_interpolations.append(interpolation)
            list_derivatives.append(interpolation.derivative())
        # store interpolation and derivatives
        self.interpolation_dict[function_name] = list_interpolations
        self.d_interpolation_dict[function_name] = list_derivatives

    def _compute_sizes(self, function_name, input_names):
        """Determine the size of the vector input and output.

        :param str function_name: function name
        :param list(str) input_names: input names
        """
        input_size = 0
        for input_name in input_names:
            input_size += self.sizes.get(input_name, 1)
        output_size = self.sizes.get(function_name, 1)  # default 1
        return input_size, output_size

    def _extrapolate(self, function_name, input_names, input_size, output_size):
        """Extrapolate a 1D function to arbitrary input and output dimensions. Generate a
        function that produces an output with a given size from an input with a given
        size, and its derivative.

        :param str function_name: name of the output function
        :param list(str) input_names: names of the inputs
        :param int input_size: size of the input vector
        :param int output_size: size of the output vector
        """
        # crop the matrices to the correct sizes and convert to array
        io_dependency = {
            input_name: dep_mat[0:output_size, 0 : self.sizes[input_name]]
            for (input_name, dep_mat) in self.io_dependency[function_name].items()
        }
        io_dependency = concatenate_dict_of_arrays_to_array(io_dependency, input_names)

        # Convert the input-output dependency matrix to a list
        # where the i-th element is a list whose j-th element corresponds to
        # the degree of dependence between the i-th output component and the
        # i-th input.
        io_dependency = [list(row) for row in io_dependency]

        # Get the 1D interpolation functions and their derivatives
        interpolated_fun_1d = self.interpolation_dict[function_name]
        interpolated_dfun_1d = self.d_interpolation_dict[function_name]

        # Get the indices of the 1D interpolation functions
        # associated with the components of the new output.
        outputs_to_original_ones = self.output_dependency[function_name]

        def scalable_function(input_data):
            """n-dimensional to size_output-dimensional extrapolated function.

            :param list(float) input_data: vector of inputs
            :returns: size_output extrapolated output
            """
            result = zeros(output_size)
            for output_index in range(output_size):
                func = interpolated_fun_1d[outputs_to_original_ones[output_index]]
                coefficients = io_dependency[output_index]
                result[output_index] = sum(
                    coefficient * func(input_value)
                    for coefficient, input_value in zip(coefficients, input_data)
                ) / sum(coefficients)

            return result

        def scalable_derivative(input_data):
            """n-dimensional to size_output-dimensional extrapolated function Jacobian.

            :param list(float) input_data: vector of inputs
            :returns: size_output - sizeof input_vars extrapolated output
            """
            result = zeros([output_size, input_size])
            for output_index in range(output_size):
                func = interpolated_dfun_1d[outputs_to_original_ones[output_index]]
                coefficients = io_dependency[output_index]
                result[output_index, :] = array(
                    [
                        coefficient * func(input_value)
                        for coefficient, input_value in zip(coefficients, input_data)
                    ]
                ) / sum(coefficients)

            return result

        self.scalable_functions[function_name] = scalable_function
        self.scalable_dfunctions[function_name] = scalable_derivative
