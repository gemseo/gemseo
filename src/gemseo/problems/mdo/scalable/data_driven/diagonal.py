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
"""Scalable diagonal model.

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
from typing import TYPE_CHECKING
from typing import Callable

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
from numpy.random import Generator
from numpy.random import default_rng
from scipy.interpolate import InterpolatedUnivariateSpline

from gemseo.problems.mdo.scalable.data_driven.model import ScalableModel
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from numpy._typing import NDArray

    from gemseo.datasets.io_dataset import IODataset


class ScalableDiagonalModel(ScalableModel):
    """Scalable diagonal model."""

    ABBR = "sdm"

    __rng: Generator
    """The random number generator."""

    def __init__(
        self,
        data: IODataset,
        sizes: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
        fill_factor: float = -1,
        comp_dep: NDArray[float] | None = None,
        inpt_dep: NDArray[float] | None = None,
        force_input_dependency: bool = False,
        allow_unused_inputs: bool = True,
        seed: int = SEED,
        group_dep: Mapping[str, Iterable[str]] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            fill_factor: The degree of sparsity of the dependency matrix.
            comp_dep: The matrix defining the selection
                of a single original component for each scalable component.
                If ``None``,
                generate a random matrix.
            inpt_dep: The input-output dependency matrix.
                If ``None``,
                generate a random matrix.
            force_input_dependency: Whether to force the dependency of each output
                with at least one input.
            bool allow_unused_inputs: The possibility to have an input
                with no dependence with any output.
            seed: The seed for reproducible results.
            group_dep: The dependency between the inputs and outputs.
        """  # noqa: D205, D212, D415
        if isinstance(fill_factor, Number):
            fill_factor = dict.fromkeys(
                data.get_variable_names(data.OUTPUT_GROUP), fill_factor
            )
        elif not isinstance(fill_factor, dict):
            msg = (
                "Fill factor must be either a number between 0 and 1, "
                "a number equal to -1 or a dictionary."
            )
            raise TypeError(msg)
        parameters = {
            "fill_factor": fill_factor,
            "comp_dep": comp_dep,
            "inpt_dep": inpt_dep,
            "force_input_dependency": force_input_dependency,
            "allow_unused_inputs": allow_unused_inputs,
            "seed": seed,
            "group_dep": group_dep,
        }
        self.__rng = default_rng(seed)
        super().__init__(data, sizes, **parameters)
        self.t_scaled, self.f_scaled = self.__build_scalable_functions()

    def __build_dependencies(self) -> tuple[NDArray[float], NDArray[float]]:
        """Build dependencies.

        Returns:
            The matrix defining the selection of a single original component
            for each scalable component,
            and the dependency matrix between the inputs and the outputs.
        """
        comp_dep = self.parameters["comp_dep"]
        inpt_dep = self.parameters["inpt_dep"]
        if comp_dep is None or inpt_dep is None:
            comp_dep, inpt_dep = self.generate_random_dependency()
        return comp_dep, inpt_dep

    def scalable_function(
        self, input_value: Mapping[str, NDArray[float]] | None = None
    ) -> dict[str, NDArray[float]]:
        """Compute the outputs.

        Args:
            input_value: The input values.

        Returns:
            The values of the outputs.
        """
        input_value = input_value or self.default_input_data
        input_value = concatenate_dict_of_arrays_to_array(input_value, self.input_names)
        scal_func = self.model.get_scalable_function
        return {fname: scal_func(fname)(input_value) for fname in self.output_names}

    def scalable_derivatives(
        self, input_value: Mapping[str, NDArray[float]] | None = None
    ) -> dict[str, NDArray[float]]:
        """Compute the derivatives.

        Args:
            input_value: The input values.

        Returns:
            The values of the derivatives.
        """
        input_value = input_value or self.default_input_data
        input_value = concatenate_dict_of_arrays_to_array(input_value, self.input_names)
        scal_der = self.model.get_scalable_derivative
        return {fname: scal_der(fname)(input_value) for fname in self.output_names}

    def build_model(self) -> ScalableDiagonalApproximation:
        """Build the model with the original sizes for input and output variables.

        Returns:
            The scalable approximation.
        """
        comp_dep, inpt_dep = self.__build_dependencies()
        return ScalableDiagonalApproximation(self.sizes, comp_dep, inpt_dep)

    def __build_scalable_functions(
        self,
    ) -> tuple[dict[str, list[NDArray[float]]], dict[str, NDArray[float]]]:
        """Build all the required functions from the original dataset.

        Returns:
            The output names bound to the input and output samples scaled in [0, 1].
        """
        t_scaled = {}
        f_scaled = {}
        for function_name in self.output_names:
            t_sc, f_sc = self.model.build_scalable_function(
                function_name, self.data, self.input_names
            )
            t_scaled[function_name] = t_sc
            f_scaled[function_name] = f_sc
        return t_scaled, f_scaled

    def __get_variables_locations(self, names: Sequence[str]) -> list[int]:
        """Get the locations of first component of each variable.

        Args:
            names: The names of the variables

        Returns:
            The locations of the first component of each variable.
        """
        positions = []
        current_position = 0
        for name in names:
            positions.append(current_position)
            current_position += self.sizes[name]
        return positions

    def __convert_dependency_to_array(self, dependency) -> NDArray[float]:
        """Convert a dictionary-like dependency structure into an array-like one.

        Args:
            dependency: The dictionary-like dependency structure.

        Returns:
            The array-like dependency structure.
        """
        matrix = None
        for output_name in self.output_names:
            row = hstack([dependency[output_name][inpt] for inpt in self.input_names])
            matrix = row if matrix is None else vstack((matrix, row))
        return matrix.T

    def plot_dependency(
        self,
        add_levels: bool = True,
        save: bool = True,
        show: bool = False,
        directory: str = ".",
        png: bool = False,
    ) -> str:
        """Plot the dependency matrix of a discipline in the form of a chessboard.

        The rows represent inputs,
        columns represent output
        and gray scale represents the dependency level between inputs and outputs.

        Args:
            add_levels: Whether to add the dependency levels in percentage.
            save: Whether to save the figure.
            show: Whether to display the figure.
            directory: The directory path.
            png: Whether to use PNG file format instead of PDF.
        """
        inputs_positions = self.__get_variables_locations(self.input_names)
        outputs_positions = self.__get_variables_locations(self.output_names)
        dependency = self.model.io_dependency
        dependency_matrix = self.__convert_dependency_to_array(dependency)
        is_binary_matrix = array_equal(
            dependency_matrix, dependency_matrix.astype(bool)
        )
        if not is_binary_matrix:
            dp_sum = dependency_matrix.sum(0)
            dependency_matrix /= dp_sum

        fig, ax = plt.subplots()
        ax.matshow(dependency_matrix, cmap="Greys", vmin=0)
        ax.set_yticks(inputs_positions)
        ax.set_yticklabels(self.input_names)
        ax.set_xticks(outputs_positions)
        ax.set_xticklabels(self.output_names, ha="left", rotation="vertical")
        ax.tick_params(axis="both", which="both", length=0)
        for pos in inputs_positions[1:]:
            ax.axhline(y=pos - 0.5, color="r", linewidth=0.5)
        for pos in outputs_positions[1:]:
            ax.axvline(x=pos - 0.5, color="r", linewidth=0.5)
        if add_levels and not is_binary_matrix:
            for i in range(dependency_matrix.shape[0]):
                for j in range(dependency_matrix.shape[1]):
                    val = round(dependency_matrix[i, j] * 100)
                    med = median(dependency_matrix[:, j] * 100)
                    col = "white" if val > med else "black"
                    ax.text(j, i, val, ha="center", va="center", color=col)
        if save:
            extension = "png" if png else "pdf"
            file_path = Path(directory) / f"{self.name}_dependency.{extension}"
        else:
            file_path = None
        save_show_figure(fig, show, file_path)
        return str(file_path)

    def plot_1d_interpolations(
        self,
        save: bool = False,
        show: bool = False,
        step: float = 0.01,
        varnames: Sequence[str] = (),
        directory: str = ".",
        png: bool = False,
    ) -> list[str]:
        r"""Plot the scaled 1D interpolations, a.k.a. the basis functions.

        A basis function is a mono dimensional function
        interpolating the samples of a given output component
        over the input sampling line
        :math:`t\in[0,1]\mapsto \underline{x}+t(\overline{x}-\underline{x})`.

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
        either on screen (``show=True``), in a file (``save=True``)
        or both.
        We can also specify the discretization ``step``
        whose default value is ``0.01``.

        Args:
            save: Whether to save the figure.
            show: Whether to display the figure.
            step: The step to evaluate the 1d interpolation function.
            varnames: The names of the variable to plot.
                If empty, all the variables are plotted.
            directory: The directory path.
            png: Whether to use PNG file format instead of PDF.

        Returns:
            The names of the files.
        """
        function_names = varnames or self.output_names
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
                    file_path = (
                        Path(directory)
                        / f"{self.name}_{func}_1D_interpolation_{index}.{extension}"
                    )
                    fnames.append(str(file_path))
                else:
                    file_path = None
                save_show_figure(fig, show, file_path)
        return fnames

    def generate_random_dependency(
        self,
    ) -> tuple[dict[str, NDArray[int]], dict[str, dict[str, NDArray[float]]]]:
        """Generate a random dependency structure for use in scalable discipline.

        Returns:
            The dependency structure.
        """
        io_dependency = self.parameters["group_dep"] or {}
        for function_name in self.output_names:
            input_names = io_dependency.get(function_name, self.input_names)
            io_dependency[function_name] = input_names

        if self.parameters.get("inpt_dep") is None:
            io_dep = self.__generate_random_io_dep(io_dependency)

        if self.parameters.get("comp_dep") is None:
            out_map = self.__generate_random_output_map()

        # If an output function does not have any dependency with inputs,
        # add a random dependency if independent outputs are forbidden
        if self.parameters["force_input_dependency"]:
            for function_name in self.output_names:
                for function_component in range(self.sizes.get(function_name)):
                    self.__complete_random_dep(
                        io_dep, function_name, function_component, io_dependency
                    )

        # If an input parameter does not have any dependency with output
        # functions, add a random dependency if unused inputs are not allowed
        if not self.parameters["allow_unused_inputs"]:
            for input_name in self.input_names:
                for input_component in range(self.sizes.get(input_name)):
                    self.__complete_random_dep(
                        io_dep, input_name, input_component, io_dependency
                    )
        return out_map, io_dep

    def __generate_random_io_dep(
        self, io_dependency: Mapping[str, Iterable[str]]
    ) -> dict[str, dict[str, NDArray[float]]]:
        """Generate the dependency between the new inputs and the new outputs.

        Args:
            io_dependency: The input-output dependency structure.

        Returns:
            The dependencies between the inputs and the outputs.
        """
        error_msg = (
            "Fill factor must be a number, either -1 or a real number between 0 and 1."
        )
        r_io_dependency = {}
        for function_name in self.output_names:
            fill_factor = self.parameters["fill_factor"].get(function_name, 0.7)
            if fill_factor != -1.0 and (fill_factor < 0.0 or fill_factor > 1):
                raise TypeError(error_msg)
            function_size = self.sizes.get(function_name)
            r_io_dependency[function_name] = {}
            for input_name in self.input_names:
                input_size = self.sizes.get(input_name)
                if input_name in io_dependency[function_name]:
                    if 0 <= fill_factor <= 1:
                        rand_dep = self.__rng.choice(
                            2,
                            (function_size, input_size),
                            p=[1.0 - fill_factor, fill_factor],
                        )
                    else:
                        rand_dep = self.__rng.random((function_size, input_size))
                    r_io_dependency[function_name][input_name] = rand_dep
                else:
                    zeros_dep = zeros((function_size, input_size))
                    r_io_dependency[function_name][input_name] = zeros_dep
        return r_io_dependency

    def __generate_random_output_map(self) -> dict[str, NDArray[int]]:
        """Generate the dependency between the original and new output components.

        Returns:
            The output names bound to the original output components.
        """
        out_map = {}
        for function_name in self.output_names:
            original_function_size = self.original_sizes.get(function_name)
            function_size = self.sizes.get(function_name)
            out_map[function_name] = self.__rng.integers(
                original_function_size, size=function_size
            )
        return out_map

    def __complete_random_dep(
        self,
        r_io_dep: NDArray[float],
        dataname: str,
        index: int,
        io_dep: Mapping[Iterable[str]] | None,
    ) -> None:
        """Complete random dependency.

        Only if row (input name) or column (function name) of
        the random dependency matrix is empty.

        Args:
            r_io_dep: The dependency between inputs and outputs.
            dataname: The name of the variable whose component must be check.
            index: The index of the component of the variable.
            io_dep: The dependency between the inputs and the outputs.
                If ``None``,
                all output components can depend on all input components.
        """
        is_input = dataname in self.input_names
        if is_input:
            varnames = []
            for function_name, inputs in io_dep.items():
                if dataname in inputs:
                    varnames.append(function_name)
            inpt_dep_mat = hstack([
                r_io_dep[varname][dataname].T for varname in varnames
            ])
        else:
            varnames = io_dep[dataname]
            inpt_dep_mat = hstack([r_io_dep[dataname][varname] for varname in varnames])

        if sum(inpt_dep_mat[index, :]) == 0:
            prob = [self.sizes.get(varname, 1) for varname in varnames]
            prob = [float(x) / sum(prob) for x in prob]
            id_var = self.__rng.choice(len(varnames), p=prob)
            id_comp = self.__rng.integers(0, self.sizes.get(varnames[id_var], 1))
            if is_input:
                varname = varnames[id_var]
                r_io_dep[varname][dataname][id_comp, index] = 1
            else:
                varname = varnames[id_var]
                r_io_dep[dataname][varname][index, id_comp] = 1


class ScalableDiagonalApproximation:
    """Methodology that captures the trends of a physical problem.

    It also extends it
    into a problem that has scalable input and outputs dimensions.
    The original and the resulting scalable problem have the same interface:

    all inputs and outputs have the same names; only their dimensions vary.
    """

    def __init__(
        self,
        sizes: Mapping[str, int],
        output_dependency,
        io_dependency,
    ) -> None:
        """
        Args:
            sizes: The sizes of the inputs and outputs.
            output_dependency: The dependency between the original and new outputs.
            io_dependency: The dependency between the new inputs and outputs.
        """  # noqa: D205, D212, D415
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

    def build_scalable_function(
        self,
        function_name: str,
        dataset: IODataset,
        input_names: Iterable[str],
        degree: int = 3,
    ) -> tuple[list[NDArray[float]], NDArray[float]]:
        """Create the interpolation functions for a specific output.

        Args:
            function_name: The name of the output.
            dataset: The input-output dataset.
            input_names: The names of the inputs.
            degree: The degree of interpolation.

        Returns:
            The input and output samples scaled in [0, 1].
        """
        x2_scaled = nan_to_num(
            dataset.get_view(group_names=dataset.INPUT_GROUP).to_numpy() ** 2
        )
        t_scaled = [atleast_1d(sqrt(mean(val.real))) for val in x2_scaled]
        f_scaled = dataset.get_view(variable_names=function_name).to_numpy().real

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

    def get_scalable_function(
        self, output_function: str
    ) -> Callable[[Iterable[float]], NDArray[float]]:
        """Return the function computing an output.

        Args:
            output_function: The name of the output.

        Returns:
            The function computing the output.
        """
        return self.scalable_functions[output_function]

    def get_scalable_derivative(
        self, output_function: str
    ) -> Callable[[Iterable[float]], NDArray[float]]:
        """Return the function computing the derivatives of an output.

        Args:
            output_function: The name of the output.

        Returns:
            The function computing the derivatives of this output.
        """
        return self.scalable_dfunctions[output_function]

    @staticmethod
    def scale_samples(samples: Iterable[NDArray[float]]) -> NDArray[float]:
        """Scale array samples into [0, 1].

        Args:
            samples: The samples.

        Returns:
            The samples with components scaled in [0, 1].
        """
        samples = array(samples)
        col_min = np_min(samples, 0)
        col_max = np_max(samples, 0)
        scaled_samples = samples - col_min
        range_col = col_max - col_min
        scaling = where(abs(range_col) > 1e-6, range_col, 1)
        scaled_samples /= scaling
        return scaled_samples

    def _interpolate(
        self,
        function_name: str,
        t_scaled: Iterable[float],
        f_scaled: Iterable[float],
        degree: int = 3,
    ) -> None:
        """Interpolate a set of samples (t, y(t)) with a polynomial spline.

        Args:
            function_name: The name of the output.
            t_scaled: The input values.
            f_scaled: The output values.
            degree: The degree of the polynomial interpolation.
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

    def _compute_sizes(
        self, function_name: str, input_names: Sequence[str]
    ) -> tuple[int, int]:
        """Compute the sizes of the input and output vectors.

        Args:
            function_name: The name of the output.
            input_names: The names of the inputs.

        Returns:
            The sizes of the input and output vectors.
        """
        input_size = 0
        for input_name in input_names:
            input_size += self.sizes.get(input_name, 1)
        output_size = self.sizes.get(function_name, 1)  # default 1
        return input_size, output_size

    def _extrapolate(
        self,
        function_name: str,
        input_names: Iterable[str],
        input_size: int,
        output_size: int,
    ) -> None:
        """Extrapolate a 1D function to arbitrary input and output dimensions.

        Generate a function
        that produces an output with a given size
        from an input with a given size,
        as well as its derivative.

        Args:
            function_name: The name of the output.
            input_names: The names of the inputs.
            input_size: The size of the input vector.
            output_size: The size of the output vector.
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

        def scalable_function(input_data: Iterable[float]) -> NDArray[float]:
            """Compute the output vector.

            Args:
                input_data: The input vector.

            Returns:
                The output vector.
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

        def scalable_derivative(input_data: Iterable[float]) -> NDArray[float]:
            """Compute the Jacobian matrix.

            Args:
                input_data: The input vector.

            Returns:
                The Jacobian matrix.
            """
            result = zeros([output_size, input_size])
            for output_index in range(output_size):
                func = interpolated_dfun_1d[outputs_to_original_ones[output_index]]
                coefficients = io_dependency[output_index]
                result[output_index, :] = array([
                    coefficient * func(input_value)
                    for coefficient, input_value in zip(coefficients, input_data)
                ]) / sum(coefficients)

            return result

        self.scalable_functions[function_name] = scalable_function
        self.scalable_dfunctions[function_name] = scalable_derivative
