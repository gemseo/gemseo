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
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A discipline interfacing a Python function."""
from __future__ import annotations

import re
from inspect import getfullargspec
from inspect import getsource
from typing import Callable
from typing import Iterable
from typing import Sequence
from typing import Union

from numpy import array
from numpy import atleast_2d
from numpy import ndarray

from gemseo.core.data_processor import DataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

DataType = Union[float, ndarray]


class AutoPyDiscipline(MDODiscipline):
    """Wrap a Python function into a discipline.

    A simplified and straightforward way of integrating a discipline
    from a Python function.

    The Python function can take and return only numbers and NumPy arrays.

    The Python function may or may not include default values for input arguments,
    however, if the resulting :class:`.AutoPyDiscipline` is going to be placed inside
    an :class:`.MDF`, a :class:`.BiLevel` formulation or an :class:`.MDA`
    with strong couplings, then the Python function **must** assign default values for
    its input arguments.

    Example:
        >>> from gemseo.disciplines.auto_py import AutoPyDiscipline
        >>> from numpy import array
        >>> def my_function(x=0., y=0.):
        >>>     z1 = x + 2*y
        >>>     z2 = x + 2*y + 1
        >>>     return z1, z2
        >>>
        >>> discipline = AutoPyDiscipline(py_func=my_function)
        >>> discipline.execute()
        {'x': array([0.]), 'y': array([0.]), 'z1': array([0.]), 'z2': array([1.])}
        >>> discipline.execute({'x': array([1.]), 'y':array([-3.2])})
        {'x': array([1.]), 'y': array([-3.2]), 'z1': array([-5.4]), 'z2': array([-4.4])}
    """

    py_func: Callable[[DataType, ..., DataType], DataType]
    """The Python function to compute the outputs from the inputs."""

    use_arrays: bool
    """Whether the function is expected
    to take arrays as inputs and give outputs as arrays."""

    py_jac: Callable[[DataType, ..., DataType], ndarray] | None
    """The Python function to compute the Jacobian from the inputs."""

    in_names: list[str]
    """The names of the inputs."""

    out_names: list[str]
    """The names of the outputs."""

    data_processor: AutoDiscDataProcessor
    """A data processor forcing input data to float and output data to arrays."""

    sizes: dict[str, int]
    """The sizes of the input and output variables."""

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + ("py_func", "out_names")

    def __init__(
        self,
        py_func: Callable[[DataType, ..., DataType], DataType],
        py_jac: Callable[[DataType, ..., DataType], ndarray] | None = None,
        name: str | None = None,
        use_arrays: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    ) -> None:
        """
        Args:
            py_func: The Python function to compute the outputs from the inputs.
            py_jac: The Python function to compute the Jacobian from the inputs;
                its output value must be a 2D NumPy array
                with rows correspond to the outputs
                and columns to the inputs.
            name: The name of the discipline. If ``None``, use the name of the Python
                function.
            use_arrays: Whether the function is expected
                to take arrays as inputs and give outputs as arrays.

        Raises:
            TypeError: When ``py_func`` is not callable.
        """  # noqa: D205 D212 D415
        if not callable(py_func):
            raise TypeError("py_func must be callable.")

        super().__init__(name=name or py_func.__name__, grammar_type=grammar_type)

        self.py_func = py_func
        self.use_arrays = use_arrays
        self.py_jac = py_jac

        args_in = getfullargspec(py_func)[0]
        self.in_names = args_in
        self.input_grammar.update(self.in_names)
        self.out_names = self._get_return_spec(py_func)
        self.output_grammar.update(self.out_names)

        if not use_arrays:
            self.data_processor = AutoDiscDataProcessor(self.out_names)

        if self.py_jac is None:
            self.set_jacobian_approximation()

        def_func = self._get_defaults()
        self.default_inputs = to_arrays_dict(def_func)
        self.__sizes = {}
        self.__jac_shape = []
        self.__in_names_with_namespaces = []
        self.__out_names_with_namespaces = []

    def _get_defaults(self) -> dict[str, DataType]:
        """Return the default values of the input variables when available.

        The values are read from the signature of the Python function.

        Returns:
            The default values of the input variables.
        """
        full_arg_specs = getfullargspec(self.py_func)
        args = full_arg_specs[0]
        defaults = full_arg_specs[3]
        if defaults is None:
            return {}

        n_defaults = len(defaults)
        return {args[-n_defaults:][i]: defaults[i] for i in range(n_defaults)}

    def _run(self):
        output_values = self.py_func(**self.get_input_data(with_namespaces=False))
        if len(self.out_names) == 1:
            output_values = {self.out_names[0]: output_values}
        else:
            output_values = dict(zip(self.out_names, output_values))
        self.store_local_data(**output_values)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        """
        Raises:
            RuntimeError: When the analytic Jacobian :attr:`.py_jac` is ``None``.
            ValueError: When the Jacobian shape is inconsistent.
        """  # noqa: D205 D212 D415
        if self.py_jac is None:
            raise RuntimeError("The analytic Jacobian is missing.")

        if not self.__sizes:
            self.__sizes = {k: v.size for k, v in self.local_data.items()}

            in_to_ns = self.input_grammar.to_namespaced
            self.__in_names_with_namespaces = [
                in_to_ns[name] if name in in_to_ns else name for name in self.in_names
            ]
            out_to_ns = self.output_grammar.to_namespaced
            self.__out_names_with_namespaces = [
                out_to_ns[name] if name in out_to_ns else name
                for name in self.out_names
            ]
            n_rows = sum(
                self.__sizes[output] for output in self.__out_names_with_namespaces
            )
            n_cols = sum(
                self.__sizes[input] for input in self.__in_names_with_namespaces
            )
            self.__jac_shape = (n_rows, n_cols)

        func_jac = self.py_jac(**self.get_input_data(with_namespaces=False))

        if len(func_jac.shape) < 2:
            func_jac = atleast_2d(func_jac)
        if func_jac.shape != self.__jac_shape:
            msg = (
                "The jacobian provided by the py_jac function is of wrong shape. "
                "Expected {}, got {}."
            ).format(self.__jac_shape, func_jac.shape)
            raise ValueError(msg)

        self.jac = split_array_to_dict_of_arrays(
            func_jac,
            self.__sizes,
            self.__out_names_with_namespaces,
            self.__in_names_with_namespaces,
        )

    @staticmethod
    def get_return_spec_fromstr(
        return_line: str,
    ) -> list[str]:
        """Return the output specifications of a Python function.

        Args:
            return_line: The Python line containing the return statement.

        Returns:
            The output names separated with commas
            if the return statement starts with "return ";
            otherwise, ``None``.
        """
        stripped_line = return_line.strip()
        if not stripped_line.startswith("return "):
            return []
        return re.sub(r"\s+", "", stripped_line.replace("return ", "")).split(",")

    @staticmethod
    def _get_return_spec(
        func: Callable,
    ) -> list[str]:
        """Return the output specifications of a Python function.

        Args:
            func: The Python function.

        Returns:
            The output names separated with commas.

        Raises:
            ValueError: When the return statements have different definitions.
        """
        output_names = []
        for line in getsource(func).split("\n"):
            outs_loc = AutoPyDiscipline.get_return_spec_fromstr(line)
            if outs_loc and output_names and tuple(outs_loc) != tuple(output_names):
                raise ValueError(
                    "Inconsistent definition of return statements in function: "
                    "{} != {}.".format(tuple(outs_loc), tuple(output_names))
                )

            if outs_loc:
                output_names = outs_loc

        return output_names


class AutoDiscDataProcessor(DataProcessor):
    """A data processor forcing input data to float and output data to arrays.

    Convert all |g| scalar input data to floats, and convert all discipline output data
    to NumPy arrays.
    """

    out_names: Sequence[str]
    """The names of the outputs."""

    one_output: bool
    """Whether there is a single output."""

    def __init__(
        self,
        out_names: Sequence[str],
    ) -> None:
        """
        Args:
            out_names: The names of the outputs.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.out_names = out_names
        self.one_output = len(out_names) == 1

    def pre_process_data(
        self,
        data: dict[str, DataType],
    ) -> dict[str, DataType]:
        """Pre-process the input data.

        Execute a pre-processing of input data
        after they are checked by :meth:`~MDODiscipline.check_input_data`,
        and before the :meth:`~MDODiscipline._run` method of the discipline is called.

        Args:
            data: The data to be processed.

        Returns:
            The processed data
            where one-length NumPy arrays have been replaced with floats.
        """
        processed_data = data.copy()
        for key, val in data.items():
            if len(val) == 1:
                processed_data[key] = float(val[0])

        return processed_data

    def post_process_data(
        self,
        data: dict[str, DataType],
    ) -> dict[str, ndarray]:
        """Post-process the output data.

        Execute a post-processing of the output data
        after the :meth:`~MDODiscipline._run` method of the discipline is called,
        and before they are checked by :meth:`~MDODiscipline.check_output_data`.

        Args:
            data: The data to be processed.

        Returns:
            The processed data with NumPy arrays as values.
        """
        processed_data = data.copy()
        for output_name, output_value in processed_data.items():
            if not isinstance(output_value, ndarray):
                processed_data[output_name] = array([output_value])

        return processed_data


def to_arrays_dict(
    data: dict[str, DataType],
) -> dict[str, ndarray]:
    """Ensure that the values of a dictionary are NumPy arrays.

    Args:
        data: The dictionary whose values must be NumPy arrays.

    Returns:
        The dictionary with NumPy arrays as values.
    """
    for key, value in data.items():
        if not isinstance(value, ndarray):
            data[key] = array([value])
    return data
