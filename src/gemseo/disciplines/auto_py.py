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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""A discipline interfacing a Python function."""

from __future__ import division, unicode_literals

import logging
import re
from inspect import getsource
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

from numpy import array, atleast_2d, ndarray

from gemseo.core.data_processor import DataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.py23_compat import getargspec

LOGGER = logging.getLogger(__name__)

DataType = Union[float, ndarray]


class AutoPyDiscipline(MDODiscipline):
    """Wrap a Python function into a discipline.

    A simplified and straightforward way of integrating a discipline
    from a Python function.

    The Python function can take and return only numbers and NumPy arrays.

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

    Attributes:
        py_func (Callable[[DataType, ..., DataType],DataType]): The Python function
            to compute the outputs from the inputs.
        use_arrays (bool):  Whether the function is expected
            to take arrays as inputs and give outputs as arrays.
        py_jac (Optional[Callable[[DataType, ..., DataType],ndarray]]): The Python
            function to compute the Jacobian from the inputs.
        in_names (List[str]): The names of the inputs.
        out_names (List[str]): The names of the outputs.
        data_processor (AutoDiscDataProcessor): A data processor
            forcing input data to float and output data to arrays.
        sizes (Dict[str,int]): The sizes of the input and output variables.
    """

    def __init__(
        self,
        py_func,  # type: Callable[[DataType, ..., DataType],DataType]
        py_jac=None,  # type: Optional[Callable[[DataType, ..., DataType],ndarray]]
        use_arrays=False,  # type: bool
        write_schema=False,  # type: bool
    ):  # type: (...) -> None
        # noqa: D205 D212 D415
        """
        Args:
            py_func: The Python function to compute the outputs from the inputs.
            py_jac: The Python function to compute the Jacobian from the inputs;
                its output value must be a 2D NumPy array
                with rows correspond to the outputs
                and columns to the inputs.
            use_arrays: Whether the function is expected
                to take arrays as inputs and give outputs as arrays.
            write_schema: Whether to write JSON schema on the disk.

        Raises:
            TypeError: When ``py_func`` is not callable.
        """
        if not callable(py_func):
            raise TypeError("py_func must be callable.")

        super(AutoPyDiscipline, self).__init__(
            name=py_func.__name__,
            auto_detect_grammar_files=False,
            grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        )

        self.py_func = py_func
        self.use_arrays = use_arrays
        self.py_jac = py_jac

        args_in = getargspec(py_func)[0]  # pylint: disable=deprecated-method
        self.in_names = args_in
        self.input_grammar.initialize_from_data_names(self.in_names)
        self.out_names = self._get_return_spec(py_func)
        self.output_grammar.initialize_from_data_names(self.out_names)

        if write_schema:
            self.input_grammar.write_schema()
            self.output_grammar.write_schema()

        if not use_arrays:
            self.data_processor = AutoDiscDataProcessor(self.out_names)

        if self.py_jac is None:
            self.set_jacobian_approximation()

        def_func = self._get_defaults()
        self.default_inputs = to_arrays_dict(def_func)
        self.sizes = None

    def _get_defaults(self):  # type: (...) -> Dict[str, DataType]
        """Return the default values of the input variables when available.

        The values are read from the signature of the Python function.

        Returns:
            The default values of the input variables.
        """
        args, _, _, defaults = getargspec(
            self.py_func
        )  # pylint: disable=deprecated-method
        if defaults is None:
            return {}

        n_defaults = len(defaults)
        return {args[-n_defaults:][i]: defaults[i] for i in range(n_defaults)}

    def _run(self):
        output_values = self.py_func(**self.get_input_data())
        if len(self.out_names) == 1:
            output_values = {self.out_names[0]: output_values}
        else:
            output_values = dict(zip(self.out_names, output_values))
        self.store_local_data(**output_values)

    def _compute_jacobian(
        self,
        inputs=None,  # type:Optional[Iterable[str]]
        outputs=None,  # type:Optional[Iterable[str]]
    ):  # type: (...)-> None
        # noqa: D205 D212 D415
        """
        Raises:
            RuntimeError: When the analytic Jacobian :attr:`.py_jac` is ``None``.
        """
        if self.py_jac is None:
            raise RuntimeError("The analytic Jacobian is missing.")

        if self.sizes is None:
            self.sizes = {k: v.size for k, v in self.local_data.items()}

        self.jac = split_array_to_dict_of_arrays(
            atleast_2d(self.py_jac(**self.get_input_data())),
            self.sizes,
            self.out_names,
            self.in_names,
        )

    @staticmethod
    def get_return_spec_fromstr(
        return_line,  # type: str
    ):  # type: (...) -> Optional[str]
        """Return the output specifications of a Python function.

        Args:
            return_line: The Python line containing the return statement.

        Returns:
            The output names separated with commas
            if the return statement starts with "return ";
            otherwise, ``None``.
        """
        stripped_line = return_line.strip()
        if stripped_line.startswith("return "):
            output_specifications = stripped_line.replace("return ", "")
            return re.sub(r"\s+", "", output_specifications).split(",")

    @staticmethod
    def _get_return_spec(
        func,  # type: Callable
    ):  # type: (...) -> Optional[str]
        """Return the output specifications of a Python function.

        Args:
            func: The Python function.

        Returns:
            The output names separated with commas.

        Raises:
            ValueError: When the return statements have different definitions.
        """
        docstring = getsource(func)
        output_names = None
        for line in docstring.split("\n"):
            outs_loc = AutoPyDiscipline.get_return_spec_fromstr(line)
            if outs_loc is not None and output_names is not None:
                if tuple(outs_loc) != tuple(output_names):
                    raise ValueError(
                        "Inconsistent definition of return statements in function: "
                        "{} != {}.".format(tuple(outs_loc), tuple(output_names))
                    )

            if outs_loc is not None:
                output_names = outs_loc

        return output_names


class AutoDiscDataProcessor(DataProcessor):
    """A data processor forcing input data to float and output data to arrays.

    Convert all |g| scalar input data to floats,
    and convert all discipline output data to NumPy arrays.

    Attributes:
        out_names (Sequence[str]): The names of the outputs.
        one_output (bool): Whether there is a single output.
    """

    def __init__(
        self,
        out_names,  # type: Sequence[str]
    ):  # type: (...) -> None
        # noqa: D205 D212 D415
        """
        Args:
            out_names: The names of the outputs.
        """
        super(AutoDiscDataProcessor, self).__init__()
        self.out_names = out_names
        self.one_output = len(out_names) == 1

    def pre_process_data(
        self,
        data,  # type: Dict[str, DataType]
    ):  # type: (...) -> Dict[str, DataType]
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
        data,  # type: Dict[str, DataType]
    ):  # type: (...) -> Dict[str, ndarray]
        """Post-process the output data.

        Execute a post-processing of the output data
        after the :meth:`~MDODiscipline._run` method of the discipline is called
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
    in_dict,  # type: Mapping[str, DataType]
):  # type: (...) -> Mapping[str, ndarray]
    """Ensure that the values of a dictionary are NumPy arrays.

    Args:
        in_dict: The dictionary whose values must be NumPy arrays.

    Returns:
        The dictionary with NumPy arrays as values.
    """
    for output_name, output_value in in_dict.items():
        if not isinstance(output_value, ndarray):
            in_dict[output_name] = array([output_value])

    return in_dict
