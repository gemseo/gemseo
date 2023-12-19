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

import ast
import logging
from inspect import getfullargspec
from inspect import getsource
from typing import TYPE_CHECKING
from typing import Callable
from typing import Final
from typing import Union
from typing import get_type_hints

from numpy import array
from numpy import atleast_2d
from numpy import ndarray
from typing_extensions import get_args
from typing_extensions import get_origin

from gemseo.core.data_processor import DataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.source_parsing import get_callable_argument_defaults

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

DataType = Union[float, ndarray]

LOGGER = logging.getLogger(__name__)


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

    Examples:
        >>> from gemseo.disciplines.auto_py import AutoPyDiscipline
        >>> from numpy import array
        >>> def my_function(x=0., y=0.):
        >>>     z1 = x + 2*y
        >>>     z2 = x + 2*y + 1
        >>>     return z1, z2
        >>>
        >>> discipline = AutoPyDiscipline(my_function)
        >>> discipline.execute()
        {'x': array([0.]), 'y': array([0.]), 'z1': array([0.]), 'z2': array([1.])}
        >>> discipline.execute({"x": array([1.0]), "y": array([-3.2])})
        {'x': array([1.]), 'y': array([-3.2]), 'z1': array([-5.4]), 'z2': array([-4.4])}
    """

    py_func: Callable
    """The Python function to compute the outputs from the inputs."""

    py_jac: Callable | None
    """The Python function to compute the Jacobian from the inputs."""

    # TODO: API: remove since it is not used.
    use_arrays: bool
    """Whether the function is expected to take arrays as inputs and give outputs as
    arrays."""

    # TODO: API: remove since this feature is provided by the base class.
    input_names: list[str]
    """The names of the inputs."""

    # TODO: API: remove since this feature is provided by the base class.
    output_names: list[str]
    """The names of the outputs."""

    data_processor: AutoDiscDataProcessor
    """A data processor forcing input data to float and output data to arrays."""

    sizes: dict[str, int]
    """The sizes of the input and output variables."""

    __LOG_PREFIX: Final[str] = "Discipline %s: py_func has"

    def __init__(
        self,
        py_func: Callable,
        py_jac: Callable | None = None,
        name: str | None = None,
        use_arrays: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
    ) -> None:
        """
        Args:
            py_func: The Python function to compute the outputs from the inputs.
            py_jac: The Python function to compute the Jacobian from the inputs;
                its output value must be a 2D NumPy array
                with rows corresponding to the outputs and columns to the inputs.
            name: The name of the discipline.
                If ``None``, use the name of the Python function.
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
        self.input_names = getfullargspec(self.py_func).args
        self.output_names = self.__create_output_names()
        have_type_hints = self.__create_grammars()

        if not have_type_hints and not use_arrays:
            # When type hints are used, the conversions will be handled automatically
            # by the grammars.
            self.data_processor = AutoDiscDataProcessor(self.output_names)

        if self.py_jac is None:
            self.set_jacobian_approximation()

        self.__sizes = {}
        self.__jac_shape = []
        self.__input_names_with_namespaces = []
        self.__output_names_with_namespaces = []

    def __create_grammars(self) -> bool:
        """Create the grammars.

        The grammars use type hints from the function if both the arguments and the
        return value have complete type hints. Otherwise, the grammars will have ndarray
        types.

        Returns:
            Whether type hints are used.
        """
        type_hints = get_type_hints(self.py_func)
        return_type = type_hints.pop("return", None)

        # First, determine if both the inputs and outputs have type hints, otherwise
        # that would make things complicated for no good reason.
        names_to_input_types = {}

        if type_hints:
            missing_args_types = set(self.input_names).difference(type_hints.keys())
            if missing_args_types:
                msg = f"{self.__LOG_PREFIX} missing type hints for the arguments: %s."
                LOGGER.warning(msg, self.name, ",".join(missing_args_types))
            else:
                names_to_input_types = type_hints

        names_to_output_types = {}

        if return_type is not None:
            # There could be only one return value of type tuple, or multiple return
            # values that would also be type hinted with tuple.
            if len(self.output_names) == 1:
                names_to_output_types = {self.output_names[0]: return_type}
            else:
                origin = get_origin(return_type)
                if origin is not tuple:
                    msg = (
                        f"{self.__LOG_PREFIX} bad return type hints: "
                        "expecting a tuple of types, got %s."
                    )
                    LOGGER.warning(msg, self.name, return_type)
                else:
                    type_args = get_args(return_type)
                    n_type_args = len(type_args)
                    n_output_names = len(self.output_names)
                    if n_type_args != n_output_names:
                        msg = (
                            f"{self.__LOG_PREFIX} bad return type hints: "
                            "the number of return values and return types shall be "
                            "equal: "
                            "%i return values but %i return type hints."
                        )
                        LOGGER.warning(msg, self.name, n_output_names, n_type_args)
                    else:
                        names_to_output_types = dict(zip(self.output_names, type_args))

        defaults = get_callable_argument_defaults(self.py_func)

        # Second, create the grammar according to the pre-processing above.
        if names_to_input_types and names_to_output_types:
            self.input_grammar.update_from_types(names_to_input_types)
            self.input_grammar.defaults = defaults
            self.output_grammar.update_from_types(names_to_output_types)
            return True

        msg = (
            f"{self.__LOG_PREFIX} inconsistent type hints: "
            "either both the signature arguments and the return values shall have "
            "type hints or none. "
            "The grammars will not use the type hints at all."
        )
        LOGGER.warning(msg, self.name)
        self.input_grammar.update_from_names(self.input_names)

        for key, value in defaults.items():
            if not isinstance(value, ndarray):
                defaults[key] = array([value])
        self.input_grammar.defaults = defaults

        self.output_grammar.update_from_names(self.output_names)
        return False

    def __create_output_names(self) -> list[str]:
        """Create the names of the outputs.

        Returns:
            The names of the outputs.
        """
        output_names = []

        for node in ast.walk(ast.parse(getsource(self.py_func).strip())):
            if not isinstance(node, ast.Return):
                continue

            value = node.value

            if isinstance(value, ast.Tuple):
                temp_output_names = [elt.id for elt in value.elts]
            else:
                temp_output_names = [value.id]

            if output_names and output_names != temp_output_names:
                raise ValueError(
                    "Two return statements use different variable names; "
                    f"{output_names} and {temp_output_names}."
                )

            output_names = temp_output_names

        return output_names

    def _run(self) -> None:
        output_values = self.py_func(**self.get_input_data(with_namespaces=False))
        if len(self.output_names) == 1:
            output_values = {self.output_names[0]: output_values}
        else:
            output_values = dict(zip(self.output_names, output_values))
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
            for name, value in self._local_data.items():
                if name in self.input_grammar:
                    converter = self.input_grammar.data_converter
                else:
                    converter = self.output_grammar.data_converter
                self.__sizes[name] = converter.get_value_size(name, value)

            in_to_ns = self.input_grammar.to_namespaced
            self.__input_names_with_namespaces = [
                in_to_ns.get(input_name, input_name) for input_name in self.input_names
            ]
            out_to_ns = self.output_grammar.to_namespaced
            self.__output_names_with_namespaces = [
                out_to_ns.get(output_name, output_name)
                for output_name in self.output_names
            ]
            self.__jac_shape = (
                sum(
                    self.__sizes[output_name]
                    for output_name in self.__output_names_with_namespaces
                ),
                sum(
                    self.__sizes[input_name]
                    for input_name in self.__input_names_with_namespaces
                ),
            )

        func_jac = self.py_jac(**self.get_input_data(with_namespaces=False))
        if len(func_jac.shape) < 2:
            func_jac = atleast_2d(func_jac)
        if func_jac.shape != self.__jac_shape:
            raise ValueError(
                f"The shape {func_jac.shape} "
                "of the Jacobian matrix "
                f"of the discipline {self.name} "
                f"provided by py_jac "
                "does not match "
                f"(output_size, input_size)={self.__jac_shape}."
            )

        self.jac = split_array_to_dict_of_arrays(
            func_jac,
            self.__sizes,
            self.__output_names_with_namespaces,
            self.__input_names_with_namespaces,
        )


class AutoDiscDataProcessor(DataProcessor):
    """A data processor forcing input data to float and output data to arrays.

    Convert all |g| scalar input data to floats, and convert all discipline output data
    to NumPy arrays.
    """

    # TODO: API: this is never used, remove?
    out_names: Sequence[str]
    """The names of the outputs."""

    # TODO: API: this is never used, remove?
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

    def pre_process_data(self, data: dict[str, DataType]) -> dict[str, DataType]:
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

    def post_process_data(self, data: dict[str, DataType]) -> dict[str, ndarray]:
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


# TODO: API: remove since it is not used.
def to_arrays_dict(data: dict[str, DataType]) -> dict[str, ndarray]:
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
