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
"""A discipline interfacing a Python function automatically."""

from __future__ import annotations

import ast
import logging
from inspect import getsource
from inspect import signature
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

from gemseo.core.discipline import Discipline
from gemseo.core.discipline.data_processor import DataProcessor
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.source_parsing import get_callable_argument_defaults
from gemseo.utils.source_parsing import get_options_doc
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping

DataType = Union[float, ndarray]

LOGGER = logging.getLogger(__name__)


class AutoPyDiscipline(Discipline):
    """Wrap a Python function into a discipline.

    A simplified and straightforward way of integrating a discipline
    from a Python function that:

    - returns variables,
      e.g. ``return x`` or ``return x, y``,
      but no expression like ``return a+b`` or ``return a+b, y``,
    - must have a default value per argument
      if the :class:`.AutoPyDiscipline` is used by an ``MDA``
      (deriving from :class:`.BaseMDA`),
      as in the case of :class:`.MDF` and :class:`.BiLevel` formulations,
      in the presence of strong couplings.

    The input names of the discipline are the names of the Python function arguments
    and the output names are the names of the variable listed in the return statement.

    By default,
    the arguments and returned variables are assumed to be
    either scalars
    or NumPy arrays of length greater than 1.
    When ``use_arrays`` is ``True``,
    the scalar arguments are assumed to be NumPy arrays of length equal to 1.
    When *all* the arguments and returned variables have type hints,
    these types are used by the input and output grammars.

    The default input values are the default values of the Python function arguments,
    if any.

    :ref:`This example <sphx_glr_examples_disciplines_types_plot_auto_py_discipline.py>`
    from the documentation
    illustrates this feature.
    """

    __input_names: tuple[str, ...]
    """The names of the input variables."""

    __input_names_with_namespaces: tuple[str, ...]
    """The namespaced names of the input variables."""

    __jac_shape: tuple[int, int]
    """The shape of the Jacobian matrix."""

    __output_names: tuple[str, ...]
    """The names of the output variables."""

    __output_names_with_namespaces: tuple[str, ...]
    """The namespaced names of the output variables."""

    __py_func: Callable
    """The Python function to compute the outputs from the inputs."""

    __py_jac: Callable | None
    """The Python function to compute the Jacobian from the inputs."""

    __sizes: dict[str, int]
    """The sizes of the input and output variables."""

    __LOG_PREFIX: Final[str] = "The py_func of the AutoPyDiscipline '%s' has"

    __LOG_SUFFIX: Final[str] = (
        "The grammars of this discipline will not use the type hints at all."
    )

    def __init__(
        self,
        py_func: Callable,
        py_jac: Callable | None = None,
        name: str = "",
        use_arrays: bool = False,
    ) -> None:
        """
        Args:
            py_func: The Python function to compute the outputs from the inputs.
            py_jac: The Python function to compute the Jacobian from the inputs;
                its output value must be a 2D NumPy array
                with rows corresponding to the outputs and columns to the inputs.
            name: The name of the discipline.
                If empty, use the name of the Python function.
            use_arrays: Whether the function ``py_func`` is expected
                to take arrays as inputs and give outputs as arrays.

        Raises:
            ValueError: Either when the function returns an expression
                or when two return statements use different variables.
        """  # noqa: D205 D212 D415
        super().__init__(name=name or py_func.__name__)
        self.__py_func = py_func
        self.__py_jac = py_jac
        self.__input_names = tuple(signature(self.__py_func).parameters)
        self.__output_names = self.__create_output_names()
        have_type_hints = self.__create_grammars()

        if not have_type_hints and not use_arrays:
            # When type hints are used, the conversions will be handled automatically
            # by the grammars.
            self.io.data_processor = AutoDiscDataProcessor()

        if self.__py_jac is None:
            self.set_jacobian_approximation()

        self.__sizes = {}
        self.__jac_shape = (0, 0)
        self.__input_names_with_namespaces = ()
        self.__output_names_with_namespaces = ()

    # TODO: API: remove and use self.io.input_grammar.names instead.
    @property
    def input_names(self) -> list[str]:
        """The names of the input variables."""
        return list(self.__input_names)

    # TODO: API: remove and use self.io.output_grammar.names instead.
    @property
    def output_names(self) -> list[str]:
        """The names of the output variables."""
        return list(self.__output_names)

    @property
    def py_func(self) -> Callable:
        """The Python function to compute the outputs from the inputs."""
        return self.__py_func

    @property
    def py_jac(self) -> Callable:
        """The Python function to compute the Jacobian from the inputs."""
        return self.__py_jac

    def __create_grammars(self) -> bool:
        """Create the grammars.

        The grammars use type hints from the function if both the arguments and the
        return value have complete type hints. Otherwise, the grammars will have ndarray
        types.

        Returns:
            Whether type hints are used.
        """
        type_hints = get_type_hints(self.__py_func)
        return_type = type_hints.pop("return", None)

        # First, determine if both the inputs and outputs have type hints, otherwise
        # that would make things complicated for no good reason.
        names_to_input_types = {}
        raise_if_inconsistency = True

        if type_hints:
            missing_args_types = set(self.__input_names).difference(type_hints.keys())
            if missing_args_types:
                msg = (
                    f"{self.__LOG_PREFIX} missing type hints for the arguments %s."
                    f"{self.__LOG_SUFFIX}"
                )
                LOGGER.warning(
                    msg, self.name, pretty_repr(missing_args_types, use_and=True)
                )
                raise_if_inconsistency = False
            else:
                names_to_input_types = type_hints

        names_to_output_types = {}

        if return_type is not None:
            # There could be only one return value of type tuple, or multiple return
            # values that would also be type hinted with tuple.
            if len(self.__output_names) == 1:
                names_to_output_types = {self.__output_names[0]: return_type}
            else:
                origin = get_origin(return_type)
                if origin is not tuple:
                    msg = (
                        f"{self.__LOG_PREFIX} bad return type hints: "
                        "expecting a tuple of types, got %s."
                        f"{self.__LOG_SUFFIX}"
                    )
                    LOGGER.warning(msg, self.name, return_type)
                    raise_if_inconsistency = False
                else:
                    type_args = get_args(return_type)
                    n_type_args = len(type_args)
                    n_output_names = len(self.__output_names)
                    if n_type_args != n_output_names:
                        msg = (
                            f"{self.__LOG_PREFIX} bad return type hints: "
                            "the number of return values (%i) and return types (%i) "
                            "shall be equal. "
                            f"{self.__LOG_SUFFIX}"
                        )
                        LOGGER.warning(msg, self.name, n_output_names, n_type_args)
                        raise_if_inconsistency = False
                    else:
                        names_to_output_types = dict(
                            zip(self.__output_names, type_args)
                        )

        defaults = get_callable_argument_defaults(self.__py_func)

        try:
            input_descriptions = get_options_doc(self.__py_func)
        except ValueError:
            input_descriptions = {}

        # Second, create the grammar according to the pre-processing above.
        if names_to_input_types and names_to_output_types:
            self.io.input_grammar.update_from_types(names_to_input_types)
            self.io.input_grammar.defaults = defaults
            self.io.input_grammar.descriptions.update(input_descriptions)
            self.io.output_grammar.update_from_types(names_to_output_types)
            return True

        if raise_if_inconsistency and (names_to_input_types or names_to_output_types):
            msg = (
                f"{self.__LOG_PREFIX} inconsistent type hints: "
                "either both the signature arguments and the return values shall have "
                "type hints or none. "
                f"{self.__LOG_SUFFIX}"
            )
            LOGGER.warning(msg, self.name)

        self.io.input_grammar.update_from_names(self.__input_names)
        for key, value in defaults.items():
            if not isinstance(value, ndarray):
                defaults[key] = array([value])
        self.io.input_grammar.defaults = defaults
        self.io.input_grammar.descriptions.update(input_descriptions)

        self.io.output_grammar.update_from_names(self.__output_names)
        return False

    def __create_output_names(self) -> tuple[str, ...]:
        """Create the names of the outputs.

        Returns:
            The names of the outputs.
        """
        output_names = []

        for node in ast.walk(ast.parse(getsource(self.__py_func).strip())):
            if not isinstance(node, ast.Return):
                continue

            value = node.value
            elements = value.elts if isinstance(value, ast.Tuple) else [value]
            temp_output_names = []
            for element in elements:
                if hasattr(element, "id"):
                    temp_output_names.append(element.id)
                    continue

                msg = (
                    "The function must return one or more variables, "
                    "e.g. 'return x' or 'return x, y',"
                    "but no expression like 'return a+b' or 'return a+b, y'."
                )
                raise ValueError(msg)

            if output_names and output_names != temp_output_names:
                msg = (
                    "Two return statements use different variable names; "
                    f"{output_names} and {temp_output_names}."
                )
                raise ValueError(msg)

            output_names = temp_output_names

        return tuple(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        output_values = self.__py_func(**input_data)
        if len(self.__output_names) == 1:
            output_values = {self.__output_names[0]: output_values}
        else:
            output_values = dict(zip(self.__output_names, output_values))
        return output_values

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Raises:
            RuntimeError: When the analytic Jacobian :attr:`.py_jac` is ``None``.
            ValueError: When the Jacobian shape is inconsistent.
        """  # noqa: D205 D212 D415
        if self.__py_jac is None:
            msg = "The analytic Jacobian is missing."
            raise RuntimeError(msg)

        if not self.__sizes:
            for name, value in self.io.data.items():
                if name in self.io.input_grammar:
                    converter = self.io.input_grammar.data_converter
                else:
                    converter = self.io.output_grammar.data_converter
                self.__sizes[name] = converter.get_value_size(name, value)

            in_to_ns = self.io.input_grammar.to_namespaced
            self.__input_names_with_namespaces = tuple(
                in_to_ns.get(input_name, input_name)
                for input_name in self.__input_names
            )
            out_to_ns = self.io.output_grammar.to_namespaced
            self.__output_names_with_namespaces = tuple(
                out_to_ns.get(output_name, output_name)
                for output_name in self.__output_names
            )
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

        func_jac = self.__py_jac(**self.io.get_input_data(with_namespaces=False))
        if len(func_jac.shape) < 2:
            func_jac = atleast_2d(func_jac)
        if func_jac.shape != self.__jac_shape:
            msg = (
                f"The shape {func_jac.shape} "
                "of the Jacobian matrix "
                f"of the discipline {self.name} "
                f"provided by py_jac "
                "does not match "
                f"(output_size, input_size)={self.__jac_shape}."
            )
            raise ValueError(msg)

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

    def pre_process_data(self, data: dict[str, DataType]) -> dict[str, DataType]:
        """Pre-process the input data.

        Execute a pre-processing of input data
        after they are checked by :meth:`~Discipline.validate_input_data`,
        and before the :meth:`~Discipline._run` method of the discipline is called.

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
        after the :meth:`~Discipline._run` method of the discipline is called,
        and before they are checked by :meth:`~Discipline.validate_output_data`.

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
