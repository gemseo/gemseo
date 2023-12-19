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
"""A discipline based on analytic expressions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

import numpy
from numpy import array
from numpy import expand_dims
from numpy import float64
from numpy import heaviside
from numpy import ndarray
from numpy import zeros
from sympy import Expr
from sympy import Symbol
from sympy import lambdify
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


class AnalyticDiscipline(MDODiscipline):
    """A discipline based on analytic expressions.

    Use `SymPy <https://www.sympy.org/>`_, a symbolic calculation engine.

    Compute the Jacobian matrices by automatically differentiating the expressions.

    Examples:
        >>> from gemseo.disciplines.analytic import AnalyticDiscipline
        >>> discipline = AnalyticDiscipline({"y_1": "2*x**2", "y_2": "4*x**2+5+z**3"})
    """

    expressions: Mapping[str, str | Expr]
    """The outputs expressed as functions of the inputs."""

    output_names_to_symbols: dict[str, list[str]]
    """The names of the inputs associated to the outputs, e.g. ``{'out': ['in_1',
    'in_2']}``."""

    input_names: list[str]
    """The names of the inputs."""

    _ATTR_NOT_TO_SERIALIZE = MDODiscipline._ATTR_NOT_TO_SERIALIZE.union([
        "_sympy_funcs",
        "_sympy_jac_funcs",
    ])

    # TODO: API: remove fast_evaluation (see the docstring of sympy.lambdify)
    def __init__(
        self,
        expressions: Mapping[str, str | Expr],
        name: str | None = None,
        fast_evaluation: bool = True,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
    ) -> None:
        """
        Args:
            expressions: The outputs expressed as functions of the inputs.
            name: The name of the discipline.
                If ``None``, use the class name.
            fast_evaluation: Whether to apply ``sympy.lambdify`` to the expressions
                in order to accelerate their numerical evaluation;
                otherwise the expressions are evaluated with ``sympy.Expr.evalf``.
        """  # noqa: D205, D212, D415
        super().__init__(name, grammar_type=grammar_type)
        self.expressions = expressions
        self.output_names_to_symbols = {}
        self.input_names = []
        self._sympy_exprs = {}
        self._sympy_funcs = {}
        self._sympy_jac_exprs = {}
        self._sympy_jac_funcs = {}
        self._fast_evaluation = fast_evaluation
        if fast_evaluation:
            self._run = self._run_with_fast_evaluation
            self._compute_jacobian = self._compute_jacobian_with_fast_evaluation
        else:
            self._run = self._run_without_fast_evaluation
            self._compute_jacobian = self._compute_jacobian_without_fast_evaluation

        self._init_expressions()
        self._init_grammars()
        self._init_default_inputs()
        self.re_exec_policy = self.ReExecutionPolicy.DONE

    def _init_grammars(self) -> None:
        """Initialize the input an output grammars from the expressions' dictionary."""
        self.input_grammar.update_from_names(self.input_names)
        self.output_grammar.update_from_names(self.expressions.keys())

    def _init_expressions(self) -> None:
        """Parse the expressions of the functions and their derivatives.

        Get SymPy expressions from string expressions.

        Raises:
            TypeError: When the expression is neither a SymPy expression nor a string.
        """
        all_real_input_symbols = []
        for output_name, output_expression in self.expressions.items():
            if isinstance(output_expression, Expr):
                output_expression_to_derive = output_expression
                real_input_symbols = self.__create_real_input_symbols(output_expression)
            elif isinstance(output_expression, str):
                string_output_expression = output_expression
                output_expression = parse_expr(string_output_expression)
                real_input_symbols = self.__create_real_input_symbols(output_expression)
                output_expression_to_derive = parse_expr(
                    string_output_expression, local_dict=real_input_symbols
                )
            else:
                raise TypeError("Expression must be a SymPy expression or a string.")

            self._sympy_exprs[output_name] = output_expression
            all_real_input_symbols.extend(real_input_symbols.values())
            self.output_names_to_symbols[output_name] = list(real_input_symbols)
            self._sympy_jac_exprs[output_name] = {
                input_symbol_name: output_expression_to_derive.diff(input_symbol)
                for input_symbol_name, input_symbol in real_input_symbols.items()
            }

        self.input_names = sorted(
            input_symbol.name for input_symbol in set(all_real_input_symbols)
        )

        self.__real_symbols = {symbol.name: symbol for symbol in all_real_input_symbols}

        if self._fast_evaluation:
            self._lambdify_expressions()

    @staticmethod
    def __create_real_input_symbols(expression: Expr) -> dict[str, Symbol]:
        """Return the symbols used by a SymPy expression with real type.

        Args:
            expression: The SymPy expression.

        Returns:
            The symbols used by ``expression`` with real type.
        """
        return {
            symbol.name: symbols(symbol.name, real=True)
            for symbol in expression.free_symbols
        }

    def _lambdify_expressions(self) -> None:
        """Lambdify the SymPy expressions."""
        numpy_str = "numpy"

        modules = [numpy_str, {"Heaviside": lambda x: heaviside(x, 1)}]
        for output_name, output_expression in self._sympy_exprs.items():
            input_names = self.output_names_to_symbols[output_name]
            self._sympy_funcs[output_name] = lambdify(
                list(output_expression.free_symbols), output_expression
            )
            input_symbols = [self.__real_symbols[k] for k in input_names]
            jac_expr = self._sympy_jac_exprs[output_name]
            self._sympy_jac_funcs[output_name] = {
                input_symbol.name: lambdify(
                    input_symbols,
                    jac_expr[input_symbol.name],
                    modules,
                )
                for input_symbol in input_symbols
            }

    def _init_default_inputs(self) -> None:
        """Initialize the default inputs of the discipline with zeros."""
        self.default_inputs = {
            input_name: zeros(1) for input_name in self.get_input_data_names()
        }

    @staticmethod
    def __cast_expression_to_array(expression: Expr) -> ndarray:
        """Cast a SymPy expression to a NumPy array.

        Args:
            expression: The SymPy expression to cast.

        Returns:
            The NumPy array.
        """
        if expression.is_integer:
            data_type = int
        elif expression.is_real:
            data_type = float
        else:
            data_type = complex

        return array([expression], data_type)

    def _run_with_fast_evaluation(self) -> None:
        """Run the discipline with fast evaluation."""
        output_data = {}
        # Do not pass useless tokens to the expr, this may
        # fail when tokens contain dots, or slow down the process
        input_data = self.__get_local_data_without_namespace()
        for output_name, output_function in self._sympy_funcs.items():
            input_symbols = self.output_names_to_symbols[output_name]
            output_value = output_function(
                *(input_data[input_symbol] for input_symbol in input_symbols)
            )
            output_data[output_name] = expand_dims(output_value, 0)
        self.store_local_data(**output_data)

    def _run_without_fast_evaluation(self) -> None:
        """Run the discipline without fast evaluation."""
        output_data = {}
        # Do not pass useless tokens to the expr, this may
        # fail when tokens contain dots, or slow down the process
        input_data = self.__get_local_data_without_namespace()
        for output_name, output_expression in self._sympy_exprs.items():
            try:
                output_value = output_expression.evalf(subs=input_data)
                output_data[output_name] = self.__cast_expression_to_array(output_value)
            except TypeError:  # noqa: PERF203
                LOGGER.exception(
                    "Failed to evaluate expression : %s", output_expression
                )
                LOGGER.exception("With inputs : %s", self.local_data)
                raise

        self.store_local_data(**output_data)

    def __get_local_data_without_namespace(self) -> dict[str, numpy.number]:
        """Return the local data without namespace prefixes.

        Returns:
            The local data without namespace prefixes.
        """
        return {
            input_name: self.local_data[input_name].item()
            for input_name in self.get_input_data_names(with_namespaces=False)
        }

    def _compute_jacobian_with_fast_evaluation(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        # otherwise there may be missing terms
        # if some formula have no dependency
        input_names, output_names = self._init_jacobian(inputs, outputs)
        input_values = self.__get_local_data_without_namespace()
        for output_name in output_names:
            gradient_function = self._sympy_jac_funcs[output_name]
            input_data = tuple(
                input_values[name] for name in self.output_names_to_symbols[output_name]
            )
            jac = self.jac[output_name]
            for input_symbol, derivative_function in gradient_function.items():
                if input_symbol in input_names:
                    jac[input_symbol] = array(
                        [[derivative_function(*input_data)]], dtype=float64
                    )

    def _compute_jacobian_without_fast_evaluation(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        # otherwise there may be missing terms
        # if some formula have no dependency
        input_names, output_names = self._init_jacobian(inputs, outputs)
        input_values = self.__get_local_data_without_namespace()
        subs = {self.__real_symbols[k]: v for k, v in input_values.items()}
        for output_name in output_names:
            output_expression = self._sympy_exprs[output_name]
            jac = self.jac[output_name]
            jac_expr = self._sympy_jac_exprs[output_name]
            for input_symbol in output_expression.free_symbols:
                input_name = input_symbol.name
                if input_name in input_names:
                    jac[input_name] = array(
                        [[jac_expr[input_name].evalf(subs=subs)]],
                        dtype=float64,
                    )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super().__setstate__(state)
        self._sympy_funcs = {}
        self._sympy_jac_funcs = {}
        self._init_expressions()
