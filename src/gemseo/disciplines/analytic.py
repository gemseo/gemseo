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
from typing import Iterable
from typing import Mapping

from numpy import array
from numpy import float64
from numpy import heaviside
from numpy import zeros
from sympy import Expr
from sympy import lambdify
from sympy import Symbol
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr

from gemseo.core.discipline import MDODiscipline

LOGGER = logging.getLogger(__name__)


class AnalyticDiscipline(MDODiscipline):
    """A discipline based on analytic expressions.

    Use `SymPy <https://www.sympy.org/>`_, a symbolic calculation engine.

    Compute the Jacobian matrices by automatically differentiating the expressions.

    Example:
        >>> from gemseo.disciplines.analytic import AnalyticDiscipline
        >>> discipline = AnalyticDiscipline({'y_1': '2*x**2', 'y_2': '4*x**2+5+z**3'})
    """

    expressions: Mapping[str, str | Expr]
    """The outputs expressed as functions of the inputs."""

    output_names_to_symbols: dict[str, list[str]]
    """The names of the inputs
    associated to the outputs, e.g. ``{'out': ['in_1', 'in_2']}``."""

    input_names: list[str]
    """The names of the inputs."""

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "expressions",
        "output_names_to_symbols",
        "_fast_evaluation",
        "_sympy_exprs",
        "_sympy_jac_exprs",
    )

    def __init__(
        self,
        expressions: Mapping[str, str | Expr],
        name: str | None = None,
        fast_evaluation: bool = True,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    ) -> None:
        """
        Args:
            expressions: The outputs expressed as functions of the inputs.
            name: The name of the discipline.
                If None, use the class name.
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
        self._init_expressions()
        self._init_grammars()
        self._init_default_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _init_grammars(self) -> None:
        """Initialize the input an output grammars from the expressions' dictionary."""
        self.input_grammar.update(self.input_names)
        self.output_grammar.update(self.expressions.keys())

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
            self.output_names_to_symbols[output_name] = [
                name for name in real_input_symbols
            ]
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

    def _run(self) -> None:
        output_data = {}
        # Do not pass useless tokens to the expr, this may
        # fail when tokens contain dots, or slow down the process
        input_data = self.__convert_input_values_to_float()
        if self._fast_evaluation:
            for output_name, output_function in self._sympy_funcs.items():
                input_symbols = self.output_names_to_symbols[output_name]
                output_value = output_function(
                    *(input_data[input_symbol] for input_symbol in input_symbols)
                )
                output_data[output_name] = array([output_value], dtype=float64)

        else:
            for output_name, output_expression in self._sympy_exprs.items():
                try:
                    output_value = output_expression.evalf(subs=input_data)
                    output_data[output_name] = array([output_value], dtype=float64)
                except TypeError:
                    LOGGER.error(
                        "Failed to evaluate expression : %s", str(output_expression)
                    )
                    LOGGER.error("With inputs : %s", str(self.local_data))
                    raise

        self.store_local_data(**output_data)

    def __convert_input_values_to_float(self) -> dict[str, float]:
        """Return the local data with float values."""
        return {
            input_name: float(self.local_data[input_name].real)
            for input_name in self.get_input_data_names(with_namespaces=False)
        }

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        # otherwise there may be missing terms
        # if some formula have no dependency
        self._init_jacobian(inputs, outputs, with_zeros=True)
        input_values = self.__convert_input_values_to_float()
        if self._fast_evaluation:
            for output_name, gradient_function in self._sympy_jac_funcs.items():
                input_data = tuple(
                    input_values[name]
                    for name in self.output_names_to_symbols[output_name]
                )
                jac = self.jac[output_name]
                for input_symbol, derivative_function in gradient_function.items():
                    jac[input_symbol] = array(
                        [[derivative_function(*input_data)]], dtype=float64
                    )
        else:
            subs = {self.__real_symbols[k]: v for k, v in input_values.items()}
            for output_name, output_expression in self._sympy_exprs.items():
                jac = self.jac[output_name]
                jac_expr = self._sympy_jac_exprs[output_name]
                for input_symbol in output_expression.free_symbols:
                    jac[input_symbol.name] = array(
                        [[jac_expr[input_symbol.name].evalf(subs=subs)]],
                        dtype=float64,
                    )

    def __setstate__(self, state):
        super().__setstate__(state)
        self._sympy_funcs = {}
        self._sympy_jac_funcs = {}
        self._init_expressions()
