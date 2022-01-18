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
"""A discipline based on analytic expressions."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping, Optional, Union

from numpy import array, float64, heaviside, zeros
from six import string_types
from sympy import Expr, lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.py23_compat import PY2

LOGGER = logging.getLogger(__name__)


class AnalyticDiscipline(MDODiscipline):
    """A discipline based on analytic expressions.

    Use `SymPy <https://www.sympy.org/>`_, a symbolic calculation engine.

    Compute the Jacobian matrices by automatically differentiating the expressions.

    Attributes:
        expressions (Mapping[str,Union[str,Expr]]): The outputs
            expressed as functions of the inputs.
        expr_symbols_dict (Dict[str, List[str]]): The names of the inputs
            associated to the outputs, e.g. ``{'out': ['in_1', 'in_2']}``.
        input_names (List[str]): The names of the inputs.

    Example:
        >>> from gemseo.disciplines.analytic import AnalyticDiscipline
        >>> discipline = AnalyticDiscipline({'y_1': '2*x**2', 'y_2': '4*x**2+5+z**3'})
    """

    def __init__(
        self,
        expressions,  # type: Mapping[str,Union[str,Expr]]
        name=None,  # type: Optional[str]
        fast_evaluation=True,  # type: bool
    ):  # type: (...) -> None
        # noqa: D205 D212 D415
        """
        Args:
            expressions: The outputs expressed as functions of the inputs.
            name: The name of the discipline.
                If None, use the class name.
            fast_evaluation: Whether to apply ``sympy.lambdify`` to the expressions
                in order to accelerate their numerical evaluation;
                otherwise the expressions are evaluated with ``sympy.Expr.evalf``.
        """
        super(AnalyticDiscipline, self).__init__(name)
        self.expressions = expressions
        self.expr_symbols_dict = {}
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

    def _init_grammars(self):  # type: (...) -> None
        """Initialize the input an output grammars from the expressions dictionary."""
        self.input_grammar.initialize_from_data_names(self.input_names)
        self.output_grammar.initialize_from_data_names(self.expressions.keys())

    def _init_expressions(self):  # type: (...) -> None
        """Parse the expressions of the functions and their derivatives.

        Get SymPy expressions from string expressions.

        Raises:
            TypeError: When the expression is neither a SymPy expression nor a string.
        """
        input_symbols = []
        for output_name, output_expression in self.expressions.items():
            if isinstance(output_expression, Expr):
                sympy_expression = output_expression
            elif isinstance(output_expression, string_types):
                sympy_expression = parse_expr(output_expression)
            else:
                raise TypeError("Expression must be a SymPy expression or a string.")

            self._sympy_exprs[output_name] = sympy_expression
            free_symbols = list(sympy_expression.free_symbols)
            input_symbols += free_symbols
            input_names_subset = [free_symbol.name for free_symbol in free_symbols]
            self.expr_symbols_dict[output_name] = input_names_subset
            self._sympy_jac_exprs[output_name] = {
                input_name: sympy_expression.diff(input_name)
                for input_name in input_names_subset
            }

        self.input_names = sorted(
            [input_symbol.name for input_symbol in set(input_symbols)]
        )

        if self._fast_evaluation:
            self._lambdify_expressions()

    def _lambdify_expressions(self):  # type: (...) -> None
        """Lambdify the SymPy expressions."""
        numpy_str = "numpy"
        if PY2:
            numpy_str = numpy_str.encode("ascii")

        modules = [numpy_str, {"Heaviside": lambda x: heaviside(x, 1)}]
        for output_name, output_expression in self._sympy_exprs.items():
            input_names = self.expr_symbols_dict[output_name]
            input_symbols = symbols(input_names)
            self._sympy_funcs[output_name] = lambdify(input_symbols, output_expression)
            self._sympy_jac_funcs[output_name] = {
                input_name: lambdify(
                    input_symbols,
                    self._sympy_jac_exprs[output_name][input_name],
                    modules,
                )
                for input_name in input_names
            }

    def _init_default_inputs(self):  # type: (...) -> None
        """Initialize the default inputs of the discipline with zeros."""
        zeros_array = zeros(1)
        self.default_inputs = {
            input_name: zeros_array for input_name in self.get_input_data_names()
        }

    def _run(self):  # type: (...) -> None
        outputs = {}
        # Do not pass useless tokens to the expr, this may
        # fail when tokens contains dots, or slow down the process
        input_values = self.__convert_input_values_to_float()
        if self._fast_evaluation:
            for output_name, output_function in self._sympy_funcs.items():
                input_names = self.expr_symbols_dict[output_name]
                output_value = output_function(
                    *(input_values[input_name] for input_name in input_names)
                )
                outputs[output_name] = array([output_value], dtype=float64)

        else:
            for output_name, output_expression in self._sympy_exprs.items():
                try:
                    output_value = output_expression.evalf(subs=input_values)
                    outputs[output_name] = array([output_value], dtype=float64)
                except TypeError:
                    LOGGER.error(
                        "Failed to evaluate expression : %s", str(output_expression)
                    )
                    LOGGER.error("With inputs : %s", str(self.local_data))
                    raise

        self.store_local_data(**outputs)

    def __convert_input_values_to_float(self):  # type: (...) -> Dict[str, float]
        """Return the local data with float values."""
        return {
            input_name: float(input_value.real)
            for input_name, input_value in self.local_data.items()
        }

    def _compute_jacobian(
        self,
        inputs=None,  # type:Optional[Iterable[str]]
        outputs=None,  # type:Optional[Iterable[str]]
    ):  # type: (...)-> None
        # otherwise there may be missing terms
        # if some formula have no dependency
        self._init_jacobian(inputs, outputs, with_zeros=True)
        input_values = self.__convert_input_values_to_float()
        if self._fast_evaluation:
            for output_name, gradient_function in self._sympy_jac_funcs.items():
                input_names = self.expr_symbols_dict[output_name]
                for input_name, derivative_function in gradient_function.items():
                    derivative_value = derivative_function(
                        *(input_values[input_name] for input_name in input_names)
                    )
                    self.jac[output_name][input_name] = array(
                        [[derivative_value]], dtype=float64
                    )
        else:
            for output_name, output_expression in self._sympy_exprs.items():
                for input_symbol in output_expression.free_symbols:
                    input_name = input_symbol.name
                    derivative_expr = self._sympy_jac_exprs[output_name][input_name]
                    self.jac[output_name][input_name] = array(
                        [[derivative_expr.evalf(subs=input_values)]], dtype=float64
                    )