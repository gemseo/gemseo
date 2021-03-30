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
"""
Analytic MDODiscipline based on symbolic expressions
****************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library
from numpy import array, float64, zeros
from six import string_types
from sympy.parsing.sympy_parser import parse_expr

from gemseo.core.discipline import MDODiscipline

standard_library.install_aliases()


from gemseo import LOGGER


class AnalyticDiscipline(MDODiscipline):
    """Discipline based on analytic expressions list,
    using the symbolic calculation sympy engine.

    Automatically differentiates the expressions to obtain
    the Jacobian matrices.

    See also
    --------
    gemseo.core.discipline.MDODiscipline : abstract class defining
        the key concept of discipline
    """

    def __init__(self, name=None, expressions_dict=None):
        """
        Constructor

        :param name: name of the discipline.
        :param expressions_dict: dictionary of outputs and their expressions
            for instance : { 'y_1':'2*x**2', 'y_2':'4*x**2+5+z**3'}
            will create a discipline with outputs y_1, y_2
            and inputs x, and z.
        """
        super(AnalyticDiscipline, self).__init__(name)
        if not expressions_dict:
            raise ValueError("expressions_dict is a mandatory argument")
        self.expressions_dict = expressions_dict
        self.expr_symbols_dict = {}
        self.input_names = []
        self._sympy_exprs = {}
        self._sympy_jac_exprs = {}
        self._init_expressions()
        self._init_grammars()
        self._init_default_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _init_grammars(self):
        """Initializes the |g| grammars from the expressions dict"""
        zero = zeros(2)
        in_dict = {k: zero for k in self.input_names}
        self.input_grammar.initialize_from_base_dict(in_dict)
        out_dict = {k: zero for k in self.expressions_dict}
        self.output_grammar.initialize_from_base_dict(out_dict)

    def _init_expressions(self):
        """Parses the expressions and get sympy exprs
        idem for jacobians


        """

        input_symbols = []
        for out_var, expr in self.expressions_dict.items():
            if not isinstance(expr, string_types):
                raise TypeError("Expressions must be an iterable of strings")
            parsed = parse_expr(expr)
            args = list(parsed.free_symbols)
            self._sympy_exprs[out_var] = parsed
            input_symbols += args
            args_names = [arg.name for arg in args]
            self.expr_symbols_dict[out_var] = args_names
            self._sympy_jac_exprs[out_var] = {}
            for arg in args_names:
                jac_expr = parsed.diff(arg)
                self._sympy_jac_exprs[out_var][arg] = jac_expr

        self.input_names = [symb.name for symb in set(input_symbols)]

    def _init_default_inputs(self):
        """Initalizes the default inputs of the discipline
        with zeros


        """
        zeros_array = zeros(1)
        self.default_inputs = {k: zeros_array for k in self.input_names}

    def _run(self):
        """
        Runs the discipline
        """
        outputs = {}
        # Do not pass useless tokens to the expr, this may
        # fail when tokens contains dots, or slow down the process
        filtered_inputs = {key: float(val.real) for key, val in self.local_data.items()}
        for out_var, expr in self._sympy_exprs.items():
            try:
                out_val = expr.evalf(subs=filtered_inputs)
                outputs[out_var] = array([out_val], dtype=float64)
            except TypeError:
                LOGGER.error("Failed to evaluate expression : %s", str(expr))
                LOGGER.error("With inputs : %s", str(self.local_data))
                raise
        self.store_local_data(**outputs)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian

        :param inputs: Default value = None)
        :param outputs: Default value = None)
        """
        # otherwise there may be missing terms
        # if some formula have no dependency
        self._init_jacobian(inputs, outputs, with_zeros=True)
        filtered_inputs = {key: float(val.real) for key, val in self.local_data.items()}
        for out_var, expr in self._sympy_exprs.items():
            for arg in expr.free_symbols:
                j_expr = self._sympy_jac_exprs[out_var][arg.name]
                jac_val = j_expr.evalf(subs=filtered_inputs)
                self.jac[out_var][arg.name] = array([[jac_val]], dtype=float64)
