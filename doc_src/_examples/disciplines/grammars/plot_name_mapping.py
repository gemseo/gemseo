# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
Rename discipline variables
===========================
"""

from __future__ import annotations

from numpy import array

from gemseo.core.discipline.data_processor import NameMapping
from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# :class:`.NameMapping` is a :class:`.DataProcessor`
# to rename one or more variables of a :class:`.Discipline`.
#
# In this example,
# we consider an :class:`.AnalyticDiscipline`
# computing the sum and the difference of two operands:
discipline = AnalyticDiscipline(
    {
        "sum": "first_operand+second_operand",
        "diff": "first_operand-second_operand",
    },
    name="SumAndDiff",
)
discipline.io.input_grammar.defaults = {
    "first_operand": array([1.0]),
    "second_operand": array([2.0]),
}
# %%
# We want to use this discipline in a study
# to sum and subtract the input variables ``"x1"`` and ``"x2"``
# and return the output variables ``"y1"`` and ``"y2"``.
# Unfortunately,
# its input names ``"first_operand"`` and ``"second_operand"``
# and the output names ``"sum"`` and ``"diff"``
# do not match the naming of our user study.
# For this simple example,
# we could have easily created a new :class:`.AnalyticDiscipline`
# with an expression dictionary using ``"x1"``, ``"x2"``, ``"y1"`` and ``"y2"``.
# but this solution is not generic
# because in practice,
# the discipline's :meth:`_run` method manipulating the data from the variable names
# cannot be modified.
# To fix this problem,
# we can rename the input and output variables:
discipline.io.input_grammar.rename_element("first_operand", "x1")
discipline.io.input_grammar.rename_element("second_operand", "x2")
discipline.io.output_grammar.rename_element("sum", "y1")
discipline.io.output_grammar.rename_element("diff", "y2")
# %%
# and set its :class:`.DataProcessor` to a :class:`.NameMapping`
# defined from a dictionary of the form ``{new_variable_name: variable_name}``:
discipline.io.data_processor = NameMapping({
    "x1": "first_operand",
    "x2": "second_operand",
    "y1": "sum",
    "y2": "diff",
})
# %%
# We can verify that the discipline can be executed correctly:
discipline.execute()
discipline.io.get_input_data(), discipline.io.get_output_data()

# %%
# This :class:`.DataProcessor` is compatible with the use of namespaces:
discipline.add_namespace_to_input("x1", "ns_in")
discipline.add_namespace_to_output("y1", "ns_out")
discipline.execute()
discipline.io.get_input_data(), discipline.io.get_output_data()

# %%
# Finally,
# we may also be interested in
# :ref:`this example <sphx_glr_examples_disciplines_variables_plot_variable_renaming.py>`,
# which illustrates the use of :func:`.rename_discipline_variables`.
# Given a collection of disciplines
# and a dictionary of translations
# generated either by hand or by a user-friendly interface,
# this function automates the process presented in the current example,
# namely renaming the input variables,
# renaming the output variables
# and using a :class:`.NameMapping`.
