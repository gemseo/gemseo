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
"""Created on Mar 25, 2019.

@author: matthias.delozzo
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import generate_coupling_graph
from gemseo.api import generate_n2_plot
from numpy import array
from numpy import exp

configure_logger()


def py_function_1(inpt=0.0, output_2=0.0):
    output_1 = exp(-(inpt**2 + output_2**2))
    return output_1


def py_function_2(inpt=0.0, output_1=0.0):
    output_2 = exp(-(inpt**2 + output_1**2))
    return output_2


def py_function_3(inpt=0.0, output_1=0.0, output_2=0.0):
    obj = inpt * (output_1 + output_2)
    return obj


discipline_1 = create_discipline("AutoPyDiscipline", py_func=py_function_1)
discipline_2 = create_discipline("AutoPyDiscipline", py_func=py_function_2)
discipline_3 = create_discipline("AutoPyDiscipline", py_func=py_function_3)
disciplines = [discipline_1, discipline_2, discipline_3]

generate_coupling_graph(disciplines)
generate_n2_plot(disciplines)

design_space = create_design_space()
design_space.add_variable("inpt", 1, "float", array([-1.0]), array([1.0]), array([0.0]))

scenario = create_scenario(disciplines, "MDF", "obj", design_space)
scenario.execute({"algo": "SLSQP", "max_iter": 100})
scenario.post_process("OptHistoryView", save=True, show=False)
