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
#  Contributors:
#     INITIAL AUTHORS - initial API and implementation
#                       and/or initial documentation
#        @author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.api import create_discipline
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
from numpy import arange

sellar = create_discipline("Sellar1")
design_space = SellarDesignSpace()
input_names = sellar.get_input_data_names()
output_names = sellar.get_output_data_names()
var_lb = {name: design_space.get_lower_bound(name) for name in input_names}
var_ub = {name: design_space.get_upper_bound(name) for name in input_names}

variables_sizes = 5
sizes = {name: variables_sizes for name in input_names + output_names}
scalable_sellar = create_discipline(
    "ScalableFittedDiscipline",
    discipline=sellar,
    var_lb=var_lb,
    var_ub=var_ub,
    sizes=sizes,
    fill_factor=0.6,
)

input_data = {
    name: arange(variables_sizes) / float(variables_sizes) for name in input_names
}
print(scalable_sellar.execute(input_data)["y_1"])

variables_sizes = 3
sizes = {name: variables_sizes for name in input_names + output_names}
scalable_sellar = create_discipline(
    "ScalableFittedDiscipline",
    discipline=sellar,
    var_lb=var_lb,
    var_ub=var_ub,
    sizes=sizes,
    fill_factor=0.6,
)

input_data = {
    name: arange(variables_sizes) / float(variables_sizes) for name in input_names
}
print(scalable_sellar.execute(input_data)["y_1"])
