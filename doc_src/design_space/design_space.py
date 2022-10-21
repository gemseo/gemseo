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
from __future__ import annotations

from gemseo.api import create_design_space
from gemseo.api import read_design_space
from numpy import array
from numpy import ones

design_space = read_design_space("design_space.txt")
print(design_space)

design_space = read_design_space(
    "design_space_without_header.txt",
    ["name", "lower_bound", "value", "upper_bound", "type"],
)
print(design_space)


design_space = create_design_space()
design_space.add_variable("x1")
design_space.add_variable("x2", var_type="integer")
design_space.add_variable("x3", size=2)
design_space.add_variable("x4", l_b=ones(1))
design_space.add_variable("x5", u_b=ones(1))
design_space.add_variable("x6", value=ones(1))
design_space.add_variable(
    "x7", size=2, var_type="integer", value=array([0, 1]), l_b=-ones(2), u_b=ones(2)
)
print(design_space.get_indexed_variables_names())
print(design_space)
print("normalize", design_space.normalize)
design_space.remove_variable("x4")
print(design_space)

design_space.filter(["x1", "x2", "x3", "x6"])
print(design_space)

design_space.filter_dim("x3", [1])
print(design_space)

design_space.set_current_value(array([1.0, 1.0, 1.0, 1.0]))
design_space.set_current_variable("x1", array([3.0]))
design_space.set_lower_bound("x1", array([-10.0]))
design_space.set_lower_bound("x2", array([-10.0]))
design_space.set_lower_bound("x3", array([-10.0]))
design_space.set_lower_bound("x6", array([-10.0]))
design_space.set_upper_bound("x1", array([10.0]))
design_space.set_upper_bound("x2", array([10.0]))
design_space.set_upper_bound("x3", array([10.0]))
design_space.set_upper_bound("x6", array([10.0]))
print(design_space)

tmp = design_space.array_to_dict(array([1.0, 2.0, 3.0, 4.0]))

print(design_space.get_size("x3"))
print(design_space.get_type("x3"))
print(design_space.get_lower_bound("x3"))
print(design_space.get_upper_bound("x3"))
print(design_space.get_lower_bounds(["x1", "x3"]))
print(design_space.get_upper_bounds(["x1", "x3"]))


print(design_space.has_current_value())

print(design_space.check())

# print design_space.check_membership()

print(design_space.get_current_value())
print(design_space.get_current_value(normalize=True))
print(design_space.get_current_value(as_dict=True))
print("normalize", design_space.normalize)

print("active", design_space.get_active_bounds())
print("active", design_space.get_active_bounds(array([1, 10, 1, 1])))

print(design_space.get_indexed_variables_names())

print(design_space.get_current_value())
design_space.to_complex()
print(design_space.get_current_value())

x_vect = array([1.0, 10.0, 1.0, 1.0])
print("x_vect", x_vect)
normalized_x_vect = design_space.normalize_vect(x_vect, minus_lb=False)
print(normalized_x_vect)
unnormalized_x_vect = design_space.unnormalize_vect(normalized_x_vect)
print(unnormalized_x_vect)

print(design_space.project_into_bounds(array([1.0, 3, -15.0, 23.0])))

print(design_space.round_vect(array([1.3, 3.4, 3.6, -1.4])))

# print design_space.round_vect()

# print design_space.pretty_table(['name', 'value'])

array_point = array([1, 2, 3, 4])
dict_point = design_space.array_to_dict(array_point)
print(dict_point)
new_array_point = design_space.dict_to_array(dict_point)
print(new_array_point)
