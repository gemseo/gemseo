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

# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Use data converters

## Problem

GEMSEO requires input and output data to be formatted as 1D NumPy arrays
to ensure compatibility with its optimization and differentiation drivers.
However,
real-world disciplines often involve complex data structures
such as multi-dimensional arrays,
nested dictionaries,
or custom objects.
How can these high-dimensional
or non-standard data types be seamlessly integrated into a GEMSEO-compliant workflow?

## Solution

GEMSEO leverages a
[BaseDataConverter][gemseo.core.data_converters.base.BaseDataConverter]
to manage the transformation of variable values into 1D NumPy arrays and vice versa.
Beyond just flattening and reshaping data,
this utility also provides the framework
with the expected dimensions of these 1D arrays,
ensuring consistency across the coupling process.

## Step-by-step guide

"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from numpy import array
from numpy import exp
from numpy import ones

from gemseo import set_data_converters
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.core.discipline import Discipline
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.scenarios.mdo import MDOScenario

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


# %%
# ### 1. Define the disciplines
#
# As compared to the original Sellar example,
# here the `y_1` variable is a dictionary.
# and the `y_2` variable is a 2D array
# The grammars are changed to `SimpleGrammar` for simplicity,
# otherwise,
# with the default `JSONGrammar`,
# more complex definitions would be required.

DEFAULT_INPUT_DATA = {
    "x": ones(1),
    "z": array([1.0, 0.0]),
    "y_1": {"dummy": ones(1)},
    "y_2": ones((1, 1)),
}


class SellarSystem(Discipline):
    default_grammar_type = Discipline.default_grammar_type.SIMPLE

    def __init__(self) -> None:
        super().__init__()
        default_input_data = DEFAULT_INPUT_DATA.copy()
        self.input_grammar.update_from_data(default_input_data)
        self.output_grammar.update_from_names(["obj", "c_1", "c_2"])
        self.default_input_data = default_input_data

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        z = input_data["z"]
        y_1 = input_data["y_1"]["dummy"]
        y_2 = input_data["y_2"][0]
        return {
            "obj": array([x[0] ** 2 + z[1] + y_1[0] ** 2 + exp(-y_2[0])]),
            "c_1": array([3.16 - y_1[0] ** 2]),
            "c_2": array([y_2[0] - 24.0]),
        }


class Sellar1(Discipline):
    default_grammar_type = Discipline.default_grammar_type.SIMPLE

    def __init__(self) -> None:
        super().__init__()
        default_input_data = DEFAULT_INPUT_DATA.copy()
        # Keep the y_1 data for easily defining the output grammar.
        output_grammar_data = {"y_1": default_input_data.pop("y_1")}
        self.input_grammar.update_from_data(default_input_data)
        self.output_grammar.update_from_data(output_grammar_data)
        self.default_input_data = default_input_data

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        z = input_data["z"]
        y_2 = input_data["y_2"][0]
        return {
            "y_1": {"dummy": array([(z[0] ** 2 + z[1] + x[0] - 0.2 * y_2[0]) ** 0.5])}
        }


class Sellar2(Discipline):
    default_grammar_type = Discipline.default_grammar_type.SIMPLE

    def __init__(self) -> None:
        super().__init__()
        default_input_data = DEFAULT_INPUT_DATA.copy()
        del default_input_data["y_2"]
        del default_input_data["x"]
        self.input_grammar.update_from_data(default_input_data)
        self.output_grammar.update_from_names(["y_2"])
        self.default_input_data = default_input_data

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        z = input_data["z"]
        y_1 = input_data["y_1"]["dummy"]
        return {"y_2": array([[abs(y_1[0]) + z[0] + z[1]]])}


# %%
# ### 2. Set the data converter
# Define the data converters for the custom types of the variables `y_1` and `y_2`.
#
# This one shall return the value of a variable as a 1D array.
to_array = {
    "y_1": operator.itemgetter("dummy"),
    "y_2": operator.itemgetter(0),
}


# %%
# This one shall return the value of a variable in the type expected by the discipline from a 1D array.
def get_dict(array_):
    return {"dummy": array_}


def get_2d_array(array_):
    return array([array_])


from_array = {
    "y_1": get_dict,
    "y_2": get_2d_array,
}


# %%
# This one shall return the size of a value of a variable.
# The builtin converters already handle NumPy arrays via the `size` attribute,
# so we only need to handle `y_1`.
def get_size(data):
    return data["dummy"].size


to_size = {
    "y_1": get_size,
}

# %%
# The [set_data_converters][gemseo.set_data_converters] function is applied:
set_data_converters(to_array, from_array, to_size)

# %%
# ### 3. Create your scenario
disciplines = [Sellar1(), Sellar2(), SellarSystem()]

design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=ones(1))
design_space.add_variable(
    "z", 2, lower_bound=(-10, 0.0), upper_bound=(10.0, 10.0), value=array([4.0, 3.0])
)

scenario = MDOScenario(
    disciplines,
    design_space,
    formulation_settings=MDF_Settings(),
)
scenario.add_objective("obj")

scenario.add_constraint("c_1", constraint_type="ineq")
scenario.add_constraint("c_2", constraint_type="ineq")

scenario.execute(NLOPT_COBYLA_Settings(max_iter=10))

# Reset the data converters to remove the custom converters.
# This should only be needed if the type of the variables y_1 and y_2 are no longer
# custom type afterward.
set_data_converters({}, {}, {})

# %%
# ## Summary
#
# The [set_data_converters][gemseo.set_data_converters] function can be used to set
# [BaseDataConverter][gemseo.core.data_converters.base.BaseDataConverter]
# so it enables seamless I/O exchange between disciplines
# without altering their underlying grammar.
#
# A [BaseDataConverter][gemseo.core.data_converters.base.BaseDataConverter]
# acts as a transparent translation layer,
# mapping complex values to 1D NumPy arrays and vice versa.
