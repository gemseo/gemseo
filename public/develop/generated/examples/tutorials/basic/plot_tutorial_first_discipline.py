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
"""# Tutorial - Your first discipline

## Goal

In this tutorial, you will learn everything about
[Disciplines][gemseo.core.discipline.discipline.Discipline],
from creation to manipulation of core concepts.
"""

from __future__ import annotations

import contextlib

from numpy import array
from numpy import atleast_2d
from pydantic import BaseModel
from pydantic import Field

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.pydantic import PydanticGrammar


# %%
# ## Step 1 - Your first discipline
#
# As seen in the [user guide][concept-discipline],
# a discipline is a set of calculations that:
#
# - produces a dictionary of arrays as outputs
# - from a dictionary of arrays as inputs
# - using a computation bloc containing whatever you may imagine.
#
# With GEMSEO, you can create a discipline
# by inheriting from the abstract class
# [Disciplines][gemseo.core.discipline.discipline.Discipline].
# Doing so, you have to define:
#
# - the input grammar,
# - the output grammar,
# - the computation method,
# contained in the `_run()` method.
#
# Let's consider this following implementation:
class NewDiscipline(Discipline):
    def __init__(self, name: str = ""):
        super().__init__(name)
        # The input and output names must been defined here.
        self.input_grammar.update_from_names(["x", "z"])
        self.output_grammar.update_from_names(["f", "g"])

    def _run(self, input_data):
        # How to compute the outputs from the inputs.
        x = input_data["x"]
        z = input_data["z"]
        return {
            "f": array([x[0] * z[0]]),
            "g": array([x[0] * (z[0] + 1.0) ** 2]),
        }


# %%
# This discipline computes `f` and `g` (the outputs)
# from `x` and `z` (the inputs),
# according the following equations:
# $f = x \times z$ and $g = x (z + 1)^2$.
#
# You can already use it! Let's make your first computation:
discipline = NewDiscipline("MyFirstDiscipline")
discipline.execute({"x": array([5.0]), "z": array([1])})

# %%
# The execution of a discipline is made by calling the
# [execute()][gemseo.core.discipline.discipline.Discipline.execute] method.
# This method wraps the protected
# `_run()`
# method that you defined earlier,
# enabling other features that you may discover in this tutorial.
# !!! note
#     Discipline inputs and outputs shall be
#     [numpy](http://www.numpy.org/) arrays of real numbers or integers.
#
# Your discipline returns $f=5$ and $g=20$, as expected.
# Congratulations!
#
# Your discipline can now compute $f$ and $g$ whenever you give $x$ and $z$.
# The variables $x$ and $z$ can be given default values:
discipline.default_input_data = {"x": array([0.0]), "z": array([0.0])}

# %%
# !!! note
#     Some variables may have default values even if others do not.
#
# !!! tips
#     The definition of default inputs usually occurs in the `__init__` method.
#
# You can then execute your discipline by giving a subset of inputs, or no input at all.
# In this case, since the two inputs have default values,
# we can execute it without giving any information on the inputs:
discipline.execute()


# %%
#
# !!! warning
#     Some cases require the discipline to get default values.
#     Otherwise, the execution will fail.
#     Whenever possible, it is a good practice to give default values.
#
# ## Step 2 - Towards gradient computation
#
# The [Discipline][gemseo.core.discipline.discipline.Discipline]
# may also provide the derivatives of their outputs
# with respect to their inputs.
# So let's complexify your gradient-free discipline,
# by adding the gradient computation.
class GradientDiscipline(Discipline):
    def __init__(self, name: str = ""):
        super().__init__(name)
        # The input and output names must been defined here.
        self.input_grammar.update_from_names(["x", "z"])
        self.output_grammar.update_from_names(["f", "g"])
        self.default_input_data = {"x": array([0.0]), "z": array([0.0])}

    def _run(self, input_data):
        # How to compute the outputs from the inputs.
        x = input_data["x"]
        z = input_data["z"]
        return {
            "f": array([x[0] * z[0]]),
            "g": array([x[0] * (z[0] + 1.0) ** 2]),
        }

    def _compute_jacobian(self, input_names=(), output_names=()):
        # Initialize all matrices to zeros.
        self._init_jacobian(fill_missing_keys=True)

        # Get the inputs from the local data.
        x = self.local_data["x"]
        z = self.local_data["z"]

        self.jac = {
            "f": {"x": atleast_2d(z), "z": atleast_2d(x)},
            "g": {
                "x": atleast_2d(array([(z[0] + 1.0) ** 2])),
                "z": atleast_2d(array([2 * x[0] * z[0] * (z[0] + 1.0)])),
            },
        }


# %%
# !!! note
#     The discipline may have multiple inputs and multiple outputs.
#     To store the multiple Jacobian matrices associated to
#     all the inputs and outputs,
#     GEMSEO uses a dictionary of dictionaries structure.
#     This data structure is sparse and makes easy the access
#     and the iteration over the elements of the Jacobian.
#
# Now, we can also compute the gradient,
# by calling the
# [linearize()][gemseo.core.discipline.discipline.Discipline.linearize] method.
discipline = GradientDiscipline("MyFirstGradientDiscipline")
discipline.linearize(compute_all_jacobians=True)

# %%
# !!! info "How-tos"
#     You can have different HOWTOs about the
#     [Discipline differentiation][discipline-differentiation].
#
# ## Step 3 - Consolidate your discipline
#
# Your discipline has default inputs, is able to compute some outputs and a gradient.
# You can also robustify your discipline
# by checking your inputs and outputs at each evaluation.
# Or you can make your discipline more permissive.
# To explore this feature, you will dive into the `Grammars`.
#
# A discipline has two grammars: one for the inputs, and one for the ouputs.
# You can simply visualize grammars:
discipline.input_grammar
# %%
# Here, the $x$ and $z$ inputs are required, and are of type "arrays".
# By default, a discipline uses
# [JSONGrammars][gemseo.core.grammars.json.JSONGrammar].
# What your seeing here is the JSON schema of your input grammar.
#
# !!! note
#     Grammars are different: some allow a lot while others are very restrictive.
#     Please refer to the [Grammar section][concept-grammars] for more information.
#
# Let's make your discipline more restrictive, by saying that $x$ is now an integer.
# One way to do so is to update the grammar with the
# [update_from_data][gemseo.core.grammars.base.BaseGrammar.update_from_data]
# method.
# By giving to $x$ an array of integer,
# the grammar will automatically understand that x is now an integer.
discipline.input_grammar.update_from_data({"x": array([5])})
discipline.input_grammar
# %%
# You can still compute your discipline when you give $x$ as an integer
discipline.execute({"x": array([5])})

# %%
# But the execution will fail when $x$ is not of type integer
try:
    discipline.execute({"x": array([1.2])})
except InvalidDataError:
    print("You are crazy, aren't you?")


# %%
# ## Step 4 - A complete discipline
#
# Let's create another discipline, a little bit more complex than the previous one.
# You will create Pydantic grammars to perfectly control both input and outputs.
#
# !!! note
#     Pydantic allows you to control whatever you want, since you can create validators.
#     Please check Pydantic documentation for further details.
#
# Let's consider $x$ as an integer, so that $0 \leq x \leq 5$.
# Moreover, let's consider that you want your inputs to be int and float;
# no more arrays.
#
class MyInputGrammar(BaseModel):
    x: int = Field(default=1, le=5, ge=0)
    z: float = Field(default=0.5)


class MyOutputGrammar(BaseModel):
    f: float
    g: float


class CompleteDiscipline(Discipline):
    # The grammar is specified to be of type `Pydantic`.
    default_grammar_type = Discipline.GrammarType.PYDANTIC

    def __init__(self, name: str = ""):
        super().__init__(name)
        # The input and output names must been defined here.
        self.input_grammar = PydanticGrammar("inputs", model=MyInputGrammar)
        self.output_grammar = PydanticGrammar("outputs", model=MyOutputGrammar)

    def _run(self, input_data):
        # How to compute the outputs from the inputs.
        x = input_data["x"]
        z = input_data["z"]
        return {
            "f": x * z,
            "g": x * (z + 1.0) ** 2,
        }

    def _compute_jacobian(self, input_names=(), output_names=()):
        # Initialize all matrices to zeros.
        self._init_jacobian(fill_missing_keys=True)

        # Get the inputs from the local data.
        x = self.local_data["x"]
        z = self.local_data["z"]

        self.jac = {
            "f": {"x": atleast_2d(z), "z": atleast_2d(x)},
            "g": {
                "x": atleast_2d(array([(z + 1.0) ** 2])),
                "z": atleast_2d(array([2 * x * z * (z + 1.0)])),
            },
        }


# %%
# !!! note
#     You may have noticed that you don't have to use
#     [default_input_data][gemseo.core.discipline.discipline.Discipline.default_input_data]
#     anymore,
#     since this information is contained in the Pydantic grammar.
discipline = CompleteDiscipline("NotMyLastDiscipline")
discipline.input_grammar

# You can verify that your input data are well-validated:
with contextlib.suppress(InvalidDataError):
    discipline.execute({"x": 6})

# %%
with contextlib.suppress(InvalidDataError):
    discipline.execute({"x": -1})

# %%
discipline.execute()
# %%
discipline.linearize(compute_all_jacobians=True)

# ## Key takeaways
#
# Now you know the [Discipline][gemseo.core.discipline.discipline.Discipline] basics:
# - the role of a grammar, used for inputs and outputs
# - the computation method `_run()` that you need to overload,
# - the jacobian method [_compute_jacobian()][gemseo.core.discipline.discipline.Discipline._compute_jacobian] that you can overload (optional).
#
# You can develop all your disciplines that way.
# However, for performance and convenience pruposes,
# you are advised to check for ready-to-use GEMSEO disciplines.
#
# The [Discipline][gemseo.core.discipline.discipline.Discipline] handles a lot more features,
# such as `namespace`, `cache`, ...
# Features that you will explore later.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Grammars][grammars-for-inputs-and-outputs],
# - [Discipline differentiation][discipline-differentiation],
# - [Different types of disciplines][different-types-of-disciplines].
