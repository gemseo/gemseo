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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Propagate a namespace over a group of disciplines

## Problem

You have a group of coupled disciplines and you want to run a second, independent
copy of the whole group next to the first one, under a namespace. Namespacing each
input and output by hand works, but requires one call per affected variable, and
missing a single one silently breaks a coupling instead of raising an error.

## Solution

[propagate_namespace()][gemseo.discipline.namespace.propagate_namespace] walks the coupling graph forward
from the variables you seed it with, and namespaces every input and output affected
by the propagation in a single call.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.discipline import propagate_namespace

# %%
# ### 1. Create the group of disciplines
#
# The group is a short coupled chain:
# `a` computes `y = x + 1`,
# `b` computes `z = y + 1`,
# `c` computes `w = z + y`.
# `y` therefore couples `a` to both `b` and `c`,
# and `z` couples `b` to `c`.
a = create_discipline("AnalyticDiscipline", expressions={"y": "x + 1"}, name="a")
b = create_discipline("AnalyticDiscipline", expressions={"z": "y + 1"}, name="b")
c = create_discipline("AnalyticDiscipline", expressions={"w": "z + y"}, name="c")
disciplines = [a, b, c]

for discipline in disciplines:
    print(
        f"{discipline.name}:  "
        f"inputs={list(discipline.io.input_grammar)}  "
        f"outputs={list(discipline.io.output_grammar)}"
    )

# %%
# The goal is to run this group alongside a second, independent copy of itself,
# by placing the whole group under the namespace `"left"`
# (a second copy could later be placed under, say, `"right"`,
# but this example focuses on producing the first one).
# `x` is the entry point of the group: it is the variable you seed the propagation with.

# %%
# ### 2. The manual route
#
# Doing this by hand means namespacing, on every discipline,
# every input and output that is not left bare on purpose.
# Here, nothing should stay bare, since the whole group moves under `"left"`.
# `c` has two inputs, `y` and `z`, and it is easy to only namespace one of them:
a.add_namespace_to_input("x", "left")
a.add_namespace_to_output("y", "left")
b.add_namespace_to_input("y", "left")
b.add_namespace_to_output("z", "left")
c.add_namespace_to_input("z", "left")  # `c`'s other input, `y`, is forgotten here.
c.add_namespace_to_output("w", "left")

# %%
# Nothing raises an error, but `a` now produces `left:y` while `c` still expects a
# bare `y`: the coupling between `a` and `c` is silently broken.
for discipline in disciplines:
    print(
        f"{discipline.name}:  "
        f"inputs={list(discipline.io.input_grammar)}  "
        f"outputs={list(discipline.io.output_grammar)}"
    )

# %%
# The fix is the seventh and last call, easy to miss among the others:
c.add_namespace_to_input("y", "left")
for discipline in disciplines:
    print(
        f"{discipline.name}:  "
        f"inputs={list(discipline.io.input_grammar)}  "
        f"outputs={list(discipline.io.output_grammar)}"
    )

# %%
# ### 3. The same result with `propagate_namespace()`
#
# Start over with a fresh copy of the group, still bare:
a = create_discipline("AnalyticDiscipline", expressions={"y": "x + 1"}, name="a")
b = create_discipline("AnalyticDiscipline", expressions={"z": "y + 1"}, name="b")
c = create_discipline("AnalyticDiscipline", expressions={"w": "z + y"}, name="c")
disciplines = [a, b, c]

# %%
# Seeding the propagation with the group's entry variable `x` reaches every
# discipline of this small chain, since `a` is the only source and `b` and `c` are
# both downstream of it:
affected = propagate_namespace(disciplines, "left", {"x"})
for discipline, ios in affected.items():
    print(
        f"{discipline.name}:  inputs={sorted(ios.inputs)}  outputs={sorted(ios.outputs)}"
    )

# %%
# One call reproduces exactly the seven renamings made by hand above:
for discipline in disciplines:
    print(
        f"{discipline.name}:  "
        f"inputs={list(discipline.io.input_grammar)}  "
        f"outputs={list(discipline.io.output_grammar)}"
    )

# %%
# ### 4. The namespaced group still solves
#
# The couplings survived the renaming, so an
# [MDA][gemseo.mda.core.base.BaseMDA] over the namespaced group still converges,
# and its namespaced output can be read back with the `"left:"` prefix:
mda = create_mda("MDAGaussSeidel", disciplines)
result = mda.execute({"left:x": array([1.0])})
print(f"left:w = {result['left:w']}")
# left:x=1 -> left:y=2 -> left:z=3 -> left:w=3+2=5

# %%
# ### 5. Before committing: the self-containment caveat
#
# Remember that the group of disciplines you pass must be self-contained:
# references held by objects that are not disciplines — design space variable
# names, objective and constraint names, couplings passed to
# [create_mda()][gemseo.create_mda] — are not reachable from the disciplines and
# so cannot be renamed nor checked automatically. See
# [Propagating a namespace over a group of disciplines][concept-namespace-propagation]
# for the details.

# %%
# ## Summary
#
# [propagate_namespace()][gemseo.discipline.namespace.propagate_namespace] cascades a namespace over a
# whole group of coupled disciplines in one call, reproducing what would otherwise
# take one [add_namespace_to_input()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_input]
# or
# [add_namespace_to_output()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_output]
# call per affected variable. The manual route remains preferable when only one
# or two disciplines need namespacing and you want fine control over which
# couplings are kept or broken.
