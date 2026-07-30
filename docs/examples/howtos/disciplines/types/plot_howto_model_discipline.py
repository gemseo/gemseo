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

r"""# Create a discipline from a Pydantic model

## Problem

You want to create a discipline without explicitly declaring its inputs and outputs.

## Solution

Subclass
[BaseModelDiscipline][gemseo.discipline.base_model_discipline.BaseModelDiscipline]
and implement its `_run_from_model` method.
GEMSEO infers the input and output grammars automatically by observing which
model fields are **read** and which are **written** during a dry-run of your
implementation.

!!! warning
    Although the BaseModelDiscipline has been tested on example problems and certain real
    world applications, there is no guarantee that it will work with every type definition
    or in very complex workflow setups. If you find a bug, do not hesitate to [create an
    issue on Gitlab](https://gitlab.com/gemseo/dev/gemseo/).

## Step-by-step guide
"""

from __future__ import annotations

# %%
# ### 1. Define the Pydantic model
#
# The model carries all data exchanged with the discipline.
# Use standard Pydantic field annotations to declare the types and defaults.
from pydantic import BaseModel
from pydantic import Field


class QuadraticModel(BaseModel):
    """Inputs and outputs of a simple quadratic function ``y = a * x**2 + b``."""

    x: float = Field(default=1.0, description="Independent variable.")
    a: float = Field(default=1.0, description="Quadratic coefficient.")
    b: float = Field(default=0.0, description="Constant offset.")
    y: float = Field(default=0.0, description="Function value.")


# %%
# ### 2. Subclass BaseModelDiscipline
#
# The only requirement is to implement ``_run_from_model``.
# Read input values from *model* and write the results back into it.
from gemseo.discipline.base_model_discipline import BaseModelDiscipline  # noqa: E402


class QuadraticDiscipline(BaseModelDiscipline):
    """Compute ``y = a * x**2 + b`` using a Pydantic model."""

    def _run_from_model(self, model: QuadraticModel) -> None:
        model.y = model.a * model.x**2 + model.b


# %%
# ### 3. Instantiate the discipline
#
# Pass a model instance that carries the **default** values; the grammars are
# inferred automatically at construction time — no manual grammar declaration is
# needed.
model = QuadraticModel()
discipline = QuadraticDiscipline(model)

# %%
# ### 4. Inspect the auto-inferred grammar names
#
# ``x``, ``a``, and ``b`` were **read** in ``_run_from_model`` → inputs.
# ``y`` was **written** → output.
print("Inputs :", sorted(discipline.io.input_grammar.keys()))
print("Outputs:", sorted(discipline.io.output_grammar.keys()))

# %%
# ### 5. Execute the discipline
#
# Call ``execute`` with a dictionary of input values.
# Unspecified inputs fall back to the defaults stored in the model.
output = discipline.execute({"x": 3.0, "a": 2.0, "b": 1.0})
print("y =", output["y"])  # 2 * 3**2 + 1 = 19

# %%
# ## Situations to avoid
#
# The grammars are inferred from a **single** dry-run at construction time,
# so ``_run_from_model`` must access the same fields whatever the input values.
#
# - Do not make the set of read or written fields depend on the input values,
#   for instance by reading or writing a field only inside a branch that tests
#   an input: the grammar inferred from the dry-run would not match the fields
#   accessed by other runs.
# - Write **every** output field on every execution. An output written only in
#   some branches is still inferred as a discipline output from the dry-run; on
#   a run that skips the write it silently keeps the model value (typically its
#   default) instead of a computed one, with no error raised.
# - Avoid side effects and non-deterministic field accesses in
#   ``_run_from_model``, for the same reason.

# %%
# ## Access fields as attributes
#
# Tracking hooks the model's attribute getter and setter, so every input and
# output field must be **read and written as an attribute** of ``model``.
# Binding a field to a local name defeats the tracking: with
#
# ```python
# y = model.y      # reads model.y
# y = value        # rebinds the local name, model.y is never written
# ```
#
# ``y`` is not detected as an output. Write ``model.y = value`` instead.
#
# With nested models, you may alias a sub-model (``sub = model.sub``) as long as
# the tracked leaf field is still accessed as an attribute of it
# (``sub.field = value``): the high-level field ``sub`` is what must be reached
# through an attribute access for the leaf to be seen as an input or output.

# %%
# ## Summary
#
# [BaseModelDiscipline][gemseo.discipline.base_model_discipline.BaseModelDiscipline]
# removes the need to declare grammars by hand: subclass it, define the Pydantic
# model that describes the data, implement `_run_from_model` and GEMSEO does the rest.
#
# For disciplines with nested models, sub-model fields are exposed with a dot
# separator (e.g. ``"sub.field"``), and the same execute / grammar API applies.
