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
"""# Batch sampling for a scenario

## Problem

By default, a DOE algorithm evaluates a scenario one sample at a time,
calling each discipline once per sample.
This can be inefficient when disciplines natively support batch evaluation.

## Solution

Pass `vectorize=True` to the DOE settings.
GEMSEO will call each discipline once with all samples concatenated into 1D arrays,
instead of calling it `n_samples` times with scalar values.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline.discipline import Discipline
from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings import MC_Settings

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# ### 1. Implement vectorizable disciplines
#
# In batch mode, each discipline receives and must return dictionaries of 1D arrays.
# For a scalar variable, the 1D array has size `n_samples`.
# For a `d`-dimensional variable `"x"`, the 1D array has size `n_samples * d`,
# with all components of sample `i` stored before those of sample `i+1`.
#
# !!! note
#     In the case of a `d`-length vectorial variable `"x"`,
#     the expected form would be
#     `{"x": array([x_1_1, ..., x_d_1, ..., x_1_n, ..., x_d_n])}`.
#
# Here, `AreaDiscipline` computes the area of a rectangle:


class AreaDiscipline(Discipline):
    """Compute the area of a rectangle."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(("length", "width"))
        self.io.output_grammar.update_from_names(("area",))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"area": input_data["length"] * input_data["width"]}


# %%
# `AreaIncreaser` doubles the area — it is chained after `AreaDiscipline`
# to show that batch sampling propagates through the entire discipline chain:


class AreaIncreaser(Discipline):
    """Increase the area by a factor of 2."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(("area",))
        self.io.output_grammar.update_from_names(("final_area",))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"final_area": 2.0 * input_data["area"]}


# %%
# ### 2. Build the design space and scenario
#
area_discipline = AreaDiscipline()

design_space = DesignSpace()
design_space.add_variable("length", lower_bound=0.0, upper_bound=10.0)
design_space.add_variable("width", lower_bound=0.0, upper_bound=10.0)

scenario = MDOScenario([area_discipline, AreaIncreaser()], design_space)
scenario.add_objective("final_area")

# %%
# ### 3. Execute with batch sampling
#
# Pass `vectorize=True` to the DOE settings to enable batch sampling:
scenario.execute(MC_Settings(n_samples=1000, vectorize=True))

# %%
# !!! note
#     Batch sampling is not specific to the `DisciplinaryOpt` formulation.
#     It works with any MDO formulation (e.g. `MDF`, `IDF`),
#     provided that all disciplines are vectorized.

# %%
# ### 4. Verify
#
# Each discipline received and returned 1D arrays of size 1000:
area_discipline.io.data["length"].shape

# %%
area_discipline.io.data["width"].shape

# %%
area_discipline.io.data["area"].shape

# %%
# The dataset contains 1000 samples:
scenario.to_dataset(opt_naming=False)

# %%
# ## Summary
#
# To enable batch sampling for a scenario,
# pass `vectorize=True` to the DOE settings.
# GEMSEO will call each discipline once with all samples concatenated into 1D arrays
# of size `n_samples` (for scalar variables),
# instead of calling it `n_samples` times.
# Disciplines must handle these 1D arrays directly — no shape conversion is needed
# since inputs and outputs remain 1D throughout.
