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
"""# DOE from a file

## Problem

You need to evaluate a scenario at input sample points defined in a file
(e.g. from an external tool or a previous study).

## Solution

Use [CustomDOE][gemseo.algos.doe.custom_doe.custom_doe.CustomDOE]
and pass the path to the file via the `doe_file` setting.

## Step-by-step guide
"""

from __future__ import annotations

from pathlib import Path

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.settings import CustomDOE_Settings

# %%
# ### 1. Build the discipline and design space
#
discipline = AnalyticDiscipline({"y": "a*b"})

design_space = DesignSpace()
design_space.add_variable("a", lower_bound=1.0, upper_bound=10.0)
design_space.add_variable("b", lower_bound=1.0, upper_bound=10.0)

# %%
# ### 2. Execute the scenario from the file
#
scenario = EvaluationScenario([discipline], design_space)
scenario.add_observable("y")

# %%
# The file `"doe.txt"` contains one sample per row,
# with columns ordered consistently with the design space variables.
print(Path("doe.txt").read_text())

# %%
# There are 3 samples in this file, giving values to both $a$ and $b$ variables.
# The delimiter defaults to `","` but can be changed via the `delimiter` setting.
# Use `skiprows` to ignore header lines if present.
#
# The scenario can use this file to evaluate the samples.
scenario.execute(CustomDOE_Settings(doe_file="doe.txt"))

# %%
# !!! note
#     The DOE file is parsed by using
#     the [read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) Pandas method.
#
# ### 3. Inspect the results
#
# Export the database to a [Dataset][gemseo.datasets.dataset.Dataset]
# and verify that the output equals the product of $a$ and $b$:
dataset = scenario.to_dataset(name="custom_sampling")
dataset

# %%
# ## Summary
#
# Pass the path to a delimited text file to `CustomDOE_Settings(doe_file=...)`
# to evaluate a discipline at the input points it contains.
# Rows are samples and columns must follow the order of the design space variables.
