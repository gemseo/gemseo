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
"""# Batch sampling for an evaluation problem

## Problem

By default, a DOE algorithm evaluates a function one sample at a time.
This can be inefficient when the function natively supports batch evaluation,
i.e., computing all outputs at once from a 2D array of inputs.

## Solution

Pass `vectorize=True` to the DOE settings.
GEMSEO will call the function once with all samples stacked into a 2D array
of shape `(n_samples, input_dimension)`,
instead of calling it `n_samples` times with a 1D array.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import newaxis

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.settings import MC_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray

# %%
# ### 1. Define the design space
#
design_space = DesignSpace()
design_space.add_variable("length", lower_bound=0.0, upper_bound=10.0)
design_space.add_variable("width", lower_bound=0.0, upper_bound=10.0)

# %%
# ### 2. Implement a vectorizable function
#
# The function must handle both a single sample (1D array of shape `(input_dimension,)`)
# and a batch of samples (2D array of shape `(n_samples, input_dimension)`):


class AreaFunction:
    """Compute the area of a rectangle."""

    def __init__(self) -> None:
        self.last_input_data = array([])

    def __call__(self, input_data: RealArray) -> RealArray:
        """Compute the area of the rectangle.

        Args:
            input_data: Either a 1D array shaped `(input_dimension,)`
                or a 2D array shaped `(n_samples, input_dimension)`.

        Returns:
            Either a 1D array shaped `(output_dimension,)`
            or a 1D array shaped `(n_samples,)`.
        """
        is_1d = input_data.ndim == 1
        if is_1d:
            input_data = input_data[newaxis, :]

        self.last_input_data = input_data
        length = input_data[:, 0]
        width = input_data[:, 1]
        area = length * width
        return area[0] if is_1d else area


function = AreaFunction()

# %%
# ### 3. Create the evaluation problem
#
problem = EvaluationProblem(design_space)
problem.add_observable(ArrayFunction(function, name="area"))

# %%
# ### 4. Execute with batch sampling
#
# Pass `vectorize=True` to the DOE settings to enable batch sampling:
SciPyDOE("MC").execute(problem, settings=MC_Settings(n_samples=1000, vectorize=True))

# %%
# ### 5. Verify
#
# The function was called once with a 2D array of shape `(1000, 2)`:
function.last_input_data.shape

# %%
# The dataset contains 1000 samples:
problem.to_dataset()

# %%
# ## Summary
#
# To enable batch sampling for an
# [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
# pass `vectorize=True` to the DOE settings.
# GEMSEO will call the function once with all samples stacked into a 2D array
# of shape `(n_samples, input_dimension)`.
# The function must handle both 1D and 2D inputs.
