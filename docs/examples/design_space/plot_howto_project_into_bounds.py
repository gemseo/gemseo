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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# How to project parameters into boundaries.

In this example, we will see how to project design paramters into a design space.

Sometimes, components of a design vector are greater than the upper bounds or
lower than the upper bounds.
For that, it is possible to project the vector into the bounds by means of
the [project_into_bounds()][gemseo.algos.design_space.DesignSpace.project_into_bounds]:
"""

from __future__ import annotations

from numpy import array
from numpy import ones

from gemseo import create_design_space

# %%
# ## Create a design space
#
# First, let's create a design space.
design_space = create_design_space()
design_space.add_variable("x1", lower_bound=-10, upper_bound=10)
design_space.add_variable("x2", lower_bound=-10, upper_bound=10)
design_space.add_variable("x3", lower_bound=-10, upper_bound=10)
design_space.add_variable("x4", value=ones(1), lower_bound=-10, upper_bound=10)

# %%
#
# ## Array projection
point = array([1.0, 3, -15.0, 23.0])
p_point = design_space.project_into_bounds(point)
p_point
