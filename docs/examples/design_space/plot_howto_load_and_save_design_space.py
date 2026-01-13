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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# How to import and export a design space from disk.

In this example, we will see how to read, filter, and export a design space
from the disk.
"""

from __future__ import annotations

from gemseo import read_design_space
from gemseo import write_design_space

# %%
# ## Read a design space from a file
#
#
# The user can read a design space from a file using the
# [create_design_space()][gemseo.create_design_space] function.
design_space = read_design_space("design_space.csv")
design_space

# %%
# !!! note
#
#     Files without any header can also be considered.
#     To read such file, the ``header`` optional argument should be given:
#     ``header=["name", "lower_bound", "value", "upper_bound", "type"]``
#
# !!! warning
#
#     -   User must provide the following minimal fields in the file defining the design space: `'name'`, `'lower_bound'` and `'upper_bound'`.
#     -   For each variable `'name'`, the inequality `'lower_bound'` <= `'upper_bound'` must be satisfied.
#     If given, the `'value'` must satisfy `'lower_bound'` <= `'value'` <= `'upper_bound'`.
#
# !!! note
#
#     -   Available fields are `'name'`, `'lower_bound'`, `'upper_bound'`, `'value'` and `'type'`.
#     -   The `'value'` field is optional. By default, it is set at `None`.
#     -   The `'type'` field is optional. By default, it is set at `float`.
#     -   Each dimension of a variable must be provided. E.g. when the `'size'` of `'x1'` is 2:
#
#         ```
#         name lower_bound value upper_bound type
#         x1 -1. 0. 1. float
#         x1 -3. -1. 1. float
#         x2 5. 6. 8. float
#         ```
#
# !!! note
#     -   Lower infinite bound is encoded `-inf'` or `'-Inf'`.
#     -   Upper infinite bound is encoded `'inf'`, `'Inf'`, `'+inf'` or `'+Inf'`.
#
# ## Filtering the design space
#
#
# The user can filter the design space in order to only keep some variables. To
# do so, the user can use the [filter()][gemseo.algos.design_space.DesignSpace.filter] method:
design_space.filter(["x1", "x2"])
design_space

# %%
# ## Export the design space
#
#
# The user can export a [DesignSpace][gemseo.algos.design_space.DesignSpace] instance by using the
# [write_design_space()][gemseo.write_design_space] function.
write_design_space(design_space, "new_design_space.csv")
