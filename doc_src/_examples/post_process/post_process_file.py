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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Post-process an HDF5 file.
==========================
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo import import_database

# %%
# Given an HDF5 file
# generated from an :class:`.EvaluationProblem`,
# an :class:`.OptimizationProblem`
# or a :class:`.Scenario`,
# we can visualize the evaluation history using the function :func:`.execute_post`
# and a specific visualization tool such as ``"BasicHistory"``:
execute_post("my_results.hdf5", post_name="BasicHistory", variable_names=["y_4"])

# %%
# .. note::
#    By default, |g| saves the images on the disk.
#    Use ``save=False`` to not save figures and ``show=True``
#    to display them on the screen.
#
# We can also get the raw data as a :class:`.Database` from this HDF5 file:
database = import_database("my_results.hdf5")

# %%
# and convert it into a :class:`.Dataset`` to handle it more easily
# (see :ref:`this example <sphx_glr_examples_dataset_creation_plot_dataset.py>`):
dataset = database.to_dataset()

# %%
# .. seealso::
#
#    - :ref:`sphx_glr_examples_post_process_save_from_scenario.py`,
#    - :ref:`sphx_glr_examples_post_process_save_from_opt_problem.py`,
#    - :ref:`sphx_glr_examples_dataset_creation_plot_optimization_dataset.py`.
