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
"""
Post-process an HDF5 file
=========================
"""
from __future__ import annotations

from gemseo.api import execute_post

# %%
# We can post-process an HDF5 file
# generated from an :class:`.OptimizationProblem` or a :class:`.Scenario`
# with the function :func:`.execute_post`:
execute_post("my_results.hdf", "BasicHistory", variable_names=["y"])

# %%
# .. note::
#    By default, |g| saves the images on the disk.
#    Use ``save=False`` to not save figures and ``show=True`` to display them on the screen.
#
# .. seealso::
#
#    - :ref:`sphx_glr_examples_post_process_save_from_scenario.py`,
#    - :ref:`sphx_glr_examples_post_process_save_from_opt_problem.py`.
