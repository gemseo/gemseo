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
Save an optimization problem for post-processing
================================================
"""
from __future__ import annotations

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import create_design_space
from gemseo.api import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction

# %%
# We consider a minimization problem over the interval :math:`[0,1]`
# of the :math:`f(x)=x^2` objective function:

objective = MDOFunction(lambda x: x**2, "f", args=["x"], outvars=["y"])

design_space = create_design_space()
design_space.add_variable("x", l_b=0.0, u_b=1.0)

optimization_problem = OptimizationProblem(design_space)
optimization_problem.objective = objective

# %%
# We solve this optimization problem with the gradient-free algorithm COBYLA:
execute_algo(optimization_problem, "NLOPT_COBYLA", max_iter=10)

# %%
# Then,
# we save the results to an HDF5 file for future post-processing:
optimization_problem.export_hdf("my_results.hdf")

# %%
# .. seealso:: :ref:`sphx_glr_examples_post_process_post_process_file.py`.
