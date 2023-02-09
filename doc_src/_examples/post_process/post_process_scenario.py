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
Post-process a scenario
=======================
"""
from __future__ import annotations

from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_post

# %%
# We consider a minimization problem over the interval :math:`[0,1]`
# of the :math:`f(x)=x^2` objective function:
discipline = create_discipline("AnalyticDiscipline", expressions={"y": "x**2"})

design_space = create_design_space()
design_space.add_variable("x", l_b=0.0, u_b=1.0)

scenario = create_scenario([discipline], "DisciplinaryOpt", "y", design_space)

# %%
# We solve this optimization problem with the gradient-free algorithm COBYLA:
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 10})

# %%
# Then,
# we can post-process this :class:`.MDOScenario`
# either with its method :meth:`~.MDOScenario.post_process`:
scenario.post_process("BasicHistory", variable_names=["y"])

# %%
# or with the function :func:`.execute_post`:
execute_post(scenario, "BasicHistory", variable_names=["y"])

# %%
# .. note::
#    By default, |g| saves the images on the disk.
#    Use ``save=False`` to not save figures and ``show=True`` to display them on the screen.
#
# .. seealso:: `Post-processing algorithms <index.html#algorithms>`_.
