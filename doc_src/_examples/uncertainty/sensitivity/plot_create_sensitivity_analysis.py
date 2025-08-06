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
r"""
Create sensitivity analysis
===========================

:func:`.create_sensitivity_analysis` is a top-level function
to create a sensitivity analysis from a sensitivity analysis class name,
e.g. ``"MorrisAnalysis"``.
"""

from __future__ import annotations

from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.uncertainty import create_sensitivity_analysis

# %%
# There are two ways of using :func:`.create_sensitivity_analysis`.
#
# The first one is to perform a sensitivity analysis
# from a collection of disciplines and an uncertain space:
analysis = create_sensitivity_analysis("MorrisAnalysis")
uncertain_space = IshigamiSpace()
discipline = IshigamiDiscipline()
samples = analysis.compute_samples([discipline], uncertain_space, n_samples=0)
indices = analysis.compute_indices()
indices

# %%
# The ``samples`` can be saved on the disk using the :func:`.to_pickle` function,
# e.g. ``to_pickle(sample, "my_samples.p")``,
# in order to use them later to compute sensitivity indices.
#
# The other way is to perform a sensitivity analysis
# from samples computed from another sensitivity analysis:
analysis = create_sensitivity_analysis("MorrisAnalysis", samples=samples)
indices = analysis.compute_indices()
indices

# %%
# The argument ``samples`` of :func:`.create_sensitivity_analysis`
# can be either an :class:`.IODataset` as above or a pickle file path,
# e.g. ``create_sensitivity_analysis("MorrisAnalysis", samples="my_samples.p")``.
