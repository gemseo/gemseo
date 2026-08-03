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
"""# Save and reuse sensitivity analysis samples

## Problem

Generating samples for a sensitivity analysis may be expensive.
You want to save the samples to disk after a first run
so that you can recompute indices later (e.g. with different output selections)
without re-evaluating the discipline.

## Solution

[compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples]
returns an
[IODataset][gemseo.dataset.io_dataset.IODataset]
that can be persisted with
[to_pickle()][gemseo.util.pickle.to_pickle]
and reloaded with
[from_pickle()][gemseo.util.pickle.from_pickle].
A new analysis object initialized from those samples computes indices instantly.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami import IshigamiSpace
from gemseo.uncertainty.sensitivity import MorrisAnalysis
from gemseo.util.pickle import from_pickle
from gemseo.util.pickle import to_pickle

# %%
# ### 1. Generate and save samples
#
# Run the analysis a first time to produce input-output samples,
# then save them to disk:
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace()

analysis = MorrisAnalysis()
samples = analysis.compute_samples([discipline], uncertain_space, n_samples=0)
analysis.compute_indices()

to_pickle(samples, "morris_samples.p")

# %%
# ### 2. Reload samples from disk
#
# In a later session, reload the samples without re-running the discipline:
samples = from_pickle("morris_samples.p")
samples

# %%
# ### 3. Create a new analysis from the saved samples
#
# Pass the reloaded dataset directly to
# [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis]
# and compute indices without any new discipline evaluations:
analysis2 = MorrisAnalysis(samples=samples)
analysis2.compute_indices()

# %%
# ### 4. Verify the results match
#
# The indices are identical to those from the original run:
analysis2.indices

# %%
# ## Summary
#
# - [compute_samples()][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.compute_samples]
#   returns an
#   [IODataset][gemseo.dataset.io_dataset.IODataset];
# - Save it with
#   [to_pickle()][gemseo.util.pickle.to_pickle]
#   and reload it with
#   [from_pickle()][gemseo.util.pickle.from_pickle];
# - Pass the reloaded dataset as `samples=` to the analysis constructor
#   to skip discipline evaluations and go straight to
#   [compute_indices()][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.compute_indices].
