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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Reduce the data dimension before training an ML model

## Problem

With a constant budget,
the quality of models often decreases when the size of inputs or outputs increases,
because the number of parameters relative to the number of samples continues to grow,
which can lead to overfitting.
Furthermore,
some learning algorithms do not scale well with the input dimension.

How can I reduce the data dimension?

## Solution

Define a data transformation policy of type "data reduction" in the regressor settings.

## Step-by-step guide
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import cos
from numpy import linspace
from numpy import pi

from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.machine_learning.regression.models.rbf import RBFRegressor
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.machine_learning.regression.quality.r2_measure import R2Measure
from gemseo.machine_learning.transformers.dimension_reduction.pca import PCA

# %%
# ### 1. Define the reference model
#
# We consider the functional discipline $f(x)=\{\cos(xt): t\in[0,2]\}$
# defined over the input space $[0, 2\pi]$
# and the function output is discretized over a mesh of 1000 equispaced nodes.


t = linspace(0, 2, num=1000)


def f(x):
    y = cos(x * t) ** 2
    return y


discipline = AutoPyDiscipline(f)
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=2 * pi)

# %%
# ### 2. Create the training dataset
#
# We generate 20 training samples
# using optimized Latin hypercube sampling strategy.
training_dataset = sample_disciplines(
    [discipline],
    design_space,
    ["y"],
    algo_settings_model=OT_OPT_LHS_Settings(n_samples=20),
)

plt.plot(t, training_dataset.get_view(variable_names="y").T)
plt.show()

# %%
# ### 3. Create the validation dataset
#
# We generate many validation samples using Monte Carlo sampling.

test_dataset = sample_disciplines(
    [discipline], design_space, ["y"], algo_settings_model=MC_Settings(n_samples=1000)
)

# %%
# ### 4. Use an ML model without normalization
#
# We create a regressor,
# e.g. radial basis function (RBF) regressor,
# and reduce the output dimension to $K\in\{5,6,7,8,9,10\}$ components
# using the principal components analysis (PCA) technique.
for n_components in [5, 6, 7, 8, 9, 10]:
    model = RBFRegressor(
        training_dataset,
        settings=RBFRegressor_Settings(
            transformer={"outputs": PCA(n_components=n_components)}
        ),
    )
    model.learn()

    r2 = R2Measure(model)
    plt.plot(t, r2.compute_test_measure(test_dataset), label=f"PCA({n_components})")
    plt.xlabel("t")
    plt.ylabel("R²")

plt.legend()
plt.show()

# %%
# We can see that the validation quality $R^2$ increases with the number of components
# and is approximately 1, i.e. excelllent, starting from 10 components.
#
# !!! note
#
#     You can also reduce the dimension of a single variable
#     using the key `"y"` instead of `"outputs"`.
#
# ## Summary
#
# An ML model can be trained from reduced dimension data,
# using dimension reduction as data transformation policy.
# The data transformation policy can be set
# using the `transformer` parameter of the ML model settings.
