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
#                         documentation
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Create a clustering model

## Problem

I would like to group data according to their similarities.
How can I use clustering for that purpose?

## Solution

Use a clustering model (a.k.a. clusterer) from the sub-package
[gemseo.machine_learning.clustering.models][gemseo.machine_learning.clustering.models].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo import create_benchmark_dataset
from gemseo.machine_learning.clustering.models.kmeans import KMeans
from gemseo.machine_learning.clustering.models.kmeans_settings import KMeans_Settings
from gemseo.machine_learning.clustering.quality.silhouette_measure import (
    SilhouetteMeasure,
)
from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings

# %%
# ### 1. Create the training dataset
# For example,
# [the Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).
training_dataset = create_benchmark_dataset("IrisDataset")

# %%
# ### 2. Create a clustering model
model = KMeans(
    training_dataset,
    settings=KMeans_Settings(
        n_clusters=3,
        # We remove the variable "specy"
        # that corresponds to the expected cluster from a biological perspective;
        # otherwise it would be cheating!
        var_names=training_dataset.get_variable_names(training_dataset.PARAMETER_GROUP),
    ),
)
model.learn()
model

# %%
# ### 3. Assess its quality
silhouette = SilhouetteMeasure(model)
silhouette.compute_learning_measure(multioutput=False)

# %%
# ### 4. Predict a cluster
input_value = {
    "sepal_length": array([4.5]),
    "sepal_width": array([3.0]),
    "petal_length": array([1.0]),
    "petal_width": array([0.2]),
}
model.predict(input_value)

# %%
# ### 5. Visualize predictions
training_dataset.add_variable("km_specy", model.labels.reshape((-1, 1)), "labels")
pair_plot = PairPlot(
    training_dataset,
    PairPlot_Settings(
        use_kde=True,
        classifier="km_specy",
        variable_names=(
            "sepal_length",
            "petal_length",
            "sepal_width",
            "petal_width",
            "km_specy",
        ),
    ),
)
pair_plot.execute(save=False, show=True)

# %%
# ## Summary
#
# Clustering models can group data according to their similarities.
# They can be found in the
# [gemseo.machine_learning.clustering.models][gemseo.machine_learning.clustering.models] package.
# Their main parameter is `n_clusters`.
