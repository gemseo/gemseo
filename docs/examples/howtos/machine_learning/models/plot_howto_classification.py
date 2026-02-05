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
"""# Create a classification model

## Problem

I would like to model the relationship between properties and a label.
How can I use classification for that purpose?

## Solution

Use a classification model (a.k.a. classifier) from the sub-package
[gemseo.machine_learning.classification.models][gemseo.machine_learning.classification.models].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.machine_learning.classification.models.knn import KNNClassifier
from gemseo.machine_learning.classification.models.knn_settings import (
    KNNClassifier_Settings,
)
from gemseo.machine_learning.classification.quality.f1_measure import F1Measure
from gemseo.problems.dataset.iris import create_iris_dataset

# %%
# ### 1. Create the training dataset
# For example,
# [the Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).
training_dataset = create_iris_dataset(as_io=True)

# %%
# ### 2. Create a classification model
model = KNNClassifier(
    training_dataset,
    settings=KNNClassifier_Settings(),
)
model.learn()
model

# %%
# ### 3. Assess its quality
f1 = F1Measure(model)
f1.compute_learning_measure(multioutput=False)

# %%
# ### 4. Predict a cluster
input_value = {
    "sepal_length": array([4.5]),
    "sepal_width": array([3.0]),
    "petal_length": array([1.0]),
    "petal_width": array([0.2]),
}
predictions = model.predict(input_value)

# %%
# ## Summary
#
# Classification models can group data according to their similarities.
# They can be found in the
# [gemseo.machine_learning.classification.models][gemseo.machine_learning.classification.models] package.
