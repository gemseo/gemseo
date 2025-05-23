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
#        :author: Syver Doving Agdestein, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Random forest
=============

A :class:`.RandomForestClassifier` is a random forest model
based on `scikit-learn <https://scikit-learn.org/>`__.

We want to classify the Iris dataset using a Random Forest classifier.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo import create_benchmark_dataset
from gemseo.mlearning import create_classification_model

configure_logger()


# %%
# Load Iris dataset
# -----------------
iris = create_benchmark_dataset("IrisDataset", as_io=True)

# %%
# Create the classification model
# -------------------------------
# Then, we build the random forest classification model from the discipline cache and
# displays this model.
model = create_classification_model("RandomForestClassifier", data=iris)
model.learn()
model

# %%
# Predict output
# --------------
# Once it is built, we can use it for prediction.
input_value = {
    "sepal_length": array([4.5]),
    "sepal_width": array([3.0]),
    "petal_length": array([1.0]),
    "petal_width": array([0.2]),
}
output_value = model.predict(input_value)
output_value
