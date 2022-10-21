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
"""
K-means
=======

Load Iris dataset and create clusters.
"""
###############################################################################
# Import
# ------
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import create_clustering_model
from numpy import array

configure_logger()


###############################################################################
# Create dataset
# --------------
# We import the Iris benchmark dataset through the API.
iris = load_dataset("IrisDataset")

# Extract inputs as a new dataset
data = iris.get_data_by_group(iris.PARAMETER_GROUP)
variables = iris.get_names(iris.PARAMETER_GROUP)
print(variables)

dataset = Dataset("sepal_and_petal")
dataset.set_from_array(data, variables)

###############################################################################
# Create clustering model
# -----------------------
# We know that there are three classes of Iris plants.
# We will thus try to identify three clusters.
model = create_clustering_model("KMeans", data=dataset, n_clusters=3)
model.learn()
print(model)

###############################################################################
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
print(output_value)

###############################################################################
# Plot clusters
# -------------
# Show cluster labels
dataset.add_variable(
    "km_specy", model.labels.reshape((-1, 1)), group="labels", cache_as_input=False
)
dataset.plot("ScatterMatrix", kde=True, classifier="km_specy")
