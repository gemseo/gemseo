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
Advanced mixture of experts
===========================
"""
from __future__ import annotations

from gemseo.api import load_dataset
from gemseo.mlearning.api import create_regression_model
from gemseo.mlearning.qual_measure.f1_measure import F1Measure
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.qual_measure.silhouette import SilhouetteMeasure

##############################################################################
# In this example,
# we seek to estimate the Rosenbrock function from the :class:`.RosenbrockDataset`.
dataset = load_dataset("RosenbrockDataset", opt_naming=False)

##############################################################################
# For that purpose,
# we will use a :class:`.MOERegressor` in an advanced way:
# we will not set the clustering, classification and regression algorithms
# but select them according to their performance
# from several candidates that we will provide.
# Moreover,
# for a given candidate,
# we will propose several settings,
# compare their performances
# and select the best one.
#
# Initialization
# --------------
# First,
# we initialize a :class:`.MOERegressor` with soft classification
# by means of the machine learning API function
# :meth:`~gemseo.mlearning.api.create_regression_model`.
model = create_regression_model("MOERegressor", dataset, hard=False)

##############################################################################
# Clustering
# ----------
# Then,
# we add two clustering algorithms
# with different numbers of clusters (called *components* for the Gaussian Mixture)
# and set the :class:`.SilhouetteMeasure` as clustering measure
# to be evaluated from the learning set.
# During the learning stage,
# the mixture of experts will select the clustering algorithm
# and the number of clusters
# minimizing this measure.
model.set_clustering_measure(SilhouetteMeasure)
model.add_clusterer_candidate("KMeans", n_clusters=[2, 3, 4])
model.add_clusterer_candidate("GaussianMixture", n_components=[3, 4, 5])

##############################################################################
# Classification
# --------------
# We also add classification algorithms
# with different settings
# and set the :class:`.F1Measure` as classification measure
# to be evaluated from the learning set.
# During the learning stage,
# the mixture of experts will select the classification algorithm and the settings
# minimizing this measure.
model.set_classification_measure(F1Measure)
model.add_classifier_candidate("KNNClassifier", n_neighbors=[3, 4, 5])
model.add_classifier_candidate("RandomForestClassifier", n_estimators=[100])

##############################################################################
# Regression
# ----------
# We also add regression algorithms
# and set the :class:`.MSEMeasure` as regression measure
# to be evaluated from the learning set.
# During the learning stage, for each cluster,
# the mixture of experts will select the regression algorithm minimizing this measure.
model.set_regression_measure(MSEMeasure)
model.add_regressor_candidate("LinearRegressor")
model.add_regressor_candidate("RBFRegressor")

##############################################################################
# .. note::
#
#    We could also add candidates for some learning stages,
#    e.g. clustering and regression,
#    and set the machine learning algorithms for the remaining ones,
#    e.g. classification.
#
# Training
# --------
# Lastly,
# we learn the data
# and select the best machine learning algorithm
# for both clustering, classification and regression steps.
model.learn()

##############################################################################
# Result
# ------
# We can get information on this model,
# on the sub-machine learning models selected among the candidates
# and on their selected settings.
# We can see that
# a :class:`.KMeans` with four clusters has been selected for the clustering stage,
# as well as a :class:`.RandomForestClassifier` for the classification stage
# and a :class:`.RBFRegressor` for each cluster.
print(model)

##############################################################################
# .. note::
#
#    By adding candidates,
#    and depending on the complexity of the function to be approximated,
#    one could obtain different regression models according to the clusters.
#    For example,
#    one could use a :class:`.PolynomialRegressor` with order 2
#    on a sub-part of the input space
#    and a :class:`.GaussianProcessRegressor`
#    on another sub-part of the input space.
#
# Once built,
# this mixture of experts can be used as any :class:`.MLRegressionAlgo`.
#
# .. seealso::
#
#    :ref:`Another example <sphx_glr_examples_mlearning_regression_model_plot_moe.py>`
#    proposes a standard use of :class:`.MOERegressor`.
