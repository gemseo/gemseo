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
Iris dataset
============

Presentation
------------

This is one of the best known dataset
to be found in the machine learning literature.

It was introduced by the statistician Ronald Fisher
in his 1936 paper "The use of multiple measurements in taxonomic problems",
Annals of Eugenics. 7 (2): 179–188.

It contains 150 instances of iris plants:

- 50 Iris Setosa,
- 50 Iris Versicolour,
- 50 Iris Virginica.

Each instance is characterized by:

- its sepal length in cm,
- its sepal width in cm,
- its petal length in cm,
- its petal width in cm.

This dataset can be used for either clustering purposes
or classification ones.

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.post.dataset.andrews_curves import AndrewsCurves
from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.post.dataset.radviz import Radar
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from numpy.random import choice

configure_logger()


##############################################################################
# Load Iris dataset
# -----------------
# We can easily load this dataset by means of the
# :meth:`~gemseo.api.load_dataset` function of the API:

iris = load_dataset("IrisDataset")

##############################################################################
# and get some information about it
print(iris)

##############################################################################
# Manipulate the dataset
# ----------------------
# We randomly select 10 samples to display.

shown_samples = choice(iris.length, size=10, replace=False)

##############################################################################
# If the pandas library is installed, we can export the iris dataset to a
# dataframe and print(it.
dataframe = iris.export_to_dataframe()
print(dataframe)

##############################################################################
# We can also easily access the 10 samples previously selected,
# either globally
data = iris.get_all_data(False)
print(data[0][shown_samples, :])

##############################################################################
# or only the parameters:
parameters = iris.get_data_by_group("parameters")
print(parameters[shown_samples, :])

##############################################################################
# or only the labels:
labels = iris.get_data_by_group("labels")
print(labels[shown_samples, :])

##############################################################################
# Plot the dataset
# ----------------
# Lastly, we can plot the dataset in various ways. We will note that the
# samples are colored according to their labels.

##############################################################################
# Plot scatter matrix
# ~~~~~~~~~~~~~~~~~~~
# We can use the :class:`.ScatterMatrix` plot where each non-diagonal block
# represents the samples according to the x- and y- coordinates names
# while the diagonal ones approximate the probability distributions of the
# variables, using either an histogram or a kernel-density estimator.
ScatterMatrix(iris, classifier="specy", kde=True).execute(save=False, show=True)

##############################################################################
# Plot parallel coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# We can use the
# :class:`~gemseo.post.dataset.parallel_coordinates.ParallelCoordinates` plot,
# a.k.a. cowebplot, where each samples
# is represented by a continuous straight line in pieces whose nodes are
# indexed by the variables names and measure the variables values.
ParallelCoordinates(iris, "specy").execute(save=False, show=True)

##############################################################################
# Plot Andrews curves
# ~~~~~~~~~~~~~~~~~~~
# We can use the :class:`.AndrewsCurves` plot
# which can be viewed as a smooth
# version of the parallel coordinates. Each sample is represented by a curve
# and if there is structure in data, it may be visible in the plot.
AndrewsCurves(iris, "specy").execute(save=False, show=True)

##############################################################################
# Plot Radar
# ~~~~~~~~~~
# We can use the :class:`.Radar` plot
Radar(iris, "specy").execute(save=False, show=True)
