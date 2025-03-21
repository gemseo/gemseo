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
High-level functions
====================

The :mod:`gemseo.mlearning` package includes high-level functions
to create clustering models from model class names.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_benchmark_dataset
from gemseo.mlearning import create_clustering_model
from gemseo.mlearning import get_clustering_models
from gemseo.mlearning import get_clustering_options

configure_logger()


# %%
# Available models
# ----------------
# Use the :func:`.get_clustering_models`
# to list the available model class names:
get_clustering_models()

# %%
# Available model options
# -----------------------
# Use the :func:`.get_clustering_options`
# to get the options of a model
# from its class name:
get_clustering_options("GaussianMixture", pretty_print=False)

# %%
#
# .. seealso::
#    The functions
#    :func:`.get_clustering_models` and :func:`.get_clustering_options`
#    can be very useful for the developers.
#    As a user,
#    it may be easier to consult :ref:`this page <gen_clustering_algos>`
#    to find out about the different algorithms and their options.
#
# Creation
# --------
# Given a training dataset, *e.g.*
dataset = create_benchmark_dataset("IrisDataset")
# %%
# use the :func:`.create_clustering_model` function
# to create a clustering model from its class name and settings:
model = create_clustering_model("KMeans", data=dataset, n_clusters=3)
model.learn()
