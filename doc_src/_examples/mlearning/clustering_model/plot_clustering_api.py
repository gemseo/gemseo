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
API
===

Here are some examples of the machine learning API
applied to clustering models.
"""
###############################################################################
# Import
# ------
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.mlearning.api import create_clustering_model
from gemseo.mlearning.api import get_clustering_models
from gemseo.mlearning.api import get_clustering_options

configure_logger()


###############################################################################
# Get available clustering models
# -------------------------------
print(get_clustering_models())

###############################################################################
# Get clustering model options
# ----------------------------
print(get_clustering_options("GaussianMixture"))

###############################################################################
# Create clustering model
# -----------------------
iris = load_dataset("IrisDataset")

model = create_clustering_model("KMeans", data=iris, n_clusters=3)
model.learn()

print(model)
