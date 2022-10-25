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
"""
Classification API
==================

Here are some examples of the machine learning API
applied to classification models.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import load_dataset
from gemseo.mlearning.api import create_classification_model
from gemseo.mlearning.api import get_classification_models
from gemseo.mlearning.api import get_classification_options

configure_logger()


###############################################################################
# Get available classification models
# -----------------------------------
print(get_classification_models())

###############################################################################
# Get classification model options
# --------------------------------
print(get_classification_options("KNNClassifier"))

###############################################################################
# Create classification model
# ---------------------------
iris = load_dataset("IrisDataset", as_io=True)

model = create_classification_model("KNNClassifier", data=iris)
model.learn()

print(model)
