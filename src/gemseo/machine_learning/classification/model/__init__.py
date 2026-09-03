# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Classifiers.

This package includes classification models, a.k.a. classifiers.

Given an input data,
a classifier is used to predict
either the class associated with this input data
or the probability of belonging to each class.

Use the
[ClassifierFactory][gemseo.machine_learning.classification.model.factory.ClassifierFactory]
to access all the available classifiers
or derive the
[BaseClassifier][gemseo.machine_learning.classification.core.base_classifier.BaseClassifier]
class to add a new one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.machine_learning.classification.model.factory import CLASSIFIER_FACTORY  # noqa: F401
    from gemseo.machine_learning.classification.model.knn import KNNClassifier  # noqa: F401
    from gemseo.machine_learning.classification.model.random_forest import (
        RandomForestClassifier,  # noqa: F401
    )
    from gemseo.machine_learning.classification.model.svm import SVMClassifier  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "CLASSIFIER_FACTORY": "factory",
    "KNNClassifier": "knn",
    "RandomForestClassifier": "random_forest",
    "SVMClassifier": "svm",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
