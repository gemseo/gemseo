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
"""Tests for the lazy re-export of settings classes from `gemseo.machine_learning`."""

from __future__ import annotations

import gemseo.machine_learning
from gemseo.util.testing.package_import import make_lazy_reexport_tests

_EXTRA_ALL = (
    "create_classification_model",
    "create_clustering_model",
    "create_mlearning_model",
    "create_regression_model",
    "get_classification_models",
    "get_classification_options",
    "get_clustering_models",
    "get_clustering_options",
    "get_mlearning_models",
    "get_mlearning_options",
    "get_regression_models",
    "get_regression_options",
)

globals().update(
    make_lazy_reexport_tests(gemseo.machine_learning, extra_all=_EXTRA_ALL)
)
