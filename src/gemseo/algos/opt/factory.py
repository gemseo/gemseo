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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#         Francois Gallard : refactoring for v1, May 2016
"""A factory of optimization libraries."""

from __future__ import annotations

from gemseo.algos.base_algo_factory import BaseAlgoFactory
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary


class OptimizationLibraryFactory(BaseAlgoFactory):
    """A factory of optimization libraries."""

    _CLASS = BaseOptimizationLibrary
    _PACKAGE_NAMES = ("gemseo.algos.opt",)
