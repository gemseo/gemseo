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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to execute DOE algorithms from their class names."""

from __future__ import annotations

from gemseo.algos.base_algo_factory import BaseAlgoFactory
from gemseo.algos.doe.doe_library import DOELibrary


class DOEFactory(BaseAlgoFactory):
    """DOE factory to create DOE libraries, see BaseAlgoFactory."""

    _CLASS = DOELibrary
    _MODULE_NAMES = ("gemseo.algos.doe",)
