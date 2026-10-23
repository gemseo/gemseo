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
"""Tests for the lazy re-export of space classes from `gemseo.space`."""

from __future__ import annotations

import gemseo.space
from gemseo.util.testing.package_import import make_lazy_reexport_tests

# No `deferred_sample`: `DesignSpace` is loaded eagerly through the import chain of
# `gemseo` itself, so the submodule is already in `sys.modules` by the time the package
# is imported. The lazy re-export (nothing bound in the package namespace) still holds.
globals().update(make_lazy_reexport_tests(gemseo.space))
