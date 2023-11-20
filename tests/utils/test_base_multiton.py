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
"""Tests for the Multiton metaclass."""

from __future__ import annotations

from gemseo.utils.base_multiton import BaseMultiton


class A(metaclass=BaseMultiton):
    """A class using a BaseMultiton metaclass."""


def test_multiton():
    """Check that a class using a BaseMultiton metaclass can create only one
    instance."""
    obj_1 = A()
    obj_2 = A()
    assert id(obj_1) == id(obj_2)


def test_multiton_cache_clear():
    """Verify the clearing of the cache of the multiton."""
    A()
    assert A._BaseMultiton__keys_to_class_instances
    A.clear_cache()
    assert not A._BaseMultiton__keys_to_class_instances
