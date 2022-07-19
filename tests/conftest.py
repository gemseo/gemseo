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
"""Test helpers."""
from __future__ import annotations

import pytest
from gemseo.utils.pytest_conftest import *  # noqa: F401,F403

MARK = "doc_examples"


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip by default some marked tests."""
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(reason=f"use '-m {MARK}' to run this test")
        for item in items:
            if MARK in item.keywords:
                item.add_marker(skip_me)
