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
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests the trust updater factory."""

from __future__ import annotations

from gemseo.algos.opt.core.updater_factory import UpdaterFactory
from gemseo.utils.testing.helpers import assert_exception


def test_unavailable_update(snapshot) -> None:
    """Tests the unavailable update exception."""
    updater_factory = UpdaterFactory()
    with assert_exception(ValueError, snapshot):
        updater_factory.create(
            "method_name",
            thresholds=(0.0, 0.25),
            multipliers=(0.5, 2.0),
            bound=1.0,
        )

    updater_factory.create("penalty", (0.0, 0.25), (0.5, 2.0), 1.0)
