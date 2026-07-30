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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the module scalable_discipline_settings."""

from __future__ import annotations

from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    DEFAULT_D_I,
)
from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    DEFAULT_P_I,
)
from gemseo.problem.mdo.scalable.parametric.standalone.scalable_discipline_settings import (  # noqa: E501
    DEFAULT_SCALABLE_DISCIPLINE_SETTINGS,
)
from gemseo.problem.mdo.scalable.parametric.standalone.scalable_discipline_settings import (  # noqa: E501
    ScalableDisciplineSettings,
)


def test_scalable_discipline_settings() -> None:
    """Check the named tupled ScalableDisciplineSettings."""
    settings = ScalableDisciplineSettings(3, 5)
    assert settings.d_i == 3
    assert settings.p_i == 5

    assert ScalableDisciplineSettings() == ScalableDisciplineSettings(
        DEFAULT_D_I, DEFAULT_P_I
    )


def test_default_scalable_discipline_settings() -> None:
    """Check the tuple DEFAULT_SCALABLE_DISCIPLINE_SETTINGS."""
    assert (
        ScalableDisciplineSettings(),
        ScalableDisciplineSettings(),
    ) == DEFAULT_SCALABLE_DISCIPLINE_SETTINGS
