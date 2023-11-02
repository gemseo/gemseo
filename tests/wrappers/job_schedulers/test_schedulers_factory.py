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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.wrappers.job_schedulers.schedulers_factory import SchedulersFactory


@pytest.fixture()
def factory() -> SchedulersFactory:
    return SchedulersFactory()


def test_available_schedulers(factory):
    """Test the availability of job schedulers using the Factory mechanism."""
    assert factory.is_available("LSF")
    assert factory.is_available("SLURM")

    assert "LSF" in factory.scheduler_names
    assert "SLURM" in factory.scheduler_names
