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
from __future__ import annotations

import pytest

from gemseo.formulations.idf_settings import IDF_Settings
from gemseo.mda.mda_chain_settings import MDAChain_Settings


@pytest.mark.parametrize("as_mapping", [False, True])
def test_idf_settings(as_mapping):
    """Verify that IDF_Settings can handle MDA settings passed as a mapping."""
    chain_linearize = not (
        IDF_Settings().mda_chain_settings_for_start_at_equilibrium.chain_linearize
    )
    if as_mapping:
        value = {"chain_linearize": chain_linearize}
    else:
        value = MDAChain_Settings(chain_linearize=chain_linearize)
    settings = IDF_Settings(mda_chain_settings_for_start_at_equilibrium=value)
    assert (
        settings.mda_chain_settings_for_start_at_equilibrium.chain_linearize
        is chain_linearize
    )
