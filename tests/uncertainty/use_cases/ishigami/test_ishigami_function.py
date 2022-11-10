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

from gemseo.uncertainty.use_cases.ishigami.functions import compute_gradient
from gemseo.uncertainty.use_cases.ishigami.functions import compute_output
from gemseo.uncertainty.use_cases.ishigami.ishigami_function import IshigamiFunction


def test_ishigami_function():
    """Check that the IshigamiFunction has the right name, function and jacobian."""
    function = IshigamiFunction()
    assert function.name == "Ishigami"
    assert function._func == compute_output
    assert function._jac == compute_gradient
