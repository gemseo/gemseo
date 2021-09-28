# -*- coding: utf-8 -*-
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
from typing import List

import pytest

from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem


@pytest.fixture
def sellar_disciplines():  # type: (...)-> List[Sellar1,Sellar2,SellarSystem]
    """The disciplines of the Sellar problem.

    Returns:
        * A Sellar1 discipline.
        * A Sellar2 discipline.
        * A SellarSystem discipline.
    """
    return [Sellar1(), Sellar2(), SellarSystem()]
