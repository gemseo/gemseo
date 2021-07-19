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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import division, unicode_literals

import pytest

from gemseo.problems.disciplines_factory import DisciplinesFactory
from gemseo.utils.py23_compat import Path

DATA = Path(__file__).parent / "data"

# TODO: some of theses tests are actually Factory tests, move them to the right
# place


def test_init(monkeypatch):
    monkeypatch.setenv("GEMSEO_PATH", DATA)

    fact1 = DisciplinesFactory()
    # Force update since we changed the GEMSEO_PATH
    fact1.update()
    assert "DummyDisciplineIMP" in fact1.disciplines

    gemseo_path = "{}:{}".format(DATA, DATA)
    monkeypatch.setenv("GEMSEO_PATH", gemseo_path)
    fact1.update()

    fact2 = DisciplinesFactory()

    monkeypatch.delenv("GEMSEO_PATH")
    fact2.update()

    assert fact1.disciplines == fact2.disciplines

    fact1.update()
    fact2.update()


def test_create(monkeypatch):
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    fact = DisciplinesFactory()
    dummy = fact.create("DummyDisciplineIMP", opts1=1)
    assert dummy.opts1 == 1

    fact2 = DisciplinesFactory()
    assert id(fact.factory) == id(fact2.factory)

    fact3 = DisciplinesFactory()
    dummy = fact3.create("DummyDisciplineIMP", opts1=1)

    with pytest.raises(ImportError):
        fact3.create("unknown")

    fact4 = DisciplinesFactory()
    dummy = fact4.create("DummyDisciplineIMP", jac_approx_n_processes=1)
