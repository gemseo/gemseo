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

from importlib import reload

import pytest

import gemseo.core._discipline_class_injector
import gemseo.core.discipline

# The reload done in the fixture breaks other test where discipline pickles are done.
# The error
# E       _pickle.PicklingError: Can't pickle <enum 'LinearizationMode'>: it's not the same object as gemseo.core.discipline.MDODiscipline.LinearizationMode #noqa: E501
# does not seem to be avoidable.
# A workaround is to execute those tests last with pytest-order but this plugin is not
# compatible with pytest-xdist.
# So let's skip them for now!
pytest.skip(allow_module_level=True)


def _reload(monkeypatch):
    # Force taking into account the new base class in the injector.
    reload(gemseo.core._discipline_class_injector)
    reload(gemseo.core.discipline)

    yield

    # Restore the original base discipline class.
    monkeypatch.undo()
    reload(gemseo.core._discipline_class_injector)
    reload(gemseo.core.discipline)


@pytest.fixture
def prepare(monkeypatch):
    """Inject a new base discipline class temporarily."""
    monkeypatch.setenv(
        "GEMSEO_BASE_DISCIPLINE_CLASS",
        "tests.core.discipline_injector_class.NewBaseDiscipline",
    )
    yield from _reload(monkeypatch)


@pytest.fixture
def prepare_error(monkeypatch):
    """Inject a new base discipline class temporarily."""
    monkeypatch.setenv(
        "GEMSEO_BASE_DISCIPLINE_CLASS",
        "dummy.Dummy",
    )
    yield from _reload(monkeypatch)


def test_class_injector_with_nothing():
    """Verify that nothing is injected by default."""
    from gemseo.core.discipline import MDODiscipline

    class Disc(MDODiscipline):
        def _run(self):
            pass

    assert not hasattr(Disc(), "hi_there")


def test_class_injector(prepare):
    """Verify the injection of a new base class."""

    from gemseo.core.discipline import MDODiscipline

    class Disc(MDODiscipline):
        def _run(self):
            pass

    assert Disc().hi_there is None


def test_class_injector_error(prepare_error):
    """Verify the error handling when injecting a bad class."""
    from gemseo.core.discipline import MDODiscipline

    match = "The class Dummy cannot be imported from the package dummy."
    with pytest.raises(ImportError, match=match):

        class Disc(MDODiscipline):
            def _run(self):
                pass
