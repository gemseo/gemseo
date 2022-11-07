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
#     Matthias De Lozzo
from __future__ import annotations

from typing import Iterable

import pytest
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.remapping import RemappingDiscipline
from numpy import array
from numpy.testing import assert_equal


class NewDiscipline(MDODiscipline):
    """A new discipline."""

    def __init__(self) -> None:
        super().__init__(name="foo")
        self.input_grammar.update(["in_1", "in_2"])
        self.output_grammar.update(["out_1", "out_2"])
        self.default_inputs = {"in_1": array([1.0]), "in_2": array([2.0, 3.0])}

    def _run(self) -> None:
        self.local_data["out_1"] = self.local_data["in_1"] + 1
        self.local_data["out_2"] = self.local_data["in_2"] - 1

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self.jac = {
            "out_1": {"in_1": array([[1.0]]), "in_2": array([[1.0, 1.0]])},
            "out_2": {
                "in_1": array([[1.0], [1.0]]),
                "in_2": array([[1.0, 1.0], [1.0, 1.0]]),
            },
        }


@pytest.fixture(scope="module", params=[False, True])
def discipline(module_tmp_wd, request) -> MDODiscipline:
    """A remapping discipline."""
    discipline = RemappingDiscipline(
        NewDiscipline(),
        {
            "new_in_1": "in_1",
            "new_in_2": ("in_2", 0),
            "new_in_3": ("in_2", 1),
        },
        {"new_out_1": "out_1", "new_out_2": "out_2"},
    )
    if not request.param:
        # Use the original remapping discipline
        return discipline

    # Use the remapping discipline loaded from the disk, after serialization
    file_name = "discipline.pkl"
    discipline.serialize(file_name)
    return MDODiscipline.deserialize(file_name)


def test_original_discipline(discipline):
    """Check the property original_discipline."""
    assert discipline.original_discipline == discipline._discipline


def test_with_discipline_wo_default_values():
    """Check that the wrapped discipline must have default input values."""
    with pytest.raises(
        ValueError, match="The original discipline has no default input values."
    ):
        RemappingDiscipline(MDODiscipline(), {}, {})


def test_discipline_name(discipline):
    """Check that the discipline name is the name of the original discipline."""
    assert discipline.name == "foo"


def test_io_names(discipline):
    """Check the input and output names."""
    assert list(discipline.get_input_data_names()) == [
        "new_in_1",
        "new_in_2",
        "new_in_3",
    ]
    assert list(discipline.get_output_data_names()) == ["new_out_1", "new_out_2"]


def test_default_inputs(discipline):
    """Check the default inputs when missing in original discipline."""
    assert_equal(
        {"new_in_1": array([1.0]), "new_in_2": array([2.0]), "new_in_3": array([3.0])},
        discipline.default_inputs,
    )


def test_execute(discipline):
    """Check the execution of the discipline."""
    discipline.execute()
    assert_equal(discipline.local_data["new_out_1"], array([2.0]))
    assert_equal(discipline.local_data["new_out_2"], array([1.0, 2.0]))


def test_linearize(discipline):
    """Check the linearization of the discipline."""
    discipline.linearize(force_all=True)
    assert_equal(
        discipline.jac,
        {
            "new_out_1": {
                "new_in_1": array([[1.0]]),
                "new_in_2": array([[1.0]]),
                "new_in_3": array([[1.0]]),
            },
            "new_out_2": {
                "new_in_1": array([[1.0], [1.0]]),
                "new_in_2": array([[1.0], [1.0]]),
                "new_in_3": array([[1.0], [1.0]]),
            },
        },
    )


@pytest.mark.parametrize(
    "mapping,expected",
    [
        ({"new_in_1": "x"}, {"new_in_1": ("x", slice(None))}),
        ({"new_in_1": ("x", 1)}, {"new_in_1": ("x", slice(1, 2))}),
        ({"new_in_1": ("x", [0, 2])}, {"new_in_1": ("x", [0, 2])}),
        ({"new_in_1": ("x", range(0, 2))}, {"new_in_1": ("x", range(0, 2))}),
    ],
)
def test_format_mapping(mapping, expected):
    """Check the formatting of a mapping."""
    formatted_mapping = RemappingDiscipline._RemappingDiscipline__format_mapping(
        mapping
    )
    assert formatted_mapping == expected
