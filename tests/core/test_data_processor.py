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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray
from scipy import linalg

from gemseo.core.discipline import Discipline
from gemseo.core.discipline.data_processor import ComplexDataProcessor
from gemseo.core.discipline.data_processor import FloatDataProcessor
from gemseo.core.discipline.data_processor import NameMapping
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


def test_float_data_processor() -> None:
    """"""
    dp = FloatDataProcessor()
    in_data = {"a": array([1.1]), "b": array([3.1, 4.1])}
    pre_data = dp.pre_process_data(in_data)
    assert len(pre_data) == len(in_data)
    for k, v in pre_data.items():
        assert k in in_data
        if k == "a":
            assert isinstance(v, float)
        else:
            assert isinstance(v, list)

    post_data = dp.post_process_data(pre_data)
    assert len(post_data) == len(in_data)
    for k, v in post_data.items():
        assert k in in_data
        assert isinstance(v, ndarray)


def test_complex_data_processor() -> None:
    """"""
    dp = ComplexDataProcessor()
    in_data = {"a": array([1.1 + 2j]), "b": array([3.1, 4.1 + 3j])}
    pre_data = dp.pre_process_data(in_data)
    assert len(pre_data) == len(in_data)
    for k, v in pre_data.items():
        assert k in in_data
        assert linalg.norm(v - in_data[k].real) == 0.0
        assert linalg.norm(v.imag) == 0
        assert v.dtype == float64

    post_data = dp.post_process_data(pre_data)
    assert len(post_data) == len(in_data)
    for k, v in post_data.items():
        assert k in in_data
        assert isinstance(v, ndarray)
        assert v.dtype == complex128

    sm = SobieskiMission("float64")
    sm.io.data_processor = dp
    sm.execute({
        "x_shared": array(sm.io.input_grammar.defaults["x_shared"], dtype="complex128")
    })

    assert sm.io.data["y_4"].dtype == complex128


@pytest.mark.parametrize(
    ("mapping", "input_names", "output_names", "input_data"),
    [
        (
            {"A": "a", "B": "b", "O1": "o1", "O2": "o2"},
            ("A", "B"),
            ("O1", "O2"),
            {"A": array([1]), "B": array([2])},
        ),
        (
            {"A": "a", "O1": "o1"},
            ("A", "b"),
            ("O1", "o2"),
            {"A": array([1]), "b": array([2])},
        ),
    ],
)
@pytest.mark.parametrize("add_namespace_to_a", [False, True])
@pytest.mark.parametrize("add_namespace_to_b", [False, True])
@pytest.mark.parametrize("add_namespace_to_o1", [False, True])
@pytest.mark.parametrize("add_namespace_to_o2", [False, True])
def test_name_mapping(
    mapping,
    input_data,
    input_names,
    output_names,
    add_namespace_to_a,
    add_namespace_to_b,
    add_namespace_to_o1,
    add_namespace_to_o2,
) -> None:
    """Check the NameMapping data convertor."""
    disc = LocalDisc()
    disc.io.input_grammar.update_from_names(input_names)
    disc.io.output_grammar.update_from_names(output_names)
    disc.io.data_processor = NameMapping(mapping)
    o1_name = output_names[0]
    o2_name = output_names[1]
    input_data_ = input_data.copy()
    if add_namespace_to_a:
        input_name = input_names[0]
        disc.add_namespace_to_input(input_name, "foo")
        input_data_[f"foo:{input_name}"] = input_data_.pop(input_name)
    if add_namespace_to_b:
        input_name = input_names[1]
        disc.add_namespace_to_input(input_name, "bar")
        input_data_[f"bar:{input_name}"] = input_data_.pop(input_name)
    if add_namespace_to_o1:
        output_name = output_names[0]
        disc.add_namespace_to_output(output_name, "baz")
        o1_name = f"baz:{output_name}"
    if add_namespace_to_o2:
        output_name = output_names[1]
        disc.add_namespace_to_output(output_name, "qux")
        o2_name = f"qux:{output_name}"

    out = disc.execute(input_data_)
    assert out[o1_name] == array([3.0])
    assert out[o2_name] == array([-1.0])


class LocalDisc(Discipline):
    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {
            "o1": input_data["a"] + input_data["b"],
            "o2": input_data["a"] - input_data["b"],
        }
