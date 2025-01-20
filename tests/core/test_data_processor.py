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
            {"A": "a", "B": "b", "O": "o"},
            ["A", "B"],
            ["O"],
            {"A": array([1]), "B": array([2])},
        ),
        ({"A": "a", "O": "o"}, ["A", "b"], ["O"], {"A": array([1]), "b": array([2])}),
    ],
)
def test_name_mapping(mapping, input_data, input_names, output_names) -> None:
    """Check the NameMapping data convertor."""
    disc = LocalDisc()
    disc.io.input_grammar.update_from_names(input_names)
    disc.io.output_grammar.update_from_names(output_names)
    disc.io.data_processor = NameMapping(mapping)
    out = disc.execute(input_data)
    assert out[output_names[0]] == array([3.0])


class LocalDisc(Discipline):
    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.io.data["o"] = self.io.data["a"] + self.io.data["b"]
