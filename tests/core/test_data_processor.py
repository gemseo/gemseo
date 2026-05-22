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
from numpy.testing import assert_array_equal

from gemseo.core.discipline import Discipline
from gemseo.core.discipline.data_processor import ComplexDataProcessor
from gemseo.core.discipline.data_processor import DataProcessor
from gemseo.core.discipline.data_processor import FloatDataProcessor
from gemseo.core.discipline.data_processor import NameMapping
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class LocalDisc(Discipline):
    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {
            "o1": input_data["a"] + input_data["b"],
            "o2": input_data["a"] - input_data["b"],
        }


def test_data_processor_is_abstract() -> None:
    """Verify ``DataProcessor`` cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        DataProcessor()


def test_float_data_processor_pre_process() -> None:
    """Verify ``FloatDataProcessor.pre_process_data`` casts to float or float list."""
    in_data = {"scalar": array([1.1]), "vector": array([3.1, 4.1])}
    pre_data = FloatDataProcessor().pre_process_data(in_data)
    assert pre_data["scalar"] == 1.1
    assert isinstance(pre_data["scalar"], float)
    assert pre_data["vector"] == [3.1, 4.1]
    assert all(isinstance(v, float) for v in pre_data["vector"])


def test_float_data_processor_post_process() -> None:
    """Verify ``FloatDataProcessor.post_process_data`` wraps values in ndarrays."""
    in_data = {"scalar": 1.1, "vector": [3.1, 4.1]}
    post_data = FloatDataProcessor().post_process_data(in_data)
    assert_array_equal(post_data["scalar"], array([1.1]))
    assert_array_equal(post_data["vector"], array([3.1, 4.1]))
    assert all(isinstance(v, ndarray) for v in post_data.values())


def test_complex_data_processor_pre_process() -> None:
    """Verify ``ComplexDataProcessor.pre_process_data`` keeps only the real part."""
    in_data = {"a": array([1.1 + 2j]), "b": array([3.1, 4.1 + 3j])}
    pre_data = ComplexDataProcessor().pre_process_data(in_data)
    assert_array_equal(pre_data["a"], array([1.1]))
    assert_array_equal(pre_data["b"], array([3.1, 4.1]))
    assert all(v.dtype == float64 for v in pre_data.values())


def test_complex_data_processor_post_process() -> None:
    """Verify ``ComplexDataProcessor.post_process_data`` casts to complex128."""
    in_data = {"a": array([1.1]), "b": array([3.1, 4.1])}
    post_data = ComplexDataProcessor().post_process_data(in_data)
    assert_array_equal(post_data["a"], array([1.1 + 0j]))
    assert_array_equal(post_data["b"], array([3.1 + 0j, 4.1 + 0j]))
    assert all(v.dtype == complex128 for v in post_data.values())


def test_complex_data_processor_with_discipline() -> None:
    """Verify ``ComplexDataProcessor`` lets a real discipline accept complex input."""
    discipline = SobieskiMission("float64")
    discipline.io.data_processor = ComplexDataProcessor()
    defaults = discipline.io.input_grammar.defaults
    discipline.execute({"x_shared": array(defaults["x_shared"], dtype="complex128")})
    assert discipline.io.data["y_4"].dtype == complex128


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


def test_name_mapping_pre_process() -> None:
    """Verify ``NameMapping.pre_process_data`` renames global keys to local keys."""
    processor = NameMapping({"global_a": "local_a", "global_b": "local_b"})
    pre = processor.pre_process_data({"global_a": 1, "global_b": 2})
    assert pre == {"local_a": 1, "local_b": 2}


def test_name_mapping_pre_process_passes_unmapped_keys() -> None:
    """Verify unmapped keys go through ``pre_process_data`` unchanged."""
    processor = NameMapping({"global_a": "local_a"})
    pre = processor.pre_process_data({"global_a": 1, "passthrough": 2})
    assert pre == {"local_a": 1, "passthrough": 2}


def test_name_mapping_post_process() -> None:
    """Verify ``NameMapping.post_process_data`` reverses the renaming."""
    processor = NameMapping({"global_a": "local_a", "global_b": "local_b"})
    post = processor.post_process_data({"local_a": 1, "local_b": 2})
    assert post == {"global_a": 1, "global_b": 2}


def test_name_mapping_post_process_passes_unmapped_keys() -> None:
    """Verify unmapped keys go through ``post_process_data`` unchanged."""
    processor = NameMapping({"global_a": "local_a"})
    post = processor.post_process_data({"local_a": 1, "passthrough": 2})
    assert post == {"global_a": 1, "passthrough": 2}
