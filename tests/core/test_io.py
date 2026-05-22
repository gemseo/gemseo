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

from typing import TYPE_CHECKING

import pytest
from numpy import array

from gemseo.core.discipline.data_processor import DataProcessor
from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.discipline.io import IO
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.factory import GrammarType
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.testing.helpers import assert_exception

if TYPE_CHECKING:
    from gemseo.typing import MutableStrKeyMapping


@pytest.mark.parametrize("grammar_type", GrammarType)
def test_grammar_type(grammar_type):
    """Verify ``grammar_type`` returns the grammar type used at construction."""
    io = IO(type, "", grammar_type)
    assert io.grammar_type == grammar_type


@pytest.fixture
def io() -> IO:
    return IO(type, "io", GrammarType.SIMPLE)


def test_prepare_input_data_empty(io: IO):
    """Verify ``prepare_input_data`` is empty when the grammar is empty."""
    assert not io.prepare_input_data({})


def test_prepare_input_data_returns_copy_of_defaults(io: IO):
    """Verify ``prepare_input_data({})`` returns a shallow copy of the defaults.

    Mutating the returned data must not mutate the stored defaults.
    """
    default_value = []
    io.input_grammar.update_from_data({"input": []})
    io.input_grammar.defaults.update({"input": default_value})
    prepared_data = io.prepare_input_data({})
    assert prepared_data == {"input": []}
    prepared_data["input"] += [0]
    assert default_value == [0]
    del prepared_data["input"]
    assert "input" in io.input_grammar.defaults


def test_prepare_input_data_drops_alien_items(io: IO):
    """Verify ``prepare_input_data`` drops items not in the input grammar."""
    io.input_grammar.update_from_data({"input": 0})
    prepared = io.prepare_input_data({"input": 0, "dummy": 0})
    assert prepared == {"input": 0}


def test_prepare_input_data_completes_with_defaults(io: IO):
    """Verify ``prepare_input_data`` completes missing items with defaults."""
    io.input_grammar.update_from_data({"with_default": 0, "without_default": 0})
    io.input_grammar.defaults.update({"with_default": 1})
    prepared = io.prepare_input_data({"without_default": 2})
    assert prepared == {"with_default": 1, "without_default": 2}


def test_prepare_input_data_overrides_defaults(io: IO):
    """Verify ``prepare_input_data`` overrides defaults with passed items."""
    io.input_grammar.update_from_data({"input": 0})
    io.input_grammar.defaults.update({"input": 0})
    assert io.prepare_input_data({"input": 1}) == {"input": 1}


def test_data_default_is_empty(io: IO):
    """Verify the default ``data`` is empty."""
    assert not io.data


def test_data_setter_wraps_in_discipline_data(io: IO):
    """Verify the ``data`` setter wraps the value in a ``DisciplineData``."""
    io.data = {0: 0}
    assert io.data == {0: 0}
    assert isinstance(io.data, DisciplineData)


def assert_get_io_data(io: IO, attr_naming: str) -> None:
    """Factorize testing get_input_data and get_output_data."""
    get_io_data = getattr(io, f"get_{attr_naming}_data")
    assert not get_io_data()
    data = {"name": 0}
    grammar = getattr(io, f"{attr_naming}_grammar")
    grammar.update_from_data(data)

    # Without namespace.
    # Add an item not in the grammar.
    io.data.update({**data, "dummy": 0})
    assert get_io_data() == data

    # With namespace.
    grammar.add_namespace("name", "n")
    assert not get_io_data()
    data_with_ns = {"n:name": 0}
    io.data.update({**data_with_ns, "dummy": 0})
    assert get_io_data() == data_with_ns
    assert get_io_data(with_namespaces=False) == data


def test_get_input_data(io: IO):
    """Verify ``get_input_data`` restricts ``data`` to input grammar items."""
    assert_get_io_data(io, "input")


def test_get_output_data(io: IO):
    """Verify ``get_output_data`` restricts ``data`` to output grammar items."""
    assert_get_io_data(io, "output")


def test_update_output_data_filters_non_outputs(io: IO):
    """Verify ``update_output_data`` ignores items not in the output grammar."""
    io.output_grammar.update_from_data({"output": 0})
    io.update_output_data({"input": 0, "dummy": 0, "output": 0})
    assert io.data == {"output": 0}


def test_update_output_data_handles_namespaces(io: IO):
    """Verify ``update_output_data`` prefixes the namespace to namespaced outputs."""
    io.output_grammar.update_from_data({"output": 0})
    io.output_grammar.add_namespace("output", "n")
    io.update_output_data({"output": 0})
    assert io.data == {"n:output": 0}


def test_update_output_data_keeps_already_namespaced_keys(io: IO):
    """Verify ``update_output_data`` accepts already-namespaced keys as-is."""
    io.output_grammar.update_from_data({"output": 0})
    io.output_grammar.add_namespace("output", "n")
    io.update_output_data({"n:output": 1})
    assert io.data == {"n:output": 1}


class Processor(DataProcessor):
    def pre_process_data(self, data: MutableStrKeyMapping) -> MutableStrKeyMapping:
        new_data = {}
        for key, value in data.items():
            new_data[key] = value + 1
        return new_data

    def post_process_data(self, data: MutableStrKeyMapping) -> MutableStrKeyMapping:
        new_data = {}
        for key, value in data.items():
            new_data[key] = value - 1
        return new_data


class OtherProcessor(Processor):
    def post_process_data(self, data: MutableStrKeyMapping) -> MutableStrKeyMapping:
        return {k: v[0] for k, v in data.items()}


def test_initialize(io: IO, snapshot):
    io.input_grammar.update_from_data({"input": 0})

    validate = False
    io.initialize({"dummy": 0}, validate)
    assert io.data == {"dummy": 0}
    io.initialize({"input": 0}, validate)
    assert io.data == {"input": 0}

    validate = True
    io.initialize({"input": 0}, validate)
    assert io.data == {"input": 0}

    with assert_exception(InvalidDataError, snapshot):
        io.initialize({}, validate)

    assert io.data == {"input": 0}


def test_finalize(io: IO, snapshot):
    io.input_grammar.update_from_data({"input": 0})
    io.output_grammar.update_from_data({"output": 0})

    io.data = {"dummy": 0, "input": 0, "output": 0}
    io.finalize(False)
    io.finalize(True)
    assert io.data == {"dummy": 0, "input": 0, "output": 0}

    # Check validation.
    io.data = {"dummy": 0, "input": "0", "output": "0"}
    with assert_exception(InvalidDataError, snapshot):
        io.finalize(True)
    assert io.data == {"dummy": 0, "input": "0", "output": "0"}


def test_initialize_finalize_data_processor(snapshot):
    # Validate input and output data and use data processor.
    discipline = AnalyticDiscipline({"y": "x+2"})
    discipline.validate_input_data = True
    discipline.validate_output_data = True
    discipline.io.data_processor = Processor()
    discipline.execute()
    # x = 0 in input data
    # x = 1 after pre-processing
    # y = 3 after _run
    # y = 2 after post-processing
    assert discipline.io.data == {"x": array([0.0]), "y": array([2.0])}

    # Raises an InvalidDataError when passing scalar input data.
    with assert_exception(InvalidDataError, snapshot):
        discipline.execute({"x": 1.0})

    # Raises an InvalidDataError when returning scalar input data.
    discipline.io.data_processor = OtherProcessor()
    with assert_exception(InvalidDataError, snapshot):
        discipline.execute({"x": array([1.0])})


def test_linear_relationships(io: IO):
    """Verify both set_linear_relationships and have_linear_relationships."""
    input_names = ("input1", "input2")
    output_names = ("output1", "output2")
    io.input_grammar.update_from_names(input_names)
    io.output_grammar.update_from_names(output_names)

    # Vary the arguments of set_linear_relationship and
    # verify with have_linear_relationship. Clear the internal state afterward.
    io.set_linear_relationships((), ())
    for input_names_ in ((), ("input1",), ("input2",), input_names):
        for output_names_ in ((), ("output1",), ("output2",), output_names):
            assert io.have_linear_relationships(input_names_, output_names_)
    for output_names_ in (("output1",), ("output2",), output_names):
        assert not io.have_linear_relationships(("dummy",), output_names_)

    io._IO__linear_relationships = ()
    io.set_linear_relationships(("input1",), ())
    for input_names_ in ((), ("input1",)):
        for output_names_ in ((), ("output1",), ("output2",), output_names):
            assert io.have_linear_relationships(input_names_, output_names_)
    for input_names_ in (("dummy",), ("input2",)):
        for output_names_ in (("output1",), ("output2",), output_names):
            assert not io.have_linear_relationships(input_names_, output_names_)

    io._IO__linear_relationships = ()
    io.set_linear_relationships((), ("output1",))
    for input_names_ in ((), ("input1",), ("input2",), input_names):
        for output_names_ in ((), ("output1",)):
            assert io.have_linear_relationships(input_names_, output_names_)
    for input_names_ in ((), ("input1",), ("input2",), input_names):
        for output_names_ in (("output2",), ("output1", "output2")):
            assert not io.have_linear_relationships(input_names_, output_names_)

    io._IO__linear_relationships = ()
    io.set_linear_relationships(("input1",), ("output1",))
    for input_names_ in ((), ("input1",)):
        for output_names_ in ((), ("output1",)):
            assert io.have_linear_relationships(input_names_, output_names_)
    for input_names_ in ((), ("input1",), ("input2",), input_names):
        for output_names_ in (("output2",), ("output1", "output2")):
            assert not io.have_linear_relationships(input_names_, output_names_)


def test_set_linear_relationships_error(io: IO, snapshot):
    with assert_exception(ValueError, snapshot):
        io.set_linear_relationships(("dummy",), ())

    with assert_exception(ValueError, snapshot):
        io.set_linear_relationships((), ("dummy",))
