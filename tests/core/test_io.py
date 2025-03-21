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

import re
from typing import TYPE_CHECKING

import pytest
from numpy import array

from gemseo.core.discipline.data_processor import DataProcessor
from gemseo.core.discipline.io import IO
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.factory import GrammarType
from gemseo.disciplines.analytic import AnalyticDiscipline

if TYPE_CHECKING:
    from gemseo.typing import MutableStrKeyMapping


@pytest.mark.parametrize("grammar_type", GrammarType)
def test_grammar_type(grammar_type):
    io = IO(type, "", grammar_type)
    assert io.grammar_type == grammar_type


@pytest.fixture
def io() -> IO:
    return IO(type, "io", GrammarType.SIMPLE)


def test_prepare_input_data_from(io: IO):
    assert not io.prepare_input_data({})

    # The value is mutable to check deep/shallow copying.
    data_with_default = {"input": []}
    data = {**data_with_default, "input-no-default": 0}
    io.input_grammar.update_from_data(data)
    io.input_grammar.defaults.update(data_with_default)

    # For empty argument: a deepcopy of the defaults are returned.
    prepared_data = io.prepare_input_data({})
    assert prepared_data == data_with_default
    prepared_data["input"] += [0]
    assert data["input"] == [0]

    # For non-empty argument: a shallow copy of the defaults are returned,
    # the alien items are removed.
    prepared_data = io.prepare_input_data({"dummy": 0, "input-no-default": 0})
    assert prepared_data == data
    prepared_data["input"] += [0]
    assert data["input"] == [0, 0]

    # Items with defaults are passed.
    prepared_data = io.prepare_input_data({"input": [0]})
    assert prepared_data == {"input": [0]}


def test_data(io: IO):
    assert not io.data

    io.data = {0: 0}
    assert io.data == {0: 0}


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
    assert_get_io_data(io, "input")


def test_get_output_data(io: IO):
    assert_get_io_data(io, "output")


def test_update(io: IO):
    io.input_grammar.update_from_data({"input": 0})
    io.output_grammar.update_from_data({"output1": 0, "output2": 0})

    # Without namespace.
    assert not io.data
    io.update_output_data({"input": 0, "dummy": 0, "output1": 0})
    assert io.data == {"output1": 0}

    # With namespace.
    io.data.clear()
    io.output_grammar.add_namespace("output1", "n")
    io.update_output_data({"input": 0, "dummy": 0, "output1": 0})
    assert io.data == {"n:output1": 0}


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


def test_initialize(io: IO):
    io.input_grammar.update_from_data({"input": 0})

    validate = False
    io.initialize({"dummy": 0}, validate)
    assert io.data == {"dummy": 0}
    io.initialize({"input": 0}, validate)
    assert io.data == {"input": 0}

    validate = True
    io.initialize({"input": 0}, validate)
    assert io.data == {"input": 0}

    match = (
        "Grammar io_discipline_input: validation failed.\nMissing required names: input"
    )
    with pytest.raises(InvalidDataError, match=match):
        io.initialize({}, validate)

    assert io.data == {"input": 0}


def test_finalize(io: IO):
    io.input_grammar.update_from_data({"input": 0})
    io.output_grammar.update_from_data({"output": 0})

    io.data = {"dummy": 0, "input": 0, "output": 0}
    io.finalize(False)
    io.finalize(True)
    assert io.data == {"dummy": 0, "input": 0, "output": 0}

    # Check validation.
    io.data = {"dummy": 0, "input": "0", "output": "0"}
    match = (
        "Grammar io_discipline_output: validation failed.\nBad type for output: "
        "<class 'str'> instead of <class 'int'>."
    )
    with pytest.raises(InvalidDataError, match=match):
        io.finalize(True)
    assert io.data == {"dummy": 0, "input": "0", "output": "0"}


def test_initialize_finalize_data_processor():
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
    with pytest.raises(
        InvalidDataError,
        match=re.escape(
            "Grammar AnalyticDiscipline_discipline_input: validation failed.\n"
            "error: data.x must be array"
        ),
    ):
        discipline.execute({"x": 1.0})

    # Raises an InvalidDataError when returning scalar input data.
    discipline.io.data_processor = OtherProcessor()
    with pytest.raises(
        InvalidDataError,
        match=re.escape(
            "Grammar AnalyticDiscipline_discipline_output: validation failed.\n"
            "error: data.y must be array"
        ),
    ):
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


def test_set_linear_relationships_error(io: IO):
    match = "The following input_names are not in the input grammar: dummy."
    with pytest.raises(ValueError, match=match):
        io.set_linear_relationships(("dummy",), ())

    match = "The following output_names are not in the output grammar: dummy."
    with pytest.raises(ValueError, match=match):
        io.set_linear_relationships((), ("dummy",))
