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
"""Tests for VariableRenamer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.core.discipline.data_processor import NameMapping
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.util.discipline import DisciplineVariableProperties
from gemseo.util.discipline import VariableRenamer
from gemseo.util.discipline import VariableTranslation
from gemseo.util.discipline import get_discipline_variable_properties
from gemseo.util.discipline import rename_discipline_variables
from gemseo.util.discipline import update_default_input_values
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Mapping


def _build_namemapping_input(
    original_dict: Mapping[str, str],
    index: int,
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Build the two mappings of a `NameMapping`.

    Args:
        original_dict: The mapping to place.
        index: `0` to place it as the input mapping, `1` as the output mapping.

    Returns:
        The input mapping and the output mapping.
    """
    lst = [{}, {}]
    lst[index] = original_dict
    return tuple(lst)


@pytest.fixture(scope="module")
def translations() -> tuple[
    VariableTranslation, VariableTranslation, VariableTranslation
]:
    """Three translations."""
    return (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="B",
            is_input=False,
            variable_name="b",
            new_variable_name="y",
        ),
        VariableTranslation(
            discipline_name="A",
            is_input=False,
            variable_name="c",
            new_variable_name="z",
        ),
    )


@pytest.fixture(scope="module")
def translators() -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    """The translators."""
    return {"A": ({"a": "x"}, {"c": "z"}), "B": ({}, {"b": "y"})}


@pytest.fixture
def disciplines() -> tuple[
    AnalyticDiscipline, AnalyticDiscipline, AnalyticDiscipline, AnalyticDiscipline
]:
    """Four analytic disciplines."""
    return (
        AnalyticDiscipline({"c": "2*a"}, name="A"),
        AnalyticDiscipline({"t": "3*g"}, name="C"),
        AnalyticDiscipline({"c": "4*a"}, name="A"),
        AnalyticDiscipline({"b": "5*j"}, name="B"),
    )


def test_variable_translation():
    """Check VariableTranslation."""
    for is_input in [True, False]:
        translation = VariableTranslation(
            discipline_name="a",
            is_input=is_input,
            variable_name="b",
            new_variable_name="c",
        )
        assert translation._fields == (
            "discipline_name",
            "is_input",
            "variable_name",
            "new_variable_name",
        )
        assert translation.discipline_name == "a"
        assert translation.is_input == is_input
        assert translation.variable_name == "b"
        assert translation.new_variable_name == "c"
        assert str(translation) == repr(translation) == f"'a'.{is_input}.'b'='c'"


def test_variable_renamer(
    translations: tuple[VariableTranslation, VariableTranslation, VariableTranslation],
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
):
    """Check VariableRenamer."""
    renamer = VariableRenamer.from_translations(*translations)
    assert renamer.translations == translations
    assert renamer.translators == translators
    expected = """
+-----------------+-----------+---------------+-------------------+
| Discipline name | Is input? | Variable name | New variable name |
+-----------------+-----------+---------------+-------------------+
|        A        |    True   |       a       |         x         |
|        B        |   False   |       b       |         y         |
|        A        |   False   |       c       |         z         |
+-----------------+-----------+---------------+-------------------+
"""  # noqa: E501

    assert repr(renamer) == expected[1:-1]
    expected = """
<div style='margin: 1em;'><table>
    <thead>
        <tr>
            <th>Discipline name</th>
            <th>Is input?</th>
            <th>Variable name</th>
            <th>New variable name</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>A</td>
            <td>True</td>
            <td>a</td>
            <td>x</td>
        </tr>
        <tr>
            <td>B</td>
            <td>False</td>
            <td>b</td>
            <td>y</td>
        </tr>
        <tr>
            <td>A</td>
            <td>False</td>
            <td>c</td>
            <td>z</td>
        </tr>
    </tbody>
</table></div>
"""
    assert renamer._repr_html_() == expected[1:-1]


def test_variable_renamer_from_translations_and_tuples(
    translations: tuple[VariableTranslation, VariableTranslation, VariableTranslation],
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
):
    """Check VariableRenamer from translations and tuples."""
    renamer = VariableRenamer.from_translations(
        ("A", True, "a", "x"),
        VariableTranslation(
            discipline_name="B",
            is_input=False,
            variable_name="b",
            new_variable_name="y",
        ),
        ("A", False, "c", "z"),
    )
    assert renamer.translations == translations
    assert renamer.translators == translators


def test_variable_renamer_from_dictionary(
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
):
    """Check VariableRenamer from dictionary."""
    renamer = VariableRenamer.from_dictionary({
        "A": ({"a": "x"}, {"c": "z"}),
        "B": ({}, {"b": "y"}),
    })
    translations = (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="A",
            is_input=False,
            variable_name="c",
            new_variable_name="z",
        ),
        VariableTranslation(
            discipline_name="B",
            is_input=False,
            variable_name="b",
            new_variable_name="y",
        ),
    )
    assert renamer.translations == translations
    assert renamer.translators == translators


@pytest.mark.parametrize(
    ("sep", "file_name"),
    [({}, "translations.csv"), ({"sep": ";"}, "translations_sep.csv")],
)
def test_variable_renamer_from_csv(
    sep,
    file_name,
    translations: tuple[VariableTranslation, VariableTranslation, VariableTranslation],
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
):
    """Check VariableRenamer from a CSV file."""
    file_path = Path(__file__).parent / "data" / file_name
    renamer = VariableRenamer.from_csv(file_path, **sep)
    assert renamer.translations == translations
    assert renamer.translators == translators


def test_variable_renamer_from_spread_sheet(
    translations: tuple[VariableTranslation, VariableTranslation, VariableTranslation],
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
):
    """Check VariableRenamer from a spreadsheet file."""
    file_path = Path(__file__).parent / "data" / "translations.xlsx"
    renamer = VariableRenamer.from_spreadsheet(file_path)
    assert renamer.translations == translations
    assert renamer.translators == translators


@pytest.mark.parametrize(
    "translation",
    [
        ("A", "out", "a", "x"),
        ("A", "", "a", "x"),
        ("A", 1, "a", "x"),
        ("A", None, "a", "x"),
        VariableTranslation(
            discipline_name="A",
            is_input="out",
            variable_name="a",
            new_variable_name="x",
        ),
    ],
    ids=["tuple_str", "tuple_empty", "tuple_int", "tuple_none", "variable_translation"],
)
def test_add_translation_is_input_not_bool(translation, snapshot):
    """Check the error raised when `is_input` is neither `True` nor `False`."""
    with assert_exception(TypeError, snapshot):
        VariableRenamer.from_translations(translation)


@pytest.mark.parametrize(
    ("file_name", "create_renamer"),
    [
        ("translations_is_input_not_bool.csv", VariableRenamer.from_csv),
        ("translations_is_input_not_bool.xlsx", VariableRenamer.from_spreadsheet),
    ],
)
def test_renaming_from_file_is_input_not_bool(file_name, create_renamer, snapshot):
    """Check the error raised when `is_input` is misspelled in a renaming file."""
    file_path = Path(__file__).parent / "data" / file_name
    with assert_exception(TypeError, snapshot):
        create_renamer(file_path)


def test_rename_twice_log(caplog: pytest.LogCaptureFixture):
    """Check the message logged when renaming a variable twice with same name."""
    translations = (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
    )
    VariableRenamer.from_translations(*translations)
    assert caplog.record_tuples[0] == (
        "gemseo.util.discipline",
        30,
        (
            "In discipline 'A', "
            "the variable 'a' cannot be renamed to 'x' "
            "because it has already been renamed to 'x'."
        ),
    )


def test_rename_twice_error(snapshot):
    """Check the error message raised when renaming a variable twice with diff name."""
    translations = (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="y",
        ),
    )
    re.escape(
        "In discipline 'A', "
        "the variable 'a' cannot be renamed to 'y' "
        "because it has already been renamed to 'x'."
    )
    with assert_exception(ValueError, snapshot):
        VariableRenamer.from_translations(*translations)


def test_add_translations_by_variable():
    """Check the method add_translations_by_variable."""
    renamer = VariableRenamer()
    renamer.add_translations_by_variable("x", {"A": ["a", True], "B": ["b", False]})
    renamer.add_translations_by_variable("z", {"C": ["c", True]})
    assert renamer.translations == (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="B",
            is_input=False,
            variable_name="b",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="C",
            is_input=True,
            variable_name="c",
            new_variable_name="z",
        ),
    )
    assert renamer.translators == {
        "A": ({"a": "x"}, {}),
        "B": ({}, {"b": "x"}),
        "C": ({"c": "z"}, {}),
    }


def test_add_translations_by_discipline():
    """Check the method add_translations_by_discipline."""
    renamer = VariableRenamer()
    renamer.add_translations_by_discipline("A", {"a": "x"}, {"b": "x"})
    renamer.add_translations_by_discipline("C", {"c": "z"}, {})
    assert renamer.translations == (
        VariableTranslation(
            discipline_name="A",
            is_input=True,
            variable_name="a",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="A",
            is_input=False,
            variable_name="b",
            new_variable_name="x",
        ),
        VariableTranslation(
            discipline_name="C",
            is_input=True,
            variable_name="c",
            new_variable_name="z",
        ),
    )
    assert renamer.translators == {"A": ({"a": "x"}, {"b": "x"}), "C": ({"c": "z"}, {})}


def test_rename_discipline_variables(
    disciplines: tuple[
        AnalyticDiscipline, AnalyticDiscipline, AnalyticDiscipline, AnalyticDiscipline
    ],
    translators: dict[str, tuple[dict[str, str], dict[str, str]]],
    caplog: pytest.LogCaptureFixture,
    snapshot,
):
    """Check rename_discipline_variables.

    Translators: {"A": ({"a": "x"}, {"c": "z"}), "B": ({}, {"b": "y"})}

    Disciplines:
        - AnalyticDiscipline({"c": "2*a"}, name="A"): rename a to x and c to z
        - AnalyticDiscipline({"t": "3*g"}, name="C"): no renaming
        - AnalyticDiscipline({"c": "4*a"}, name="A"): rename a to x and c to z
        - AnalyticDiscipline({"b": "5*j"}, name="B"): rename b to y
    """
    rename_discipline_variables(disciplines, translators)
    disc_a, disc_c, other_disc_a, disc_b = disciplines
    assert_equal(disc_a.execute({"x": array([3.0])})["z"], array([6.0]))
    assert_equal(disc_c.execute({"g": array([3.0])})["t"], array([9.0]))
    assert_equal(other_disc_a.execute({"x": array([3.0])})["z"], array([12.0]))
    assert_equal(disc_b.execute({"j": array([3.0])})["y"], array([15.0]))
    assert caplog.record_tuples[0] == (
        "gemseo.util.discipline",
        30,
        "The discipline 'C' has no translator.",
    )

    with assert_exception(ValueError, snapshot):
        rename_discipline_variables(disciplines, {"A": ({"foo": "bar"}, {})})

    with assert_exception(TypeError, snapshot):
        rename_discipline_variables(disciplines, {"A": {"a": "x"}})


@pytest.mark.parametrize("in_", ["x", "y"])
@pytest.mark.parametrize("out", ["x", "y"])
@pytest.mark.parametrize(("new_in", "new_out"), [("a", "a"), ("a", "b")])
def test_rename_discipline_variables_shared_names(in_, out, new_in, new_out):
    """Check rename_discipline_variables when input and output names coincide."""
    discipline = AnalyticDiscipline({out: f"2*{in_}"}, name="Discipline")
    translator = ({in_: new_in}, {out: new_out})
    rename_discipline_variables((discipline,), {"Discipline": translator})
    discipline.execute({new_in: array([3.0])})
    assert_equal(discipline.io.input_data[new_in], array([3.0]))
    assert_equal(discipline.io.output_data[new_out], array([6.0]))


def test_get_discipline_variable_properties():
    """Check get_discipline_variable_properties."""
    discipline = AnalyticDiscipline({"foo": "foo"})
    grammars = [discipline.io.input_grammar, discipline.io.output_grammar]
    description = "The description of foo."
    for index, grammar in enumerate(grammars):
        grammar.descriptions["foo"] = description
        name_to_properties = get_discipline_variable_properties(discipline)[index]
        assert name_to_properties["foo"] == DisciplineVariableProperties(
            current_name="foo",
            original_name="foo",
            current_name_without_namespace="foo",
            description=description,
        )
        grammar.rename_element("foo", "bar")
        discipline.io.data_processor = NameMapping(
            *_build_namemapping_input({"bar": "foo"}, index)
        )
        name_to_properties = get_discipline_variable_properties(discipline)[index]
        assert name_to_properties["bar"] == DisciplineVariableProperties(
            current_name="bar",
            original_name="foo",
            current_name_without_namespace="bar",
            description=description,
        )
        grammar.rename_element("bar", "baz")
        discipline.io.data_processor = NameMapping(
            *_build_namemapping_input({"baz": "foo"}, index)
        )
        name_to_properties = get_discipline_variable_properties(discipline)[index]
        assert name_to_properties["baz"] == DisciplineVariableProperties(
            current_name="baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description=description,
        )
        grammar.add_namespace("baz", "ns")
        name_to_properties = get_discipline_variable_properties(discipline)[index]
        assert name_to_properties["ns:baz"] == DisciplineVariableProperties(
            current_name="ns:baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description=description,
        )
        del grammar.descriptions["ns:baz"]
        name_to_properties = get_discipline_variable_properties(discipline)[index]
        assert name_to_properties["ns:baz"] == DisciplineVariableProperties(
            current_name="ns:baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description="",
        )


def test_update_default_input_values():
    """Check that update_default_input_values works correctly."""
    discipline_1 = AnalyticDiscipline({"y1": "a+b"})
    discipline_2 = AnalyticDiscipline({"y2": "a+b+c"})
    update_default_input_values([discipline_1, discipline_2], {"a": 1.0, "c": 2.0})
    assert discipline_1.io.input_grammar.defaults == {"a": 1.0, "b": 0.0}
    assert discipline_2.io.input_grammar.defaults == {"a": 1.0, "b": 0.0, "c": 2.0}
