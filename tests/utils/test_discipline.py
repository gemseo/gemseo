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

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.core.discipline.data_processor import NameMapping
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.discipline import DisciplineVariableProperties
from gemseo.utils.discipline import VariableRenamer
from gemseo.utils.discipline import VariableTranslation
from gemseo.utils.discipline import get_discipline_variable_properties
from gemseo.utils.discipline import rename_discipline_variables


@pytest.fixture(scope="module")
def translations() -> tuple[
    VariableTranslation, VariableTranslation, VariableTranslation
]:
    """Three translations."""
    return (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="B", variable_name="b", new_variable_name="y"
        ),
        VariableTranslation(
            discipline_name="A", variable_name="c", new_variable_name="z"
        ),
    )


@pytest.fixture(scope="module")
def translators() -> dict[str, dict[str, str]]:
    """The translators."""
    return {"A": {"a": "x", "c": "z"}, "B": {"b": "y"}}


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
    translation = VariableTranslation(
        discipline_name="a", variable_name="b", new_variable_name="c"
    )
    assert translation._fields == (
        "discipline_name",
        "variable_name",
        "new_variable_name",
    )
    assert translation.discipline_name == "a"
    assert translation.variable_name == "b"
    assert translation.new_variable_name == "c"
    assert str(translation) == repr(translation) == "'a'.'b'='c'"


def test_variable_renamer(translations, translators):
    """Check VariableRenamer."""
    renamer = VariableRenamer.from_translations(*translations)
    assert renamer.translations == translations
    assert renamer.translators == translators
    expected = """
+-----------------+---------------+-------------------+
| Discipline name | Variable name | New variable name |
+-----------------+---------------+-------------------+
|        A        |       a       |         x         |
|        B        |       b       |         y         |
|        A        |       c       |         z         |
+-----------------+---------------+-------------------+
"""  # noqa: E501

    assert repr(renamer) == expected[1:-1]
    expected = """
<div style='margin: 1em;'><table>
    <thead>
        <tr>
            <th>Discipline name</th>
            <th>Variable name</th>
            <th>New variable name</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>A</td>
            <td>a</td>
            <td>x</td>
        </tr>
        <tr>
            <td>B</td>
            <td>b</td>
            <td>y</td>
        </tr>
        <tr>
            <td>A</td>
            <td>c</td>
            <td>z</td>
        </tr>
    </tbody>
</table></div>
"""
    assert renamer._repr_html_() == expected[1:-1]


def test_variable_renamer_from_translations_and_tuples(translations, translators):
    """Check VariableRenamer from translations and tuples."""
    renamer = VariableRenamer.from_translations(
        ("A", "a", "x"),
        VariableTranslation(
            discipline_name="B", variable_name="b", new_variable_name="y"
        ),
        ("A", "c", "z"),
    )
    assert renamer.translations == translations
    assert renamer.translators == translators


def test_variable_renamer_from_dictionary(translators):
    """Check VariableRenamer from dictionary."""
    renamer = VariableRenamer.from_dictionary({
        "A": {"a": "x", "c": "z"},
        "B": {"b": "y"},
    })
    translations = (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="A", variable_name="c", new_variable_name="z"
        ),
        VariableTranslation(
            discipline_name="B", variable_name="b", new_variable_name="y"
        ),
    )
    assert renamer.translations == translations
    assert renamer.translators == translators


@pytest.mark.parametrize(
    ("sep", "file_name"),
    [({}, "translations.csv"), ({"sep": ";"}, "translations_sep.csv")],
)
def test_variable_renamer_from_csv(sep, file_name, translations, translators):
    """Check VariableRenamer from a CSV file."""
    file_path = Path(__file__).parent / "data" / file_name
    renamer = VariableRenamer.from_csv(file_path, **sep)
    assert renamer.translations == translations
    assert renamer.translators == translators


def test_variable_renamer_from_spread_sheet(translations, translators):
    """Check VariableRenamer from a spreadsheet file."""
    file_path = Path(__file__).parent / "data" / "translations.xlsx"
    renamer = VariableRenamer.from_spreadsheet(file_path)
    assert renamer.translations == translations
    assert renamer.translators == translators


def test_rename_twice_log(caplog):
    """Check the message logged when renaming a variable twice with same name."""
    translations = (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
    )
    VariableRenamer.from_translations(*translations)
    assert caplog.record_tuples[0] == (
        "gemseo.utils.discipline",
        30,
        "In discipline 'A', "
        "the variable 'a' cannot be renamed to 'x' "
        "because it has already been renamed to 'x'.",
    )


def test_rename_twice_error():
    """Check the error message raised when renaming a variable twice with diff name."""
    translations = (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="y"
        ),
    )
    msg = re.escape(
        "In discipline 'A', "
        "the variable 'a' cannot be renamed to 'y' "
        "because it has already been renamed to 'x'."
    )
    with pytest.raises(ValueError, match=msg):
        VariableRenamer.from_translations(*translations)


def test_add_translations_by_variable():
    """Check the method add_translations_by_variable."""
    renamer = VariableRenamer()
    renamer.add_translations_by_variable("x", {"A": "a", "B": "b"})
    renamer.add_translations_by_variable("z", {"C": "c"})
    assert renamer.translations == (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="B", variable_name="b", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="C", variable_name="c", new_variable_name="z"
        ),
    )
    assert renamer.translators == {"A": {"a": "x"}, "B": {"b": "x"}, "C": {"c": "z"}}


def test_add_translations_by_discipline():
    """Check the method add_translations_by_discipline."""
    renamer = VariableRenamer()
    renamer.add_translations_by_discipline("A", {"a": "x", "b": "x"})
    renamer.add_translations_by_discipline("C", {"c": "z"})
    assert renamer.translations == (
        VariableTranslation(
            discipline_name="A", variable_name="a", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="A", variable_name="b", new_variable_name="x"
        ),
        VariableTranslation(
            discipline_name="C", variable_name="c", new_variable_name="z"
        ),
    )
    assert renamer.translators == {"A": {"a": "x", "b": "x"}, "C": {"c": "z"}}


def test_rename_discipline_variables(disciplines, translators, caplog):
    """Check rename_discipline_variables.

    Translators: {"A": {"a": "x", "c": "z"}, "B": {"b": "y"}}

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
        "gemseo.utils.discipline",
        30,
        "The discipline 'C' has no translator.",
    )

    with pytest.raises(
        ValueError, match=re.escape("The discipline 'A' has no variable 'foo'.")
    ):
        rename_discipline_variables(disciplines, {"A": {"foo": "bar"}})


def test_get_discipline_variable_properties():
    """Check get_discipline_variable_properties."""
    discipline = AnalyticDiscipline({"foo": "foo"})
    grammars = [discipline.io.input_grammar, discipline.io.output_grammar]
    description = "The description of foo."
    for index, grammar in enumerate(grammars):
        grammar.descriptions["foo"] = description
        names_to_properties = get_discipline_variable_properties(discipline)[index]
        assert names_to_properties["foo"] == DisciplineVariableProperties(
            current_name="foo",
            original_name="foo",
            current_name_without_namespace="foo",
            description=description,
        )
        grammar.rename_element("foo", "bar")
        discipline.io.data_processor = NameMapping({"bar": "foo"})
        names_to_properties = get_discipline_variable_properties(discipline)[index]
        assert names_to_properties["bar"] == DisciplineVariableProperties(
            current_name="bar",
            original_name="foo",
            current_name_without_namespace="bar",
            description=description,
        )
        grammar.rename_element("bar", "baz")
        discipline.io.data_processor = NameMapping({"baz": "foo"})
        names_to_properties = get_discipline_variable_properties(discipline)[index]
        assert names_to_properties["baz"] == DisciplineVariableProperties(
            current_name="baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description=description,
        )
        grammar.add_namespace("baz", "ns")
        names_to_properties = get_discipline_variable_properties(discipline)[index]
        assert names_to_properties["ns:baz"] == DisciplineVariableProperties(
            current_name="ns:baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description=description,
        )
        del grammar.descriptions["ns:baz"]
        names_to_properties = get_discipline_variable_properties(discipline)[index]
        assert names_to_properties["ns:baz"] == DisciplineVariableProperties(
            current_name="ns:baz",
            original_name="foo",
            current_name_without_namespace="baz",
            description="",
        )
