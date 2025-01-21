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

from gemseo.utils.variable_renaming import VariableRenamer
from gemseo.utils.variable_renaming import VariableTranslation


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


def test_variable_renamer_from_tuples(translations, translators):
    """Check VariableRenamer from tuples."""
    renamer = VariableRenamer.from_tuples(
        ("A", "a", "x"), ("B", "b", "y"), ("A", "c", "z")
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
        "gemseo.utils.variable_renaming",
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
