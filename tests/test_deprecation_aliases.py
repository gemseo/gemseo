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
"""Unit tests for the alias-table builder behind the deprecated-import machinery."""

from __future__ import annotations

from pathlib import Path

import gemseo
from gemseo._deprecation import aliases


def test_parse_section_skips_comments_and_blank_lines():
    """Comment and blank lines inside a section are ignored."""
    text = "modules:\n  # a comment\n\n  old.a: new_a\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_skips_unindented_comments():
    """An unindented comment does not truncate a section."""
    text = "modules:\n  old.a: new_a\n# an unindented comment\n  old.b: new_b\n"
    assert aliases._parse_section(text, "modules") == {
        "old.a": "new_a",
        "old.b": "new_b",
    }


def test_parse_section_stops_before_next_section():
    """Parsing stops at the first unindented line following the section name."""
    text = "modules:\n  old.a: new_a\nclasses:\n  Foo: bar\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_reaches_end_of_text():
    """Parsing consumes the whole text when the section is the last one."""
    text = "modules:\n  old.a: new_a\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_skips_other_sections():
    """Only the requested section is parsed."""
    text = "modules:\n  old.a: new_a\nattributes:\n  old.b.Old: New\n"
    assert aliases._parse_section(text, "attributes") == {"old.b.Old": "New"}


def test_parse_section_list_items():
    """A list section yields its items with an empty value."""
    text = "dissolved:\n  - old.a\n  - old.b\n"
    assert aliases._parse_section(text, "dissolved") == {"old.a": "", "old.b": ""}


def test_parent():
    """The parent of a dotted name drops its last segment."""
    assert aliases._parent("a.b.c") == "a.b"


def test_last():
    """The last segment of a dotted name is returned."""
    assert aliases._last("a.b.c") == "c"


def test_raw_new_module_absolute_path():
    """An absolute `gemseo.*` value replaces the whole old module name."""
    assert aliases._raw_new_module("gemseo.a.b", "gemseo.x.y") == "gemseo.x.y"


def test_raw_new_module_last_segment_only():
    """A bare value replaces only the last segment, keeping the parent."""
    assert aliases._raw_new_module("gemseo.a.b", "c") == "gemseo.a.c"


def test_apply_longest_prefix_picks_longest_match():
    """The mapping entry whose key is the longest prefix wins."""
    mapping = {"a": "z", "a.b": "y"}
    assert aliases._apply_longest_prefix("a.b.c", mapping) == "y.c"


def test_apply_longest_prefix_no_match():
    """The name is returned unchanged when no key is a prefix of it."""
    mapping = {"a": "z"}
    assert aliases._apply_longest_prefix("other", mapping) == "other"


def test_resolve_applies_ancestor_renames_repeatedly():
    """Resolution keeps rewriting until an ancestor rename also applies."""
    mapping = {"a": "b", "b.c": "b.d"}
    assert aliases._resolve("a.c", mapping) == "b.d"


def test_resolve_cycle_guard_terminates():
    """A rename cycle terminates via the `seen` guard instead of looping forever."""
    mapping = {"a": "b", "b": "a"}
    assert aliases._resolve("a", mapping) == "a"


def test_build_from_synthetic_config(tmp_path, monkeypatch):
    """`_build` resolves the module renames and groups the attribute ones."""
    config = tmp_path / "bump-version.yml"
    config.write_text(
        "modules:\n"
        "  gemseo.old_pkg.old_mod: new_mod\n"
        "  gemseo.old_pkg: new_pkg\n"
        "  gemseo.same.name: name\n"
        "attributes:\n"
        "  gemseo.old_pkg.Old: New\n"
        "  gemseo.old_pkg.old_mod.old_func: new_func\n"
        "dissolved:\n"
        "  - gemseo.old_pkg\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_CONFIG_PATH", config)

    module_renames, attribute_renames, dissolved = aliases._build()

    assert module_renames["gemseo.old_pkg"] == "gemseo.new_pkg"
    assert module_renames["gemseo.old_pkg.old_mod"] == "gemseo.new_pkg.new_mod"
    # A no-op entry is dropped.
    assert "gemseo.same.name" not in module_renames
    assert attribute_renames == {
        "gemseo.old_pkg": {"Old": "New"},
        "gemseo.old_pkg.old_mod": {"old_func": "new_func"},
    }
    assert dissolved == ("gemseo.old_pkg",)


def test_module_renames_target_existing_modules():
    """Every module rename points at a module of the package.

    A rename whose target does not exist would shadow the normal import error on the old
    name with a confusing one on the new name. The check is done on the file tree rather
    than by importing, so that the optional dependencies are not needed.
    """
    root = Path(gemseo.__file__).parent
    missing = [
        f"{old} -> {new}"
        for old, new in aliases.MODULE_RENAMES.items()
        if not (root / Path(*new.split(".")[1:])).is_dir()
        and not (root / Path(*new.split(".")[1:])).with_suffix(".py").is_file()
    ]
    assert not missing


def test_attribute_renames_do_not_shadow_modules():
    """No attribute rename is also tabulated as a module rename.

    The `modules:` and `attributes:` sections must be disjoint: an entry in both would
    make an old name resolve either as a submodule or as an attribute depending on how
    it is accessed.
    """
    overlap = [
        f"{module}.{name}"
        for module, renames in aliases.ATTRIBUTE_RENAMES.items()
        for name in renames
        if f"{module}.{name}" in aliases.MODULE_RENAMES
    ]
    assert not overlap
