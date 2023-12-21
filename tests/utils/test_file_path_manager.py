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

from pathlib import Path

import pytest

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.repr_html import REPR_HTML_WRAPPER


def test_repr():
    """Verify the string representation of a FileManager."""
    manager = FilePathManager(FilePathManager.FileType.FIGURE)
    expected = [
        "FilePathManager",
        "   File type: FIGURE",
        "   Default file name: figure",
        "   Default file extension: png",
        f"   Default directory: {Path.cwd()}",
    ]
    assert repr(manager) == str(manager) == "\n".join(expected)


def test_repr_html():
    """Check FileManager._repr_html_."""
    assert FilePathManager(
        FilePathManager.FileType.FIGURE, default_directory="."
    )._repr_html_() == REPR_HTML_WRAPPER.format(
        "FilePathManager<br/>"
        "<ul>"
        "<li>File type: FIGURE</li>"
        "<li>Default file name: figure</li>"
        "<li>Default file extension: png</li>"
        "<li>Default directory: .</li>"
        "</ul>"
    )


@pytest.fixture(scope="module")
def file_path_manager() -> FilePathManager:
    """A manager of figure file paths."""
    return FilePathManager(FilePathManager.FileType.FIGURE, default_directory=Path())


@pytest.mark.parametrize(
    ("file_path", "directory_path", "file_name", "file_extension", "expected"),
    [
        (None, None, None, None, Path("figure.png")),
        (None, Path("directory"), None, None, Path("directory") / "figure.png"),
        (None, Path("directory"), "fname", None, Path("directory") / "fname.png"),
        (None, Path("directory"), None, "pdf", Path("directory") / "figure.pdf"),
        (None, Path("directory"), "fname", "pdf", Path("directory") / "fname.pdf"),
        (None, None, "fname", None, Path("fname.png")),
        (None, None, None, "pdf", Path("figure.pdf")),
        (None, None, "fname", "pdf", Path("fname.pdf")),
    ],
)
def test_create_file_path(
    file_path_manager,
    file_path,
    directory_path,
    file_name,
    file_extension,
    expected,
):
    """Verify the creation of file paths."""
    assert expected == file_path_manager.create_file_path(
        file_path=file_path,
        directory_path=directory_path,
        file_name=file_name,
        file_extension=file_extension,
    )


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("foo", "foo"),
        ("Foo", "foo"),
        ("FooBar", "foo_bar"),
        ("Foo Bar", "foo_bar"),
        ("Foo-Bar", "foo_bar"),
    ],
)
def test_to_snake_case(original, expected):
    assert FilePathManager.to_snake_case(original) == expected


def test_add_suffix():
    file_path = Path("directory") / "filename.pdf"
    expected_file_path = Path("directory") / "filename_suffix.pdf"
    assert FilePathManager.add_suffix(file_path, "suffix") == expected_file_path


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "png"),
        ({"default_extension": "png"}, "png"),
        ({"default_extension": "jpg"}, "jpg"),
    ],
)
def test_default_extension(kwargs, expected):
    """Check the default extension."""
    manager = FilePathManager(FilePathManager.FileType.FIGURE, **kwargs)
    assert manager._FilePathManager__default_extension == expected
