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
"""Services for handling file paths."""
from __future__ import annotations

import re
from collections import namedtuple
from enum import Enum
from pathlib import Path
from re import findall

from gemseo.utils.string_tools import MultiLineString

FileDefinition = namedtuple("FileDefinition", ["name", "extension"])


class FileType(Enum):
    """The type of a file, defined by its default name and format."""

    FIGURE = FileDefinition("figure", "png")
    SCHEMA = FileDefinition("schema", "json")
    TEXT = FileDefinition("document", "txt")
    WEBPAGE = FileDefinition("page", "html")


class FilePathManager:
    """A factory of file paths for a given type of file and with default settings."""

    def __init__(
        self,
        file_type: FileType,
        default_name: str = None,
        default_directory: Path | None = None,
        default_extension: str | None = None,
    ) -> None:
        """
        Args:
            file_type: The type of file,
                defined by its default file name and format;
                select a file type by iterating over ``FileType``.
            default_name: The default file name.
                If None, use the default file name related to the given type of file.
            default_directory: The default directory path.
                If None, use the current working directory.
            default_extension: The default extension.
                If None, use the default extension related to the given type of file.
        """
        if default_name is None:
            self.__default_name = file_type.value.name
        else:
            self.__default_name = default_name

        self.__default_directory = Path(default_directory or Path.cwd())

        if default_extension is None:
            self.__default_extension = file_type.value.extension
        else:
            self.__default_extension = default_extension

        self.__file_type = file_type.name

    def __str__(self) -> str:
        string = MultiLineString()
        string.add(self.__class__.__name__)
        string.indent()
        string.add("File type: {}", self.__file_type)
        string.add("Default file name: {}", self.__default_name)
        string.add("Default file extension: {}", self.__default_extension)
        string.add("Default directory: {}", self.__default_directory)
        return str(string)

    def create_file_path(
        self,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
    ) -> Path:
        """Make a file path from a directory path, a file name and a file extension.

        Args:
            file_path: The path of the file to be returned.
                If None,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory.
                If None, use the default directory path.
            file_name: The file name to be used.
                If None, use the default file name.
            file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If None, use the default file extension.

        Returns:
            The file path.
        """
        suffix = f".{file_extension or self.__default_extension}"

        if file_path is not None:
            file_path = Path(file_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(suffix)
            return file_path

        file_name = file_name or self.__default_name

        if directory_path is None:
            directory_path = self.__default_directory
        else:
            directory_path = Path(directory_path)

        return (directory_path / file_name).with_suffix(suffix)

    @staticmethod
    def to_snake_case(
        message: str,
    ) -> str:
        """Snake case a string.

        That means:

        1. Split the message.
        2. Lowercase the resulting elements.

        ``-`` and `` `` are replaced with ``_``.

        Args:
            message: The message to be snake-cased.

        Returns:
            The snake-cased message.
        """
        message = message.replace("-", "_").replace(" ", "_")
        message = "_".join(
            [elem.lower() for elem in findall("[A-Z][^A-Z]*", message)] or [message]
        )
        return re.sub("_+", "_", message)

    @classmethod
    def add_suffix(
        cls,
        file_path: Path,
        suffix: str,
    ) -> Path:
        """Add a suffix to an existing file path between the filename and the extension.

        E.g. `directory/filename_suffix.pdf`.

        Args:
            file_path: The file path to be suffixed.
            suffix: The suffix to be added to the file path.

        Returns:
            The directory path, the file name and the file extension
            obtained from the file path.
        """
        extension = file_path.suffix
        file_name = f"{str(file_path.stem)}_{suffix}"
        directory_path = file_path.parent
        return (directory_path / file_name).with_suffix(extension)
