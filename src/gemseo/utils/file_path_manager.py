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
from pathlib import Path
from re import findall
from typing import ClassVar

from strenum import LowercaseStrEnum

from gemseo.utils.string_tools import MultiLineString

FileDefinition = namedtuple("FileDefinition", ["name", "extension"])


class FilePathManager:
    """A manager of file paths for a given type of file and with default settings."""

    class FileType(LowercaseStrEnum):
        """The type of file, defined by its default name and format."""

        FIGURE = "figure"
        SCHEMA = "schema"
        TEXT = "document"
        WEBPAGE = "page"

    __FILE_TYPE_TO_EXTENSION: ClassVar[dict[FileType, str]] = {
        FileType.FIGURE: "png",
        FileType.SCHEMA: "json",
        FileType.TEXT: "txt",
        FileType.WEBPAGE: "html",
    }

    def __init__(
        self,
        file_type: FileType,
        default_name: str = "",
        default_directory: Path | str = "",
        default_extension: str = "",
    ) -> None:
        """
        Args:
            file_type: The type of file,
                defined by its default file name and format;
                select a file type by iterating over ``FileType``.
            default_name: The default file name.
                If empty, use the default file name related to the given type of file.
            default_directory: The default directory path.
                If empty, use the current working directory.
            default_extension: The default extension.
                If empty, use the default extension related to the given type of file.
        """  # noqa:D205 D212 D415
        if default_name:
            self.__default_name = default_name
        else:
            self.__default_name = file_type.value

        if default_directory:
            self.__default_directory = Path(default_directory)
        else:
            self.__default_directory = Path.cwd()

        if default_extension:
            self.__default_extension = default_extension
        else:
            self.__default_extension = self.__FILE_TYPE_TO_EXTENSION[file_type]

        self.__file_type = file_type

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the object."""
        mls = MultiLineString()
        mls.add(self.__class__.__name__)
        mls.indent()
        mls.add("File type: {}", self.__file_type.name)
        mls.add("Default file name: {}", self.__default_name)
        mls.add("Default file extension: {}", self.__default_extension)
        mls.add("Default directory: {}", self.__default_directory)
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    def create_file_path(
        self,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_extension: str = "",
    ) -> Path:
        """Make a file path from a directory path, a file name and a file extension.

        Args:
            file_path: The path of the file to be returned.
                If empty, create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory.
                If empty, use the default directory path.
            file_name: The file name to be used.
                If empty, use the default file name.
            file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use the default file extension.

        Returns:
            The file path.
        """
        suffix = f".{file_extension or self.__default_extension}"

        if file_path:
            file_path = Path(file_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(suffix)
            return file_path

        file_name = file_name or self.__default_name

        if directory_path:
            directory_path = Path(directory_path)
        else:
            directory_path = self.__default_directory

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
        file_name = f"{file_path.stem!s}_{suffix}"
        directory_path = file_path.parent
        return (directory_path / file_name).with_suffix(extension)
