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
"""Handle saving and loading pickles of objects."""

from __future__ import annotations

from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from pickle import Pickler
from pickle import Unpickler
from typing import Any


def to_pickle(
    obj: Any, file_path: str | Path, protocol: int = HIGHEST_PROTOCOL
) -> None:
    """Save the pickled representation of an object on the disk.

    Args:
        obj: An object.
        file_path: The path to the file to store the pickled representation.
        protocol: The protocol to use for pickling.
    """
    with Path(file_path).open("wb") as f:
        pickler = Pickler(f, protocol=protocol)
        pickler.dump(obj)


def from_pickle(file_path: str | Path) -> Any:
    """Load an object from its pickled representation stored on the disk.

    Args:
        file_path: The path to the file containing the pickled representation.

    Returns:
        The object.
    """
    with Path(file_path).open("rb") as f:
        return Unpickler(f).load()
