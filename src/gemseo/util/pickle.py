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
from typing import TYPE_CHECKING
from typing import Any

from gemseo.util._compatibility.numpy import NUMPY_GREATER_THAN_2

if TYPE_CHECKING:
    from gemseo.util.typing import StrPath


class _NumpyCompatUnpickler(Unpickler):
    """An [Unpickler][pickle.Unpickler] mapping `numpy.core` to `numpy._core`.

    NumPy renamed its private `numpy.core` package to `numpy._core` in NumPy 2.
    Loading a pickle that references the former emits a deprecation warning and will
    break once the backward-compatibility shim is removed from NumPy.
    The mapping is only done with NumPy 2 and greater
    since `numpy._core` does not exist before.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Return the class `name` from `module`, remapping `numpy.core`.

        Args:
            module: The name of the module.
            name: The name of the class.

        Returns:
            The class.
        """
        if NUMPY_GREATER_THAN_2 and (
            module == "numpy.core" or module.startswith("numpy.core.")
        ):
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def to_pickle(
    obj: Any,
    file_path: StrPath,
    protocol: int = HIGHEST_PROTOCOL,
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


def from_pickle(file_path: StrPath) -> Any:
    """Load an object from its pickled representation stored on the disk.

    Args:
        file_path: The path to the file containing the pickled representation.

    Returns:
        The object.
    """
    with Path(file_path).open("rb") as f:
        return _NumpyCompatUnpickler(f).load()
