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
"""Compatibility between different versions of SciPy."""

from __future__ import annotations

from importlib.metadata import version
from typing import Final
from typing import Union

from numpy import ndarray
from packaging.version import Version
from packaging.version import parse as parse_version
from scipy.sparse import coo_matrix

SCIPY_VERSION: Final[Version] = parse_version(version("scipy"))
SCIPY_LOWER_THAN_1_11: Final[bool] = parse_version("1.11") > SCIPY_VERSION
SCIPY_LOWER_THAN_1_12: Final[bool] = parse_version("1.12") > SCIPY_VERSION
TOL_OPTION: Final[str] = "tol" if SCIPY_LOWER_THAN_1_12 else "rtol"

if SCIPY_LOWER_THAN_1_11:
    from scipy.sparse import spmatrix

    sparse_classes = (spmatrix,)
    SparseArrayType = spmatrix

    def get_row(matrix, i):  # noqa: D103
        return matrix.getrow(i)

else:
    from scipy.sparse import sparray
    from scipy.sparse import spmatrix

    sparse_classes = (spmatrix, sparray)
    SparseArrayType = Union[coo_matrix, spmatrix, sparray]

    def get_row(matrix, i):  # noqa: D103
        if hasattr(matrix, "getrow"):
            return matrix.getrow(i)

        return matrix[[i], :]


array_classes = (ndarray, *sparse_classes)
