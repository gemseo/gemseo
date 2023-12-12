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

from typing import Union

import scipy
from numpy import ndarray
from packaging import version

if version.parse(scipy.__version__) < version.parse("1.11"):
    from scipy.sparse import spmatrix

    sparse_classes = (spmatrix,)
    SparseArrayType = spmatrix

else:
    from scipy.sparse import sparray
    from scipy.sparse import spmatrix

    sparse_classes = (spmatrix, sparray)
    SparseArrayType = Union[spmatrix, sparray]

array_classes = (ndarray, *sparse_classes)
ArrayType = Union[ndarray, SparseArrayType]
