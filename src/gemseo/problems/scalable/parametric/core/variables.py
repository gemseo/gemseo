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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalable problem - Variables
****************************
"""
from __future__ import annotations

X_LOCAL_NAME_BASIS = "x_local"
U_LOCAL_NAME_BASIS = "u_local"


def get_u_local_name(index):
    """Returns the name of the local uncertain parameter associated with an index.

    :param int index: index.
    """
    return f"{U_LOCAL_NAME_BASIS}_{index}"


def get_x_local_name(index):
    """Returns the name of the local design parameter associated with an index.

    :param int index: index.
    """
    return f"{X_LOCAL_NAME_BASIS}_{index}"


X_SHARED_NAME = "x_shared"


def get_coupling_name(index):
    """Returns the name of the coupling variable associated with an index.

    :param int index: index.
    """
    return f"y_{index}"


def get_constraint_name(index):
    """Returns the name of the constraint associated with an index.

    :param int index: index.
    """
    return f"cstr_{index}"


OBJECTIVE_NAME = "obj"


def check_consistency(n_shared, n_local, n_coupling):
    """Check if n_shared is an integer and if n_local and n_coupling are list of integers
    with the same length."""
    if not isinstance(n_shared, int) or n_shared < 1:
        raise TypeError("n_shared must be an integer > 0.")
    if not isinstance(n_local, list):
        raise TypeError("n_local must be a list of integers > 0.")
    if not isinstance(n_coupling, list):
        raise TypeError("n_coupling must be a list of integers > 0.")
    if not all([isinstance(val, int) for val in n_local]):
        raise TypeError("n_local must be a list of integers > 0.")
    if not all([val > 0 for val in n_local]):
        raise TypeError("n_local must be a list of integers > 0.")
    if not all([isinstance(val, int) for val in n_coupling]):
        raise TypeError("n_coupling must be a list of integers > 0.")
    if not all([val > 0 for val in n_coupling]):
        raise TypeError("n_coupling must be a list of integers > 0.")
    if len(n_local) != len(n_coupling):
        raise ValueError(
            "n_local and n_coupling must have the same length "
            "which is equal to the number of disciplines"
        )
