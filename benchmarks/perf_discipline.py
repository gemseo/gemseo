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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""TODO."""
from __future__ import annotations

from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.core.discipline import MDODiscipline


def run():
    """Runner."""
    for _ in range(10000):
        MDODiscipline()


print("Toto")  # noqa: T201

if __name__ == "__main__":
    from time import time

    t0 = time()
    MDODiscipline()
    MDODiscipline()
    MDODiscipline()
    HDF5Cache()
    HDF5Cache()
    HDF5Cache()
    HDF5Cache()
    HDF5Cache()
    tf = time()
    # noqa: N802
    print("Total time =" + str(tf - t0))  # noqa: T201

    t0 = time()
    run()
    tf = time()
    # noqa: N802
    print("Total time =" + str(tf - t0))  # noqa: T201
