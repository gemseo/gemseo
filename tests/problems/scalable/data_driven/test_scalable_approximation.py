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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois Gallard, Charlie Vanaret, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from os.path import dirname
from os.path import join

import numpy as np
import pytest
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.data_driven.diagonal import ScalableDiagonalApproximation
from gemseo.problems.scalable.data_driven.discipline import ScalableDiscipline
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics

HDF_CACHE_PATH = join(dirname(__file__), "dataset.hdf5")


@pytest.fixture()
def sobieski_aerodynamics():
    """Create a SobieskiAerodynamics problem.

    Returns:
         SobieskiAerodynamics: An instance of the SobieskiAerodynamics class.
    """
    return SobieskiAerodynamics()


def test_build_model(sobieski_aerodynamics):
    """Test the build a 1D interpolation of Sobieski's drag wrt z."""
    sizes = {}
    for k, value in sobieski_aerodynamics.default_inputs.items():
        sizes[k] = len(value)

    hdf_cache = HDF5Cache(HDF_CACHE_PATH, sobieski_aerodynamics.name)
    dataset = hdf_cache.export_to_dataset()

    scd = ScalableDiscipline("ScalableDiagonalModel", dataset, sizes, fill_factor=0.7)
    comp_dep, in_dep = scd.scalable_model.generate_random_dependency()
    scale_pb = ScalableDiagonalApproximation(sizes, comp_dep, in_dep)
    scale_pb.build_scalable_function("y_23", dataset, ["x_shared"])

    def get_samples(n_samples):
        """Generate a vector of uniformly scattered samples.

        :param n_samples: number of samples
        """
        return np.arange(n_samples) / (n_samples - 1.0)

    # MDL: I think that the following lines are wrong because because
    # scalable_func must take in inputs a 1D numpy array whose length
    # is equal to the input dimension...
    #
    # for n_in in range(10, 1010, 100):
    #    fout = scalable_func(get_samples(n_in))
    #    assert -1e-3 <= fout <= 1.001

    scd = ScalableDiscipline("ScalableDiagonalModel", dataset, sizes, fill_factor=0.0)
    comp_dep, in_dep = scd.scalable_model.generate_random_dependency()
    scale_pb = ScalableDiagonalApproximation(sizes, comp_dep, in_dep)
    scale_pb.build_scalable_function("y_23", dataset, ["x_shared"])

    scd = ScalableDiscipline(
        "ScalableDiagonalModel",
        dataset,
        sizes,
        fill_factor=0.0,
        allow_unused_inputs=False,
        force_input_dependency=True,
    )
    scd.scalable_model.generate_random_dependency()

    scd = ScalableDiscipline("ScalableDiagonalModel", dataset, sizes, fill_factor=0.7)
    comp_dep, in_dep = scd.scalable_model.generate_random_dependency()
    ScalableDiagonalApproximation(sizes, comp_dep, in_dep)


def test_serialize(tmp_wd, sobieski_aerodynamics):
    """Test the serialization of a SobieskiAerodynamics instance."""
    s_file = "aero.o"
    sobieski_aerodynamics.serialize(s_file)
    aero2 = MDODiscipline.deserialize(s_file)
    aero2.execute()
