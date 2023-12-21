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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Rosenbrock dataset.

This :class:`.Dataset` contains 100 evaluations
of the well-known Rosenbrock function:

.. math::

   f(x,y)=(1-x)^2+100(y-x^2)^2

This function is known for its global minimum at point (1,1),
its banana valley and the difficulty to reach its minimum.

This :class:`.Dataset` is based on a full-factorial
design of experiments.

`More information about the Rosenbrock function
<https://en.wikipedia.org/wiki/Rosenbrock_function>`_
"""

from __future__ import annotations

from numpy import hstack
from numpy import linspace
from numpy import meshgrid

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.datasets.optimization_dataset import OptimizationDataset


def create_rosenbrock_dataset(
    n_samples: int = 100, opt_naming: bool = True, categorize: bool = True
) -> Dataset:
    """Rosenbrock dataset parametrization.

    Args:
        n_samples: The number of samples.
        opt_naming: Whether to use an optimization naming.
        categorize: Whether to distinguish
            between the different groups of variables.

    Returns:
        The Rosenbrock dataset.
    """
    # Create function.
    root_n_samples = int(n_samples**0.5)
    x_i = linspace(-2.0, 2.0, root_n_samples)
    x_i, y_i = meshgrid(x_i, x_i)
    x_i = x_i.reshape((-1, 1))
    y_i = y_i.reshape((-1, 1))
    z_i = 100 * (y_i - x_i**2) ** 2 + (1 - x_i) ** 2
    data = hstack((x_i, y_i, z_i))

    # Create groups.
    if categorize:
        if opt_naming:
            groups = {
                "x": OptimizationDataset.DESIGN_GROUP,
                "rosen": OptimizationDataset.OBJECTIVE_GROUP,
            }
            cls = OptimizationDataset
        else:
            groups = {"x": IODataset.INPUT_GROUP, "rosen": IODataset.OUTPUT_GROUP}
            cls = IODataset
    else:
        groups = None
        cls = Dataset()

    dataset = cls.from_array(data, ["x", "rosen"], {"x": 2, "rosen": 1}, groups)
    dataset.name = "Rosenbrock"
    dataset.misc["root_n_samples"] = root_n_samples

    return dataset
