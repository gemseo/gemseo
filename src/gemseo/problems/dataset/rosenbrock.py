# -*- coding: utf-8 -*-
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
"""
Rosenbrock dataset
==================

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
from __future__ import division, unicode_literals

from numpy import hstack, linspace, meshgrid

from gemseo.core.dataset import Dataset


class RosenbrockDataset(Dataset):
    """Rosenbrock dataset parametrization."""

    def __init__(
        self,
        name="Rosenbrock",
        by_group=True,
        n_samples=100,
        categorize=True,
        opt_naming=True,
    ):
        """Constructor.

        :param str name: name of the dataset.
        :param bool by_group: if True, store the data by group. Otherwise,
            store them by variables. Default: True
        :param int n_samples: number of samples
        :param bool categorize: distinguish between the different groups of
            variables. Default: True.
        :parma bool opt_naming: use an optimization naming. Default: True.
        """
        super(RosenbrockDataset, self).__init__(name, by_group)
        root_n_samples = int(n_samples ** 0.5)
        x_i = linspace(-2.0, 2.0, root_n_samples)
        x_i, y_i = meshgrid(x_i, x_i)
        x_i = x_i.reshape((-1, 1))
        y_i = y_i.reshape((-1, 1))
        z_i = 100 * (y_i - x_i ** 2) ** 2 + (1 - x_i) ** 2
        data = hstack((x_i, y_i, z_i))
        if categorize:
            if opt_naming:
                groups = {"x": Dataset.DESIGN_GROUP, "rosen": Dataset.FUNCTION_GROUP}
            else:
                groups = {"x": Dataset.INPUT_GROUP, "rosen": Dataset.OUTPUT_GROUP}
        else:
            groups = None
        self.set_from_array(data, ["x", "rosen"], {"x": 2, "rosen": 1}, groups=groups)
        self.set_metadata("root_n_samples", root_n_samples)
