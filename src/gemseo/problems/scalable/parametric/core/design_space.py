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
Scalable problem - Design space
*******************************
"""
from __future__ import annotations

import collections
import logging

from numpy import ones
from numpy import zeros

from .variables import check_consistency
from .variables import get_coupling_name
from .variables import get_x_local_name
from .variables import X_SHARED_NAME

LOGGER = logging.getLogger(__name__)


class TMDesignSpace:
    """The design space for the scalable problem introduced by Tedford and Martins (2010)
    defines the lower and upper bounds of both local design parameters, shared design
    parameters and coupling variables, as well as default values.

    The lower bounds are all equal to 0, the upper bounds are all equal to 1 and default
    values are all equal to 0.5.
    """

    def __init__(
        self,
        n_shared=1,
        n_local=None,
        n_coupling=None,
        default_inputs=None,
        dtype="float64",
    ):
        """The construction of the design space requires the number of shared design
        parameters, the number of local design parameters per discipline and the number
        of coupling variables per discipline. The two latter arguments must be list of
        integers with the same length which corresponds to the number of strongly coupled
        disciplines. By default, the design space considers two disciplines.

        :param int n_shared: size of the shared design parameters.
            Default: 1.
        :param list(int) n_local: sizes of the local design parameters.
            If None, use [1, 1]. Default: None.
        :param list(int) n_coupling: sizes of the coupling parameters.
            If None, use [1, 1]. Default: None.
        :param dict default_inputs: default inputs.
            Default: None.
        :param dtype: numpy data type. Default: 'float64'.
        """
        # Set and check the dimensions of the problem
        n_local = n_local or [1, 1]
        n_coupling = n_coupling or [1, 1]
        check_consistency(n_shared, n_local, n_coupling)

        default_inputs = (
            {}
            if not isinstance(default_inputs, collections.abc.Mapping)
            else default_inputs
        )
        self.names = []
        self.sizes = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.default_values = {}

        # Set the local design parameters
        for index, size in enumerate(n_local):
            name = get_x_local_name(index)
            value = default_inputs.get(name, zeros(size) + 0.5)
            value.astype(dtype)
            self.names.append(name)
            self.sizes[name] = size
            self.lower_bounds[name] = zeros(size)
            self.upper_bounds[name] = ones(size)
            self.default_values[name] = value

        # Set the shared design parameters
        name = X_SHARED_NAME
        value = default_inputs.get(X_SHARED_NAME, zeros(n_shared) + 0.5)
        value.astype(dtype)
        self.names.append(name)
        self.sizes[name] = n_shared
        self.lower_bounds[name] = zeros(n_shared)
        self.upper_bounds[name] = ones(n_shared)
        self.default_values[name] = value

        # Set the coupling variables
        for index, size in enumerate(n_coupling):
            name = get_coupling_name(index)
            value = default_inputs.get(name, zeros(size) + 0.5)
            value.astype(dtype)
            self.names.append(name)
            self.sizes[name] = size
            self.lower_bounds[name] = zeros(size)
            self.upper_bounds[name] = ones(size)
            self.default_values[name] = value
