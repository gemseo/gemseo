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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalable design space from Tedford and Martins (2010)
*****************************************************

"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.scalable.scalable_tm.variables import (
    X_SHARED_NAME,
    check_consistency,
    get_coupling_name,
    get_x_local_name,
)

standard_library.install_aliases()

from gemseo import LOGGER


class TMDesignSpace(DesignSpace):

    """The design space for the scalable problem introduced by Tedford and
    Martins (2010) defines the lower and upper bounds of both local design
    parameters, shared design parameters and coupling variables, as well as
    default values. The lower bounds are all equal to 0, the upper bounds are
    all equal to 1 and default values are all equal to 0.5.
    """

    def __init__(
        self,
        n_shared=1,
        n_local=None,
        n_coupling=None,
        default_inputs=None,
        dtype="float64",
    ):
        """The TMDesignSpace constructor requires the number of shared design
        parameters, the number of local design parameters per discipline
        and the number of coupling variables per discipline. The two latter
        arguments must be list of integers with the same length which
        corresponds to the number of strongly coupled disciplines. By default,
        the design space considers two disciplines.

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
        super(TMDesignSpace, self).__init__()
        n_local = n_local or [1, 1]
        n_coupling = n_coupling or [1, 1]
        check_consistency(n_shared, n_local, n_coupling)

        if not isinstance(default_inputs, dict):
            default_inputs = {}
        for id_x, n_x in enumerate(n_local):
            value = default_inputs.get(get_x_local_name(id_x), zeros(n_x) + 0.5)
            value.astype(dtype)
            self.add_variable(
                get_x_local_name(id_x), n_x, l_b=0.0, u_b=1.0, value=value
            )
        value = default_inputs.get(X_SHARED_NAME, zeros(n_shared) + 0.5)
        value.astype(dtype)
        self.add_variable(X_SHARED_NAME, n_shared, l_b=0.0, u_b=1.0, value=value)
        for id_y, n_y in enumerate(n_coupling):
            value = default_inputs.get(get_coupling_name(id_y), zeros(n_y) + 0.5)
            value.astype(dtype)
            self.add_variable(
                get_coupling_name(id_y), n_y, l_b=0.0, u_b=1.0, value=value
            )
