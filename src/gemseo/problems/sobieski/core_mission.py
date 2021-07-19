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
#    INITIAL AUTHORS - initial API and implementation
#               and/or initial documentation
#        :author: Sobieski, Agte, and Sandusky
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Damien Guenot
#        :author: Francois Gallard
# From NASA/TM-1998-208715
# Bi-Level Integrated System Synthesis (BLISS)
# Sobieski, Agte, and Sandusky
"""
SSBJ Mission computation
************************
"""
from __future__ import division, unicode_literals

import logging
from math import pi

from numpy import zeros

LOGGER = logging.getLogger(__name__)
DEG_TO_RAD = pi / 180.0


class SobieskiMission(object):
    """Class defining mission analysis for Sobieski problem and related method to the
    mission problem such as disciplines computation, constraints, reference optimum."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"

    def __init__(self, sobieski_base):
        """Constructor."""
        self.base = sobieski_base
        self.dtype = self.base.dtype
        self.math = self.base.math

    @staticmethod
    def compute_weight_ratio(y_14):
        """Computation of weight ratio of Breguet formula.

        :param y_14: shared variables coming from blackbox_structure

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :returns: Wt / (Wt -Wf)
        :rtype: numpy array
        """
        return y_14[0] / (y_14[0] - y_14[1])

    @staticmethod
    def compute_dweightratio_dwt(y_14):
        """Computation of derivative of weight ratio wrt total weight.

        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :returns: dweightratio_dtotalweight
        :rtype: numpy array
        """
        return -y_14[1] / ((y_14[0] - y_14[1]) * (y_14[0] - y_14[1]))

    @staticmethod
    def compute_dweightratio_dwf(y_14):
        """Computation of partial derivative of weight ratio wrt fuel weight.

        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :returns: dweightratio_dfuelweight
        :rtype: numpy array
        """
        return y_14[0] / ((y_14[0] - y_14[1]) * (y_14[0] - y_14[1]))

    def compute_dlnweightratio_dwt(self, y_14):
        """Computation of partial derivative of log of weight ratio wrt total weight.

        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :returns: d(ln(weight ratio)/d(total weight)
        :rtype: numpy array
        """
        return self.compute_dweightratio_dwt(y_14) / self.compute_weight_ratio(y_14)

    def compute_dlnweightratio_dwf(self, y_14):
        """Computation of partial derivative of log of weight ratio wrt fuel weight.

        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :returns: d(ln(weight ratio)/d(fuel weight)
        :rtype: numpy array
        """
        return self.compute_dweightratio_dwf(y_14) / self.compute_weight_ratio(y_14)

    def compute_range(self, x_shared, y_14, y_24, y_34):
        """Computation of range from Breguet formula.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :param y_24: shared variables coming from
            blackbox_aerodynamics (lift/drag ratio)
        :type y_24: numpy array
        :param y_34: shared variables coming from
            blackbox_propulsion (SFC)
        :type y_34: numpy array
        :returns: range value
        :rtype: numpy array
        """
        sqrt_theta = self.compute_sqrt_theta(x_shared)
        return ((x_shared[2] * y_24[0]) * 661.0 * sqrt_theta / y_34[0]) * self.math.log(
            y_14[0] / (y_14[0] - y_14[1])
        )

    def compute_drange_dtotalweight(self, x_shared, y_14, y_24, y_34, sqrt_theta):
        """Computation of range derivative wrt total weight.

        :param x_shared: shared design variable vector
        :type x_shared: numpy array
        :param y_14: shared variables coming from blackbox_structure
        :type y_14: numpy array
        :param y_24: shared variables coming from blackbox_aerodynamics
        :type y_24: numpy array
        :param y_34: shared variables coming from blackbox_propulsion
        :type y_34: numpy array
        :param sqrt_theta: square root of air temperature
        :type sqrt_theta: numpy array
        :returns: d(range) / d(total weight)
        :rtype: numpy array
        """
        dlnweightratio_dtotalweight = self.compute_dlnweightratio_dwt(y_14)
        return (
            x_shared[2]
            * y_24[0]
            / y_34[0]
            * 661.0
            * sqrt_theta
            * dlnweightratio_dtotalweight
        )

    def compute_drange_dfuelweight(self, x_shared, y_14, y_24, y_34, sqrt_theta):
        """Computation of range derivative wrt fuel weight.

        :param x_shared: shared design variable vector
        :type x_shared: numpy array
        :param y_14: shared variables coming from blackbox_structure
        :type y_14: numpy array
        :param y_24: shared variables coming from blackbox_aerodynamics
        :type y_24: numpy array
        :param y_34: shared variables coming from blackbox_propulsion
        :type y_34: numpy array
        :param sqrt_theta: square root of air temperature
        :type sqrt_theta: numpy array
        :returns: d(range) / d(fuel weight)
        :rtype: numpy array
        """
        dlnweightratio_dfuelweight = self.compute_dlnweightratio_dwf(y_14)
        return (
            x_shared[2]
            * y_24[0]
            / y_34[0]
            * 661.0
            * sqrt_theta
            * dlnweightratio_dfuelweight
        )

    def compute_dtheta_dh(self, x_shared):
        """Computation of air temperature and its derivative wrt altitude.

        :param x_shared: shared design variable vector
        :type x_shared: numpy array
        :returns: square root of air temperature, dtheta_dh
        :rtype: numpy array
        """
        if x_shared[1] < 36089.0:
            theta = 1 - 0.000006875 * x_shared[1]
            dtheta_dh = -0.000006875
        else:
            theta = 0.7519
            dtheta_dh = 0.0
        return self.math.sqrt(theta), dtheta_dh

    def compute_sqrt_theta(self, x_shared):
        """Computation of air temperature a.

        :param x_shared: shared design variable vector
        :type x_shared: numpy array
        :returns: square root of air temperature
        :rtype: numpy array
        """
        if x_shared[1] < 36089.0:
            theta = 1 - 0.000006875 * x_shared[1]
        else:
            theta = 0.7519
        return self.math.sqrt(theta)

    def blackbox_mission(self, x_shared, y_14, y_24, y_34):
        """THIS SECTION COMPUTES THE A/C RANGE from Breguet's law.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :param y_24: shared variables coming from
            blackbox_aerodynamics (lift/drag ratio)
        :type y_24: numpy array
        :param y_34: shared variables coming from
            blackbox_propulsion (SFC)
        :type y_34: numpy array
        :returns: y_4: range value
        :rtype: numpy array
        """
        y_4 = zeros(1, dtype=self.dtype)
        y_4[0] = self.compute_range(x_shared, y_14, y_24, y_34)
        return y_4

    def __initialize_jacobian(self):
        """
        Initialization of jacobian matrix
        :returns:  jacobian
        :rtype: dict(dict(ndarray))
        """
        jacobian = {"y_4": {}}

        jacobian["y_4"]["x_shared"] = zeros((1, 6), dtype=self.dtype)
        jacobian["y_4"]["y_14"] = zeros((1, 2), dtype=self.dtype)
        jacobian["y_4"]["y_24"] = zeros((1, 1), dtype=self.dtype)
        jacobian["y_4"]["y_34"] = zeros((1, 1), dtype=self.dtype)

        return jacobian

    def derive_blackbox_mission(self, x_shared, y_14, y_24, y_34):
        """THIS SECTION COMPUTES THE A/C RANGE from Breguet's law.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_14: shared variables coming from blackbox_structure:

            - y_14[0]: total aircraft weight
            - y_14[1]: fuel weight

        :type y_14: numpy array
        :param y_24: shared variables coming from
            blackbox_aerodynamics (lift/drag ratio)
        :param y_34: shared variables coming from
            blackbox_propulsion (SFC)
        :type y_34: numpy array
        :returns: jacobian matrix of partial derivatives
        :rtype: dict(dict(ndarray))
        """
        jacobian = self.__initialize_jacobian()
        sqrt_theta, dtheta_dh = self.compute_dtheta_dh(x_shared)
        ac_range = self.compute_range(x_shared, y_14, y_24, y_34)
        # dR_d(t/c)  = 0
        #        jacobian['y_4']['x_shared'][0, 0] = 0
        # dR_d(h)
        jacobian["y_4"]["x_shared"][0, 1] = (
            0.5 * ac_range * dtheta_dh / (sqrt_theta * sqrt_theta)
        )
        # dR_dM
        jacobian["y_4"]["x_shared"][0, 2] = ac_range / x_shared[2]
        # dR_dAR  = 0
        #        jacobian['y_4']['x_shared'][0, 3] = 0
        # dR_dsweep  = 0
        #        jacobian['y_4']['x_shared'][0, 4] = 0
        # dR_dsref  = 0
        #        jacobian['y_4']['x_shared'][0, 5] = 0

        # dR_dWt
        dy4dy14 = self.compute_drange_dtotalweight(
            x_shared, y_14, y_24, y_34, sqrt_theta
        )
        jacobian["y_4"]["y_14"][0, 0] = dy4dy14
        # dR_dWf
        dy4dy14 = self.compute_drange_dfuelweight(
            x_shared, y_14, y_24, y_34, sqrt_theta
        )
        jacobian["y_4"]["y_14"][0, 1] = dy4dy14

        # dR_d(L/D)
        jacobian["y_4"]["y_24"][0, 0] = ac_range / y_24[0]

        # dR_d(SFC)
        jacobian["y_4"]["y_34"][0, 0] = -ac_range / y_34[0]
        return jacobian
