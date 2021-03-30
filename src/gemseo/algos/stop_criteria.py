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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Various termination criteria for drivers
****************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

standard_library.install_aliases()


class TerminationCriterion(Exception):
    """Stop driver for some reason"""


class FunctionIsNan(TerminationCriterion):
    """Stops driver when a function has NaN value or NaN Jacobian"""


class DesvarIsNan(TerminationCriterion):
    """Stops driver when the design variables are nan"""


class MaxIterReachedException(TerminationCriterion):
    """
    Exception raised when the maximum number of iterations is reached
    by the driver
    """
