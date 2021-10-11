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
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Legacy module with the MDOFunction classes."""
from gemseo.core.mdofunctions.function_generator import (  # noqa: F401
    MDOFunctionGenerator,
)
from gemseo.core.mdofunctions.mdo_function import ApplyOperator  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import Concatenate  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import ConvexLinearApprox  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import FunctionRestriction  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import MDOFunction  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import MDOLinearFunction  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import MDOQuadraticFunction  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import MultiplyOperator  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import NotImplementedCallable  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import Offset  # noqa: F401
from gemseo.core.mdofunctions.mdo_function import SetPtFromDatabase  # noqa: F401

# TODO: to be deprecated.
