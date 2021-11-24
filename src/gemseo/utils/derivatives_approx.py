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

from gemseo.utils.derivatives.complex_step import ComplexStep  # noqa: F401
from gemseo.utils.derivatives.derivatives_approx import EPSILON  # noqa: F401
from gemseo.utils.derivatives.derivatives_approx import approx_hess  # noqa: F401
from gemseo.utils.derivatives.derivatives_approx import comp_best_step  # noqa: F401
from gemseo.utils.derivatives.derivatives_approx import (  # noqa: F401
    DisciplineJacApprox,
    compute_cancellation_error,
    compute_truncature_error,
)
from gemseo.utils.derivatives.finite_differences import FirstOrderFD  # noqa: F401

# TODO: deprecate this module at some point.
