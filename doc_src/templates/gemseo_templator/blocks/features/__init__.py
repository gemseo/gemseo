# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
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
from __future__ import annotations

from .doe import block as doe
from .formulation import block as formulation
from .linear_solver import block as linear_solver
from .mda import block as mda
from .mlearning import block as mlearning
from .opt import block as opt
from .post import block as post
from .scalable import block as scalable
from .study_analysis import block as study_analysis
from .surrogate import block as surrogate
from .uncertainty import block as uncertainty

blocks = [
    study_analysis,
    opt,
    doe,
    formulation,
    mda,
    linear_solver,
    post,
    surrogate,
    scalable,
    mlearning,
    uncertainty,
]
