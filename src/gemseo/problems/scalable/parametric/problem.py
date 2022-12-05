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
Scalable problem
================
"""
from __future__ import annotations

import logging

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.problems.scalable.parametric.core.problem import TMProblem
from gemseo.problems.scalable.parametric.core.variables import get_u_local_name
from gemseo.problems.scalable.parametric.disciplines import TMMainDiscipline
from gemseo.problems.scalable.parametric.disciplines import TMSubDiscipline

LOGGER = logging.getLogger(__name__)

MDA_TOLERANCE = {"tolerance": 1e-14, "linear_solver_tolerance": 1e-14}
ALGO_OPTIONS = {
    "xtol_rel": 1e-4,
    "ftol_rel": 1e-4,
    "xtol_abs": 1e-4,
    "ftol_abs": 1e-4,
    "ineq_tolerance": 1e-3,
    "eq_tolerance": 1e-3,
}


OBJECTIVE_NAME = "obj"

COUPLING_DIR = "coupling"
COEFF_DIR = "coefficients"
OPTIM_DIR = "opthistoryview"


class TMScalableProblem(TMProblem):

    """The scalable problem from Tedford and Martins, 2010, builds a list of strongly
    coupled scalable disciplines completed by a system discipline computing the objective
    function and the constraints.

    These disciplines are defined on a unit design space (parameters comprised in [0,
    1]).
    """

    @classmethod
    def _create_main_model(cls, c_constraint, default_inputs):
        """Create main model.

        :param ndarray c_constraint: coefficients for constraint.
        :param dict(ndarray) default_inputs: default inputs.
        """
        return TMMainDiscipline(c_constraint, default_inputs)

    @classmethod
    def _create_sub_model(cls, index, c_shared, c_local, c_cpl, default_inputs):
        """Create sub model.

        :param int index: model index.
        :param ndarray c_shared: coefficients for shared design parameters.
        :param ndarray c_local: coefficients for local design parameters.
        :param ndarray c_cpl: coefficients for coupling variables.
        :param dict(ndarray) default_inputs: default inputs.
        """
        return TMSubDiscipline(index, c_shared, c_local, c_cpl, default_inputs)

    @property
    def disciplines(self):
        """Alias for self.models."""
        return self.models

    @property
    def n_disciplines(self):
        """Alias for self.n_submodels."""
        return self.n_submodels

    @property
    def main_discipline(self):
        """Main disciplines.

        :return: main discipline.
        :rtype: TMDiscipline
        """
        return self.models[0]

    @property
    def sub_disciplines(self):
        """Sub-disciplines.

        :return: list of disciplines.
        :rtype: list(TMDiscipline)
        """
        return self.models[1:]

    def get_design_space(self):
        """Get the TM design space.

        :return: instance of the design space
        :rtype: DesignSpace
        """
        d_s = super().get_design_space()
        design_space = ParameterSpace()
        for name in d_s.names:
            size = d_s.sizes[name]
            l_b = d_s.lower_bounds[name]
            u_b = d_s.upper_bounds[name]
            value = d_s.default_values[name]
            design_space.add_variable(name, size, "float", l_b, u_b, value)
        if self._noised_coupling:
            for index in range(self.n_submodels):
                name = get_u_local_name(index)
                size = self.n_coupling
                distribution = "OTNormalDistribution"
                design_space.add_random_variable(name, distribution, size[index])
        return design_space

    def reset_disciplines(self):
        """Reset the disciplines, setting n_calls=0, n_calls_linearize=0, exec_time=0 and
        local_data={}."""
        for discipline in self.disciplines:
            discipline.n_calls = 0
            discipline.exec_time = 0.0
            discipline.n_calls_linearize = 0
            discipline.local_data = {}
