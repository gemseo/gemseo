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
Scalable problem - Problem
**************************
"""
from __future__ import annotations

import logging

from numpy import zeros
from numpy.random import rand
from numpy.random import seed as npseed

from .design_space import TMDesignSpace
from .models import TMMainModel
from .models import TMSubModel
from .variables import check_consistency
from .variables import get_constraint_name
from .variables import get_coupling_name
from .variables import get_u_local_name
from .variables import get_x_local_name
from .variables import X_SHARED_NAME

LOGGER = logging.getLogger(__name__)


class TMProblem:

    """The scalable problem from Tedford and Martins, 2010, builds a list of strongly
    coupled models (:class:`.TMSubModel`) completed by a main model
    (:class:`.TMMainModel`) computing the objective function and the constraints.

    These disciplines are defined on a unit design space whose parameters comprised in
    [0, 1] (:class:`.TMDesignSpace`). This problem is defined by the number of shared
    design parameters, the number of local design parameters per discipline and the
    number of output coupling variables per discipline. The strongly coupled disciplines
    can be either fully coupled (one discipline depends on all the others) or circularly
    coupled (one discipline depends only on the previous one and the first discipline
    depends only on the last one).
    """

    def __init__(
        self,
        n_shared=1,
        n_local=None,
        n_coupling=None,
        full_coupling=True,
        noised_coupling=False,
        seed=1,
    ):
        """Constructor.

        :param int n_shared: size of the shared design parameters.
            Default: 1.
        :param list(int) n_local: sizes of the local design parameters for the different
            disciplines. Same length as n_coupling. If None, use [1, 1]. Default: None.
        :param list(int) n_coupling: sizes of the coupling parameters for the different
            disciplines. Same length as n_local. If None, use [1, 1]. Default: None.
        :param bool full_coupling: if True, fully couple the disciplines. Otherwise,
            use circular coupling. Default: True.
        :param bool noised_coupling: if True, add a noise component u_local_i
            on the i-th discipline output.
        :param int seed: seed for replicability.
        """
        npseed(seed)
        self._seed = seed

        # Set the coupling style
        self._full_coupling = full_coupling
        self._noised_coupling = noised_coupling

        # Set and check the dimensions of the problem
        n_local = n_local or [1, 1]
        n_coupling = n_coupling or [1, 1]
        self.n_shared = n_shared
        self.n_local = n_local
        self.n_coupling = n_coupling
        check_consistency(n_shared, n_local, n_coupling)
        self.n_submodels = len(n_local)

        # Create instances of the random coefficients
        c_shared, c_local, c_cpl, c_constraint = self._generate_coefficients()

        # Instantiate the system model
        names = [X_SHARED_NAME]
        names += [get_coupling_name(index) for index in range(self.n_submodels)]
        default_inputs = self.get_default_inputs(names=names)
        self.models = [self._create_main_model(c_constraint, default_inputs)]

        # Instantiate the strongly coupled models
        for index in range(self.n_submodels):
            names = [X_SHARED_NAME]
            names += [get_x_local_name(index)]
            if full_coupling:
                names += [
                    get_coupling_name(other_index)
                    for other_index in range(self.n_submodels)
                    if other_index != index
                ]
            else:
                other_id = self.n_submodels - 1 if index == 0 else index - 1
                names += [get_coupling_name(other_id)]
            if self._noised_coupling:
                names += [get_u_local_name(index)]
            default_inputs = self.get_default_inputs(names=names)
            model = self._create_sub_model(
                index, c_shared[index], c_local[index], c_cpl[index], default_inputs
            )
            self.models.append(model)

        # Instantiate the design space
        self.design_space = self.get_design_space()

    @classmethod
    def _create_main_model(cls, c_constraint, default_inputs):
        """Create main model.

        :param ndarray c_constraint: coefficients for constraint.
        :param dict(ndarray) default_inputs: default inputs.
        :return: instance of the main model.
        :return: TMMainModel
        """
        return TMMainModel(c_constraint, default_inputs)

    @classmethod
    def _create_sub_model(cls, index, c_shared, c_local, c_cpl, default_inputs):
        """Create sub model.

        :param int index: model index
        :param ndarray c_shared: coefficients for shared design parameters.
        :param ndarray c_local: coefficients for local design parameters.
        :param ndarray c_cpl: coefficients for coupling variables.
        :param dict(ndarray) default_inputs: default inputs.
        :return: instance of a sub-model.
        :return: TMSubModel
        """
        return TMSubModel(index, c_shared, c_local, c_cpl, default_inputs)

    def __str__(self):
        """String representation."""
        msg = ["Scalable problem"]
        for model in self.models:
            msg.append(f".... {model.name}")
            msg.append("........ Inputs:")
            for name in model.inputs_names:
                size = model.inputs_sizes[name]
                msg.append(f"............ {name} ({size})")
            msg.append("........ Outputs:")
            for name in model.outputs_names:
                size = model.outputs_sizes[name]
                msg.append(f"............ {name} ({size})")
        return "\n".join(msg)

    def get_default_inputs(self, names=None):
        """Get default input values.

        :param list(str) names: names of the inputs.
        :return: name and values of the inputs.
        :rtype: dict
        """
        inputs = {X_SHARED_NAME: zeros(self.n_shared) + 0.5}
        for index in range(self.n_submodels):
            inputs[get_x_local_name(index)] = zeros(self.n_local[index]) + 0.5
            inputs[get_coupling_name(index)] = zeros(self.n_coupling[index]) + 0.5
            inputs[get_constraint_name(index)] = zeros(self.n_coupling[index]) + 0.5
            inputs[get_u_local_name(index)] = zeros(self.n_coupling[index])
        if names is not None:
            inputs = {name: inputs[name] for name in names}
        return inputs

    def get_design_space(self):
        """Get the TM design space.

        :return: instance of the design space
        :rtype: TMDesignSpace
        """
        return TMDesignSpace(self.n_shared, self.n_local, self.n_coupling)

    def reset_design_space(self):
        """Reset the TM design space."""
        self.design_space = self.get_design_space()

    def _generate_coefficients(self):
        """Generate coefficients associated with the shared design parameters, the local
        design parameters and the coupling variables.

        :return: coefficients for both shared design parameters,
            local design parameters and coupling variables.
        :rtype: list(ndarray), list(ndarray), list(ndarray), list(ndarray)
        """
        c_shared = []
        c_local = []
        c_coupling = []
        for disc in range(self.n_submodels):
            if self._full_coupling:
                other_indices = set(range(self.n_submodels)) - {disc}
                other_indices = list(other_indices)
            else:
                other_index = self.n_submodels - 1 if disc == 0 else disc - 1
                other_indices = [other_index]
            n_coupling = self.n_coupling[disc]
            c_shared.append(rand(n_coupling, self.n_shared))
            c_local.append(rand(n_coupling, self.n_local[disc]))
            c_coupling.append({})
            for index in other_indices:
                coeff = rand(n_coupling, self.n_coupling[index])
                c_coupling[-1][get_coupling_name(index)] = coeff
        c_constraint = [rand(val) for val in self.n_coupling]
        return c_shared, c_local, c_coupling, c_constraint
