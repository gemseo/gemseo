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
Scalable disciplines from Tedford and Martins (2010)
****************************************************
"""
from __future__ import annotations

import logging

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.parametric.core.models import TMMainModel
from gemseo.problems.scalable.parametric.core.models import TMSubModel
from gemseo.problems.scalable.parametric.core.variables import get_coupling_name
from gemseo.problems.scalable.parametric.core.variables import get_u_local_name
from gemseo.problems.scalable.parametric.core.variables import get_x_local_name
from gemseo.problems.scalable.parametric.core.variables import X_SHARED_NAME

LOGGER = logging.getLogger(__name__)


class TMDiscipline(MDODiscipline):
    """Abstract base class for disciplines in the TM problem."""

    @property
    def inputs_sizes(self):
        """Sizes of the model inputs."""
        return self.model.inputs_sizes

    @property
    def outputs_sizes(self):
        """Sizes of the model outputs."""
        return self.model.outputs_sizes

    @property
    def inputs_names(self):
        """Names of the model inputs."""
        return self.model.inputs_names

    @property
    def outputs_names(self):
        """Names of the model outputs."""
        return self.model.outputs_names


class TMMainDiscipline(TMDiscipline):
    r"""The system discipline from the scalable problem introduced by Tedford
    and Martins (2010) takes the  local design parameters
    :math:`x_1,x_2,\ldots,x_N` and the global design parameters :math:`z`
    as inputs, as well as the coupling variables :math:`y_1,y_2,\ldots,y_N`
    and returns the objective function value :math:`f(x,y(x,y))` to minimize as
    well as the inequality constraints ones
    :math:`c_1(y_1),c_2(y_2),\ldots,c_N(y_N)` which are expressed as:

    .. math::

       f(z,y) = |z|_2^2 + \sum_{i=1}^N |y_i|_2^2


    and:

    .. math::

       c_i(y_i) = 1- C_i^{-T}Iy_i

    """

    def __init__(self, c_constraint, default_inputs):
        """Constructor.

        :param list(array) c_constraint: constraint coefficients
        :param dict default_inputs: default inputs
        """
        self.model = TMMainModel(c_constraint, default_inputs)
        super().__init__(self.model.name)
        self.input_grammar.update(self.model.inputs_names)
        self.output_grammar.update(self.model.outputs_names)
        self.default_inputs = default_inputs
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    @property
    def n_disciplines(self):
        """Return the number of disciplines; alias for self.models.n_submodels."""
        return self.model.n_submodels

    def _run(self):
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        coupling = {
            get_coupling_name(index): self.get_inputs_by_name(get_coupling_name(index))
            for index in range(self.n_disciplines)
        }
        self.store_local_data(**self.model(x_shared, coupling))

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Computes the jacobian.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        coupling = {
            get_coupling_name(index): self.get_inputs_by_name(get_coupling_name(index))
            for index in range(self.n_disciplines)
        }
        jac = self.model(x_shared, coupling, jacobian=True)
        for output in jac:
            for inpt in jac[output]:
                self.jac[output][inpt] = jac[output][inpt]


class TMSubDiscipline(TMDiscipline):
    r"""An elementary discipline from the scalable problem introduced by Tedford and
    Martins (2010) takes local design parameters :math:`x_i` and shared design
    parameters :math:`z` in input as well as coupling variables
    :math:`\left(y_i\right)_{1\leq j \leq N\atop j\neq i}` from :math:`N-1` elementary
    disciplines, and returns the coupling variables:

    .. math::

        y_i =\frac{\tilde{y}_i+C_{z,i}.1+C_{x_i}.1}{\sum_{j=1 \atop j
        \neq i}^NC_{y_j,i}.1+C_{z,i}.1+C_{x_i}.1} \in [0,1]^{n_{y_i}}

    where:

    .. math::

        \tilde{y}_i = - C_{z,i}.z - C_{x_i}.x_i +
        \sum_{j=1 \atop j \neq i}^N C_{y_j,i}.y_j
    """

    def __init__(self, index, c_shared, c_local, c_coupling, default_inputs):
        """Constructor.

        :param int index: discipline index for naming.
        :param array c_shared: weights for the shared design parameters.
        :param array c_local: weights for the local design parameters.
        :param dict(array) c_coupling: weights for the coupling parameters.
        :param dict default_inputs: default inputs
        """
        self.model = TMSubModel(index, c_shared, c_local, c_coupling, default_inputs)
        super().__init__(name=self.model.name)
        self.input_grammar.update(self.model.inputs_names)
        self.output_grammar.update(self.model.outputs_names)
        self.default_inputs = default_inputs
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        x_local = self.get_inputs_by_name(get_x_local_name(self.model.index))
        u_local_name = get_u_local_name(self.model.index)
        if u_local_name in self.input_grammar:
            u_local = self.get_inputs_by_name(u_local_name)
        else:
            u_local = None
        coupling = {
            cpl_name: self.get_inputs_by_name(cpl_name)
            for cpl_name in list(self.model.c_coupling.keys())
        }
        self.store_local_data(**self.model(x_shared, x_local, coupling, u_local))

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Computes the jacobian.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        x_local = self.get_inputs_by_name(get_x_local_name(self.model.index))
        coupling = {
            cpl_name: self.get_inputs_by_name(cpl_name)
            for cpl_name in list(self.model.c_coupling.keys())
        }
        jac = self.model(x_shared, x_local, coupling, jacobian=True)
        for output in jac:
            for inpt in jac[output]:
                self.jac[output][inpt] = jac[output][inpt]
