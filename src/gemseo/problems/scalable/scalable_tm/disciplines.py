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
Scalable disciplines from Tedford and Martins (2010)
****************************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array, atleast_2d, diag, dot
from numpy import mean as npmean
from numpy import ndarray
from numpy import sum as npsum
from past.utils import old_div

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.scalable_tm.variables import (
    OBJECTIVE_NAME,
    X_SHARED_NAME,
    get_constraint_name,
    get_coupling_name,
    get_x_local_name,
)

standard_library.install_aliases()

from gemseo import LOGGER


class TMSystem(MDODiscipline):

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
        """Constructor

        :param list(array) c_constraint: constraint coefficients
        :param dict default_inputs: default inputs
        """
        super(TMSystem, self).__init__("TM_System")
        self.input_grammar.initialize_from_base_dict(default_inputs)
        default_outputs = {}
        default_outputs[OBJECTIVE_NAME] = array([0.0])
        for index, value in enumerate(c_constraint):
            tmp = array([0.0] * len(value))
            default_outputs[get_constraint_name(index)] = tmp
        self.output_grammar.initialize_from_base_dict(default_outputs)
        self.default_inputs = default_inputs
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        self.coefficients = c_constraint
        self.n_disciplines = len(c_constraint)

    def _run(self):
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        coupling = [
            self.get_inputs_by_name(get_coupling_name(index))
            for index in range(self.n_disciplines)
        ]
        obj = npmean(x_shared ** 2)
        obj += npmean(
            [npmean(coupling[index] ** 2) for index in range(self.n_disciplines)]
        )
        data = {}
        data[OBJECTIVE_NAME] = array([obj])
        for index in range(self.n_disciplines):
            constraint = get_constraint_name(index)
            tmp = old_div(coupling[index], self.coefficients[index])
            data[constraint] = 1 - tmp
        self.store_local_data(**data)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        coupling = [
            self.get_inputs_by_name(get_coupling_name(index))
            for index in range(self.n_disciplines)
        ]
        tmp = old_div(2 * x_shared, len(x_shared))
        self.jac[OBJECTIVE_NAME][X_SHARED_NAME] = atleast_2d(tmp)
        for index in range(self.n_disciplines):
            tmp = old_div(2 * coupling[index], len(coupling[index]))
            tmp /= self.n_disciplines
            self.jac[OBJECTIVE_NAME][get_coupling_name(index)] = atleast_2d(tmp)
            tmp = diag(old_div(-1.0, self.coefficients[index]))
            constraint = get_constraint_name(index)
            coupling_name = get_coupling_name(index)
            self.jac[constraint][coupling_name] = atleast_2d(tmp)


class TMDiscipline(MDODiscipline):

    r"""An elementary discipline from the scalable problem introduced by
    Tedford and Martins (2010) takes local design parameters :math:`x_i`
    and shared design parameters :math:`z` in input as well as coupling
    variables :math:`\left(y_i\right)_{1\leq j \leq N\atop j\neq i}`
    from :math:`N-1` elementary disciplines,
    and returns the coupling variables:

    .. math::

        y_i =\frac{\tilde{y}_i+C_{z,i}.1+C_{x_i}.1}{\sum_{j=1 \atop j
        \neq i}^NC_{y_j,i}.1+C_{z,i}.1+C_{x_i}.1} \in [0,1]^{n_{y_i}}

    where:

    .. math::

        \tilde{y}_i = - C_{z,i}.z - C_{x_i}.x_i +
        \sum_{j=1 \atop j \neq i}^N C_{y_j,i}.y_j
    """

    def __init__(self, index, c_shared, c_local, c_coupling, default_inputs):
        """Constructor

        :param int index: discipline index for naming.
        :param array c_shared: weights for the shared design parameters.
        :param array c_local: weights for the local design parameters.
        :param dict(array) c_coupling: weights for the coupling parameters.
        :param dict default_inputs: default inputs
        """
        self.index = index
        self.c_shared = c_shared
        self.c_local = c_local
        self.c_coupling = c_coupling
        self.n_disciplines = len(c_coupling)
        self._check_consistency(default_inputs)
        super(TMDiscipline, self).__init__(name="TM_Discipline_" + str(index))
        self.input_grammar.initialize_from_base_dict(default_inputs)
        value = array([0.0] * self.c_shared.shape[0])
        default_outputs = {get_coupling_name(index): value}
        self.output_grammar.initialize_from_base_dict(default_outputs)
        self.default_inputs = default_inputs
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _check_consistency(self, default_inputs):
        """Check consistency of default inputs.

        :param dict default_inputs: default inputs.
        """
        assert isinstance(self.c_shared, ndarray)
        assert len(self.c_shared.shape) == 2
        assert self.c_shared.shape[1] == len(default_inputs[X_SHARED_NAME])
        nout = self.c_shared.shape[0]
        assert isinstance(self.c_local, ndarray)
        assert len(self.c_local.shape) == 2
        assert self.c_local.shape[0] == nout
        xlocal = get_x_local_name(self.index)
        assert self.c_local.shape[1] == len(default_inputs[xlocal])
        assert isinstance(self.c_coupling, dict)
        for key, val in self.c_coupling.items():
            assert isinstance(val, ndarray)
            assert len(val.shape) == 2
            assert val.shape[0] == nout
            assert val.shape[1] == len(default_inputs[key])

    def _run(self):
        index = self.index
        x_shared = self.get_inputs_by_name(X_SHARED_NAME)
        x_local = self.get_inputs_by_name(get_x_local_name(index))
        coupling = {
            cpl_name: self.get_inputs_by_name(cpl_name)
            for cpl_name in list(self.c_coupling.keys())
        }
        data = -dot(self.c_shared, x_shared.reshape((-1, 1)))
        data -= dot(self.c_local, x_local.reshape((-1, 1)))
        cpl_sum = 0
        for cpl_name, cpl_value in self.c_coupling.items():
            data += dot(cpl_value, coupling[cpl_name].reshape((-1, 1)))
            cpl_sum += npsum(self.c_coupling[cpl_name], 1)
        data += npsum(self.c_shared, 1).reshape((-1, 1))
        data += npsum(self.c_local, 1).reshape((-1, 1))
        norm = cpl_sum.reshape((-1, 1))
        norm += npsum(self.c_shared, 1).reshape((-1, 1))
        norm += npsum(self.c_local, 1).reshape((-1, 1))
        data /= norm
        data = data.flatten()
        self.store_local_data(**{get_coupling_name(index): data})

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        other_indices = set(range(self.n_disciplines)) - set([self.index])
        other_indices = list(other_indices)
        cpl_sum = 0
        for cpl_value in self.c_coupling.values():
            cpl_sum += npsum(cpl_value, 1)
        norm = cpl_sum.reshape((-1, 1))
        norm += npsum(self.c_shared, 1).reshape((-1, 1))
        norm += npsum(self.c_local, 1).reshape((-1, 1))
        der = old_div(-self.c_local, norm)
        self.jac[get_coupling_name(self.index)][get_x_local_name(self.index)] = der
        der = old_div(-self.c_shared, norm)
        self.jac[get_coupling_name(self.index)][X_SHARED_NAME] = der
        for cpl_name, cpl_value in self.c_coupling.items():
            self.jac[get_coupling_name(self.index)][cpl_name] = old_div(cpl_value, norm)
