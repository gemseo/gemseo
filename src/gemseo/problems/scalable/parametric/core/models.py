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
Scalable problem - Models
*************************
"""
from __future__ import annotations

import logging

from numpy import array
from numpy import atleast_2d
from numpy import diag
from numpy import dot
from numpy import eye
from numpy import mean as npmean
from numpy import ndarray
from numpy import sum as npsum

from .variables import get_constraint_name
from .variables import get_coupling_name
from .variables import get_u_local_name
from .variables import get_x_local_name
from .variables import OBJECTIVE_NAME
from .variables import X_SHARED_NAME

LOGGER = logging.getLogger(__name__)


class TMMainModel:

    r"""The main discipline from the scalable problem introduced by Tedford
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

        :param list(ndarray) c_constraint: constraint coefficients
        :param dict(ndarray) default_inputs: default inputs
        """
        self.name = "MainModel"
        self.default_inputs = default_inputs
        self.inputs_sizes = {name: len(val) for name, val in default_inputs.items()}
        self.inputs_names = sorted(self.inputs_sizes.keys())
        self.outputs_sizes = {
            get_constraint_name(index): len(value)
            for index, value in enumerate(c_constraint)
        }
        self.outputs_sizes[OBJECTIVE_NAME] = 1
        self.outputs_names = sorted(self.outputs_sizes.keys())
        self.coefficients = c_constraint
        self.n_submodels = len(c_constraint)

    def __call__(self, x_shared=None, coupling=None, jacobian=False):
        """Compute the discipline output or derivatives for new values of shared design
        parameters and coupling variables.

        :param ndarray x_shared: shared design parameters.
        :param dict(ndarray) coupling: list of coupling variables
            (one element per sub-discipline).
        :param bool jacobian: if True, return the jacobian of the discipline output.
            Otherwise, return the discipline output.
        """
        if x_shared is None:
            x_shared = self.default_inputs[X_SHARED_NAME]
        if coupling is None:
            names = set(self.inputs_names) - {X_SHARED_NAME}
            coupling = {name: self.default_inputs[name] for name in names}
        if jacobian:
            result = self._compute_jacobian(x_shared, coupling)
        else:
            result = self._compute_output(x_shared, coupling)
        return result

    def _compute_output(self, x_shared, coupling):
        """Compute the discipline output for new values of shared design parameters and
        coupling variables.

        :param ndarray x_shared: shared design parameters.
        :param dict(ndarray) coupling: list of coupling variables
            (one element per sub-discipline).
        """
        obj = npmean(x_shared**2)
        obj += npmean(
            [
                npmean(coupling[get_coupling_name(index)] ** 2)
                for index in range(self.n_submodels)
            ]
        )
        output = {OBJECTIVE_NAME: array([obj])}
        for index in range(self.n_submodels):
            constraint = get_constraint_name(index)
            coupling_name = get_coupling_name(index)
            tmp = coupling[coupling_name] / self.coefficients[index]
            output[constraint] = 1 - tmp
        return output

    def _compute_jacobian(self, x_shared, coupling):
        """Computes the discipline jacobian.

        :param ndarray x_shared: shared design parameters.
        :param dict(ndarray) coupling: list of coupling variables
            (one element per sub-discipline).
        """
        tmp = 2 * x_shared / len(x_shared)
        jacobian = {OBJECTIVE_NAME: {}}
        jacobian[OBJECTIVE_NAME][X_SHARED_NAME] = atleast_2d(tmp)
        for index in range(self.n_submodels):
            constraint = get_constraint_name(index)
            coupling_name = get_coupling_name(index)
            jacobian[constraint] = {}
            tmp = 2 * coupling[coupling_name] / len(coupling[coupling_name])
            tmp /= self.n_submodels
            jacobian[OBJECTIVE_NAME][coupling_name] = atleast_2d(tmp)
            tmp = diag(-1.0 / self.coefficients[index])
            jacobian[constraint][coupling_name] = atleast_2d(tmp)
        return jacobian


class TMSubModel:

    r"""A sub-discipline from the scalable problem introduced by Tedford and Martins
    (2010) takes local design parameters :math:`x_i` and shared design parameters
    :math:`z` in input as well as coupling variables :math:`\left(y_i\right)_{1\leq j
    \leq N\atop j\neq i}` from :math:`N-1` elementary disciplines, and returns the
    coupling variables:

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
        :param ndarray c_shared: weights for the shared design parameters.
        :param ndarray c_local: weights for the local design parameters.
        :param dict(ndarray) c_coupling: weights for the coupling parameters.
        :param dict(ndarray) default_inputs: default inputs
        """
        self.name = f"SubModel_{index}"
        self.index = index
        self.c_shared = c_shared
        self.c_local = c_local
        self.c_coupling = c_coupling
        self.default_inputs = default_inputs
        self._check_consistency()
        self.inputs_sizes = {name: len(val) for name, val in default_inputs.items()}
        self.inputs_names = sorted(self.inputs_sizes.keys())
        output = get_coupling_name(index)
        self.outputs_sizes = {output: len(c_local)}
        self.outputs_names = sorted(self.outputs_sizes.keys())

    def _check_consistency(self):
        """Check consistency of model and default inputs."""
        assert isinstance(self.c_shared, ndarray)
        assert len(self.c_shared.shape) == 2
        assert self.c_shared.shape[1] == len(self.default_inputs[X_SHARED_NAME])
        nout = self.c_shared.shape[0]
        assert isinstance(self.c_local, ndarray)
        assert len(self.c_local.shape) == 2
        assert self.c_local.shape[0] == nout
        xlocal = get_x_local_name(self.index)
        assert self.c_local.shape[1] == len(self.default_inputs[xlocal])
        assert isinstance(self.c_coupling, dict)
        for key, val in self.c_coupling.items():
            assert isinstance(val, ndarray)
            assert len(val.shape) == 2
            assert val.shape[0] == nout
            assert val.shape[1] == len(self.default_inputs[key])

    def __call__(
        self, x_shared=None, x_local=None, coupling=None, noise=None, jacobian=False
    ):
        """Compute the discipline output or derivatives for new values of shared design
        parameters and coupling variables.

        :param ndarray x_shared: shared design parameters.
        :param ndarray x_local: local design parameters.
        :param dict(ndarray) coupling: list of coupling variables
            (one element per sub-discipline).
        :param ndarray noise: random noise applied to the vectorial output.
        :param bool jacobian: if True, return the jacobian of the discipline output.
            Otherwise, return the discipline output.
        """
        if x_shared is None:
            x_shared = self.default_inputs[X_SHARED_NAME]
        if x_local is None:
            x_local = self.default_inputs[get_x_local_name(self.index)]
        if coupling is None:
            coupling = {cpl: self.default_inputs[cpl] for cpl in self.c_coupling}
        if jacobian:
            result = self._compute_jacobian()
        else:
            result = self._compute_output(x_shared, x_local, coupling, noise)
        return result

    def _compute_output(self, x_shared, x_local, coupling, noise):
        """Compute the discipline output for new values of shared design parameters and
        coupling variables.

        :param ndarray x_shared: shared design parameters.
        :param ndarray x_local: local design parameters.
        :param dict(ndarray) coupling: list of coupling variables
            (one element per sub-discipline).
        :param ndarray noise: random noise applied to the vectorial output.
        """
        output = -dot(self.c_shared, x_shared.reshape((-1, 1)))
        output -= dot(self.c_local, x_local.reshape((-1, 1)))
        cpl_sum = 0
        for name, coeff in self.c_coupling.items():
            output += dot(coeff, coupling[name].reshape((-1, 1)))
            cpl_sum += npsum(coeff, 1)
        output += npsum(self.c_shared, 1).reshape((-1, 1))
        output += npsum(self.c_local, 1).reshape((-1, 1))
        norm = cpl_sum.reshape((-1, 1))
        norm += npsum(self.c_shared, 1).reshape((-1, 1))
        norm += npsum(self.c_local, 1).reshape((-1, 1))
        output /= norm
        output = output.flatten()
        if noise is not None:
            output += noise
        output = {get_coupling_name(self.index): output}
        return output

    def _compute_jacobian(self):
        """Computes the discipline jacobian."""
        coupling_name = get_coupling_name(self.index)
        x_local_name = get_x_local_name(self.index)
        u_local_name = get_u_local_name(self.index)
        cpl_sum = 0
        for coeff in self.c_coupling.values():
            cpl_sum += npsum(coeff, 1)
        norm = cpl_sum.reshape((-1, 1))
        norm += npsum(self.c_shared, 1).reshape((-1, 1))
        norm += npsum(self.c_local, 1).reshape((-1, 1))
        der = -self.c_local / norm
        jacobian = {coupling_name: {}}
        jacobian[coupling_name][x_local_name] = der
        jacobian[coupling_name][u_local_name] = eye(der.shape[0])
        der = -self.c_shared / norm
        jacobian[coupling_name][X_SHARED_NAME] = der
        for cpl_name, cpl_value in self.c_coupling.items():
            jacobian[coupling_name][cpl_name] = cpl_value / norm
        return jacobian
