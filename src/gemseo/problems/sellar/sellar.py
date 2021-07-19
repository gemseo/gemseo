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
#        :author: Charlie Vanaret
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""The disciplines for the MDO problem proposed by Sellar et al. in

Sellar, R., Batill, S., & Renaud, J. (1996).
Response surface based, concurrent subspace optimization
for multidisciplinary system design.
In 34th aerospace sciences meeting and exhibit (p. 714).

The MDO problem is written as follows:

.. math::

   \begin{aligned}
   \text{minimize the objective function }&obj=x_{local}^2 + x_{shared,2}
   +y_1^2+e^{-y_2} \\
   \text{with respect to the design variables }&x_{shared},\,x_{local} \\
   \text{subject to the general constraints }
   & c_1 \leq 0\\
   & c_2 \leq 0\\
   \text{subject to the bound constraints }
   & -10 \leq x_{shared,1} \leq 10\\
   & 0 \leq x_{shared,2} \leq 10\\
   & 0 \leq x_{local} \leq 10.
   \end{aligned}

where the coupling variables are

.. math::

    \text{Discipline 1: } y_1 = \sqrt{x_{shared,1}^2 + x_{shared,2} +
     x_{local} - 0.2\,y_2},

and

.. math::

    \text{Discipline 2: }y_2 = |y_1| + x_{shared,1} + x_{shared,2}.

and where the general constraints are

.. math::

   c_1 = 3.16 - y_1^2

   c_2 = y_2 - 24

This module implements three disciplines to compute the different coupling variables,
constraints and objective:

- :class:`.Sellar1`:
  this :class:`.MDODiscipline` computes :math:`y_1`
  from :math:`y_2`, :math:`x_{shared,1}`, :math:`x_{shared,2}` and :math:`x_{local}`.
- :class:`.Sellar2`:
  this :class:`.MDODiscipline` computes :math:`y_2`
  from :math:`y_1`, :math:`x_{shared,1}` and :math:`x_{shared,2}`.
- :class:`.SellarSystem`:
  this :class:`.MDODiscipline` computes both objective and constraints
  from :math:`y_1`, :math:`y_2`, :math:`x_{local}` and :math:`x_{shared,2}`.
"""
from __future__ import division, unicode_literals

from cmath import exp, sqrt
from typing import Dict, Iterable, List, Optional

from numpy import array, atleast_2d, complex128, ndarray, ones, zeros

from gemseo.core.discipline import MDODiscipline

Y_1 = "y_1"
Y_2 = "y_2"
X_SHARED = "x_shared"
X_LOCAL = "x_local"
OBJ = "obj"
C_1 = "c_1"
C_2 = "c_2"
R_1 = "r_1"
R_2 = "r_2"


def get_inputs(
    names=None,  # type: Optional[Iterable[str]]
):  # type: (...) -> Dict[str,ndarray]
    """Generate an initial solution for the MDO problem.

    Args:
        names: The names of the discipline inputs.

    Returns:
        The default values of the discipline inputs.
    """
    inputs = {
        X_LOCAL: array([0.0], dtype=complex128),
        X_SHARED: array([1.0, 0.0], dtype=complex128),
        Y_1: ones(1, dtype=complex128),
        Y_2: ones(1, dtype=complex128),
    }
    if names is None:
        return inputs
    return {name: inputs[name] for name in names}


class SellarSystem(MDODiscipline):
    """The discipline to compute the objective and constraints of the Sellar problem."""

    def __init__(self):  # type: (...) -> None
        super(SellarSystem, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = get_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):  # type: (...) -> None
        x_local, x_shared, y_1, y_2 = self.get_inputs_by_name(
            [X_LOCAL, X_SHARED, Y_1, Y_2]
        )
        obj = array([self.compute_obj(x_local, x_shared, y_1, y_2)], dtype=complex128)
        c_1 = array([self.compute_c_1(y_1)], dtype=complex128)
        c_2 = array([self.compute_c_2(y_2)], dtype=complex128)
        self.store_local_data(obj=obj, c_1=c_1, c_2=c_2)

    @staticmethod
    def compute_obj(
        x_local,  # type: ndarray
        x_shared,  # type: ndarray
        y_1,  # type: ndarray
        y_2,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the objective :math:`obj`.

        Args:
            x_local: The design variables local to the first discipline.
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the objective :math:`obj`.
        """
        return x_local[0] ** 2 + x_shared[1] + y_1[0] ** 2 + exp(-y_2[0])

    @staticmethod
    def compute_c_1(
        y_1,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the constraint :math:`c_1`.

        Args:
            y_1: The coupling variable coming from the first discipline.

        Returns:
            The value of the constraint :math:`c_1`.
        """
        return 3.16 - y_1[0] ** 2

    @staticmethod
    def compute_c_2(
        y_2,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the constraint :math:`c_2`.

        Args:
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the constraint :math:`c_2`.
        """
        return y_2[0] - 24.0

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_local, _, y_1, y_2 = self.get_inputs_by_name([X_LOCAL, X_SHARED, Y_1, Y_2])
        self.jac[C_1][Y_1] = atleast_2d(array([-2.0 * y_1]))
        self.jac[C_2][Y_2] = ones((1, 1))
        self.jac[OBJ][X_LOCAL] = atleast_2d(array([2.0 * x_local[0]]))
        self.jac[OBJ][X_SHARED] = atleast_2d(array([0.0, 1.0]))
        self.jac[OBJ][Y_1] = atleast_2d(array([2.0 * y_1[0]]))
        self.jac[OBJ][Y_2] = atleast_2d(array([-exp(-y_2[0])]))


class Sellar1(MDODiscipline):
    """The discipline to compute the coupling variable :math:`y_1`."""

    def __init__(
        self,
        residual_form=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            residual_form: If True,
                then only residuals are computed,
                not the coupling variables.
        """
        self.residual_form = residual_form
        super(Sellar1, self).__init__(auto_detect_grammar_files=True)
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        if residual_form:
            self.residual_variables = [R_1]
            self.output_grammar.remove_item(Y_1)

        else:
            self.output_grammar.remove_item(R_1)
            self.input_grammar.remove_item(Y_1)

        self.default_inputs = get_inputs(self.input_grammar.get_data_names())

    def get_attributes_to_serialize(self):  # type: (...) -> List[str]
        base_d = super(Sellar1, self).get_attributes_to_serialize()
        base_d.append("residual_form")
        return base_d

    def _run(self):  # type: (...) -> None
        x_local, x_shared, y_2 = self.get_inputs_by_name([X_LOCAL, X_SHARED, Y_2])

        if self.residual_form:
            y_1 = self.get_inputs_by_name(Y_1)
            r_1_out = array(
                [self.compute_r_1(x_local, x_shared, y_1, y_2)], dtype=complex128
            )
            self.store_local_data(r_1=r_1_out)
        else:
            # functional form
            y_1_out = array(
                [self.compute_y_1(x_local, x_shared, y_2)], dtype=complex128
            )
            self.store_local_data(y_1=y_1_out)

    @staticmethod
    def compute_y_1(
        x_local,  # type: ndarray
        x_shared,  # type: ndarray
        y_2,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the first coupling equation in functional form.

        Args:
            x_local: The design variables local to first discipline.
            x_shared: The shared design variables.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the coupling variable :math:`y_1`.
        """
        return sqrt(x_shared[0] ** 2 + x_shared[1] + x_local[0] - 0.2 * y_2[0])

    @staticmethod
    def compute_r_1(
        x_local,  # type: ndarray
        x_shared,  # type: ndarray
        y_1,  # type: ndarray
        y_2,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the first coupling equation in residual form.

        Args:
            x_local: The design variables local to the first discipline.
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the residues related to the first discipline.
        """
        return Sellar1.compute_y_1(x_local, x_shared, y_2) - y_1[0]

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_local, x_shared, y_2 = self.get_inputs_by_name([X_LOCAL, X_SHARED, Y_2])

        inv_denom = 1.0 / (self.compute_y_1(x_local, x_shared, y_2))
        self.jac[Y_1] = {}
        self.jac[Y_1][X_LOCAL] = atleast_2d(array([0.5 * inv_denom]))
        self.jac[Y_1][X_SHARED] = atleast_2d(
            array([x_shared[0] * inv_denom, 0.5 * inv_denom])
        )
        self.jac[Y_1][Y_2] = atleast_2d(array([-0.1 * inv_denom]))

        if self.residual_form:
            self.jac[R_1][X_LOCAL] = atleast_2d(self.jac[Y_1][X_LOCAL])
            self.jac[R_1][X_SHARED] = atleast_2d(self.jac[Y_1][X_SHARED])
            self.jac[R_1][Y_2] = atleast_2d(self.jac[Y_1][Y_2])
            self.jac[R_1][Y_1] = -ones((1, 1))
            del self.jac[Y_1]


class Sellar2(MDODiscipline):
    """The discipline to compute the coupling variable :math:`y_2`."""

    def __init__(
        self,
        residual_form=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            residual_form: If True,
                then only residuals are computed,
                not the coupling variables.
        """
        self.residual_form = residual_form
        super(Sellar2, self).__init__(auto_detect_grammar_files=True)
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        if residual_form:
            self.residual_variables = [R_2]
            self.output_grammar.remove_item(Y_2)
        else:
            self.output_grammar.remove_item(R_2)
            self.input_grammar.remove_item(Y_2)
        self.default_inputs = get_inputs(self.input_grammar.get_data_names())

    def get_attributes_to_serialize(self):
        base_d = super(Sellar2, self).get_attributes_to_serialize()
        base_d.append("residual_form")
        return base_d

    def _run(self):  # type: (...) -> None
        x_shared, y_1 = self.get_inputs_by_name([X_SHARED, Y_1])

        if self.residual_form:
            y_2 = self.get_inputs_by_name(Y_2)
            r_2_out = array([self.compute_r_2(x_shared, y_1, y_2)], dtype=complex128)
            self.store_local_data(r_2=r_2_out)
        else:
            # functional form
            y_2_out = array([self.compute_y_2(x_shared, y_1)], dtype=complex128)
            self.store_local_data(y_2=y_2_out)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        self._init_jacobian(inputs, outputs, with_zeros=True)
        y_1 = self.get_inputs_by_name(Y_1)
        self.jac[Y_2] = {}
        self.jac[Y_2][X_LOCAL] = zeros((1, 1))
        self.jac[Y_2][X_SHARED] = ones((1, 2))
        if y_1[0] < 0.0:
            self.jac[Y_2][Y_1] = -ones((1, 1))
        elif y_1[0] == 0.0:
            self.jac[Y_2][Y_1] = zeros((1, 1))
        else:
            self.jac[Y_2][Y_1] = ones((1, 1))

        if self.residual_form:
            self.jac[R_2][X_LOCAL] = self.jac[Y_2][X_LOCAL]
            self.jac[R_2][X_SHARED] = self.jac[Y_2][X_SHARED]
            self.jac[R_2][Y_1] = self.jac[Y_2][Y_1]
            self.jac[R_2][Y_2] = -ones((1, 1))
            del self.jac[Y_2]

    @staticmethod
    def compute_y_2(
        x_shared,  # type: ndarray
        y_1,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the second coupling equation in functional form.

        Args:
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.

        Returns:
            The value of the coupling variable :math:`y_2`.
        """
        out = x_shared[0] + x_shared[1]
        if y_1[0].real == 0:
            y_2 = out
        elif y_1[0].real > 0:
            y_2 = y_1[0] + out
        else:
            y_2 = -y_1[0] + out
        return y_2

    @staticmethod
    def compute_r_2(
        x_shared,  # type: ndarray
        y_1,  # type: ndarray
        y_2,  # type: ndarray
    ):  # type: (...) -> float
        """Evaluate the second coupling equation in residual form.

        Args:
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the residues related to the second discipline.
        """
        return Sellar2.compute_y_2(x_shared, y_1) - y_2[0]
