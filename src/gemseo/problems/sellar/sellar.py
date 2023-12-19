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
r"""The disciplines for the MDO problem proposed by Sellar et al. in.

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

from __future__ import annotations

from cmath import exp
from cmath import sqrt
from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import atleast_2d
from numpy import complex128
from numpy import ndarray
from numpy import ones
from numpy import zeros

from gemseo.core.data_converters.json import JSONGrammarDataConverter
from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.mda.mda import MDA

Y_1 = "y_1"
Y_2 = "y_2"
X_SHARED = "x_shared"
X_LOCAL = "x_local"
OBJ = "obj"
C_1 = "c_1"
C_2 = "c_2"
R_1 = "r_1"
R_2 = "r_2"

# This allows to test a specific data converter.
WITH_2D_ARRAY = False


def get_inputs(
    names: Iterable[str] | None = None,
) -> dict[str, ndarray | complex]:
    """Generate an initial solution for the MDO problem.

    Args:
        names: The names of the discipline inputs.

    Returns:
        The default values of the discipline inputs.
    """
    inputs = {
        X_LOCAL: complex(0),
        X_SHARED: array([1.0, 0.0], dtype=complex128),
        Y_1: complex(1),
        Y_2: complex(1),
    }
    if WITH_2D_ARRAY:
        inputs[X_SHARED] = atleast_2d(inputs[X_SHARED])
    if names is None:
        return inputs
    return {name: inputs[name] for name in names}


def get_y_opt(mda: MDA) -> ndarray:
    """Return the optimal ``y`` array.

    Args:
        mda: The mda.

    Returns:
        The optimal ``y`` array.
    """
    return array([mda.local_data[Y_1].real, mda.local_data[Y_2].real])


if WITH_2D_ARRAY:

    class DataConverter(JSONGrammarDataConverter):
        """A data converter where ``x_shared`` is not a ndarray."""

        def convert_value_to_array(self, name: str, value: Any) -> ndarray:  # noqa: D102
            if name == X_SHARED:
                return value[0]
            return super().convert_value_to_array(name, value)

        def convert_array_to_value(self, name: str, array_: Any) -> Any:  # noqa: D102
            if name == X_SHARED:
                return array([array_])
            return super().convert_array_to_value(name, array_)

else:
    DataConverter = JSONGrammarDataConverter


class SellarSystem(MDODiscipline):
    """The discipline to compute the objective and constraints of the Sellar problem."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        input_data = get_inputs()
        self.input_grammar.update_from_data(input_data)
        self.output_grammar.update_from_names([OBJ, C_1, C_2])
        self.default_inputs = input_data
        self.re_exec_policy = self.ReExecutionPolicy.DONE

    def _run(self) -> None:
        x_local, x_shared, y_1, y_2 = self.get_inputs_by_name([
            X_LOCAL,
            X_SHARED,
            Y_1,
            Y_2,
        ])
        if WITH_2D_ARRAY:
            x_shared = x_shared[0]
        obj = array([self.compute_obj(x_local, x_shared, y_1, y_2)], dtype=complex128)
        c_1 = array([self.compute_c_1(y_1)], dtype=complex128)
        c_2 = array([self.compute_c_2(y_2)], dtype=complex128)
        self.store_local_data(obj=obj, c_1=c_1, c_2=c_2)

    @staticmethod
    def compute_obj(
        x_local: complex,
        x_shared: ndarray,
        y_1: complex,
        y_2: complex,
    ) -> complex:
        """Evaluate the objective :math:`obj`.

        Args:
            x_local: The design variables local to the first discipline.
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the objective :math:`obj`.
        """
        return x_local**2 + x_shared[1] + y_1**2 + exp(-y_2)

    @staticmethod
    def compute_c_1(
        y_1: complex,
    ) -> complex:
        """Evaluate the constraint :math:`c_1`.

        Args:
            y_1: The coupling variable coming from the first discipline.

        Returns:
            The value of the constraint :math:`c_1`.
        """
        return 3.16 - y_1**2

    @staticmethod
    def compute_c_2(
        y_2: complex,
    ) -> complex:
        """Evaluate the constraint :math:`c_2`.

        Args:
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the constraint :math:`c_2`.
        """
        return y_2 - 24.0

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(inputs, outputs)
        x_local, y_1, y_2 = self.get_inputs_by_name([X_LOCAL, Y_1, Y_2])
        self.jac[C_1][Y_1] = atleast_2d(array([-2.0 * y_1]))
        self.jac[C_2][Y_2] = ones((1, 1))
        self.jac[OBJ][X_LOCAL] = atleast_2d(array([2.0 * x_local]))
        self.jac[OBJ][X_SHARED] = atleast_2d(array([0.0, 1.0]))
        self.jac[OBJ][Y_1] = atleast_2d(array([2.0 * y_1]))
        self.jac[OBJ][Y_2] = atleast_2d(array([-exp(-y_2)]))


class Sellar1(MDODiscipline):
    """The discipline to compute the coupling variable :math:`y_1`."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        input_data = get_inputs((X_LOCAL, X_SHARED, Y_2))
        self.input_grammar.update_from_data(input_data)
        self.output_grammar.update_from_data(get_inputs([Y_1]))
        self.default_inputs = input_data

    def _run(self) -> None:
        x_local, x_shared, y_2 = self.get_inputs_by_name([X_LOCAL, X_SHARED, Y_2])
        if WITH_2D_ARRAY:
            x_shared = x_shared[0]

        # functional form
        y_1_out = self.compute_y_1(x_local, x_shared, y_2)
        self.store_local_data(y_1=y_1_out)

    @staticmethod
    def compute_y_1(
        x_local: complex,
        x_shared: ndarray,
        y_2: complex,
    ) -> complex:
        """Evaluate the first coupling equation in functional form.

        Args:
            x_local: The design variables local to first discipline.
            x_shared: The shared design variables.
            y_2: The coupling variable coming from the second discipline.

        Returns:
            The value of the coupling variable :math:`y_1`.
        """
        return sqrt(x_shared[0] ** 2 + x_shared[1] + x_local - 0.2 * y_2)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(inputs, outputs)
        x_local, x_shared, y_2 = self.get_inputs_by_name([X_LOCAL, X_SHARED, Y_2])
        if WITH_2D_ARRAY:
            x_shared = x_shared[0]

        inv_denom = 1.0 / (self.compute_y_1(x_local, x_shared, y_2))
        self.jac[Y_1] = {}
        self.jac[Y_1][X_LOCAL] = atleast_2d(array([0.5 * inv_denom]))
        self.jac[Y_1][X_SHARED] = atleast_2d(
            array([x_shared[0] * inv_denom, 0.5 * inv_denom])
        )
        self.jac[Y_1][Y_2] = atleast_2d(array([-0.1 * inv_denom]))


class Sellar2(MDODiscipline):
    """The discipline to compute the coupling variable :math:`y_2`."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        input_data = get_inputs((X_SHARED, Y_1))
        self.input_grammar.update_from_data(input_data)
        self.output_grammar.update_from_data(get_inputs([Y_2]))
        self.default_inputs = input_data

    def _run(self) -> None:
        x_shared, y_1 = self.get_inputs_by_name([X_SHARED, Y_1])
        if WITH_2D_ARRAY:
            x_shared = x_shared[0]
        self.store_local_data(y_2=self.compute_y_2(x_shared, y_1))

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(inputs, outputs)
        y_1 = self.get_inputs_by_name(Y_1)
        self.jac[Y_2] = {}
        self.jac[Y_2][X_LOCAL] = zeros((1, 1))
        self.jac[Y_2][X_SHARED] = ones((1, 2))
        if y_1.real < 0.0:
            self.jac[Y_2][Y_1] = -ones((1, 1))
        elif y_1 == 0.0:
            self.jac[Y_2][Y_1] = zeros((1, 1))
        else:
            self.jac[Y_2][Y_1] = ones((1, 1))

    @staticmethod
    def compute_y_2(
        x_shared: ndarray,
        y_1: complex,
    ) -> complex:
        """Evaluate the second coupling equation in functional form.

        Args:
            x_shared: The shared design variables.
            y_1: The coupling variable coming from the first discipline.

        Returns:
            The value of the coupling variable :math:`y_2`.
        """
        out = x_shared[0] + x_shared[1]
        if y_1.real == 0:
            y_2 = out
        elif y_1.real > 0:
            y_2 = y_1 + out
        else:
            y_2 = -y_1 + out
        return y_2
