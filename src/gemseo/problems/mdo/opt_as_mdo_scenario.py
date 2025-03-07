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
r"""Make a monodisciplinary optimization problem multidisciplinary.

The literature on problems to benchmark optimization algorithms is much richer
than the literature on problems to benchmark MDO algorithms.
This difference is even greater in the case of MDO under uncertainties.

Faced with this limitation,
the :class:`.OptAsMDOScenario` allows the user to
rewrite a monodisciplinary optimization problem into an MDO problem.

The original discipline

- must have as outputs one or more objectives,
- may have as outputs one or more constraints,
- may have as outputs one or more observables,,
- must have as inputs at least three design variables.

The MDO problem will include :math:`2+N` disciplines,
namely

- :math:`N` strongly coupled disciplines
  computing the values of some coupling variables
  from the values of the other coupling variables
  and the values of the design variables,
- one weakly coupled discipline,
  called *link discipline*,
  computing the values of the design variables in the original optimization problem
  from
  the values of the design and coupling variables in the MDO problem,
- the original discipline.

This scenario will also modify the design space passed as argument
by renaming
the first design variable as :math:`x_0`,
the second design variable as :math:`x_1`,
and so on.
Then,
:math:`x_0` will be the global design variable
and :math:`x_i` will be the local design variable
specific to the :math:`i`-th discipline.

In other words,
an optimization problem of the form

.. math::

   \min_{z\in Z} f(z_0,z_1,\ldots,z_N) \text{ s.t. } g(z_0,z_1,\ldots,z_N) \leq 0

will be transformed into an MDO problem of the form

.. math::

   \min_{x\in Z, y\in Y} F(x_0,x_1,\ldots,x_N,y_1,\ldots,y_N)
   \text{ s.t. } G(x_0,x_1,\ldots,x_N,y_1,\ldots,y_N) \leq 0

where
:math:`y_i=h_i(x_0,x_i,y_{-i})` is the coupling variable
outputted by the :math:`i`-th strongly coupled discipline
and where :math:`F = f \circ L` and :math:`G = f \circ L`,
with :math:`L` the link discipline.

The function :math:`h_i` is defined as

.. math::

   h_i(x_0,x_i,y_{-i})
   =a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j

where :math:`a_i`, :math:`D_{i,0}`, :math:`D_{i,i}`
and :math:`\left(C_{i,j}\right)_{j=1\atop j\neq i}`
are realizations of independent random variables
uniformly distributed between 0 and 1,
and the link discipline :math:`L` is defined as

.. math::

   z = L(x, y) = x + y - c(x)

where :math:`c` is the implicit function
such that :math:`c_i(x)=h_i(x_0,x_i,c_{-i}(x))` for all :math:`i\in\{1,\ldots,N\}`.

If the original discipline is analytically differentiable,
so are the objective and constraint functions of this MDO problem.

By default,
this scenario applies
the technique proposed by Amine Aziz-Alaoui in his doctoral thesis
to the case of linear coupling and link disciplines,
using the previous expressions of :math:`h_1,\ldots,h_N` and :math:`L`.
More advanced disciplines could be imagined,
using the arguments ``coupling_equations`` and ``link_discipline``.

:ref:`This example <sphx_glr_examples_mdo_plot_opt_as_mdo.py>`
from the documentation
illustrates this feature.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from numpy import array
from numpy import hstack
from numpy import ones

from gemseo.core.discipline.discipline import Discipline
from gemseo.problems.mdo.scalable.parametric.scalable_problem import ScalableProblem
from gemseo.scenarios.mdo_scenario import MDOScenario

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.typing import RealArray


class BaseLinkDiscipline(Discipline):
    r"""The base class for the link discipline :math:`c`.

    This discipline computes
    the values of the design variables :math:`z_0,z_1,\ldots,z_N`
    of the original optimization problem
    from
    the values of the design variables :math:`x_0,x_1,\ldots,x_N`
    and coupling variables :math:`y_0,y_1,\ldots,y_N`
    of the MDO problem.
    """

    _differentiate_mda_analytically: Callable[[RealArray], RealArray] | None
    """The function differentiating the MDA analytically at a given design point.

    If ``None``, the discipline is not differentiable.
    """

    _n_strongly_coupled_disciplines: int
    """The number of strongly coupled disciplines."""

    _perform_mda_analytically: Callable[[RealArray], RealArray]
    """The function performing the MDA analytically at a given design point."""

    _x_names: tuple[str, ...]
    """The names of the design variables in the MDO problem."""

    _y_names: tuple[str, ...]
    """The names of the coupling variables in the MDO problem."""

    _differentiate_mda_analytically: Callable[[RealArray], RealArray] | None
    """The function differentiating the MDA analytically at a given design point.

    If ``None``, the discipline is not differentiable.
    """

    _n_strongly_coupled_disciplines: int
    """The number of strongly coupled disciplines."""

    _original_x_names: list[str]
    """The names of the design variables in the original problem."""

    _perform_mda_analytically: Callable[[RealArray], RealArray]
    """The function performing the MDA analytically at a given design point."""

    _x_names: tuple[str, ...]
    """The names of the design variables in the MDO problem."""

    _y_names: tuple[str, ...]
    """The names of the coupling variables in the MDO problem."""

    def __init__(
        self,
        design_space: DesignSpace,
        perform_mda_analytically: Callable[[RealArray], RealArray],
        differentiate_mda_analytically: Callable[[RealArray], RealArray] | None = None,
    ) -> None:
        r"""
        Args:
            design_space: The design space of the original optimization problem.
            perform_mda_analytically: The function :math:`c`
                performing the MDA analytically
                at a given design point :math:`x=(x_0,x_1,\ldots,x_N)`.
            differentiate_mda_analytically: The function :math:`\nabla c`
                differentiating the MDA analytically
                at a given design point :math:`x=(x_0,x_1,\ldots,x_N)`.
                If ``None``, the discipline will not be differentiable.

        """  # noqa: D205 D212
        super().__init__(name="L")
        n_strongly_coupled_disciplines = len(design_space) - 1
        self._n_strongly_coupled_disciplines = n_strongly_coupled_disciplines

        # Names of the design variables and coupling variables in the MDO problem:
        original_x_names = design_space.variable_names
        for i, original_x_name in enumerate(original_x_names):
            design_space.rename_variable(original_x_name, f"x_{i}")
        self._x_names = tuple(design_space.variable_names)
        self._y_names = tuple(
            f"y_{i}" for i in range(1, n_strongly_coupled_disciplines + 1)
        )
        self.input_grammar.update_from_names([*self._x_names, *self._y_names])

        # Names of the design variables in the original optimization problem:
        self._original_x_names = original_x_names
        self.output_grammar.update_from_names(original_x_names)

        self._perform_mda_analytically = perform_mda_analytically
        self._differentiate_mda_analytically = differentiate_mda_analytically

    def _run(self, input_data: Mapping[str, RealArray]) -> dict[str, RealArray]:
        # The values of the design variables.
        x = tuple(input_data[name] for name in self._x_names)

        # The values of the coupling variables from a numerical MDA.
        approximated_y = tuple(input_data[name] for name in self._y_names)

        # The values of the coupling variables from the analytical MDA.
        expected_y = self._perform_mda_analytically(hstack(x))

        return self._run_from_intermediate_data(x, approximated_y, expected_y)

    @abstractmethod
    def _run_from_intermediate_data(
        self,
        x: tuple[RealArray, ...],
        approximated_y: tuple[RealArray, ...],
        expected_y: RealArray,
    ) -> dict[str, RealArray]:
        r"""Compute the design variable values in the original optimization problem.

        Args:
            x: The values of the design variables :math:`x_0,x_1,\ldots,x_N`
                in the MDO problem.
            approximated_y: The values of the coupling variables :math:`y_1,\ldots,y_N`
                from a numerical MDA.
            expected_y: The values of the coupling variables
                :math:`c_1(x),\ldots,c_N(x)`
                from the analytical MDA.

        Returns:
            The values of the design variables :math:`z_0,z_1,\ldots,z_N`
                in the original optimization problem.
        """

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        if self._differentiate_mda_analytically is None:
            return

        self._init_jacobian(input_names, output_names)
        self.jac[self._original_x_names[0]][self._x_names[0]] = ones((1, 1))

        # The derivatives of the coupling variables from the analytical MDA.
        input_data = self.io.get_input_data(with_namespaces=False)
        x = tuple(input_data[name] for name in self._x_names)
        d_expected_y_dx = self._differentiate_mda_analytically(hstack(x))
        self._compute_jacobian_from_intermediate_data(x, d_expected_y_dx)

    @abstractmethod
    def _compute_jacobian_from_intermediate_data(
        self, x: tuple[RealArray, ...], d_expected_y_dx: RealArray
    ) -> None:
        r"""Compute the Jacobian :attr:`.jac`.

        Args:
            x: The values of the design variables :math:`x_0,x_1,\ldots,x_N`
                in the MDO problem.
            d_expected_y_dx: The derivatives of the coupling variables
                from the analytical MDA.
        """


class LinearLinkDiscipline(BaseLinkDiscipline):
    r"""A linear link discipline of the form :math:`L(x, y) = x + y - c(x)`.

    In more details,
    :math:`L_0(x, y) = x_0`
    and :math:`L_i(x, y) = x_i + y_i - c_i(x)` for :math:`i\in\{1,\ldots,N\}`.
    """

    def _run_from_intermediate_data(
        self,
        x: tuple[RealArray, ...],
        approximated_y: tuple[RealArray, ...],
        expected_y: RealArray,
    ) -> dict[str, RealArray]:
        output_data = {self._original_x_names[0]: x[0]}
        for i in range(1, self._n_strongly_coupled_disciplines + 1):
            original_x_i_name = self._original_x_names[i]
            output_data[original_x_i_name] = (
                x[i] + approximated_y[i - 1] - expected_y[i - 1]
            )

        return output_data

    def _compute_jacobian_from_intermediate_data(
        self, x: tuple[RealArray, ...], d_expected_y_dx: RealArray
    ) -> None:
        for i in range(1, self._n_strongly_coupled_disciplines + 1):
            jac = self.jac[self._original_x_names[i]]
            jac[self._y_names[i - 1]] = ones((1, 1))
            jac[self._x_names[i]] = ones((1, 1)) - d_expected_y_dx[i - 1, i]
            for j in range(self._n_strongly_coupled_disciplines + 1):
                if j != i:
                    jac[self._x_names[j]] = -array([[d_expected_y_dx[i - 1, j]]])


class OptAsMDOScenario(MDOScenario):
    """A monodisciplinary optimization scenario made multidisciplinary."""

    def __init__(
        self,
        discipline: Discipline,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str = "",
        maximize_objective: bool = False,
        formulation_settings_model: BaseFormulationSettings | None = None,
        coupling_equations: tuple[
            Iterable[Discipline, ...],
            Callable[[RealArray], RealArray],
            Callable[[RealArray], RealArray],
        ] = (),
        link_discipline_class: type[BaseLinkDiscipline] = LinearLinkDiscipline,
        **formulation_settings: Any,
    ) -> None:
        r"""
        Args:
            discipline: The discipline
                computing the objective, constraints and observables
                from the design variables.
            design_space: The design space
                including the design variables :math:`z_0,z_1,\ldots,z_N`
                which will be replaced by :math:`x_0,x_1,\ldots,x_N` respectively
                in the MDO problem.
            coupling_equations: The material
                to evaluate and solve the coupling equations,
                namely the disciplines :math:`h_1,\ldots,h_N`,
                the function :math:`c`
                and the Jacobian function :math:`\nabla c(x)`.
                If empty,
                the :math:`i`-th discipline is linear.
            link_discipline_class: The class of the link discipline.

        .. note::

           There is no naming convention
           for the input and output variables of ``discipline``.
           So,
           you can use :math:`a,b,c` in ``design_space`` instead of :math:`z_0,z_1,z_2`.
        """  # noqa: D205, D212, E501
        disciplines = create_disciplines(
            discipline, design_space, coupling_equations, link_discipline_class
        )
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            name=name,
            maximize_objective=maximize_objective,
            formulation_settings_model=formulation_settings_model,
            **formulation_settings,
        )


def create_disciplines(
    discipline: Discipline,
    design_space: DesignSpace,
    coupling_equations: tuple[
        Iterable[Discipline, ...],
        Callable[[RealArray], RealArray],
        Callable[[RealArray], RealArray],
    ],
    link_discipline_class: type[BaseLinkDiscipline],
) -> tuple[Discipline, BaseLinkDiscipline, Discipline]:
    r"""Create the disciplines to make an optimization problem multidisciplinary.

    Args:
        discipline: The discipline
            computing the objective, constraints and observables
            from the design variables.
        design_space: The design space
            including the design variables :math:`z_0,z_1,\ldots,z_N`
            which will be replaced by :math:`x_0,x_1,\ldots,x_N` respectively
            in the MDO problem.
        coupling_equations: The material
            to evaluate and solve the coupling equations,
            namely the disciplines :math:`h_1,\ldots,h_N`,
            the function :math:`c`
            and the Jacobian function :math:`\nabla c(x)`.
            If empty,
            the :math:`i`-th discipline is linear.
        link_discipline_class: The class of the link discipline.

    Returns:
        The original discipline,
        the link discipline,
        and the strongly coupled disciplines.

    Raises:
        ValueError: When the design space includes less than three variables
            or when a design variable is not scalar.
    """
    n_variables = len(design_space)
    if n_variables < 3:
        msg = (
            "The design space must have at least three scalar design variables; "
            f"got {n_variables}."
        )
        raise ValueError(msg)

    if design_space.dimension != n_variables:
        msg = "The design space must include scalar variables only."
        raise ValueError(msg)

    if coupling_equations:
        strongly_coupled_disciplines, compute_y, differentiate_y = coupling_equations
    else:
        scalable_problem = ScalableProblem()
        strongly_coupled_disciplines = scalable_problem.scalable_disciplines
        for i, strongly_coupled_discipline in enumerate(strongly_coupled_disciplines):
            strongly_coupled_discipline.name = f"D{i + 1}"

        compute_y = scalable_problem.compute_y
        differentiate_y = scalable_problem.differentiate_y

    link_discipline = link_discipline_class(design_space, compute_y, differentiate_y)
    return discipline, link_discipline, *strongly_coupled_disciplines
