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
"""Make a monodisciplinary optimization problem multidisciplinary."""

from __future__ import annotations

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
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class OptAsMDOScenario(MDOScenario):
    r"""A monodisciplinary optimization scenario made multidisciplinary.

    The literature on problems to benchmark optimization algorithms is much richer
    than the literature on problems to benchmark MDO algorithms.
    This difference is even greater in the case of MDO under uncertainties.

    Faced with this limitation,
    this scenario allows the user to
    rewrite a mono-disciplinary optimization problem into an MDO problem.

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
    uniformly distribution between 0 and 1,
    and the link discipline :math:`L` is defined as

    .. math::

       z = L(x, y) = x + y - c(x)

    where :math:`c` is the implicit function
    such that :math:`c_i(x)=h_i(x_0,x_i,c_{-i}(x))` for all :math:`i\in\{1,\ldots,N\}`.

    If the original discipline is analytically differentiable,
    so are the objective and constraint functions of this MDO problem.

    This scenario applies
    the technique proposed by Amine Aziz-Alaoui in his doctoral thesis
    to the case of linear coupling and link disciplines.
    More advanced disciplines could be imagined.

    :ref:`This example <sphx_glr_examples_mdo_plot_opt_as_mdo.py>`
    from the documentation
    illustrates this feature.
    """

    def __init__(
        self,
        discipline: Discipline,
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        name: str = "",
        maximize_objective: bool = False,
        formulation_settings_model: BaseFormulationSettings | None = None,
        **formulation_settings: Any,
    ) -> None:
        """
        Args:
            discipline: The discipline
                computing the objective, constraints and observables
                from the design variables.
        """  # noqa: D205 D212
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

        scalable_problem = ScalableProblem()
        strongly_coupled_disciplines = scalable_problem.scalable_disciplines
        for i, strongly_coupled_discipline in enumerate(strongly_coupled_disciplines):
            strongly_coupled_discipline.name = f"D{i + 1}"

        link_discipline = _LinkDiscipline(
            design_space, scalable_problem.compute_y, scalable_problem.differentiate_y
        )

        super().__init__(
            (discipline, link_discipline, *strongly_coupled_disciplines),
            objective_name,
            design_space,
            name=name,
            maximize_objective=maximize_objective,
            formulation_settings_model=formulation_settings_model,
            **formulation_settings,
        )


class _LinkDiscipline(Discipline):
    """A link discipline.

    This discipline computes
    the values of the design variables in the original optimization problem
    from
    the values of the design and coupling variables in the MDO problem.
    It is analytically differentiable.
    """

    __differentiate_mda_analytically: Callable[[RealArray], RealArray] | None
    """The function differentiating the MDA analytically at a given design point.

    If ``None``, the discipline is not differentiable.
    """

    __n_strongly_coupled_disciplines: int
    """The number of strongly coupled disciplines."""

    __original_x_names: list[str]
    """The names of the design variables in the original problem."""

    __perform_mda_analytically: Callable[[RealArray], RealArray]
    """The function performing the MDA analytically at a given design point."""

    __x_names: tuple[str, ...]
    """The names of the design variables in the MDO problem."""

    __y_names: tuple[str, ...]
    """The names of the coupling variables in the MDO problem."""

    def __init__(
        self,
        design_space: DesignSpace,
        perform_mda_analytically: Callable[[RealArray], RealArray],
        differentiate_mda_analytically: Callable[[RealArray], RealArray] | None = None,
    ) -> None:
        """
        Args:
            design_space: The design space of the original optimization problem,
                whose first design variable will be the shared global design variable
                of the MDO problem.
            perform_mda_analytically: The function
                performing the MDA analytically at a given design point.
            differentiate_mda_analytically: The function
                differentiating the MDA analytically at a given design point.
                If ``None``, the discipline will not be differentiable.

        """  # noqa: D205 D212
        super().__init__(name="L")
        n_strongly_coupled_disciplines = len(design_space) - 1
        self.__n_strongly_coupled_disciplines = n_strongly_coupled_disciplines

        # Names of the design variables and coupling variables in the MDO problem:
        original_x_names = design_space.variable_names
        for i, original_x_name in enumerate(original_x_names):
            design_space.rename_variable(original_x_name, f"x_{i}")
        self.__x_names = tuple(design_space.variable_names)
        self.__y_names = tuple(
            f"y_{i}" for i in range(1, n_strongly_coupled_disciplines + 1)
        )
        self.input_grammar.update_from_names([*self.__x_names, *self.__y_names])

        # Names of the design variables in the original optimization problem:
        self.__original_x_names = original_x_names
        self.output_grammar.update_from_names(original_x_names)

        self.__perform_mda_analytically = perform_mda_analytically
        self.__differentiate_mda_analytically = differentiate_mda_analytically

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # The values of the design variables.
        x = tuple(input_data[name] for name in self.__x_names)

        # The values of the coupling variables from a numerical MDA.
        approximated_y = tuple(input_data[name] for name in self.__y_names)

        # The values of the coupling variables from the analytical MDA.
        expected_y = self.__perform_mda_analytically(hstack(x))

        output_data = {self.__original_x_names[0]: x[0]}
        for i in range(1, self.__n_strongly_coupled_disciplines + 1):
            original_x_i_name = self.__original_x_names[i]
            output_data[original_x_i_name] = (
                x[i] + approximated_y[i - 1] - expected_y[i - 1]
            )

        return output_data

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        if self.__differentiate_mda_analytically is None:
            return

        self._init_jacobian(input_names, output_names)
        input_data = self.io.get_input_data(with_namespaces=False)
        self.jac[self.__original_x_names[0]][self.__x_names[0]] = ones((1, 1))

        # The derivatives of the coupling variables from the analytical MDA.
        x = tuple(input_data[name] for name in self.__x_names)
        d_expected_y_dx = self.__differentiate_mda_analytically(hstack(x))

        for i in range(1, self.__n_strongly_coupled_disciplines + 1):
            jac = self.jac[self.__original_x_names[i]]
            jac[self.__y_names[i - 1]] = ones((1, 1))
            jac[self.__x_names[i]] = ones((1, 1)) - d_expected_y_dx[i - 1, i]
            for j in range(self.__n_strongly_coupled_disciplines + 1):
                if j != i:
                    jac[self.__x_names[j]] = -array([[d_expected_y_dx[i - 1, j]]])
