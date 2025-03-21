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
#    INITIAL AUTHORS - initial API and implementation and/or
#    initial documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""The aerostructure MDO problem.

The **aerostructure** module implements all :class:`.Discipline`
included in the Aerostructure problem:

.. math::

   \text{OVERALL AIRCRAFT DESIGN} = \left\{
   \begin{aligned}
   &\text{minimize }range(thick\_airfoils,
   thick\_panels, sweep) = 8
   \times10^{11}\times lift\times mass/drag \\
   &\text{with respect to }thick\_airfoils,\,thick\_panels,
   \,sweep \\
   &\text{subject to }\\
   & rf-0.5 = 0\\
   & lift-0.5 \leq 0
   \end{aligned}\right.

where

.. math::

       \text{AERODYNAMICS} = \left\{
           \begin{aligned}
        &drag=0.1\times((sweep/360)^2 + 200 +
        thick\_airfoils^2 - thick\_airfoils -
         4\times displ)\\
        & forces=10\times sweep +
        0.2\times thick\_airfoils-0.2\times displ\\
        & lift=(sweep + 0.2\times thick\_airfoils-
        2\times displ)/3000
           \end{aligned}
           \right.

and

.. math::

       \text{STRUCTURE} = \left\{
           \begin{aligned}
        &mass =4000\times(sweep/360)^3 + 200000 +
        100\times thick\_panels + 200\times forces\\
        &rf=3\times sweep - 6\times thick\_panels +
        0.1\times forces + 55\\
        &displ=2\times sweep  + 3\times thick\_panels -
        2\times forces
           \end{aligned}
           \right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import atleast_2d
from numpy import complex128
from numpy import ones

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


def get_inputs(*names: str):
    """Generate initial solution.

    Args:
        *names: The names of the variables.

    Returns:
        An initial design solution.
    """
    inputs = {
        "drag": ones(1, dtype=complex128),
        "forces": ones(1, dtype=complex128),
        "lift": ones(1, dtype=complex128),
        "mass": ones(1, dtype=complex128),
        "displ": ones(1, dtype=complex128),
        "sweep": ones(1, dtype=complex128),
        "thick_airfoils": ones(1, dtype=complex128),
        "thick_panels": ones(1, dtype=complex128),
        "reserve_fact": ones(1, dtype=complex128),
    }
    if not names:
        return inputs
    return {name: inputs.get(name) for name in names}


class Mission(Discipline):
    """The mission discipline of the aerostructure use case.

    Compute the objective and the constraints.
    """

    auto_detect_grammar_files = True

    def __init__(self, r_val: float = 0.5, lift_val: float = 0.5) -> None:
        """
        Args:
            r_val: The threshold to compute the reserve factor constraint.
            lift_val: The threshold to compute the lift constraint.
        """  # noqa: D205 D212
        super().__init__()
        self.io.input_grammar.defaults = get_inputs(
            "lift", "mass", "drag", "reserve_fact"
        )
        self.r_val = r_val
        self.lift_val = lift_val

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        obj = array(
            [
                self.compute_range(
                    input_data["lift"], input_data["mass"], input_data["drag"]
                )
            ],
            dtype=complex128,
        )
        c_lift = array(
            [self.c_lift(input_data["lift"], self.lift_val)], dtype=complex128
        )
        c_rf = array([self.c_rf(input_data["reserve_fact"])], dtype=complex128)
        return {"range": obj, "c_lift": c_lift, "c_rf": c_rf}

    @staticmethod
    def compute_range(lift, mass, drag) -> float:
        """Compute the objective function: :math:`range=8.10^{11}*lift/(mass*drag)`.

        Args:
            lift: The lift.
            mass: The mass.
            drag: The drag.

        Returns:
            The range.
        """
        return 8e11 * lift[0] / (mass[0] * drag[0])

    @staticmethod
    def c_lift(lift, lift_val: float = 0.5):
        """Compute the lift constraint: :math:`lift-0.5`.

        Args:
            lift: The lift.
            lift_val: The threshold for the lift constraint.

        Returns:
            The value of the lift constraint.
        """
        return lift[0] - lift_val

    @staticmethod
    def c_rf(reserve_fact, rf_val: float = 0.5):
        """Compute the reserve factor constraint: :math:`rf-0.5`.

        Args:
            reserve_fact: The reserve factor.
            rf_val: The threshold for the reserve factor constraint.

        Returns:
            The value of the reserve factor constraint.
        """
        return reserve_fact[0] - rf_val

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # Initialize all matrices to zeros
        self._init_jacobian(input_names, output_names)
        local_data = self.io.data
        drag = local_data["drag"]
        lift = local_data["lift"]
        mass = local_data["mass"]
        self.jac["c_lift"]["lift"] = ones((1, 1))
        self.jac["c_rf"]["reserve_fact"] = ones((1, 1))
        self.jac["range"]["lift"] = atleast_2d(array([8e11 / (mass[0] * drag[0])]))
        self.jac["range"]["mass"] = atleast_2d(
            array([-8e11 * lift[0] / (mass[0] ** 2 * drag[0])])
        )
        self.jac["range"]["drag"] = atleast_2d(
            array([-8e11 * lift[0] / (mass[0] * drag[0] ** 2)])
        )


class Aerodynamics(Discipline):
    """The aerodynamics discipline of the aerostructure use case.

    Evaluate: ``[drag, forces, lift] = f(sweep, thick_airfoils, displ)``.
    """

    auto_detect_grammar_files = True

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.io.input_grammar.defaults = get_inputs("sweep", "thick_airfoils", "displ")

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        sweep = input_data["sweep"]
        thick_airfoils = input_data["thick_airfoils"]
        displ = input_data["displ"]
        drag_out = array([self.compute_drag(sweep, thick_airfoils, displ)])
        lift_out = array([self.compute_lift(sweep, thick_airfoils, displ)])
        forces_out = array([self.compute_forces(sweep, thick_airfoils, displ)])
        return {
            "drag": drag_out,
            "forces": forces_out,
            "lift": lift_out,
        }

    @staticmethod
    def compute_drag(sweep, thick_airfoils, displ) -> float:
        r"""Compute the coupling.

        :math:`drag=0.1*((sweep/360)^2 + 200 + thick\_airfoils^2
        - thick\_airfoils - 4*displ)`

        Args:
            sweep: The sweep.
            thick_airfoils: The thickness of the airfoils.
            displ: The displacement.

        Returns:
            The drag.
        """
        return 0.1 * (
            (sweep[0] / 360) ** 2
            + 200
            + thick_airfoils[0] ** 2
            - thick_airfoils[0]
            - 4 * displ[0]
        )

    @staticmethod
    def compute_forces(sweep, thick_airfoils, displ):
        r"""Compute the coupling forces.

        :math:`forces=10*sweep + 0.2*thick\_airfoils-0.2*displ`

        Args:
            sweep: The sweep.
            thick_airfoils: The thickness of the airfoils.
            displ: The displacement.

        Returns:
            The forces.
        """
        return 10 * sweep[0] + 0.2 * thick_airfoils[0] - 0.2 * displ[0]

    @staticmethod
    def compute_lift(sweep, thick_airfoils, displ):
        r"""Compute the lift.

        :math:`lift=(sweep + 0.2*thick\_airfoils-2.*displ)/3000.`.

        Args:
            sweep: The sweep.
            thick_airfoils: The thickness of the airfoils.
            displ: The displacement.

        Returns:
            The lift.
        """
        return (sweep[0] + 0.2 * thick_airfoils[0] - 2.0 * displ[0]) / 3000.0

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # Initialize all matrices to zeros
        self._init_jacobian(input_names, output_names)
        local_data = self.io.data
        self.jac["drag"]["sweep"] = atleast_2d(
            array([0.1 * 2.0 * local_data["sweep"][0] / 360.0**2.0])
        )
        self.jac["drag"]["thick_airfoils"] = atleast_2d(
            array([0.1 * (2.0 * local_data["thick_airfoils"][0] - 1.0)])
        )
        self.jac["drag"]["displ"] = atleast_2d(0.1 * array([-4.0]))
        self.jac["forces"]["sweep"] = atleast_2d(array([10.0]))
        self.jac["forces"]["thick_airfoils"] = atleast_2d(array([0.2]))
        self.jac["forces"]["displ"] = atleast_2d(array([-0.2]))
        self.jac["lift"]["sweep"] = atleast_2d(array([1.0 / 3000.0]))
        self.jac["lift"]["thick_airfoils"] = atleast_2d(array([0.2 / 3000.0]))
        self.jac["lift"]["displ"] = atleast_2d(array([-2.0 / 3000.0]))


class Structure(Discipline):
    """The structure discipline of the aerostructure use case.

    Evaluate: ``[mass, rf, displ] = f(sweep, thick_panels, forces)``.
    """

    auto_detect_grammar_files = True

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.io.input_grammar.defaults = get_inputs("sweep", "forces", "thick_panels")

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        sweep = input_data["sweep"]
        thick_panels = input_data["thick_panels"]
        forces = input_data["forces"]
        mass_out = array([self.compute_mass(sweep, thick_panels, forces)])
        rf_out = array([self.compute_rf(sweep, thick_panels, forces)])
        displ_out = array([self.compute_displ(sweep, thick_panels, forces)])
        return {
            "mass": mass_out,
            "reserve_fact": rf_out,
            "displ": displ_out,
        }

    @staticmethod
    def compute_mass(sweep, thick_panels, forces):
        r"""Compute the mass.

        :math:`mass=4000*(sweep/360)^3 + 200000 + 100*thick\_panels + 200.0*forces`.

        Args:
             sweep: The sweep.
             thick_panels: The thickness of the panels.
             forces: The forces.

        Returns:
            The mass.
        """
        return (
            4000 * (sweep[0] / 360) ** 3
            + 200000
            + 100 * thick_panels[0]
            + 200.0 * forces[0]
        )

    @staticmethod
    def compute_rf(sweep, thick_panels, forces):
        r"""Compute the coupling.

        :math:`rf=-3*sweep - 6*thick\_panels + 0.1*forces + 55`

        Args:
             sweep: The sweep.
             thick_panels: The thickness of the panels.
             forces: The forces.

        Returns:
            The reserve factor.
        """
        return -3 * sweep[0] - 6 * thick_panels[0] + 0.1 * forces[0] + 55

    @staticmethod
    def compute_displ(sweep, thick_panels, forces):
        r"""Compute the coupling.

        :math:`displ=2*sweep + 3*thick\_panels - 2.*forces`

        Args:
            sweep: The sweep.
            thick_panels: The thickness of the panels.
            forces: The forces.

        Returns:
            The displacement.
        """
        return 2 * sweep[0] + 3 * thick_panels[0] - 2.0 * forces[0]

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # Initialize all matrices to zeros
        self._init_jacobian(input_names, output_names)
        jac = self.jac["mass"]
        jac["sweep"] = atleast_2d(
            array([4000.0 * 3.0 * self.io.data["sweep"][0] ** 2 / 360.0**3])
        )
        jac["thick_panels"] = atleast_2d(array([100.0]))
        jac["forces"] = atleast_2d(array([200.0]))

        jac = self.jac["reserve_fact"]
        jac["sweep"] = atleast_2d(array([-3.0]))
        jac["thick_panels"] = atleast_2d(array([-6.0]))
        jac["forces"] = atleast_2d(array([0.1]))

        jac = self.jac["displ"]
        jac["sweep"] = atleast_2d(array([2.0]))
        jac["thick_panels"] = atleast_2d([3.0])
        jac["forces"] = atleast_2d(array([-2.0]))
