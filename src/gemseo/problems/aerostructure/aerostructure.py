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
#    INITIAL AUTHORS - initial API and implementation and/or
#    initial documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""
The aerostructure MDO problem
*****************************

The **aerostructure** module implements all :class:`.MDODiscipline`
included in the Aerostructure problem:

.. math::

   \text{OVERALL AIRCRAFT DESIGN} = \left\{
   \begin{aligned}
   &\text{minimize }\text{range}(\text{thick\_airfoils},
   \text{thick\_panels}, \text{sweep}) = 8
   \times10^{11}\times\text{lift}\times\text{mass}/\text{drag} \\
   &\text{with respect to }\text{thick\_airfoils},\,\text{thick\_panels},
   \,\text{sweep} \\
   &\text{subject to }\\
   & \text{rf}-0.5 = 0\\
   & \text{lift}-0.5 \leq 0
   \end{aligned}\right.

where

.. math::

       \text{AERODYNAMICS} = \left\{
           \begin{aligned}
        &\text{drag}=0.1\times((\text{sweep}/360)^2 + 200 +
        \text{thick\_airfoils}^2 - \text{thick\_airfoils} -
         4\times\text{displ})\\
        &\text{forces}=10\times\text{sweep} +
        0.2\times\text{thick\_airfoils}-0.2\times\text{displ}\\
        &\text{lift}=(\text{sweep} + 0.2\times\text{thick\_airfoils}-
        2\times\text{displ})/3000
           \end{aligned}
           \right.

and

.. math::

       \text{STRUCTURE} = \left\{
           \begin{aligned}
        &\text{mass}=4000\times(\text{sweep}/360)^3 + 200000 +
        100\times\text{thick\_panels} + 200\times\text{forces}\\
        &\text{rf}=3\times\text{sweep} - 6\times\text{thick\_panels} +
        0.1\times\text{forces} + 55\\
        &\text{displ}=2\times\text{sweep} + 3\times\text{thick\_panels} -
        2\times\text{forces}
           \end{aligned}
           \right.



"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super

from future import standard_library
from numpy import array, atleast_2d, complex128, ones

from gemseo.core.discipline import MDODiscipline

standard_library.install_aliases()


def get_inputs(names=None):
    """Generate initial solution

    :param names: input names (Default value = None)
    :type names: list(str)
    """
    inputs = {
        "drag": ones((1), dtype=complex128),
        "forces": ones((1), dtype=complex128),
        "lift": ones((1), dtype=complex128),
        "mass": ones((1), dtype=complex128),
        "displ": ones((1), dtype=complex128),
        "sweep": ones((1), dtype=complex128),
        "thick_airfoils": ones((1), dtype=complex128),
        "thick_panels": ones((1), dtype=complex128),
        "reserve_fact": ones((1), dtype=complex128),
    }
    if names is None:
        return inputs
    return {name: inputs.get(name) for name in names}


class Mission(MDODiscipline):
    """**Aerostructure** is the :class:`.MDODiscipline`
    implementing the computation of the Aerostructure's objective
    and constraints discipline."""

    def __init__(self, r_val=0.5, lift_val=0.5):
        """Constructor"""
        super(Mission, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = get_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        self.r_val = r_val
        self.lift_val = lift_val

    def _run(self):
        """Defines the execution of the process, given that data
        has been checked.
        Compute the outputs (= objective value and constraints at system level)
        of the aerostructure analytical problem.
        """
        lift, mass, drag, reserve_fact = self.get_inputs_by_name(
            ["lift", "mass", "drag", "reserve_fact"]
        )
        obj = array([self.compute_range(lift, mass, drag)], dtype=complex128)
        c_lift = array([self.c_lift(lift, self.lift_val)], dtype=complex128)
        c_rf = array([self.c_rf(reserve_fact)], dtype=complex128)
        self.store_local_data(range=obj, c_lift=c_lift, c_rf=c_rf)

    @staticmethod
    def compute_range(lift, mass, drag):
        """Objective function: :math:`range=8.10^{11}*lift/(mass*drag)`

        :param lift: lift
        :type lift: ndarray
        :param mass: mass
        :type mass: ndarray
        :param drag: drag
        :type drag: ndarray
        :returns: Objective value (range)
        :rtype: float
        """
        return 8e11 * lift[0] / (mass[0] * drag[0])

    @staticmethod
    def c_lift(lift, lift_val=0.5):
        """lift constraint: :math:`lift-0.5`

        :param lift: lift
        :type lift: ndarray
        :returns: Value of the lift constraint
        """
        return lift[0] - lift_val

    @staticmethod
    def c_rf(reserve_fact, rf_val=0.5):
        """reserve_fact constraint: :math:`rf-0.5`

        :param reserve_fact: reserve_fact
        :type reserve_fact: ndarray
        :param rf_val: rf_val
        :type rf_val: ndarray
        :returns: Value of the rf constraint
        """
        return reserve_fact[0] - rf_val

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian of [range, c_rf, c_lift]
        w.r.t [lift, mass, drag, rf]

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        drag, lift, mass = self.get_inputs_by_name(["drag", "lift", "mass"])
        self.jac["c_lift"]["lift"] = ones((1, 1))
        self.jac["c_rf"]["reserve_fact"] = ones((1, 1))
        self.jac["range"]["lift"] = atleast_2d(array([8e11 / (mass[0] * drag[0])]))
        self.jac["range"]["mass"] = atleast_2d(
            array([-8e11 * lift[0] / (mass[0] ** 2 * drag[0])])
        )
        self.jac["range"]["drag"] = atleast_2d(
            array([-8e11 * lift[0] / (mass[0] * drag[0] ** 2)])
        )


class Aerodynamics(MDODiscipline):
    """**Aerodynamics** is the :class:`.MDODiscipline`
    implementing the computation of the Aerodynamics' equations:
    [drag, forces, lift] = f(sweep, thick_airfoils, displ)."""

    def __init__(self):
        super(Aerodynamics, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = get_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Defines the execution of the process, given that
        data has been checked.
        Solve a coupling equation in functional form and
        compute coupling variable
        """
        sweep, thick_airfoils, displ = self.get_inputs_by_name(
            ["sweep", "thick_airfoils", "displ"]
        )
        drag_out = array([self.compute_drag(sweep, thick_airfoils, displ)])
        lift_out = array([self.compute_lift(sweep, thick_airfoils, displ)])
        forces_out = array([self.compute_forces(sweep, thick_airfoils, displ)])
        self.store_local_data(drag=drag_out, forces=forces_out, lift=lift_out)

    @staticmethod
    def compute_drag(sweep, thick_airfoils, displ):
        """Compute the coupling
        :math:`drag=0.1*((sweep/360)^2 + 200 + thick_airfoils^2
        - thick_airfoils - 4*displ)`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_airfoils: thick_airfoils
        :type thick_airfoils: ndarray
        :param displ: displ
        :type displ: ndarray
        :returns: drag
        :rtype: float
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
        """Compute the coupling
        :math:`forces=10*sweep + 0.2*thick_airfoils-0.2*displ`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_airfoils: thick_airfoils
        :type thick_airfoils: ndarray
        :param displ: displ
        :type displ: ndarray
        :returns: drag
        :rtype: float
        """
        return 10 * sweep[0] + 0.2 * thick_airfoils[0] - 0.2 * displ[0]

    @staticmethod
    def compute_lift(sweep, thick_airfoils, displ):
        """Compute the coupling
        :math:`lift=(sweep + 0.2*thick_airfoils-2.*displ)/3000.`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_airfoils: thick_airfoils
        :type thick_airfoils: ndarray
        :param displ: displ
        :type displ: ndarray
        :returns: drag
        :rtype: float
        """
        return (sweep[0] + 0.2 * thick_airfoils[0] - 2.0 * displ[0]) / 3000.0

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the Jacobian of [drag, forces, lift]

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        sweep, thick_airfoils = self.get_inputs_by_name(["sweep", "thick_airfoils"])
        self.jac["drag"]["sweep"] = atleast_2d(
            array([0.1 * 2.0 * sweep[0] / 360.0 ** 2.0])
        )
        self.jac["drag"]["thick_airfoils"] = atleast_2d(
            array([0.1 * (2.0 * thick_airfoils[0] - 1.0)])
        )
        self.jac["drag"]["displ"] = atleast_2d(0.1 * array([-4.0]))
        self.jac["forces"]["sweep"] = atleast_2d(array([10.0]))
        self.jac["forces"]["thick_airfoils"] = atleast_2d(array([0.2]))
        self.jac["forces"]["displ"] = atleast_2d(array([-0.2]))
        self.jac["lift"]["sweep"] = atleast_2d(array([1.0 / 3000.0]))
        self.jac["lift"]["thick_airfoils"] = atleast_2d(array([0.2 / 3000.0]))
        self.jac["lift"]["displ"] = atleast_2d(array([-2.0 / 3000.0]))


class Structure(MDODiscipline):
    """**Structure** is the :class:`.MDODiscipline`
    implementing the computation of the Structure' equations:
    [mass, rf, displ] = f(sweep, thick_panels, forces)"""

    def __init__(self):
        super(Structure, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = get_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Defines the execution of the process, given
        that data has been checked.
        Solve a coupling equation in functional form and compute coupling
        variables.
        """
        sweep, thick_panels, forces = self.get_inputs_by_name(
            ["sweep", "thick_panels", "forces"]
        )
        mass_out = array([self.compute_mass(sweep, thick_panels, forces)])
        rf_out = array([self.compute_rf(sweep, thick_panels, forces)])
        displ_out = array([self.compute_displ(sweep, thick_panels, forces)])
        self.store_local_data(mass=mass_out, reserve_fact=rf_out, displ=displ_out)

    @staticmethod
    def compute_mass(sweep, thick_panels, forces):
        """Compute the coupling
        :math:`mass=4000*(sweep/360)^3 + 200000 + 100*thick_panels
        + 200.0*forces`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_panels: thick_panels
        :type thick_panels: ndarray
        :param forces: forces
        :type forces: ndarray
        :returns: mass
        :rtype: float
        """
        return (
            4000 * (sweep[0] / 360) ** 3
            + 200000
            + 100 * thick_panels[0]
            + 200.0 * forces[0]
        )

    @staticmethod
    def compute_rf(sweep, thick_panels, forces):
        """Compute the coupling
        :math:`rf=-3*sweep - 6*thick_panels + 0.1*forces + 55`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_panels: thick_panels
        :type thick_panels: ndarray
        :param forces: forces
        :type forces: ndarray
        :returns: RF
        :rtype: float
        """
        return -3 * sweep[0] - 6 * thick_panels[0] + 0.1 * forces[0] + 55

    @staticmethod
    def compute_displ(sweep, thick_panels, forces):
        """Compute the coupling
        :math:`displ=2*sweep + 3*thick_panels - 2.*forces`

        :param sweep: sweep
        :type sweep: ndarray
        :param thick_panels: thick_panels
        :type thick_panels: ndarray
        :param forces: forces
        :type forces: ndarray
        :returns: displ
        :rtype: float
        """
        return 2 * sweep[0] + 3 * thick_panels[0] - 2.0 * forces[0]

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the Jacobian of [mass, rf, displ]

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        sweep = self.get_inputs_by_name("sweep")
        self.jac["mass"]["sweep"] = atleast_2d(
            array([4000.0 * 3.0 * sweep[0] ** 2 / 360.0 ** 3])
        )
        self.jac["mass"]["thick_panels"] = atleast_2d(array([100.0]))
        self.jac["mass"]["forces"] = atleast_2d(array([200.0]))
        self.jac["reserve_fact"]["sweep"] = atleast_2d(array([-3.0]))
        self.jac["reserve_fact"]["thick_panels"] = atleast_2d(array([-6.0]))
        self.jac["reserve_fact"]["forces"] = atleast_2d(array([0.1]))
        self.jac["displ"]["sweep"] = atleast_2d(array([2.0]))
        self.jac["displ"]["thick_panels"] = atleast_2d([3.0])
        self.jac["displ"]["forces"] = atleast_2d(array([-2.0]))
