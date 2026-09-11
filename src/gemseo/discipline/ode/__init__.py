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
r"""Disciplines for ordinary differential equations (ODE).

An [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
solves an initial value problem $\dot{s}(t)=f(t,s(t))$ over a time interval,
where the right-hand side $f$ is defined by another
[Discipline][gemseo.core.discipline.discipline.Discipline].
It takes the initial state $s(t_0)$ as input
and returns the final state $s(t_f)$,
as well as the state trajectory $(s(t))_t$ when required.
The integration can also be stopped
before the end of the time interval by termination events.

Since an
[ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
is a [Discipline][gemseo.core.discipline.discipline.Discipline],
it can be used in any GEMSEO process,
e.g. in an [MDOScenario][gemseo.scenario.mdo.MDOScenario]
to optimize the initial state of a dynamical system.
"""

from __future__ import annotations
