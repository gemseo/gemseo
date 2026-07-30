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
"""Workflow observers for tracking execution of GEMSEO objects.

This package provides a framework for observing the lifecycle of GEMSEO objects
(scenarios, disciplines, MDA solvers, optimizers, DOE algorithms) during execution.
The observers integrate with directory managers to track execution history and
manage output directories.

Workflow observers are automatically injected into observed classes via the
`_WorkflowObserverInjector` when the directory manager is enabled.
"""
