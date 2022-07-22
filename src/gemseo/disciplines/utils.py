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
"""A set of functions to handle disciplines."""
from __future__ import annotations

from typing import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.core.discipline import MDODiscipline


def __get_all_disciplines(
    disciplines: Iterable[MDODiscipline],
    skip_scenarios: bool,
) -> list[MDODiscipline]:
    """Return the non-scenario disciplines or also the disciplines of the scenario ones.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If True, skip the :class:`.Scenario` objects.
            Otherwise, return their disciplines.

    Returns:
        The non-scenario disciplines
        or also the disciplines of the scenario ones if any and ``skip_scenario=False``.
    """
    non_scenarios = [disc for disc in disciplines if not disc.is_scenario()]
    scenarios = [disc for disc in disciplines if disc.is_scenario()]

    if skip_scenarios:
        return non_scenarios

    disciplines_in_scenarios = list(
        set.union(*(set(scenario.disciplines) for scenario in scenarios))
    )
    return disciplines_in_scenarios + non_scenarios


def get_all_inputs(
    disciplines: Iterable[MDODiscipline],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the input names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If True, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the inputs.
    """
    return list(
        set.union(
            *(
                set(discipline.get_input_data_names())
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )


def get_all_outputs(
    disciplines: Iterable[MDODiscipline],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the output names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If True, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the outputs.
    """
    return list(
        set.union(
            *(
                set(discipline.get_output_data_names())
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )
