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

import logging
from typing import TYPE_CHECKING

from gemseo.core.process_discipline import ProcessDiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import MutableSequence

    from gemseo.core.discipline import Discipline
    from gemseo.scenarios.base_scenario import BaseScenario

LOGGER = logging.getLogger(__name__)


def __get_all_disciplines(
    disciplines: Iterable[Discipline | BaseScenario],
    skip_scenarios: bool,
) -> list[Discipline]:
    """Return the non-scenario disciplines or also the disciplines of the scenario ones.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, return their disciplines.

    Returns:
        The non-scenario disciplines
        or also the disciplines of the scenario ones if any and ``skip_scenario=False``.
    """
    from gemseo.scenarios.base_scenario import BaseScenario

    non_scenarios = [disc for disc in disciplines if not isinstance(disc, BaseScenario)]
    scenarios = [disc for disc in disciplines if isinstance(disc, BaseScenario)]

    if skip_scenarios:
        return non_scenarios

    disciplines_in_scenarios = list(
        set.union(*(set(scenario.disciplines) for scenario in scenarios))
    )
    return disciplines_in_scenarios + non_scenarios


def get_all_inputs(
    disciplines: Iterable[Discipline],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the input names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the inputs.
    """
    return sorted(
        set.union(
            *(
                set(discipline.io.input_grammar)
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )


def get_all_outputs(
    disciplines: Iterable[Discipline | BaseScenario],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the output names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the outputs.
    """
    return sorted(
        set.union(
            *(
                set(discipline.io.output_grammar)
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )


def get_sub_disciplines(
    disciplines: Iterable[Discipline], recursive: bool = False
) -> list[Discipline]:
    """Determine the sub-disciplines.

    This method lists the sub-disciplines' disciplines. It will list up to one level
    of disciplines contained inside another one unless the ``recursive`` argument is
    set to ``True``.

    Args:
        disciplines: The disciplines from which the sub-disciplines will be determined.
        recursive: If ``True``, the method will look inside any discipline that has
            other disciplines inside until it reaches a discipline without
            sub-disciplines, in this case the return value will not include any
            discipline that has sub-disciplines. If ``False``, the method will list
            up to one level of disciplines contained inside another one, in this
            case the return value may include disciplines that contain
            sub-disciplines.

    Returns:
        The sub-disciplines.
    """
    from gemseo.formulations.base_formulation import BaseFormulation
    from gemseo.scenarios.base_scenario import BaseScenario

    sub_disciplines = []

    for discipline in disciplines:
        if (
            not isinstance(
                discipline, (BaseScenario, BaseFormulation, ProcessDiscipline)
            )
            or not discipline.disciplines
        ):
            _add_to_sub([discipline], sub_disciplines)
        elif recursive:
            _add_to_sub(
                get_sub_disciplines(discipline.disciplines, recursive=True),
                sub_disciplines,
            )
        else:
            _add_to_sub(discipline.disciplines, sub_disciplines)

    return sub_disciplines


def _add_to_sub(
    disciplines: Iterable[Discipline],
    sub_disciplines: MutableSequence[Discipline],
) -> None:
    """Add the disciplines of the sub-scenarios to the sub-disciplines.

    A sub-discipline is only added if it is not already in ``sub_disciplines``.

    Args:
        disciplines: The disciplines.
        sub_disciplines: The current sub-disciplines.
    """
    sub_disciplines.extend(disc for disc in disciplines if disc not in sub_disciplines)


_MESSAGE = "Two disciplines, among which {}, compute the same outputs: {}"


def check_disciplines_consistency(
    disciplines: Iterable[Discipline],
    log_message: bool,
    raise_error: bool,
) -> bool:
    """Check if disciplines are consistent.

    The disciplines are consistent
    if each output is computed by one and only one discipline.

    Args:
        disciplines: The disciplines of interest.
        log_message: Whether to log a message when the disciplines are not consistent.
        raise_error: Whether to raise an error when the disciplines are not consistent.

    Returns:
        Whether the disciplines are consistent.

    Raises:
        ValueError: When two disciplines compute the same output
            and ``raise_error`` is ``True``.
    """
    output_names_until_now = set()
    for discipline in disciplines:
        new_output_names = set(discipline.io.output_grammar)
        already_existing_output_names = new_output_names & output_names_until_now
        if already_existing_output_names:
            if raise_error:
                raise ValueError(
                    _MESSAGE.format(discipline.name, already_existing_output_names)
                )

            if log_message:
                LOGGER.warning(
                    _MESSAGE.replace("{}", "%s"),
                    discipline.name,
                    already_existing_output_names,
                )

            return False

        output_names_until_now |= new_output_names

    return True
