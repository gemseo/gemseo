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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""MDA input data initialization."""
from __future__ import annotations

from gemseo.core.chain import MDOChain
from gemseo.core.discipline import MDODiscipline


def order_disciplines_from_default_inputs(
    disciplines: list[MDODiscipline], raise_error: bool = True
) -> list[MDODiscipline] | list[str]:
    """Order disciplines such that all their input values are defined.

    It is particularly useful in the case of MDAs when not all
    default_inputs are available, and the execution order is not obvious to compute
    initial values for all couplings.

    From the default inputs of the disciplines, use a greedy algorithm to detect
    sequentially the disciplines that can be executed, and records the execution order.

    Raises:
        ValueError: When not all the disciplines can be initialized.

    Args:
        disciplines: The disciplines to compute the initialization of.
        raise_error: Whether to raise an exception when the algorithm fails.

    Returns:
        The ordered disciplines when the algorithm succeeds, or, if raise_error=False,
        the inputs that are missing.
    """
    ordered_discs = []
    remaining_discs = disciplines.copy()
    available_data = []
    while remaining_discs:
        removed_discs = []
        for disc in remaining_discs:
            required_inputs = set(disc.get_input_data_names())
            if not required_inputs.difference(disc.default_inputs).difference(
                available_data
            ):
                available_data.extend(disc.get_output_data_names())
                removed_discs.append(disc)

        if not removed_discs:
            disc_names = sorted(d.name for d in remaining_discs)
            missing_inputs = sorted(
                {
                    in_name
                    for disc in remaining_discs
                    for in_name in disc.get_input_data_names()
                }.difference(available_data)
            )
            if raise_error:
                raise ValueError(
                    f"Cannot compute the inputs {', '.join(missing_inputs)}, "
                    f"for the following disciplines {', '.join(disc_names)}."
                )
            else:
                return missing_inputs

        ordered_discs.extend(removed_discs)
        for disc in removed_discs:
            remaining_discs.remove(disc)
    return ordered_discs


class MDOInitializationChain(MDOChain):
    """An initialization process for a set of disciplines.

    This MDOChain subclass computes the initialization for the computation of a set
    of disciplines. It is particularly useful in the case of MDAs when not all
    default_inputs are available, and the execution order is not obvious to compute
    initial values for all couplings.

    From the default inputs of the disciplines, use a greedy algorithm to detect
    sequentially the disciplines that can be executed, and records the execution order.

    The couplings are ignored, and therefore, a true MDA must be used afterwards to
    ensure consistency.
    """

    def __init__(
        self,
        disciplines: list[MDODiscipline],
        name: str | None = None,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    ) -> None:
        disc_ordered = order_disciplines_from_default_inputs(disciplines)
        super().__init__(disciplines=disc_ordered, name=name, grammar_type=grammar_type)
