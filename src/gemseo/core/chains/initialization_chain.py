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

from typing import TYPE_CHECKING

from gemseo.core.chains.chain import MDOChain
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


def order_disciplines_from_default_inputs(
    disciplines: Sequence[Discipline],
    raise_error: bool = True,
    available_data_names: Iterable[str] = (),
) -> list[Discipline] | list[str]:
    """Order disciplines such that all their input values are defined.

    It is particularly useful in the case of MDAs when not all
    default_input_data are available, and the execution order is not obvious to compute
    initial values for all couplings.

    From the default inputs of the disciplines, use a greedy algorithm to detect
    sequentially the disciplines that can be executed, and records the execution order.

    Raises:
        ValueError: When a discipline cannot be initialized.

    Args:
        disciplines: The disciplines to compute the initialization of.
        raise_error: Whether to raise an exception when the algorithm fails.
        available_data_names: The data names that are assumed to be available
            at runtime, in addition to the default_input_data.

    Returns:
        The ordered disciplines when the algorithm succeeds, or, if raise_error=False,
        the inputs that are missing.
    """
    ordered_discs = []
    remaining_discs = list(disciplines)
    available_data_names = list(available_data_names)
    while remaining_discs:
        removed_discs = []
        for disc in remaining_discs:
            required_inputs = set(disc.io.input_grammar)
            if not required_inputs.difference(
                disc.io.input_grammar.defaults
            ).difference(available_data_names):
                available_data_names.extend(disc.io.output_grammar)
                removed_discs.append(disc)

        if not removed_discs:
            disc_names = sorted(d.name for d in remaining_discs)
            missing_inputs = sorted(
                {
                    in_name
                    for disc in remaining_discs
                    for in_name in disc.io.input_grammar
                }.difference(available_data_names)
            )

            if raise_error:
                msg = (
                    "Cannot compute the inputs "
                    f"{pretty_str(missing_inputs, sort=False)}, "
                    "for the following disciplines "
                    f"{pretty_str(disc_names, sort=False)}."
                )
                raise ValueError(msg)

            return missing_inputs

        ordered_discs.extend(removed_discs)
        for disc in removed_discs:
            remaining_discs.remove(disc)
    return ordered_discs


class MDOInitializationChain(MDOChain):
    """An initialization process for a set of disciplines.

    This MDOChain subclass computes the initialization for the computation of a set of
    disciplines. It is particularly useful in the case of MDAs when not all
    default_input_data are available, and the execution order is not obvious to compute
    initial values for all couplings.

    From the default inputs of the disciplines, use a greedy algorithm to detect
    sequentially the disciplines that can be executed, and records the execution order.

    The couplings are ignored, and therefore, a true MDA must be used afterward to
    ensure consistency.
    """

    def __init__(  # noqa:D107
        self,
        disciplines: Sequence[Discipline],
        name: str = "",
        available_data_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            available_data_names: The data names that are assumed to be available
                at runtime, in addition to the default_input_data.
        """  # noqa:D205 D212 D415
        disc_ordered = order_disciplines_from_default_inputs(
            disciplines, available_data_names=available_data_names
        )
        super().__init__(disciplines=disc_ordered, name=name)
