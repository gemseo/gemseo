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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Rules and checks for disciplines inputs/outputs validation."""

from __future__ import division, unicode_literals

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

if TYPE_CHECKING:
    from gemseo.core.grammars.simple_grammar import SimpleGrammar

import six
from custom_inherit import DocInheritMeta
from numpy import ndarray, zeros

LOGGER = logging.getLogger(__name__)


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class AbstractGrammar(object):
    """The abstraction of a grammar.

    A grammar is a set of elements characterised by their names and types.
    One can use it to check if elements values are consistent with it.

    It is mainly used to store the names and types
    of the inputs and outputs of an :class:`.MDODiscipline`.

    Attributes:
        name (str): The name of the grammar.
    """

    def __init__(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            name: The name to be given to the grammar.
        """
        self.name = name

    def __str__(self):  # type: (...) -> str
        return "grammar name: {}".format(self.name)

    def load_data(
        self,
        data,  # type: Mapping[str,Any]
        raise_exception=True,  # type: bool
    ):  # type: (...) -> Mapping[str,Any]
        """Load elements values and check their consistency with the grammar.

        Args:
            data: The elements values to be checked.
            raise_exception: Whether to raise an exception
                when the elements values are not consistent with the grammar.

        Returns:
            The elements values after successful consistency checking.
        """
        raise NotImplementedError()

    def get_data_names(self):  # type: (...) -> List[str]
        """Return the names of the elements.

        Returns:
            The names of the elements sorted alphabetically.
        """
        raise NotImplementedError()

    def update_from(
        self,
        input_grammar,  # type: AbstractGrammar
    ):  # type: (...) -> None
        """Update the grammar with a second one.

        Add the new elements and update the existing ones.

        Args:
            input_grammar: The grammar to take the elements from.
        """
        raise NotImplementedError()

    def update_from_if_not_in(
        self,
        input_grammar,  # type: AbstractGrammar
        exclude_grammar,  # type: AbstractGrammar
    ):  # type: (...) -> None
        """Add the elements from a second grammar that are not present in a third one.

        Args:
            input_grammar: The grammar to take the elements from.
            exclude_grammar: The grammar containing the elements not to be taken.
        """
        raise NotImplementedError()

    def is_data_name_existing(
        self,
        data_name,  # type: str
    ):  # type: (...) -> bool
        """Check if the name of an element is present in the grammar.

        Args:
            data_name: The name of the element.

        Returns:
            Whether the name of the element is present in the grammar.
        """
        raise NotImplementedError()

    def is_all_data_names_existing(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> bool
        """Check if the names of the elements are present in the grammar.

        Args:
            data_names: The names of the elements.

        Returns:
            Whether all the elements names are in the grammar.
        """
        raise NotImplementedError()

    def clear(self):  # type: (...) -> None
        """Clear the grammar."""
        raise NotImplementedError()

    def to_simple_grammar(self):  # type: (...) -> SimpleGrammar
        """Convert to the base :class:`.SimpleGrammar` type.

        Returns:
            A :class:`.SimpleGrammar` version of the current grammar.
        """
        raise NotImplementedError()

    def initialize_from_data_names(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> None
        """Initialize the grammar from the names of the elements and float type.

        Args:
            data_names: The names of the elements.
        """
        element_value = zeros(1)
        elements_values = {element_name: element_value for element_name in data_names}
        self.initialize_from_base_dict(elements_values)

    def initialize_from_base_dict(
        self,
        typical_data_dict,  # type: Dict[str,ndarray]
    ):  # type: (...) -> None
        """Initialize the grammar with types and names from typical elements values.

        Args:
            typical_data_dict: Typical elements values
                indexed by the elements names.
        """
        raise NotImplementedError()

    @staticmethod
    def _get_update_error_msg(
        grammar1,  # type: AbstractGrammar
        grammar2,  # type: AbstractGrammar
        grammar3=None,  # type: Optional[AbstractGrammar]
    ):  # type: (...) -> str
        """Create a message for grammar update error.

        Args:
            grammar1: The grammar to be updated.
            grammar2: A grammar to update the first one.
            grammar3: Another grammar to update the first one.
                If None,
                consider that only the ``grammar2`` is used to update ``grammar1``.

        Returns:
            The error message based on the passed grammars.
        """
        msg = "Cannot update grammar {} of type {} with {} of type {}".format(
            grammar1.name,
            grammar1.__class__.__name__,
            grammar2.name,
            grammar2.__class__.__name__,
        )
        if grammar3 is not None:
            msg += " and {} of type {}".format(
                grammar1.name,
                grammar1.__class__.__name__,
            )
        return "{}.".format(msg)
