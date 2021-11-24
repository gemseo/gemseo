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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from gemseo.core.grammars.simple_grammar import SimpleGrammar

import six
from custom_inherit import DocInheritMeta
from numpy import ndarray, zeros

from gemseo.utils.py23_compat import Path

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

    Grammars supports __contains__ special method, so that one can test
    if an element is in the grammar with the statement `"x" in grammar`.

    Attributes:
        name (str): The name of the grammar.
    """

    def __init__(
        self,
        name,  # type: str
        **kwargs  # type: Union[str,Path]
    ):  # type: (...) -> None
        """
        Args:
            name: The name to be given to the grammar.
            **kwargs: The options of the grammar.
        """
        self.name = name

    def __str__(self):  # type: (...) -> str
        return "grammar name: {}".format(self.name)

    def __contains__(
        self,
        item,  # type: str
    ):  # type: (...) -> bool
        return self.is_data_name_existing(item)

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

    def is_type_array(
        self, data_name  # type: str
    ):  # type: (...) -> bool
        """Check if an element is an array.

        Args:
            data_name: The name of the element.

        Returns:
            Whether the element is an array.

        Raises:
            ValueError: If the name does not correspond to an element name.
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

    def restrict_to(
        self,
        data_names,  # type: Sequence[str]
    ):  # type: (...) -> None
        """Restrict the grammar to the given names.

        Args:
            data_names: The names of the elements to restrict the grammar to.
        """
        raise NotImplementedError

    def remove_item(
        self,
        item_name,  # type: str
    ):  # type: (...) -> None
        """Remove an element.

        Args:
            item_name: The name of the element to be removed.

        Raises:
            KeyError: When the element is not in the grammar.
        """
        raise NotImplementedError

    def get_type_from_python_type(
        self, python_type  # type: type
    ):  # type: (...) -> type
        """Return the grammar type that corresponds to a given Python type.

        Args:
            python_type: The Python type.

        Return:
            The equivalent grammar type.
        """
        raise NotImplementedError

    def update_elements(
        self,
        python_typing=False,  # type: bool
        **elements  # type: Mapping[str,type]
    ):  # type: (...) -> None
        """Add or update elements from their names and types.

        Args:
            python_typing: If True, handle automatically the conversion from
                Python type to grammar type.
            **elements: The names to types bindings of the elements to add or update.

        Examples:
            >>> grammar.update_elements(a=str, b=int)
            >>> grammar.update_elements(a=str, b=int, python_typing=True)
            >>> grammar.update_elements(**names_to_types)
        """
        raise NotImplementedError

    def update_required_elements(
        self, **elements  # type: Mapping[str, bool]
    ):  # type: (...) -> None
        """Add or update the required elements in the grammar.

        Args:
            **elements: The names of the elements bound to whether they shall be required.

        Raises:
            KeyError: If a given element name is not in the grammar.
            TypeError: If a given element name is not associated to a boolean value.
        """
        raise NotImplementedError

    def is_required(
        self, element_name  # type: str
    ):  # type: (...) -> bool
        """Check if an element is required in the grammar.

        Args:
            element_name: The data name to check.

        Returns:
            Whether the element name is required.

        Raises:
            ValueError: If the given element is not in the grammar.
        """
        raise NotImplementedError
