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
"""A factory of formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.base_factory import BaseFactory
from gemseo.formulations.base_formulation import BaseFormulation

if TYPE_CHECKING:
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.typing import StrKeyMapping


class BaseFormulationFactory(BaseFactory[BaseFormulation]):
    """A factory of :class:`~gemseo.formulations.base_formulation.BaseFormulation`."""

    def get_sub_options_grammar(
        self,
        name: str,
        **options: str,
    ) -> JSONGrammar:
        """Return the JSONGrammar of the sub options of a class.

        Args:
            name: The name of the class.
            **options: The options to be passed to the class required to deduce
                the sub options.

        Returns:
            The JSON grammar.
        """
        return self.get_class(name).get_sub_options_grammar(**options)

    def get_default_sub_option_values(
        self,
        name: str,
        **options: str,
    ) -> StrKeyMapping:
        """Return the default values of the sub options of a class.

        Args:
            name: The name of the class.
            **options: The options to be passed to the class required to deduce
                the sub options.

        Returns:
            The JSON grammar.
        """
        return self.get_class(name).get_default_sub_option_values(**options)
