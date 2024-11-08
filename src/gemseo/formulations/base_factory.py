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
from gemseo.utils.source_parsing import get_callable_argument_defaults
from gemseo.utils.source_parsing import get_options_doc

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

    def get_options_doc(self, name: str) -> dict[str, str]:  # noqa: D102
        cls = self.get_class(name)
        options_doc = get_options_doc(cls.__init__)
        del options_doc["settings_model"]
        del options_doc["settings"]
        options_doc.update({
            field_name: field.description
            for field_name, field in cls.Settings.model_fields.items()
        })
        return options_doc

    def get_default_option_values(self, name: str) -> StrKeyMapping:  # noqa: D102
        cls = self.get_class(name)
        default_option_values = get_callable_argument_defaults(cls.__init__)
        del default_option_values["settings_model"]
        default_option_values.update({
            field_name: field.get_default(call_default_factory=True)
            for field_name, field in cls.Settings.model_fields.items()
            if not field.is_required()
        })
        return default_option_values
