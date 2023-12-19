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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Factory to create disciplines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core import discipline
from gemseo.core.base_factory import BaseFactory
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.json_grammar import JSONGrammar

if TYPE_CHECKING:
    from collections.abc import Mapping


class DisciplinesFactory(BaseFactory):
    """The **DisciplinesFactory** is used to create :class:`.MDODiscipline` objects.

    Three types of directories are scanned
    to import the :class:`.MDODiscipline` classes:

    - the environment variable "GEMSEO_PATH" may contain
      the list of directories to scan,
    - the present directory (:doc:`gemseo.problems`) contains some
      benchmark test cases,
    """

    _CLASS = MDODiscipline
    _MODULE_NAMES = (
        "gemseo.problems",
        "gemseo.core",
        "gemseo.disciplines",
        "gemseo.wrappers",
    )

    def __init__(self) -> None:  # noqa: D107
        # Defines the benchmark problems to be imported
        super().__init__()
        self.__base_grammar = JSONGrammar("MDODiscipline_options")
        base_gram_path = Path(discipline.__file__).parent / "MDODiscipline_options.json"
        self.__base_grammar.update_from_file(base_gram_path)
        self.__base_grammar_names = self.__base_grammar.keys()

    def create(self, discipline_name: str, **options):
        """Create an :class:`.MDODiscipline` from its name.

        Args:
            discipline_name: The name of the discipline
            **options: The options of the discipline,
                both the options to be passed to the constructor
                and the options that are generic to all the disciplines.

        Returns:
            The discipline.
        """
        common_options, specific_options = self.__filter_common_options(options)
        self.__base_grammar.validate(common_options)
        discipline = super().create(discipline_name, **specific_options)
        if "linearization_mode" in common_options:
            discipline.linearization_mode = common_options["linearization_mode"]

        cache_options = self.__filter_options_with_prefix(common_options, "cache")
        if cache_options:
            discipline.set_cache_policy(**cache_options)

        jacobian_options = self.__filter_options_with_prefix(common_options, "jac")
        if jacobian_options:
            discipline.set_jacobian_approximation(**jacobian_options)

        return discipline

    @staticmethod
    def __filter_options_with_prefix(
        options: Mapping[str, Any], prefix: str
    ) -> dict[str, Any]:
        """Filter the options whose names start with a prefix.

        Args:
            options: The options of the disciplines.
            prefix: The prefix.

        Returns:
            The options whose names start with a prefix.
        """
        return {k: v for k, v in options.items() if k.startswith(prefix)}

    def __filter_common_options(self, options):
        """Separate options.

        Args:
            options: The options of the discipline.

        Returns:
            The options common to all the disciplines,
            and the options specific to the current discipline.
        """
        common_option_names = self.__base_grammar_names
        common_options = {k: v for k, v in options.items() if k in common_option_names}
        specific_options = {k: v for k, v in options.items() if k not in common_options}
        return common_options, specific_options

    @property
    def disciplines(self) -> list[str]:
        """The names of the available disciplines."""
        return self.class_names

    def get_options_grammar(
        self, name: str, write_schema: bool = False, schema_path: str | None = None
    ) -> JSONGrammar:
        """Get the options default values for the given class name.

        Args:
            name: The name of the class.
            schema_path: the output json file path. If ``None``: input.json or
                output.json depending on grammar type.
            write_schema: Whether to write the schema files

        Returns:
            The JSON grammar of the options.
        """
        disc_gram = super().get_options_grammar(name, write_schema, schema_path)
        option_grammar = deepcopy(self.__base_grammar)
        option_grammar.update(disc_gram)
        return option_grammar
