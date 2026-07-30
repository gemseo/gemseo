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
#                           documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to instantiate grammar classes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from strenum import StrEnum

from gemseo.core.base_factory import BaseFactory
from gemseo.core.grammar.base import BaseGrammar

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline
    from gemseo.util.typing import StrPath


class GrammarFactory(BaseFactory[BaseGrammar]):
    """A factory of [BaseGrammar][gemseo.core.grammar.base.BaseGrammar]."""

    _CLASS = BaseGrammar
    _PACKAGE_NAMES = ("gemseo.core.grammar",)

    __FILE_PATH_STR: Final[str] = "file_path"

    def create(
        self,
        class_name: str,
        name: str,
        search_file: bool = False,
        discipline_class: type[Discipline] | None = None,
        directory_path: StrPath = "",
        file_name_suffix: str = "",
        **options: Any,
    ) -> BaseGrammar:
        """Create a grammar.

        Args:
            class_name: The name of a class deriving
                from [BaseGrammar][gemseo.core.grammar.base.BaseGrammar].
            name: The name to be given to the grammar.
            search_file: Whether to search for a JSON grammar file.
                This argument is considered to be `False` when the option
                `file_path` is given.
            discipline_class: The class of the discipline used for searching the grammar
                in the parent classes.
                This argument is used when `search_file` is `True`.
            directory_path: The path to the directory where to search for JSON grammar
                files.
                This argument is used when `search_file` is `True`.
            file_name_suffix: The suffix of the JSON grammar file.
                This argument is used when `search_file` is `True`.
            **options: The options to be passed to the initialization.

        Returns:
            The grammar.

        Raises:
            ValueError: If `search_file` is `True` and `class_name`
                is not `"JSONGrammar"`,
                or if `search_file` is `True` and `discipline_class` is `None`.
        """
        if search_file and not options.get(self.__FILE_PATH_STR):
            if class_name != "JSONGrammar":
                msg = (
                    "search_file=True is only supported for JSONGrammar; "
                    f"got {class_name}."
                )
                raise ValueError(msg)
            if discipline_class is None:
                msg = "discipline_class is required when search_file is True."
                raise ValueError(msg)
            options[self.__FILE_PATH_STR] = self.__search_file(
                discipline_class, file_name_suffix, directory_path
            )
        if class_name != "JSONGrammar" and not options.get(self.__FILE_PATH_STR):
            # `file_path` is JSONGrammar-specific; drop it when callers pass it as the
            # empty default so they can build any grammar type generically.
            options.pop(self.__FILE_PATH_STR, None)
        return super().create(class_name, name=name, **options)

    @staticmethod
    def __search_file(
        discipline_class: type[Discipline],
        file_name_suffix: str,
        directory_path: StrPath,
    ) -> Path:
        """Use a naming convention to associate a grammar file to the discipline.

        Search in the directory `directory_path` for
        either an input grammar file named `name + "_input.json"`
        or an output grammar file named `name + "_output.json"`.

        Args:
            discipline_class: The class of the discipline used for searching the
                grammar in the parent classes.
            file_name_suffix: The suffix of the file name (xxx_suffix.json)
            directory_path: The directory in which to search the grammar file.
                If empty,
                use the directory of the module defining each class of the
                discipline class hierarchy.

        Returns:
            The grammar file path.
        """
        # To avoid circular dependencies.
        from gemseo.core.discipline.base_discipline import BaseDiscipline

        # The mro starts with discipline_class itself.
        classes = [
            base
            for base in discipline_class.__mro__
            if issubclass(base, BaseDiscipline)
        ]

        for cls in classes:
            name = cls.__name__
            if not directory_path:
                class_module = sys.modules[cls.__module__]
                directory_path_ = Path(class_module.__file__).parent  # type: ignore[arg-type] # __file__ could be None
            else:
                directory_path_ = Path(directory_path)
            grammar_file_path = directory_path_ / f"{name}_{file_name_suffix}.json"
            if grammar_file_path.is_file():
                return grammar_file_path

        file_name = f"{discipline_class.__name__}_{file_name_suffix}.json"
        msg = f"The grammar file {file_name} is missing."
        raise FileNotFoundError(msg)


class GrammarType(StrEnum):
    """The name of the grammar class."""

    JSON = "JSONGrammar"
    SIMPLE = "SimpleGrammar"
    SIMPLER = "SimplerGrammar"
    PYDANTIC = "PydanticGrammar"


GRAMMAR_FACTORY: Final[GrammarFactory] = GrammarFactory()
"""The factory for `BaseGrammar` objects."""
