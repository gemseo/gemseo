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
"""A factory to instantiate a derived class of :class:`.BaseGrammar`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from strenum import StrEnum

from gemseo.core.base_factory import BaseFactory
from gemseo.core.grammars.base_grammar import BaseGrammar

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline


class GrammarFactory(BaseFactory[BaseGrammar]):
    """A factory of :class:`.BaseGrammar`."""

    _CLASS = BaseGrammar
    _PACKAGE_NAMES = ("gemseo.core.grammars",)

    def create(
        self,
        class_name: str,
        name: str,
        search_file: bool = False,
        discipline_class: type[Discipline] | None = None,
        directory_path: Path | str = "",
        file_name_suffix: str = "",
        **options: Any,
    ) -> BaseGrammar:
        """Create a grammar.

        Args:
            class_name: The name of a class deriving from :class:`.BaseGrammar`.
            name: The name to be given to the grammar.
            search_file: Whether to search for a JSON grammar file.
                This argument is considered to be ``False`` when the option
                ``file_path`` is given.
            discipline_class: The class of the discipline used for searching the grammar
                in the parent classes.
                This argument is used when ``search_file`` is ``True``.
            directory_path: The path to the directory where to search for JSON grammar
                files.
                This argument is used when ``search_file`` is ``True``.
            file_name_suffix: The suffix of the JSON grammar file.
                This argument is used when ``search_file`` is ``True``.
            **options: The options to be passed to the initialization.
        """
        if class_name == "JSONGrammar" and search_file and not options.get("file_path"):
            if discipline_class is None:
                msg = (
                    "The argument discipline_class is needed when the argument "
                    "when the argument search_file is True."
                )
                raise ValueError(msg)
            options["file_path"] = self.__search_file(
                discipline_class, file_name_suffix, directory_path
            )
        return super().create(class_name, name=name, **options)

    @staticmethod
    def __search_file(
        discipline_class: type[Discipline],
        file_name_suffix: str,
        directory_path: str | Path,
    ) -> Path:
        """Use a naming convention to associate a grammar file to the discipline.

        Search in the directory ``directory_path`` for
        either an input grammar file named ``name + "_input.json"``
        or an output grammar file named ``name + "_output.json"``.

        Args:
            file_name_suffix: The suffix of the file name (xxx_suffix.json)
            directory_path: The directory in which to search the grammar file.
                If ``None``,
                use the :attr:`.GRAMMAR_DIRECTORY` if any,
                or the directory of the discipline class module.

        Returns:
            The grammar file path.
        """
        # To avoid circular dependencies.
        from gemseo.core.discipline.base_discipline import BaseDiscipline

        classes = [discipline_class] + [
            base
            for base in discipline_class.__bases__
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
