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
"""Tools to facilitate the renaming of discipline input and output variables."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import NamedTuple

from pandas import DataFrame
from pandas import read_csv
from pandas import read_excel
from prettytable import PrettyTable

from gemseo.utils.repr_html import REPR_HTML_WRAPPER

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class VariableRenamer:
    """Renamer of discipline input and output variable names."""

    __translations: tuple[VariableTranslation, ...]
    """The translations of the discipline input and output variables."""

    __translators: Mapping[str, Mapping[str, str]]
    """The translators."""

    def __initialize(
        self,
        data: tuple[VariableTranslation, ...] | Mapping[str, Mapping[str, str]],
    ) -> None:
        """Set the translations and translators.

        Args:
            data: Either the translations or translators.
        """
        if isinstance(data, Mapping):
            self.__translations = tuple(
                VariableTranslation(
                    discipline_name=discipline_name,
                    variable_name=variable_name,
                    new_variable_name=new_variable_name,
                )
                for discipline_name, discipline_translations in data.items()
                for variable_name, new_variable_name in discipline_translations.items()
            )
            self.__translators = data
        else:
            self.__translations = data
            self.__translators = self.__compute_translators()

    def __get_pretty_table(self) -> PrettyTable:
        """Return a tabular view.

        Returns:
            A tabular view of the variable renamer.
        """
        pretty_table = PrettyTable()
        pretty_table.field_names = [
            "Discipline name",
            "Variable name",
            "New variable name",
        ]
        for translation in self.__translations:
            pretty_table.add_row([
                translation.discipline_name,
                translation.variable_name,
                translation.new_variable_name,
            ])

        return pretty_table

    def __repr__(self) -> str:
        return str(self.__get_pretty_table())

    def _repr_html_(self) -> str:
        return REPR_HTML_WRAPPER.format(self.__get_pretty_table().get_html_string())

    @property
    def translations(self) -> tuple[VariableTranslation, ...]:
        """The translations of the discipline input and output variables."""
        return self.__translations

    @property
    def translators(self) -> Mapping[str, Mapping[str, str]]:
        """The translators.

        As ``{discipline_name: {variable_name, new_variable_name}}``.
        """
        return self.__translators

    @classmethod
    def from_translations(cls, *translations: VariableTranslation) -> VariableRenamer:
        """Create from translations.

        Args:
            *translations: The translations
                of the discipline input and output variables.

        Returns:
            A renamer.
        """
        renamer = cls()
        renamer.__initialize(translations)
        return renamer

    @classmethod
    def from_tuples(cls, *translations: tuple[str, str, str]) -> VariableRenamer:
        """Create from tuples.

        Args:
            *translations: The translations of the discipline input and output variables
                as ``(discipline_name, variable_name, new_variable_name)``.

        Returns:
            A renamer.
        """
        return cls.from_translations(
            *(
                VariableTranslation(
                    discipline_name=discipline_name,
                    variable_name=variable_name,
                    new_variable_name=new_variable_name,
                )
                for (discipline_name, variable_name, new_variable_name) in translations
            )
        )

    @classmethod
    def from_dictionary(
        cls, translations: Mapping[str, Mapping[str, str]]
    ) -> VariableRenamer:
        """Create from dictionaries.

        Args:
            translations: The translations of the discipline input and output variables
                as ``{discipline_name: {variable_name: new_variable_name}}``.

        Returns:
            A renamer.
        """
        renamer = cls()
        renamer.__initialize(translations)
        return renamer

    @classmethod
    def from_spreadsheet(cls, file_path: str | Path) -> Self:
        """Create from a spreadsheet file.

        Structured as `discipline_name, variable_name, new_variable_name`.

        Args:
            file_path: The path to the spreadsheet file.

        Returns:
            A renamer.
        """
        return cls.__from_dataframe(read_excel(file_path, header=None))

    @classmethod
    def from_csv(cls, file_path: str | Path, sep: str = ",") -> Self:
        """Create from a CSV file.

        Structured as `discipline_name, variable_name, new_variable_name`.

        Args:
            file_path: The path to the CSV file.
            sep: The separator character.

        Returns:
            A renamer.
        """
        return cls.__from_dataframe(read_csv(file_path, sep=sep, header=None))

    @classmethod
    def __from_dataframe(cls, dataframe: DataFrame) -> Self:
        """Create from a pandas dataframe.

        Args:
            dataframe: A pandas dataframe.

        Returns:
            A renamer.
        """
        translations = [
            VariableTranslation(
                discipline_name=discipline_name,
                variable_name=variable_name,
                new_variable_name=new_variable_name,
            )
            for (
                discipline_name,
                variable_name,
                new_variable_name,
            ) in dataframe.to_numpy()
        ]
        return cls.from_translations(*translations)

    def __compute_translators(self) -> dict[str, dict[str, str]]:
        """Compute one translator per discipline.

        Returns:
            One translator per discipline

        Raises:
            ValueError: When a variable has already been renamed.
        """
        translators = defaultdict(dict)
        for t in self.__translations:
            translator = translators[t.discipline_name]
            new_variable_name = translator.get(t.variable_name)
            if new_variable_name is not None:
                msg = (
                    f"In discipline {t.discipline_name!r}, "
                    f"the variable {t.variable_name!r} cannot be renamed "
                    f"to {t.new_variable_name!r} "
                    f"because it has already been renamed to {new_variable_name!r}."
                )
                if new_variable_name == t.new_variable_name:
                    LOGGER.warning(msg)
                else:
                    raise ValueError(msg)

            translator[t.variable_name] = t.new_variable_name

        return translators


class VariableTranslation(NamedTuple):
    """The translation of a discipline input or output variable."""

    discipline_name: str
    """The name of the discipline."""

    variable_name: str
    """The name of the variable."""

    new_variable_name: str
    """The new name of the variable."""

    def __repr__(self) -> str:
        return (
            f"{self.discipline_name!r}.{self.variable_name!r}"
            f"={self.new_variable_name!r}"
        )
