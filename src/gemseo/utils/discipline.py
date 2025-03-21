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

"""Discipline utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NamedTuple

from pandas import DataFrame
from pandas import read_csv
from pandas import read_excel
from prettytable import PrettyTable

from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.core.discipline.data_processor import NameMapping
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.process_discipline import ProcessDiscipline
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import MutableSequence
    from pathlib import Path
    from typing import Self

    from gemseo.scenarios.base_scenario import BaseScenario
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


# TODO: try to keep only one of these two dummy disciplines
class DummyBaseDiscipline(BaseDiscipline):
    """A dummy base discipline that does nothing."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        pass  # pragma: no cover


class DummyDiscipline(Discipline):
    """A dummy discipline that does nothing."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        pass


def __get_all_disciplines(
    disciplines: Iterable[Discipline | BaseScenario],
    skip_scenarios: bool,
) -> list[Discipline]:
    """Return the non-scenario disciplines or also the disciplines of the scenario ones.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, return their disciplines.

    Returns:
        The non-scenario disciplines
        or also the disciplines of the scenario ones if any and ``skip_scenario=False``.
    """
    from gemseo.scenarios.base_scenario import BaseScenario

    non_scenarios = [disc for disc in disciplines if not isinstance(disc, BaseScenario)]
    scenarios = [disc for disc in disciplines if isinstance(disc, BaseScenario)]

    if skip_scenarios:
        return non_scenarios

    disciplines_in_scenarios = list(
        set.union(*(set(scenario.disciplines) for scenario in scenarios))
    )
    return disciplines_in_scenarios + non_scenarios


def get_all_inputs(
    disciplines: Iterable[Discipline],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the input names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the inputs.
    """
    return sorted(
        set.union(
            *(
                set(discipline.io.input_grammar)
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )


def get_all_outputs(
    disciplines: Iterable[Discipline | BaseScenario],
    skip_scenarios: bool = True,
) -> list[str]:
    """Return all the output names of the disciplines.

    Args:
        disciplines: The disciplines including potentially :class:`.Scenario` objects.
        skip_scenarios: If ``True``, skip the :class:`.Scenario` objects.
            Otherwise, consider their disciplines.

    Returns:
        The names of the outputs.
    """
    return sorted(
        set.union(
            *(
                set(discipline.io.output_grammar)
                for discipline in __get_all_disciplines(
                    disciplines, skip_scenarios=skip_scenarios
                )
            )
        )
    )


def get_sub_disciplines(
    disciplines: Iterable[Discipline], recursive: bool = False
) -> list[Discipline]:
    """Determine the sub-disciplines.

    This method lists the sub-disciplines' disciplines. It will list up to one level
    of disciplines contained inside another one unless the ``recursive`` argument is
    set to ``True``.

    Args:
        disciplines: The disciplines from which the sub-disciplines will be determined.
        recursive: If ``True``, the method will look inside any discipline that has
            other disciplines inside until it reaches a discipline without
            sub-disciplines, in this case the return value will not include any
            discipline that has sub-disciplines. If ``False``, the method will list
            up to one level of disciplines contained inside another one, in this
            case the return value may include disciplines that contain
            sub-disciplines.

    Returns:
        The sub-disciplines.
    """
    from gemseo.formulations.base_formulation import BaseFormulation
    from gemseo.scenarios.base_scenario import BaseScenario

    sub_disciplines = []

    for discipline in disciplines:
        if (
            not isinstance(
                discipline, (BaseScenario, BaseFormulation, ProcessDiscipline)
            )
            or not discipline.disciplines
        ):
            _add_to_sub([discipline], sub_disciplines)
        elif recursive:
            _add_to_sub(
                get_sub_disciplines(discipline.disciplines, recursive=True),
                sub_disciplines,
            )
        else:
            _add_to_sub(discipline.disciplines, sub_disciplines)

    return sub_disciplines


def _add_to_sub(
    disciplines: Iterable[Discipline],
    sub_disciplines: MutableSequence[Discipline],
) -> None:
    """Add the disciplines of the sub-scenarios to the sub-disciplines.

    A sub-discipline is only added if it is not already in ``sub_disciplines``.

    Args:
        disciplines: The disciplines.
        sub_disciplines: The current sub-disciplines.
    """
    sub_disciplines.extend(disc for disc in disciplines if disc not in sub_disciplines)


_MESSAGE = "Two disciplines, among which {}, compute the same outputs: {}"


def check_disciplines_consistency(
    disciplines: Iterable[Discipline],
    log_message: bool,
    raise_error: bool,
) -> bool:
    """Check if disciplines are consistent.

    The disciplines are consistent
    if each output is computed by one and only one discipline.

    Args:
        disciplines: The disciplines of interest.
        log_message: Whether to log a message when the disciplines are not consistent.
        raise_error: Whether to raise an error when the disciplines are not consistent.

    Returns:
        Whether the disciplines are consistent.

    Raises:
        ValueError: When two disciplines compute the same output
            and ``raise_error`` is ``True``.
    """
    output_names_until_now = set()
    for discipline in disciplines:
        new_output_names = set(discipline.io.output_grammar)
        already_existing_output_names = new_output_names & output_names_until_now
        if already_existing_output_names:
            if raise_error:
                raise ValueError(
                    _MESSAGE.format(discipline.name, already_existing_output_names)
                )

            if log_message:
                LOGGER.warning(
                    _MESSAGE.replace("{}", "%s"),
                    discipline.name,
                    already_existing_output_names,
                )

            return False

        output_names_until_now |= new_output_names

    return True


class VariableRenamer:
    """Renamer of discipline input and output variable names."""

    __translations: tuple[VariableTranslation, ...]
    """The translations of the discipline input and output variables."""

    __translators: Mapping[str, Mapping[str, str]]
    """The translators."""

    def __init__(self) -> None:  # noqa: D107
        self.__translations = ()
        self.__translators = defaultdict(dict)

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
    def from_translations(
        cls, *translations: VariableTranslation | tuple[str, str, str]
    ) -> VariableRenamer:
        """Create from translations.

        Args:
            *translations: The translations
                of the discipline input and output variables.
                If ``tuple``,
                formatted as ``(discipline_name, variable_name, new_variable_name)``.

        Returns:
            A renamer.
        """
        renamer = cls()
        for translation in translations:
            renamer.add_translation(translation)

        return renamer

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
        for (
            discipline_name,
            variable_names_to_new_variable_names,
        ) in translations.items():
            renamer.add_translations_by_discipline(
                discipline_name, variable_names_to_new_variable_names
            )

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

    def add_translation(
        self, translation: VariableTranslation | tuple[str, str, str]
    ) -> None:
        """Add a translation.

        Args:
            translation: A variable translation.
                If tuple,
                formatted as ``(discipline_name, variable_name, new_variable_name)``.

        Raises:
            ValueError: When a variable has already been renamed.
        """
        if not isinstance(translation, VariableTranslation):
            translation = VariableTranslation(
                discipline_name=translation[0],
                variable_name=translation[1],
                new_variable_name=translation[2],
            )

        self.__translations = (*self.__translations, translation)
        translator = self.__translators[translation.discipline_name]
        new_variable_name = translator.get(translation.variable_name)
        if new_variable_name is not None:
            msg = (
                f"In discipline {translation.discipline_name!r}, "
                f"the variable {translation.variable_name!r} cannot be renamed "
                f"to {translation.new_variable_name!r} "
                f"because it has already been renamed to {new_variable_name!r}."
            )
            if new_variable_name == translation.new_variable_name:
                LOGGER.warning(msg)
            else:
                raise ValueError(msg)

        translator[translation.variable_name] = translation.new_variable_name

    def add_translations_by_discipline(
        self,
        discipline_name: str,
        variable_names_to_new_variable_names: Mapping[str, str],
    ) -> None:
        """Add one or more translations for a given discipline.

        Args:
            discipline_name: The name of the discipline.
            variable_names_to_new_variable_names: The new variable names
                bound to the old variable names.
        """
        for (
            variable_name,
            new_variable_name,
        ) in variable_names_to_new_variable_names.items():
            self.add_translation(
                VariableTranslation(
                    discipline_name=discipline_name,
                    variable_name=variable_name,
                    new_variable_name=new_variable_name,
                )
            )

    def add_translations_by_variable(
        self,
        new_variable_name: str,
        discipline_names_to_variable_names: Mapping[str, str],
    ) -> None:
        """Add one or more translations for a same variable.

        Args:
            new_variable_name: The new name of the variable
                to rename discipline variables.
            discipline_names_to_variable_names: The variable names
                bound to the discipline names.
        """
        for (
            discipline_name,
            variable_name,
        ) in discipline_names_to_variable_names.items():
            self.add_translation(
                VariableTranslation(
                    discipline_name=discipline_name,
                    variable_name=variable_name,
                    new_variable_name=new_variable_name,
                )
            )


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


def rename_discipline_variables(
    disciplines: Iterable[Discipline], translators: Mapping[str, Mapping[str, str]]
) -> None:
    """Rename input and output variables of disciplines.

    Args:
        disciplines: The disciplines.
        translators: The translators
            of the form ``{discipline_name: {variable_name: new_variable_name}}``.

    Raises:
        ValueError: when a translator uses a wrong ``variable_name``.
    """
    for discipline in disciplines:
        translator = translators.get(discipline_name := discipline.name)
        if translator is None:
            LOGGER.warning("The discipline '%s' has no translator.", discipline_name)
            continue

        grammars = [discipline.io.input_grammar, discipline.io.output_grammar]
        for variable_name, new_variable_name in translator.items():
            variable_name_does_not_exist = True
            for grammar in grammars:
                if variable_name in grammar:
                    variable_name_does_not_exist = False
                    grammar.rename_element(variable_name, new_variable_name)

            if variable_name_does_not_exist:
                msg = (
                    f"The discipline {discipline_name!r} "
                    f"has no variable {variable_name!r}."
                )
                raise ValueError(msg)

        discipline.io.data_processor = NameMapping({
            new_variable_name: variable_name
            for variable_name, new_variable_name in translator.items()
        })


@dataclass
class DisciplineVariableProperties:
    """The properties of a discipline variable."""

    current_name: str
    """The current name of the variable."""

    current_name_without_namespace: str
    """The current name of the variable without namespace."""

    description: str
    """The description of the variable."""

    original_name: str
    """The original name of the variable."""


def get_discipline_variable_properties(
    discipline: Discipline,
) -> tuple[
    dict[str, DisciplineVariableProperties], dict[str, DisciplineVariableProperties]
]:
    """Return the properties of the input and output variables of a discipline.

    Args:
        discipline: The discipline.

    Returns:
        The properties of the input variables,
        then the properties of the output variables.
    """
    input_names_to_properties = {}
    output_names_to_properties = {}
    data_processor = discipline.io.data_processor
    if isinstance(data_processor, NameMapping):
        data_processor_mapping = data_processor.mapping
    else:
        data_processor_mapping = None

    for grammar, names_to_properties in zip(
        (discipline.io.input_grammar, discipline.io.output_grammar),
        (input_names_to_properties, output_names_to_properties),
    ):
        from_namespaced = grammar.from_namespaced
        for current_name in grammar:
            current_name_without_namespace = from_namespaced.get(
                current_name, current_name
            )
            if data_processor_mapping is None:
                original_name = current_name_without_namespace
            else:
                original_name = data_processor_mapping.get(
                    current_name_without_namespace, current_name_without_namespace
                )

            names_to_properties[current_name] = DisciplineVariableProperties(
                current_name=current_name,
                current_name_without_namespace=current_name_without_namespace,
                original_name=original_name,
                description=grammar.descriptions.get(current_name, ""),
            )

    return input_names_to_properties, output_names_to_properties
