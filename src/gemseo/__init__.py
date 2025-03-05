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
"""|g| main package.

This module contains the high-level functions to easily use |g|
without requiring a deep knowledge of this software.

Besides,
these functions shall change much less often than the internal classes,
which is key for backward compatibility,
which means ensuring that
your current scripts using |g| will be usable with the future versions of |g|.

The high-level functions also facilitate the interfacing of |g|
with a platform or other software.

To interface a simulation software with |g|,
please refer to: :ref:`software_connection`.

See also :ref:`extending-gemseo`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from numpy import ndarray

from gemseo.core.execution_statistics import ExecutionStatistics as _ExecutionStatistics
from gemseo.datasets import DatasetClassName
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.problems.dataset import DatasetType
from gemseo.scenarios.base_scenario import BaseScenario as BaseScenario
from gemseo.scenarios.factory import ScenarioFactory as ScenarioFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import DEFAULT_DATE_FORMAT
from gemseo.utils.logging_tools import DEFAULT_MESSAGE_FORMAT
from gemseo.utils.logging_tools import LOGGING_SETTINGS
from gemseo.utils.pickle import from_pickle  # noqa: F401
from gemseo.utils.pickle import to_pickle  # noqa: F401

if TYPE_CHECKING:
    from logging import Logger

    from pydantic import BaseModel

    from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings
    from gemseo.algos.base_driver_library import DriverSettingType
    from gemseo.algos.database import Database
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_settings import BaseDOESettings
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.caches.base_cache import BaseCache
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.datasets.dataset import Dataset
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.datasets.optimization_dataset import (
        OptimizationDataset as OptimizationDataset,
    )
    from gemseo.disciplines.surrogate import SurrogateDiscipline
    from gemseo.disciplines.wrappers.job_schedulers.discipline_wrapper import (
        JobSchedulerDisciplineWrapper,
    )
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.mda.base_mda import BaseMDA
    from gemseo.mda.base_mda_settings import BaseMDASettings
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.post._graph_view import GraphView
    from gemseo.post.base_post import BasePost
    from gemseo.post.base_post_settings import BasePostSettings
    from gemseo.problems.mdo.scalable.data_driven.discipline import (
        DataDrivenScalableDiscipline,
    )
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.scenarios.doe_scenario import DOEScenario as DOEScenario
    from gemseo.scenarios.scenario_results.scenario_result import (
        ScenarioResult as ScenarioResult,
    )
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.matplotlib_figure import FigSizeType
    from gemseo.utils.xdsm import XDSM

# Most modules are imported directly in the methods, which adds a very small
# overhead, but prevents users from importing them from this root module.
# All factories are Singletons which means that the scan for
# plugins is done once only

LOGGER = logging.getLogger(__name__)
# By default, no logging is produced.
LOGGER.addHandler(logging.NullHandler())


def generate_n2_plot(
    disciplines: Sequence[Discipline],
    file_path: str | Path = "n2.pdf",
    show_data_names: bool = True,
    save: bool = True,
    show: bool = False,
    fig_size: FigSizeType = (15.0, 10.0),
    show_html: bool = False,
) -> None:
    """Generate a N2 plot from disciplines.

    It can be static (e.g. PDF, PNG, ...), dynamic (HTML) or both.

    The disciplines are located on the diagonal of the N2 plot
    while the coupling variables are situated on the other blocks
    of the matrix view.
    A coupling variable is outputted by a discipline horizontally
    and enters another vertically.
    On the static plot,
    a blue diagonal block represents a self-coupled discipline,
    i.e. a discipline having some of its outputs as inputs.

    Args:
        disciplines: The disciplines from which the N2 chart is generated.
        file_path: The file path to save the static N2 chart.
        show_data_names: Whether to show the names of the coupling variables
            between two disciplines;
            otherwise,
            circles are drawn,
            whose size depends on the number of coupling names.
        save: Whether to save the static N2 chart.
        show: Whether to show the static N2 chart.
        fig_size: The width and height of the static N2 chart.
        show_html: Whether to display the interactive N2 chart in a web browser.

    Examples:
        >>> from gemseo import create_discipline, generate_n2_plot
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> generate_n2_plot(disciplines)
    """
    from gemseo.core.coupling_structure import CouplingStructure

    coupling_structure = CouplingStructure(disciplines)
    coupling_structure.plot_n2_chart(
        file_path, show_data_names, save, show, fig_size, show_html
    )


def generate_coupling_graph(
    disciplines: Sequence[Discipline],
    file_path: str | Path = "coupling_graph.pdf",
    full: bool = True,
) -> GraphView | None:
    """Generate a graph of the couplings between disciplines.

    Args:
        disciplines: The disciplines from which the graph is generated.
        file_path: The path of the file to save the figure.
            If empty, the figure is not saved.
        full: Whether to generate the full coupling graph.
            Otherwise, the condensed coupling graph is generated.

    Returns:
        Either the graph of the couplings between disciplines
        or ``None`` when graphviz is not installed.

    Examples:
        >>> from gemseo import create_discipline, generate_coupling_graph
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> generate_coupling_graph(disciplines)
    """
    from gemseo.core.coupling_structure import CouplingStructure

    coupling_structure = CouplingStructure(disciplines)
    if full:
        return coupling_structure.graph.render_full_graph(file_path)
    return coupling_structure.graph.render_condensed_graph(file_path)


def get_available_formulations() -> list[str]:
    """Return the names of the available formulations.

    Returns:
        The names of the available MDO formulations.

    Examples:
        >>> from gemseo import get_available_formulations
        >>> get_available_formulations()
    """
    from gemseo.formulations.factory import MDOFormulationFactory

    return MDOFormulationFactory().class_names


def get_available_opt_algorithms() -> list[str]:
    """Return the names of the available optimization algorithms.

    Returns:
        The names of the available optimization algorithms.

    Examples:
        >>> from gemseo import get_available_opt_algorithms
        >>> get_available_opt_algorithms()
    """
    from gemseo.algos.opt.factory import OptimizationLibraryFactory

    return OptimizationLibraryFactory().algorithms


def get_available_doe_algorithms() -> list[str]:
    """Return the names of the available design of experiments (DOEs) algorithms.

    Returns:
        The names of the available DOE algorithms.

    Examples;
        >>> from gemseo import get_available_doe_algorithms
        >>> get_available_doe_algorithms()
    """
    from gemseo.algos.doe.factory import DOELibraryFactory

    return DOELibraryFactory().algorithms


def get_available_surrogates() -> list[str]:
    """Return the names of the available surrogate disciplines.

    Returns:
        The names of the available surrogate disciplines.

    Examples:
        >>> from gemseo import get_available_surrogates
        >>> print(get_available_surrogates())
        ['RBFRegressor', 'GaussianProcessRegressor', 'LinearRegressor', 'PCERegressor']
    """
    from gemseo.mlearning import get_regression_models

    return get_regression_models()


def get_available_disciplines() -> list[str]:
    """Return the names of the available disciplines.

    Returns:
        The names of the available disciplines.

    Examples:
        >>> from gemseo import get_available_disciplines
        >>> print(get_available_disciplines())
        ['RosenMF', 'SobieskiAerodynamics', 'ScalableKriging', 'DOEScenario',
        'MDOScenario', 'SobieskiMission', 'SobieskiDiscipline', 'Sellar1',
        'Sellar2', 'MDOChain', 'SobieskiStructure', 'AutoPyDiscipline',
        'Structure', 'SobieskiPropulsion', 'Scenario', 'AnalyticDiscipline',
        'MDOScenarioAdapter', 'ScalableDiscipline', 'SellarSystem', 'Aerodynamics',
        'Mission', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3',
        'PropaneReaction', 'MDOParallelChain']
    """
    from gemseo.disciplines.factory import DisciplineFactory

    return DisciplineFactory().class_names


def get_surrogate_options_schema(
    surrogate_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the available options for a surrogate discipline.

    Args:
        surrogate_name: The name of the surrogate discipline.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the surrogate discipline.

    Examples:
        >>> from gemseo import get_surrogate_options_schema
        >>> tmp = get_surrogate_options_schema('LinRegSurrogateDiscipline',
        >>>                                    pretty_print=True)
    """
    from gemseo.mlearning import get_regression_options

    return get_regression_options(surrogate_name, output_json, pretty_print)


def get_algorithm_options_schema(
    algorithm_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the options of an algorithm.

    Args:
        algorithm_name: The name of the algorithm.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the algorithm.

    Raises:
        ValueError: When the algorithm is not available.

    Examples:
        >>> from gemseo import get_algorithm_options_schema
        >>> schema = get_algorithm_options_schema("NLOPT_SLSQP", pretty_print=True)
    """
    from gemseo.algos.doe.factory import DOELibraryFactory
    from gemseo.algos.opt.factory import OptimizationLibraryFactory

    for factory in (DOELibraryFactory(), OptimizationLibraryFactory()):
        if factory.is_available(algorithm_name):
            algo_lib = factory.create(algorithm_name)
            settings = algo_lib.ALGORITHM_INFOS[algorithm_name].Settings
            return _get_json_schema_from_settings(
                settings,
                output_json,
                pretty_print,
            )
    msg = f"Algorithm named {algorithm_name} is not available."
    raise ValueError(msg)


def _get_json_schema_from_settings(
    settings: type[BaseModel],
    output_json: bool,
    pretty_print: bool,
) -> str | dict[str, Any]:
    """Return the schema of a JSON grammar.

    Args:
        settings: The algorithm settings.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the JSON grammar.
    """
    schema = settings.model_json_schema()
    if pretty_print:
        _pretty_print_schema(schema)

    if output_json:
        return json.dumps(settings.model_json_schema(), indent=4)

    return schema


def get_discipline_inputs_schema(
    discipline: Discipline,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the inputs of a discipline.

    Args:
        discipline: The discipline.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the inputs of the discipline.

    Examples:
        >>> from gemseo import create_discipline, get_discipline_inputs_schema
        >>> discipline = create_discipline("Sellar1")
        >>> schema = get_discipline_inputs_schema(discipline, pretty_print=True)
    """
    return _get_schema(discipline.io.input_grammar, output_json, pretty_print)


def get_discipline_outputs_schema(
    discipline: Discipline,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the outputs of a discipline.

    Args:
        discipline: The discipline.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the outputs of the discipline.

    Examples:
        >>> from gemseo import get_discipline_outputs_schema, create_discipline
        >>> discipline = create_discipline("Sellar1")
        >>> get_discipline_outputs_schema(discipline, pretty_print=True)
    """
    return _get_schema(discipline.io.output_grammar, output_json, pretty_print)


def get_available_post_processings() -> list[str]:
    """Return the names of the available optimization post-processings.

    Returns:
        The names of the available post-processings.

    Examples:
        >>> from gemseo import get_available_post_processings
        >>> print(get_available_post_processings())
        ['ScatterPlotMatrix', 'VariableInfluence', 'ConstraintsHistory',
        'RadarChart', 'Robustness', 'Correlations', 'SOM', 'KMeans',
        'ParallelCoordinates', 'GradientSensitivity', 'OptHistoryView',
        'BasicHistory', 'ObjConstrHist', 'QuadApprox']
    """
    from gemseo.post.factory import PostFactory

    return PostFactory().class_names


def get_post_processing_options_schema(
    post_proc_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the options of a post-processing.

    Args:
        post_proc_name: The name of the post-processing.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the post-processing.

    Examples:
        >>> from gemseo import get_post_processing_options_schema
        >>> schema = get_post_processing_options_schema('OptHistoryView',
        >>>                                             pretty_print=True)
    """
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.post.factory import PostFactory

    problem = OptimizationProblem(DesignSpace())
    problem.objective = MDOFunction(lambda x: x, "f")
    post_proc = PostFactory().create(post_proc_name, problem)
    return _get_json_schema_from_settings(post_proc.Settings, output_json, pretty_print)


def get_formulation_options_schema(
    formulation_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the options of a formulation.

    Args:
        formulation_name: The name of the formulation.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the formulation.

    Examples:
        >>> from gemseo import get_formulation_options_schema
        >>> schema = get_formulation_options_schema("MDF", pretty_print=True)
    """
    from gemseo.formulations.factory import MDOFormulationFactory

    grammar = MDOFormulationFactory().get_options_grammar(formulation_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_formulation_sub_options_schema(
    formulation_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
    **formulation_settings: Any,
) -> str | dict[str, Any]:
    """Return the schema of the sub-options of a formulation.

    Args:
        formulation_name: The name of the formulation.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.
        **formulation_settings: The settings of the formulation
            required for its instantiation.

    Returns:
        The schema of the sub-options of the formulation, if any.

    Examples:
        >>> from gemseo import get_formulation_sub_options_schema
        >>> schema = get_formulation_sub_options_schema('MDF',
        >>>                                             main_mda_name='MDAJacobi',
        >>>                                             pretty_print=True)
    """
    from gemseo.formulations.factory import MDOFormulationFactory

    grammar = MDOFormulationFactory().get_sub_options_grammar(
        formulation_name, **formulation_settings
    )
    return _get_schema(grammar, output_json, pretty_print)


def get_formulations_sub_options_defaults(
    formulation_name: str,
    **formulation_settings: Any,
) -> dict[str, Any]:
    """Return the default values of the sub-options of a formulation.

    Args:
        formulation_name: The name of the formulation.
        **formulation_settings: The settings of the formulation
            required for its instantiation.

    Returns:
        The default values of the sub-options of the formulation.

    Examples:
        >>> from gemseo import get_formulations_sub_options_defaults
        >>> get_formulations_sub_options_defaults('MDF',
        >>>                                       main_mda_name='MDAJacobi')
    """
    from gemseo.formulations.factory import MDOFormulationFactory

    return MDOFormulationFactory().get_default_sub_option_values(
        formulation_name, **formulation_settings
    )


def get_formulations_options_defaults(
    formulation_name: str,
) -> dict[str, Any]:
    """Return the default values of the options of a formulation.

    Args:
        formulation_name: The name of the formulation.

    Returns:
        The default values of the options of the formulation.

    Examples:
        >>> from gemseo import get_formulations_options_defaults
        >>> get_formulations_options_defaults("MDF")
        {'main_mda_name': 'MDAChain',
         'maximize_objective': False,
         'inner_mda_name': 'MDAJacobi'}
    """
    from gemseo.formulations.factory import MDOFormulationFactory

    return MDOFormulationFactory().get_default_option_values(formulation_name)


def get_discipline_options_schema(
    discipline_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of a discipline.

    Args:
        discipline_name: The name of the formulation.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the discipline.

    Examples:
        >>> from gemseo import get_discipline_options_schema
        >>> schema = get_discipline_options_schema("Sellar1", pretty_print=True)
    """
    from gemseo.disciplines.factory import DisciplineFactory

    disc_fact = DisciplineFactory()
    grammar = disc_fact.get_options_grammar(discipline_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_scenario_options_schema(
    scenario_type: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the options of a scenario.

    Args:
        scenario_type: The type of the scenario.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the scenario.

    Examples:
        >>> from gemseo import get_scenario_options_schema
        >>> get_scenario_options_schema("MDO")
    """
    if scenario_type not in get_available_scenario_types():
        msg = f"Unknown scenario type {scenario_type}"
        raise ValueError(msg)
    scenario_class = {"MDO": "MDOScenario", "DOE": "DOEScenario"}[scenario_type]
    grammar = ScenarioFactory().get_options_grammar(scenario_class)
    return _get_schema(grammar, output_json, pretty_print)


def get_scenario_inputs_schema(
    scenario: BaseScenario,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the inputs of a scenario.

    Args:
        scenario: The scenario.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the inputs of the scenario.

    Examples:
        >>> from gemseo import create_discipline, create_scenario,
        get_scenario_inputs_schema
        >>> from gemseo.problems.mdo.sellar.sellar_design_space import (
        ...     SellarDesignSpace,
        ... )
        >>> design_space = SellarDesignSpace()
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> scenario = create_scenario(
        ...     disciplines,
        ...     "obj",
        ...     design_space,
        ...     formulation_name="MDF",
        ...     scenario_type="MDO",
        ... )
        >>> get_scenario_inputs_schema(scenario)
    """
    return scenario.Settings.model_json_schema()


def get_discipline_options_defaults(
    discipline_name: str,
) -> dict[str, Any]:
    """Return the default values of the options of a discipline.

    Args:
        discipline_name: The name of the discipline.

    Returns:
        The default values of the options of the discipline.

    Examples:
        >>> from gemseo import get_discipline_options_defaults
        >>> get_discipline_options_defaults("Sellar1")
    """
    from gemseo.disciplines.factory import DisciplineFactory

    return DisciplineFactory().get_default_option_values(discipline_name)


def get_scenario_differentiation_modes() -> tuple[
    OptimizationProblem.DifferentiationMethod
]:
    """Return the names of the available differentiation modes of a scenario.

    Returns:
        The names of the available differentiation modes of a scenario.

    Examples:
        >>> from gemseo import get_scenario_differentiation_modes
        >>> get_scenario_differentiation_modes()
    """
    from gemseo.algos.optimization_problem import OptimizationProblem

    return tuple(OptimizationProblem.DifferentiationMethod)


def get_available_scenario_types() -> list[str]:
    """Return the names of the available scenario types.

    Returns:
        The names of the available scenario types.

    Examples:
        >>> from gemseo import get_available_scenario_types
        >>> get_available_scenario_types()
    """
    return ["MDO", "DOE"]


def _get_schema(
    json_grammar: JSONGrammar,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of a JSON grammar.

    Args:
        json_grammar: The JSON grammar to be considered.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the JSON grammar if any.
    """
    if not json_grammar:
        return {}

    dict_schema = json_grammar.schema

    if pretty_print:
        _pretty_print_schema(dict_schema)

    if output_json:
        return json_grammar.to_json()

    return dict_schema


def _pretty_print_schema(schema: dict[str, Any]):
    """Pretty print a json schema.

    Args:
        schema: The json schema to pretty print.
    """
    from prettytable import PrettyTable

    title = schema["name"].replace("_", " ") if "name" in schema else None
    table = PrettyTable(title=title, max_table_width=150)
    names = []
    descriptions = []
    types = []
    for name, value in schema["properties"].items():
        names.append(name)
        descriptions.append(value.get("description"))
        if descriptions[-1] is not None:
            descriptions[-1] = descriptions[-1].split(":type")[0]
            descriptions[-1] = descriptions[-1].capitalize()
            descriptions[-1] = descriptions[-1].replace("\n", " ")
        types.append(value.get("type"))
    table.add_column("Name", names)
    table.add_column("Description", descriptions)
    table.add_column("Type", types)
    table.sortby = "Name"
    table.min_width = 25
    print(table)  # noqa: T201
    LOGGER.info("%s", table)


def get_available_mdas() -> list[str]:
    """Return the names of the available multidisciplinary analyses (MDAs).

    Returns:
        The names of the available MDAs.

    Examples:
        >>> from gemseo import get_available_mdas
        >>> get_available_mdas()
    """
    from gemseo.mda.factory import MDAFactory

    return MDAFactory().class_names


def get_mda_options_schema(
    mda_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any]:
    """Return the schema of the options of a multidisciplinary analysis (MDA).

    Args:
        mda_name: The name of the MDA.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the options of the MDA.

    Examples:
        >>> from gemseo import get_mda_options_schema
        >>> get_mda_options_schema("MDAJacobi")
    """
    from gemseo.mda.factory import MDAFactory

    grammar = MDAFactory().get_options_grammar(mda_name)
    return _get_schema(grammar, output_json, pretty_print)


def create_scenario(
    disciplines: Sequence[Discipline] | Discipline,
    objective_name: str,
    design_space: DesignSpace | str | Path,
    name: str = "",
    scenario_type: str = "MDO",
    maximize_objective: bool = False,
    formulation_settings_model: BaseFormulationSettings | None = None,
    **formulation_settings: Any,
) -> BaseScenario:
    """Initialize a scenario.

    Args:
        disciplines: The disciplines
            used to compute the objective, constraints and observables
            from the design variables.
        objective_name: The name(s) of the discipline output(s) used as objective.
            If multiple names are passed, the objective will be a vector.
        design_space: The search space including at least the design variables
            (some formulations requires additional variables,
            e.g. :class:`.IDF` with the coupling variables).
        name: The name to be given to this scenario.
            If empty, use the name of the class.
        scenario_type: The type of the scenario, e.g. ``"MDO"`` or ``"DOE"``.
        maximize_objective: Whether to maximize the objective.
        formulation_settings_model: The formulation settings as a Pydantic model,
            including the formulation name (use the keyword ``"formulation"``).
            If ``None``, use ``**settings``.
        **formulation_settings: The formulation settings.
            These arguments are ignored when ``settings_model`` is not ``None``.

    Examples:
        >>> from gemseo import create_discipline, create_scenario
        >>> from gemseo.problems.mdo.sellar.sellar_design_space import (
        ...     SellarDesignSpace,
        ... )
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> design_space = SellarDesignSpace()
        >>> scenario = create_scenario(
        >>>     disciplines, "obj", design_space, formulation_name="MDF"
        >>> )
    """
    from gemseo.scenarios.doe_scenario import DOEScenario
    from gemseo.scenarios.mdo_scenario import MDOScenario

    if not isinstance(disciplines, Collection):
        disciplines = [disciplines]

    if isinstance(design_space, (str, Path)):
        design_space = read_design_space(design_space)

    if scenario_type == "MDO":
        cls = MDOScenario
    elif scenario_type == "DOE":
        cls = DOEScenario
    else:
        msg = f"Unknown scenario type: {scenario_type}, use one of : 'MDO' or 'DOE'."
        raise ValueError(msg)

    return cls(
        disciplines,
        objective_name,
        design_space,
        name=name,
        maximize_objective=maximize_objective,
        formulation_settings_model=formulation_settings_model,
        **formulation_settings,
    )


def configure_logger(
    logger_name: str = "",
    level: str | int = logging.INFO,
    date_format: str = DEFAULT_DATE_FORMAT,
    message_format: str = DEFAULT_MESSAGE_FORMAT,
    filename: str | Path = "",
    filemode: str = "a",
) -> Logger:
    """Configure |g| logging.

    Args:
        logger_name: The name of the logger to configure.
            If empty, configure the root logger.
        level: The numerical value or name of the logging level,
            as defined in :py:mod:`logging`.
            Values can either be
            ``logging.NOTSET`` (``"NOTSET"``),
            ``logging.DEBUG`` (``"DEBUG"``),
            ``logging.INFO`` (``"INFO"``),
            ``logging.WARNING`` (``"WARNING"`` or ``"WARN"``),
            ``logging.ERROR`` (``"ERROR"``), or
            ``logging.CRITICAL`` (``"FATAL"`` or ``"CRITICAL"``).
        date_format: The logging date format.
        message_format: The logging message format.
        filename: The path to the log file, if outputs must be written in a file.
        filemode: The logging output file mode,
            either 'w' (overwrite) or 'a' (append).

    Returns:
        The configured logger.

    Examples:
        >>> import logging
        >>> configure_logger(level=logging.WARNING)
    """
    from gemseo.utils.logging_tools import MultiLineFileHandler
    from gemseo.utils.logging_tools import MultiLineStreamHandler

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt=message_format, datefmt=date_format)

    # remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = MultiLineStreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename:
        file_handler = MultiLineFileHandler(
            filename, mode=filemode, delay=True, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    LOGGING_SETTINGS.message_format = message_format
    LOGGING_SETTINGS.date_format = date_format
    LOGGING_SETTINGS.logger = logger
    return logger


# TODO: rename to create_disciplines (plural)
def create_discipline(
    discipline_name: str | Iterable[str],
    **options: Any,
) -> Discipline | list[Discipline]:
    """Instantiate one or more disciplines.

    Args:
        discipline_name: Either the name of a discipline
            or the names of several disciplines.
        **options: The options to be passed to the disciplines constructors.

    Returns:
        The disciplines.

    Examples:
        >>> from gemseo import create_discipline
        >>> discipline = create_discipline("Sellar1")
        >>> discipline.execute()
        {'x_local': array([0.+0.j]),
         'x_shared': array([1.+0.j, 0.+0.j]),
         'y_0': array([0.89442719+0.j]),
         'y_1': array([1.+0.j])}
    """
    from gemseo.disciplines.factory import DisciplineFactory

    factory = DisciplineFactory()
    if isinstance(discipline_name, str):
        return factory.create(discipline_name, **options)

    return [factory.create(d_name, **options) for d_name in discipline_name]


def import_discipline(
    file_path: str | Path, cls: type[Discipline] | None = None
) -> Discipline:
    """Import a discipline from a pickle file.

    Args:
        file_path: The path to the file containing the discipline
            saved with the method :meth:`.Discipline.to_pickle`.
        cls: A class of discipline.
            If ``None``, use ``Discipline``.

    Returns:
        The discipline.
    """
    return from_pickle(file_path)


def create_scalable(
    name: str,
    data: Dataset,
    sizes: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
    **parameters: Any,
) -> DataDrivenScalableDiscipline:
    """Create a scalable discipline from a dataset.

    Args:
        name: The name of the class of the scalable model.
        data: The training dataset.
        sizes: The sizes of the input and output variables.
        **parameters: The parameters of the scalable model.

    Returns:
        The scalable discipline.
    """
    from gemseo.problems.mdo.scalable.data_driven.discipline import (  # noqa:F811
        DataDrivenScalableDiscipline,
    )

    return DataDrivenScalableDiscipline(name, data, sizes, **parameters)


def create_surrogate(
    surrogate: str | BaseRegressor,
    data: IODataset | None = None,
    transformer: TransformerType = BaseRegressor.DEFAULT_TRANSFORMER,
    disc_name: str = "",
    default_input_data: dict[str, ndarray] = READ_ONLY_EMPTY_DICT,
    input_names: Iterable[str] = (),
    output_names: Iterable[str] = (),
    **parameters: Any,
) -> SurrogateDiscipline:
    """Create a surrogate discipline, either from a dataset or a regression model.

    Args:
            surrogate: Either the name of a subclass of :class:`.BaseRegressor`
                or an instance of this subclass.
            data: The training dataset to train the regression model.
                If ``None``, the regression model is supposed to be trained.
            transformer: The strategies to transform the variables.
                This argument is ignored
                when ``surrogate`` is a :class:`.BaseRegressor`;
                in this case,
                these strategies are defined
                with the ``transformer`` argument of this :class:`.BaseRegressor`,
                whose default value is :attr:`.BaseMLAlgo.IDENTITY`,
                which means no transformation.
                In the other cases,
                the values of the dictionary are instances of :class:`.BaseTransformer`
                while the keys can be variable names,
                the group name ``"inputs"``
                or the group name ``"outputs"``.
                If a group name is specified,
                the :class:`.BaseTransformer` will be applied
                to all the variables of this group.
                If :attr:`.BaseMLAlgo.IDENTITY`, do not transform the variables.
                The :attr:`.BaseRegressor.DEFAULT_TRANSFORMER` uses
                the :class:`.MinMaxScaler` strategy for both input and output variables.
            disc_name: The name to be given to the surrogate discipline.
                If empty,
                the name will be ``f"{surrogate.SHORT_ALGO_NAME}_{data.name}``.
            default_input_data: The default values of the input variables.
                If empty,
                use the center of the learning input space.
            input_names: The names of the input variables.
                If empty,
                consider all input variables mentioned in the training dataset.
            output_names: The names of the output variables.
                If empty,
                consider all input variables mentioned in the training dataset.
            **parameters: The parameters of the machine learning algorithm.
    """
    from gemseo.disciplines.surrogate import SurrogateDiscipline  # noqa:F811

    return SurrogateDiscipline(
        surrogate,
        data=data,
        transformer=transformer,
        disc_name=disc_name,
        default_input_data=default_input_data,
        input_names=input_names,
        output_names=output_names,
        **parameters,
    )


def create_mda(
    mda_name: str,
    disciplines: Sequence[Discipline],
    settings_model: BaseMDASettings | None = None,
    **settings: Any,
) -> BaseMDA:
    """Create a multidisciplinary analysis (MDA).

    Args:
        mda_name: The name of the MDA.
        disciplines: The disciplines.
        settings_model: The MDA settings as a Pydantic model.
            If ``None``, use ``**settings``.
            The MDA settings model can be imported from :mod:`gemseo.settings.mda`.
        **settings: The MDA settings as key/value pairs.
            These arguments are ignored when ``settings_model`` is not ``None``.

    Returns:
        The MDA.

    Examples:
        >>> from gemseo import create_discipline, create_mda
        >>> disciplines = create_discipline(["Sellar1", "Sellar2"])
        >>> mda = create_mda("MDAGaussSeidel", disciplines)
        >>> mda.execute()
        {'x_local': array([0.+0.j]),
         'x_shared': array([1.+0.j, 0.+0.j]),
         'y_0': array([0.79999995+0.j]),
         'y_1': array([1.79999995+0.j])}
    """
    from gemseo.mda.factory import MDAFactory

    return MDAFactory().create(
        mda_name,
        disciplines,
        settings_model=settings_model,
        **settings,
    )


def execute_post(
    to_post_proc: BaseScenario | OptimizationProblem | str | Path,
    settings_model: BasePostSettings | None = None,
    **settings: Any,
) -> BasePost:
    """Post-process a result.

    Args:
        to_post_proc: The result to be post-processed,
            either a DOE scenario,
            an MDO scenario,
            an optimization problem
            or a path to an HDF file containing a saved optimization problem.
        settings_model: The post-processor settings as a Pydantic model.
            If ``None``, use ``**settings``.
        **settings: The post-processor settings,
            including the algorithm name (use the keyword ``"post_name"``).
            These arguments are ignored when ``settings_model`` is not ``None``.

    Returns:
        The post-processor.

    Examples:
        >>> from gemseo import create_discipline, create_scenario, execute_post
        >>> from gemseo.problems.mdo.sellar.sellar_design_space import (
        ...     SellarDesignSpace,
        ... )
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> design_space = SellarDesignSpace()
        >>> scenario = create_scenario(
        ...     disciplines,
        ...     "obj",
        ...     design_space,
        ...     formulation_name="MDF",
        ...     name="SellarMDFScenario",
        ... )
        >>> scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
        >>> execute_post(scenario, post_name="OptHistoryView", show=False, save=True)
    """
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.post.factory import PostFactory

    if isinstance(to_post_proc, BaseScenario):
        opt_problem = to_post_proc.formulation.optimization_problem
    elif isinstance(to_post_proc, OptimizationProblem):
        opt_problem = to_post_proc
    elif isinstance(to_post_proc, (str, PathLike)):
        opt_problem = OptimizationProblem.from_hdf(to_post_proc)
    else:
        msg = f"Cannot post process type: {type(to_post_proc)}"
        raise TypeError(msg)
    return PostFactory().execute(opt_problem, settings_model=settings_model, **settings)


def execute_algo(
    opt_problem: OptimizationProblem,
    algo_type: str = "opt",
    settings_model: BaseAlgorithmSettings | None = None,
    **settings: Any,
) -> OptimizationResult:
    """Solve an optimization problem.

    Args:
        opt_problem: The optimization problem to be solved.
        algo_type: The type of algorithm,
            either "opt" for optimization
            or "doe" for design of experiments.
        settings_model: The algorithm settings as a Pydantic model.
            If ``None``, use ``**settings``.
        **settings: The algorithm settings,
            including the algorithm name (use the keyword ``"algo_name"``).
            These arguments are ignored when ``settings_model`` is not ``None``.

    Examples:
        >>> from gemseo import execute_algo
        >>> from gemseo.problems.optimization.rosenbrock import Rosenbrock
        >>> opt_problem = Rosenbrock()
        >>> opt_result = execute_algo(opt_problem, algo_name="SLSQP")
        >>> opt_result
        Optimization result:
        |_ Design variables: [0.99999787 0.99999581]
        |_ Objective function: 5.054173713127532e-12
        |_ Feasible solution: True
    """
    if algo_type == "opt":
        from gemseo.algos.opt.factory import OptimizationLibraryFactory

        factory = OptimizationLibraryFactory()

    elif algo_type == "doe":
        from gemseo.algos.doe.factory import DOELibraryFactory

        factory = DOELibraryFactory()
    else:
        msg = f"Unknown algo type: {algo_type}, please use 'doe' or 'opt' !"
        raise ValueError(msg)

    return factory.execute(opt_problem, settings_model=settings_model, **settings)


def monitor_scenario(
    scenario: BaseScenario,
    observer,
) -> None:
    """Add an observer to a scenario.

    The observer must have an ``update`` method
    that handles the execution status change of an atom.
    `update(atom)` is called everytime an atom execution changes.

    Args:
        scenario: The scenario to monitor.
        observer: The observer that handles an update of status.
    """
    from gemseo.core.monitoring import Monitoring

    # Monitoring object is a singleton
    monitor = Monitoring(scenario)
    monitor.add_observer(observer)


def print_configuration() -> None:
    """Print the current configuration.

    The log message contains the successfully loaded modules
    and failed imports with the reason.

    Examples:
        >>> from gemseo import print_configuration
        >>> print_configuration()
    """
    from gemseo.algos.doe.factory import DOELibraryFactory
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.disciplines.factory import DisciplineFactory
    from gemseo.formulations.factory import MDOFormulationFactory
    from gemseo.mda.factory import MDAFactory
    from gemseo.mlearning.regression.algos.factory import RegressorFactory
    from gemseo.post.factory import PostFactory

    settings = _log_settings()
    LOGGER.info("%s", settings)
    print(settings)  # noqa: T201
    for factory in (
        DisciplineFactory,
        OptimizationLibraryFactory,
        DOELibraryFactory,
        RegressorFactory,
        MDOFormulationFactory,
        MDAFactory,
        PostFactory,
    ):
        factory_repr = repr(factory())
        LOGGER.info("%s", factory_repr)
        print(factory_repr)  # noqa: T201


def read_design_space(
    file_path: str | Path,
    header: Iterable[str] = (),
) -> DesignSpace:
    """Read a design space from a CSV or HDF file.

    In the case of a CSV file,
    the following columns must be in the file:
    "name", "lower_bound" and "upper_bound".
    This file shall contain space-separated values
    (the number of spaces is not important)
    with a row for each variable
    and at least the bounds of the variable.

    Args:
        file_path: The path to the file.
        header: The names of the fields saved in the CSV file.
            If empty, read them in the first row of the CSV file.

    Returns:
        The design space.

    Examples:
        >>> from gemseo import (create_design_space, write_design_space,
        >>>     read_design_space)
        >>> original_design_space = create_design_space()
        >>> original_design_space.add_variable(
        ...     "x", lower_bound=-1, value=0.0, upper_bound=1.0
        ... )
        >>> write_design_space(original_design_space, "file.csv")
        >>> design_space = read_design_space("file.csv")
        >>> print(design_space)
        Design Space:
        +------+-------------+-------+-------------+-------+
        | name | lower_bound | value | upper_bound | type  |
        +------+-------------+-------+-------------+-------+
        | x    |      -1     |   0   |      1      | float |
        +------+-------------+-------+-------------+-------+
    """
    from gemseo.algos.design_space import DesignSpace

    return DesignSpace.from_file(file_path, header=header)


def write_design_space(
    design_space: DesignSpace,
    output_file: str | Path,
    fields: Sequence[str] = (),
    header_char: str = "",
    **table_options: Any,
) -> None:
    """Save a design space to a CSV or HDF file.

    Args:
        design_space: The design space to be saved.
        output_file: The path to the file.
        fields: The fields to be exported.
            If empty, export all fields.
        header_char: The header character.
        **table_options: The names and values of additional attributes
            for the :class:`.PrettyTable` view
            generated by :meth:`.DesignSpace.get_pretty_table`.

    Examples:
        >>> from gemseo import create_design_space, write_design_space
        >>> design_space = create_design_space()
        >>> design_space.add_variable("x", lower_bound=-1, upper_bound=1, value=0.0)
        >>> write_design_space(design_space, "file.csv")
    """
    design_space.to_file(
        output_file, fields=fields, header_char=header_char, **table_options
    )


def create_design_space() -> DesignSpace:
    """Create an empty design space.

    Returns:
        An empty design space.

    Examples:
        >>> from gemseo import create_design_space
        >>> design_space = create_design_space()
        >>> design_space.add_variable("x", lower_bound=-1, upper_bound=1, value=0.0)
        >>> print(design_space)
        Design Space:
        +------+-------------+-------+-------------+-------+
        | name | lower_bound | value | upper_bound | type  |
        +------+-------------+-------+-------------+-------+
        | x    |      -1     |   0   |      1      | float |
        +------+-------------+-------+-------------+-------+
    """
    from gemseo.algos.design_space import DesignSpace

    return DesignSpace()


def create_parameter_space() -> ParameterSpace:
    """Create an empty parameter space.

    Returns:
        An empty parameter space.
    """
    from gemseo.algos.parameter_space import ParameterSpace  # noqa: F811

    return ParameterSpace()


def get_available_caches() -> list[str]:
    """Return the names of the available caches.

    Returns:
        The names of the available caches.

    Examples:
        >>> from gemseo import get_available_caches
        >>> get_available_caches()
        ['BaseFullCache', 'HDF5Cache', 'MemoryFullCache', 'SimpleCache']
    """
    from gemseo.caches.factory import CacheFactory

    return CacheFactory().class_names


def create_cache(
    cache_type: str,
    name: str = "",
    **options: Any,
) -> BaseCache:
    """Return a cache.

    Args:
        cache_type: The type of the cache.
        name: The name to be given to the cache.
            If empty, use ``cache_type``.
        **options: The options of the cache.

    Returns:
        The cache.

    Examples:
        >>> from gemseo import create_cache
        >>> cache = create_cache("MemoryFullCache")
        >>> print(cache)
        +--------------------------------+
        |        MemoryFullCache         |
        +--------------+-----------------+
        |   Property   |      Value      |
        +--------------+-----------------+
        |     Type     | MemoryFullCache |
        |  Tolerance   |       0.0       |
        | Input names  |       None      |
        | Output names |       None      |
        |    Length    |        0        |
        +--------------+-----------------+
    """
    from gemseo.caches.factory import CacheFactory

    return CacheFactory().create(cache_type, name=name, **options)


def create_dataset(
    name: str = "",
    data: ndarray | str | Path = "",
    variable_names: str | Iterable[str] = (),
    variable_names_to_n_components: dict[str, int] = READ_ONLY_EMPTY_DICT,
    variable_names_to_group_names: dict[str, str] = READ_ONLY_EMPTY_DICT,
    delimiter: str = ",",
    header: bool = True,
    class_name: DatasetClassName = DatasetClassName.Dataset,
) -> Dataset:
    """Create a dataset from a NumPy array or a data file.

    Args:
        name: The name to be given to the dataset.
        data: The data to be stored in the dataset,
            either a NumPy array or a file path.
            If empty, return an empty dataset.
        variable_names: The names of the variables.
            If empty, use default names.
        variable_names_to_n_components: The number of components of the variables.
            If empty,
            assume that all the variables have a single component.
        variable_names_to_group_names: The groups of the variables.
            If empty,
            use :attr:`.Dataset.DEFAULT_GROUP` for all the variables.
        delimiter: The field delimiter.
        header: If ``True`` and `data` is a string,
            read the variables names on the first line of the file.
        class_name: The name of the dataset class.

    Returns:
        The dataset generated from the NumPy array or data file.

    Raises:
        ValueError: If ``data`` is neither a file nor an array.
    """
    from gemseo.datasets.factory import DatasetFactory

    dataset_class = DatasetFactory().get_class(class_name)

    if isinstance(data, ndarray):
        dataset = dataset_class.from_array(
            data,
            variable_names,
            variable_names_to_n_components,
            variable_names_to_group_names,
        )
    elif not data:
        dataset = dataset_class()
    elif isinstance(data, (PathLike, str)):
        data = Path(data)
        extension = data.suffix
        if extension == ".csv":
            dataset = dataset_class.from_csv(data, delimiter=delimiter)
        elif extension == ".txt":
            dataset = dataset_class.from_txt(
                data,
                variable_names,
                variable_names_to_n_components,
                variable_names_to_group_names,
                delimiter,
                header,
            )
        else:
            msg = (
                "The dataset can be created from a file with a .csv or .txt extension, "
                f"not {extension}."
            )
            raise ValueError(msg)
    else:
        msg = (
            "The dataset can be created from an array or a .csv or .txt file, "
            f"not a {type(data)}."
        )
        raise ValueError(msg)

    if name:
        dataset.name = name
    return dataset


def create_benchmark_dataset(
    dataset_type: DatasetType,
    **options: Any,
) -> Dataset:
    """Instantiate a dataset.

    Typically, benchmark datasets can be found in :mod:`gemseo.datasets.dataset`.

    Args:
        dataset_type: The type of the dataset.
        **options: The options for creating the dataset.

    Returns:
        The dataset.
    """
    from gemseo.problems.dataset.burgers import create_burgers_dataset
    from gemseo.problems.dataset.iris import create_iris_dataset
    from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset

    return {
        DatasetType.BURGER: create_burgers_dataset,
        DatasetType.IRIS: create_iris_dataset,
        DatasetType.ROSENBROCK: create_rosenbrock_dataset,
    }[dataset_type](**options)


def import_database(
    file_path: str | Path,
    hdf_node_path: str = "",
) -> Database:
    """Load a database from an HDF file path.

    This file could be generated using
    :meth:`.Database.to_hdf`,
    :meth:`.OptimizationProblem.to_hdf`
    or :meth:`.Scenario.save_optimization_history`.

    Args:
        file_path: The path of the HDF file.
        hdf_node_path: The path of the HDF node from which
            the database should be exported.
            If empty, the root node is considered.

    Returns:
        The database.
    """
    from gemseo.algos.database import Database
    from gemseo.algos.optimization_problem import OptimizationProblem

    try:
        return OptimizationProblem.from_hdf(
            file_path, hdf_node_path=hdf_node_path
        ).database
    except KeyError:
        return Database.from_hdf(file_path, hdf_node_path=hdf_node_path)


def compute_doe(
    variables_space: DesignSpace | int,
    unit_sampling: bool = False,
    settings_model: BaseDOESettings | None = None,
    **settings: DriverSettingType,
) -> ndarray:
    """Compute a design of experiments (DOE) in a variables space.

    Args:
        variables_space: Either the variables space to be sampled or its dimension.
        unit_sampling: Whether to sample in the unit hypercube.
            If the value provided in ``variables_space`` is the dimension,
            the samples will be generated in the unit hypercube
            whatever the value of ``unit_sampling``.
        settings_model: The DOE settings as a Pydantic model.
            If ``None``, use ``**settings``.
        **settings: The DOE settings,
            including the algorithm name (use the keyword ``"algo_name"``).
            These arguments are ignored when ``settings_model`` is not ``None``.

    Returns:
          The design of experiments
          whose rows are the samples and columns the variables.

    Examples:
        >>> from gemseo import compute_doe, create_design_space
        >>> variables_space = create_design_space()
        >>> variables_space.add_variable("x", 2, lower_bound=-1.0, upper_bound=1.0)
        >>> doe = compute_doe(variables_space, algo_name="lhs", n_samples=5)
    """
    from gemseo.algos.doe.factory import DOELibraryFactory
    from gemseo.utils.pydantic import get_algo_name

    algo_name = get_algo_name(settings_model, settings)
    library = DOELibraryFactory().create(algo_name)
    return library.compute_doe(
        variables_space,
        unit_sampling=unit_sampling,
        settings_model=settings_model,
        **settings,
    )


def _log_settings() -> str:
    from gemseo.algos.base_driver_library import BaseDriverLibrary
    from gemseo.algos.problem_function import ProblemFunction
    from gemseo.core.discipline import Discipline
    from gemseo.utils.string_tools import MultiLineString

    add_not_prefix = lambda x: "" if x else " not"  # noqa: E731
    text = MultiLineString()
    text.add("Settings")
    text.indent()
    text.add("Discipline")
    text.indent()
    text.add(
        "The caches are {}enabled.",
        add_not_prefix(Discipline.default_cache_type is not Discipline.CacheType.NONE),
    )
    text.add(
        "The counters are {}enabled.",
        add_not_prefix(_ExecutionStatistics.is_enabled),
    )
    text.add(
        "The input data are{} checked before running the discipline.",
        add_not_prefix(Discipline.validate_input_data),
    )
    text.add(
        "The output data are{} checked after running the discipline.",
        add_not_prefix(Discipline.validate_output_data),
    )
    text.dedent()
    text.add("ProblemFunction")
    text.indent()
    text.add(
        "The counters are {}enabled.",
        add_not_prefix(ProblemFunction.enable_statistics),
    )
    text.dedent()
    text.add("BaseDriverLibrary")
    text.indent()
    text.add(
        "The progress bar is {}enabled.",
        add_not_prefix(BaseDriverLibrary.enable_progress_bar),
    )
    return str(text)


def configure(
    enable_discipline_statistics: bool = True,
    enable_function_statistics: bool = True,
    enable_progress_bar: bool = True,
    enable_discipline_cache: bool = True,
    validate_input_data: bool = True,
    validate_output_data: bool = True,
    check_desvars_bounds: bool = True,
) -> None:
    """Update the configuration of |g| if needed.

    This could be useful to speed up calculations in presence of cheap disciplines
    such as analytic formula and surrogate models.

    Warnings:
        This function should be called before calling anything from |g|.

    Args:
        enable_discipline_statistics: Whether to record execution statistics of the
            disciplines such as the execution time,
            the number of executions and the number of linearizations.
        enable_function_statistics: Whether to record the statistics
            attached to the functions,
            in charge of counting their number of evaluations.
        enable_progress_bar: Whether to enable the progress bar
            attached to the drivers,
            in charge to log the execution of the process:
            iteration, execution time and objective value.
        enable_discipline_cache: Whether to enable the discipline cache.
        validate_input_data: Whether to validate the input data of a discipline
            before execution.
        validate_output_data: Whether to validate the output data of a discipline
            after execution.
        check_desvars_bounds: Whether to check the membership of design variables
            in the bounds when evaluating the functions in OptimizationProblem.
    """
    from gemseo.algos.base_driver_library import BaseDriverLibrary
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.problem_function import ProblemFunction
    from gemseo.core.discipline import Discipline

    _ExecutionStatistics.is_enabled = enable_discipline_statistics
    ProblemFunction.enable_statistics = enable_function_statistics
    BaseDriverLibrary.enable_progress_bar = enable_progress_bar
    Discipline.validate_input_data = validate_input_data
    Discipline.validate_output_data = validate_output_data
    Discipline.default_cache_type = (
        Discipline.CacheType.SIMPLE
        if enable_discipline_cache
        else Discipline.CacheType.NONE
    )
    OptimizationProblem.check_bounds = check_desvars_bounds


def wrap_discipline_in_job_scheduler(
    discipline: Discipline,
    scheduler_name: str,
    workdir_path: Path,
    **options: Any,
) -> JobSchedulerDisciplineWrapper:
    """Wrap the discipline within another one to delegate its execution to a job
    scheduler.

    The discipline is serialized to the disk, its input too, then a job file is
    created from a template to execute it with the provided options.
    The submission command is launched, it will setup the environment, deserialize
    the discipline and its inputs, execute it and serialize the outputs.
    Finally, the deserialized outputs are returned by the wrapper.

    All process classes :class:`.MDOScenario`,
    or :class:`.BaseMDA`, inherit from
    :class:`.Discipline` so can be sent to HPCs in this way.

    The job scheduler template script can be provided directly or the predefined
    templates file names in gemseo.wrappers.job_schedulers.template can be used.
    SLURM and LSF templates are provided, but one can use other job schedulers
    or to customize the scheduler commands according to the user needs
    and infrastructure requirements.

    The command to submit the job can also be overloaded.

    Args:
        discipline: The discipline to wrap in the job scheduler.
        scheduler_name: The name of the job scheduler (for instance LSF, SLURM, PBS).
        workdir_path: The path to the workdir
        **options: The submission options.

    Raises:
        OSError: if the job template does not exist.

    Warnings:
        This method serializes the passed discipline so it has to be serializable.
        All disciplines provided in GEMSEO are serializable but it is possible that
        custom ones are not and this will make the submission proess fail.
        Also, see :ref:`platform-paths` to handle paths for cross-platforms.

    Examples:
        This example execute a DOE of 100 points on an MDA, each MDA is executed on 24
        CPUS using the SLURM wrapper, on a HPC, and at most 10 points run in parallel,
        everytime a point of the DOE is computed, another one is submitted to the queue.

        >>> from gemseo.disciplines.wrappers.job_schedulers.factory import (
        ...     JobSchedulerDisciplineWrapperFactory,
        ... )
        >>> from gemseo import create_discipline, create_scenario, create_mda
        >>> from gemseo.problems.mdo.sellar.sellar_design_space import (
        ...     SellarDesignSpace,
        ... )
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> mda = create_mda(disciplines)
        >>> wrapped_mda= wrap_discipline_in_job_scheduler(mda, scheduler_name="SLURM",
        >>>                                               workdir_path="workdir",
        >>>                                               cpus_per_task=24)
        >>> scn=create_scenario(
        >>> ... mda, "obj", SellarDesignSpace(), formulation_name="DisciplinaryOpt", scenario_type="DOE"
        >>> )
        >>> scn.execute(algo_name="lhs", n_samples=100, n_processes=10)

        In this variant, each discipline is wrapped independently in the job scheduler,
        which allows to parallelize more the process because each discipline will run on
        indpendent nodes, whithout being parallelized using MPI. The drawback is that
        each discipline execution will be queued on the HPC.
        A HDF5 cache is attached to the MDA, so all executions will be recorded.
        Each wrapped discipline can also be cached using a HDF cache.

        >>> from gemseo.core.discipline import Discipline
        >>> disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
        >>> wrapped_discs=[wrap_discipline_in_job_scheduler(disc,
        >>>                                                 workdir_path="workdir",
        >>>                                                 cpus_per_task=24,
        >>>                                                 scheduler_name="SLURM"),
        >>>                for disc in disciplines]
        >>> scn=create_scenario(
        >>>     wrapped_discs, "obj", SellarDesignSpace(), formulation_name="MDF", scenario_type="DOE"
        >>> )
        >>> scn.formulation.mda.set_cache(
        ...     Discipline.HDF5_CACHE, hdf_file_path="mda_cache.h5"
        ... )
        >>> scn.execute(algo_name="lhs", n_samples=100, n_processes=10)
    """  # noqa:D205 D212 D415 E501
    from gemseo.disciplines.wrappers.job_schedulers.factory import (
        JobSchedulerDisciplineWrapperFactory,
    )

    return JobSchedulerDisciplineWrapperFactory().wrap_discipline(
        discipline=discipline,
        scheduler_name=scheduler_name,
        workdir_path=workdir_path,
        **options,
    )


def create_scenario_result(
    scenario: BaseScenario | str | Path,
    name: str = "",
    **options: Any,
) -> ScenarioResult | None:
    """Create the result of a scenario execution.

    Args:
        scenario: The scenario to post-process or its path to its HDF5 file.
        name: The class name of the :class:`.ScenarioResult`.
            If empty,
            use the :attr:`~.BaseFormulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME`
            of the :class:`.BaseMDOFormulation` attached to the :class:`.Scenario`.
        **options: The options of the :class:`.ScenarioResult`.

    Returns:
        The result of a scenario execution or ``None`` if not yet executed`.
    """
    # TODO: use Scenario.get_result
    if scenario.optimization_result is None:
        return None

    from gemseo.scenarios.scenario_results.factory import ScenarioResultFactory

    return ScenarioResultFactory().create(
        name or scenario.formulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME,
        scenario=scenario,
        **options,
    )


def sample_disciplines(
    disciplines: Sequence[Discipline],
    input_space: DesignSpace,
    output_names: str | Iterable[str],
    formulation_name: str = "MDF",
    formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    name: str = "Sampling",
    backup_settings: BackupSettings | None = None,
    algo_settings_model: BaseDOESettings | None = None,
    **algo_settings: Any,
) -> IODataset:
    """Sample a set of disciplines associated with an MDO formulation.

    Args:
        disciplines: The disciplines to be sampled.
        input_space: The input space on which to sample the discipline.
        output_names: The names of the outputs of interest.
        n_samples: The number of samples.
        formulation_name: The name of the MDO formulation.
        formulation_settings: The settings of the MDO formulation.
            If empty, use the default ones.
        name: The name of the returned dataset.
            If empty, use the name of the discipline.
        backup_settings: The settings of the backup file to store the evaluations
            if any.
        algo_settings_model: The DOE settings as a Pydantic model.
            If ``None``, use ``**settings``.
        **algo_settings: The DOE settings.
            These arguments are ignored when ``settings_model`` is not ``None``.

    Returns:
        The input-output samples of the disciplines.
    """
    from gemseo.scenarios.doe_scenario import DOEScenario
    from gemseo.utils.string_tools import convert_strings_to_iterable

    output_names = convert_strings_to_iterable(output_names)
    output_names_iterator = iter(output_names)
    scenario = DOEScenario(
        disciplines,
        next(output_names_iterator),
        input_space,
        formulation_name=formulation_name,
        name=name,
        **formulation_settings,
    )
    for output_name in output_names_iterator:
        scenario.add_observable(output_name)

    if not algo_settings_model:
        if "log_problem" not in algo_settings:
            algo_settings["log_problem"] = False

        if "use_one_line_progress_bar" not in algo_settings:
            algo_settings["use_one_line_progress_bar"] = True

    if backup_settings is not None and backup_settings.file_path:
        scenario.set_optimization_history_backup(
            backup_settings.file_path,
            at_each_iteration=backup_settings.at_each_iteration,
            at_each_function_call=backup_settings.at_each_function_call,
            erase=backup_settings.erase,
            load=backup_settings.load,
        )
    scenario.execute(algo_settings_model=algo_settings_model, **algo_settings)
    return scenario.formulation.optimization_problem.to_dataset(
        name=name, opt_naming=False, export_gradients=True
    )


def generate_xdsm(
    discipline: Discipline,
    directory_path: str | Path = ".",
    file_name: str = "xdsm",
    show_html: bool = False,
    save_html: bool = True,
    save_json: bool = False,
    save_pdf: bool = False,
    pdf_build: bool = True,
    pdf_cleanup: bool = True,
    pdf_batchmode: bool = True,
) -> XDSM:
    """Create the XDSM diagram of a discipline.

    Args:
        directory_path: The path of the directory to save the files.
        file_name: The file name without the file extension.
        show_html: Whether to open the web browser and display the XDSM.
        save_html: Whether to save the XDSM as a HTML file.
        save_json: Whether to save the XDSM as a JSON file.
        save_pdf: Whether to save the XDSM as a PDF file;
            use ``save_pdf=True`` and ``pdf_build=False``
            to generate the ``file_name.tex`` and ``file_name.tikz`` files
            without building the PDF file.
        pdf_build: Whether to generate the PDF file when ``save_pdf`` is ``True``.
        pdf_cleanup: Whether to clean up the intermediate files
            (``file_name.tex``, ``file_name.tikz`` and built files)
            used to build the PDF file.
        pdf_batchmode: Whether pdflatex is run in `batchmode`.

    Returns:
        The XDSM diagram of the discipline.
    """
    from gemseo.utils.xdsmizer import XDSMizer

    return XDSMizer(discipline).run(
        directory_path=directory_path,
        save_pdf=save_pdf,
        show_html=show_html,
        save_html=save_html,
        save_json=save_json,
        file_name=file_name,
        pdf_build=pdf_build,
        pdf_cleanup=pdf_cleanup,
        pdf_batchmode=pdf_batchmode,
    )
