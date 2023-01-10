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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Introduction
************

Here is the Application Programming Interface (API) of |g|,
a set of high level functions for ease of use.

Make |g| ever more accessible
-----------------------------

The aim of this API is to provide high level functions that are sufficient
to use |g| in most cases, without requiring a deep knowledge of |g|.

Besides, these functions shall change much less often than the internal
classes, which is key for backward compatibility,
which means ensuring that your current scripts using |g| will be usable
with the future versions of |g|.

Connect |g| to your favorite tools
----------------------------------

The API also facilitates the interfacing of |g|
with a platform or other software.

To interface a simulation software with |g|,
please refer to: :ref:`software_connection`.

Extending |g|
-------------

See :ref:`extending-gemseo`.

Table of contents
*****************

Algorithms
----------

- :meth:`~gemseo.api.create_scenario`
- :meth:`~gemseo.api.execute_algo`
- :meth:`~gemseo.api.get_available_opt_algorithms`
- :meth:`~gemseo.api.get_available_doe_algorithms`
- :meth:`~gemseo.api.get_algorithm_options_schema`

Cache
-----

- :meth:`~gemseo.api.create_cache`
- :meth:`~gemseo.api.get_available_caches`

Configuration
-------------

- :meth:`~gemseo.api.configure_logger`

Coupling
--------

- :meth:`~gemseo.api.generate_n2_plot`
- :meth:`~gemseo.api.generate_coupling_graph`

Design space
------------

- :meth:`~gemseo.api.read_design_space`
- :meth:`~gemseo.api.export_design_space`
- :meth:`~gemseo.api.create_design_space`

Disciplines
-----------

- :meth:`~gemseo.api.create_discipline`
- :meth:`~gemseo.api.import_discipline`
- :meth:`~gemseo.api.get_available_disciplines`
- :meth:`~gemseo.api.get_discipline_inputs_schema`
- :meth:`~gemseo.api.get_discipline_outputs_schema`
- :meth:`~gemseo.api.get_discipline_options_schema`
- :meth:`~gemseo.api.get_discipline_options_defaults`

Formulations
------------

- :meth:`~gemseo.api.create_scenario`
- :meth:`~gemseo.api.get_available_formulations`
- :meth:`~gemseo.api.get_formulation_options_schema`
- :meth:`~gemseo.api.get_formulation_sub_options_schema`
- :meth:`~gemseo.api.get_formulations_sub_options_defaults`
- :meth:`~gemseo.api.get_formulations_options_defaults`

MDA
---

- :meth:`~gemseo.api.create_mda`
- :meth:`~gemseo.api.get_available_mdas`
- :meth:`~gemseo.api.get_mda_options_schema`

Post-processing
---------------

- :meth:`~gemseo.api.execute_post`
- :meth:`~gemseo.api.get_available_post_processings`
- :meth:`~gemseo.api.get_post_processing_options_schema`

Scalable
--------

- :meth:`~gemseo.api.create_scalable`

Scenario
--------

- :meth:`~gemseo.api.create_scenario`
- :meth:`~gemseo.api.monitor_scenario`
- :meth:`~gemseo.api.get_available_scenario_types`
- :meth:`~gemseo.api.get_scenario_options_schema`
- :meth:`~gemseo.api.get_scenario_inputs_schema`
- :meth:`~gemseo.api.get_scenario_differentiation_modes`

Surrogates
----------

- :meth:`~gemseo.api.create_surrogate`
- :meth:`~gemseo.api.get_available_surrogates`
- :meth:`~gemseo.api.get_surrogate_options_schema`

API functions
*************
"""
from __future__ import annotations

import logging
import re
from collections import namedtuple
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING

from numpy import ndarray

if TYPE_CHECKING:
    from logging import Logger
    from matplotlib.figure import Figure
    from gemseo.algos.doe.doe_lib import DOELibraryOptionType
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.algos.opt_result import OptimizationResult
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.cache import AbstractCache
    from gemseo.core.dataset import Dataset
    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.core.scenario import Scenario
    from gemseo.disciplines.surrogate import SurrogateDiscipline
    from gemseo.mda.mda import MDA
    from gemseo.mlearning.core.ml_algo import TransformerType
    from gemseo.problems.scalable.data_driven.discipline import (
        ScalableDiscipline,
    )

from gemseo.mlearning.regression.regression import MLRegressionAlgo

from pathlib import Path

# Most modules are imported directly in the methods, which adds a very small
# overhead, but prevents users from importing them from the API.
# All factories are Singletons which means that the scan for
# plugins is done once only

LOGGER = logging.getLogger(__name__)


# pylint: disable=import-outside-toplevel


def generate_n2_plot(
    disciplines: Sequence[MDODiscipline],
    file_path: str | Path = "n2.pdf",
    show_data_names: bool = True,
    save: bool = True,
    show: bool = False,
    fig_size: tuple[float, float] = (15.0, 10.0),
    open_browser: bool = False,
) -> None:
    """Generate a N2 plot from disciplines.

    It can be static (e.g. PDF, PNG, ...), dynamic (HTML) or both.

    The disciplines are located on the diagonal of the N2 plot
    while the coupling variables are situated on the other blocks
    of the matrix view.
    A coupling variable is outputted by a discipline horizontally
    and enters another vertically.
    On the static plot,
    a blue diagonal block represent a self-coupled discipline,
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
        open_browser: Whether to display the interactive N2 chart in a browser.

    Examples
    --------
    >>> from gemseo.api import create_discipline, generate_n2_plot
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> generate_n2_plot(disciplines)

    See also
    --------
    generate_coupling_graph
    """
    from gemseo.core.coupling_structure import MDOCouplingStructure

    coupling_structure = MDOCouplingStructure(disciplines)
    coupling_structure.plot_n2_chart(
        file_path, show_data_names, save, show, fig_size, open_browser
    )


def generate_coupling_graph(
    disciplines: Sequence[MDODiscipline],
    file_path: str | Path = "coupling_graph.pdf",
    full: bool = True,
) -> None:
    """Generate a graph of the couplings between disciplines.

    Args:
        disciplines: The disciplines from which the graph is generated.
        file_path: The path of the file to save the figure.
        full: If True, generate the full coupling graph.
            Otherwise, generate the condensed one.

    Examples
    --------
    >>> from gemseo.api import create_discipline, generate_coupling_graph
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> generate_coupling_graph(disciplines)

    See also
    --------
    generate_n2_plot
    """
    from gemseo.core.coupling_structure import MDOCouplingStructure

    coupling_structure = MDOCouplingStructure(disciplines)
    if full:
        coupling_structure.graph.export_initial_graph(file_path)
    else:
        coupling_structure.graph.export_reduced_graph(file_path)


def get_available_formulations() -> list[str]:
    """Return the names of the available formulations.

    Returns:
        The names of the available MDO formulations.

    Examples
    --------
    >>> from gemseo.api import get_available_formulations
    >>> get_available_formulations()

    See also
    --------
    create_scenario
    get_formulation_options_schema
    get_formulation_sub_options_schema
    get_formulations_options_defaults
    get_formulations_sub_options_defaults
    """
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory

    return MDOFormulationsFactory().formulations


def get_available_opt_algorithms() -> list[str]:
    """Return the names of the available optimization algorithms.

    Returns:
        The names of the available optimization algorithms.

    Examples
    --------
    >>> from gemseo.api import get_available_opt_algorithms
    >>> get_available_opt_algorithms()

    See also
    --------
    create_scenario
    execute_algo
    get_available_doe_algorithms
    get_algorithm_options_schema
    """
    from gemseo.algos.opt.opt_factory import OptimizersFactory

    return OptimizersFactory().algorithms


def get_available_doe_algorithms() -> list[str]:
    """Return the names of the available design of experiments (DOEs) algorithms.

    Returns:
        The names of the available DOE algorithms.

    Examples
    --------
    >>> from gemseo.api import get_available_doe_algorithms
    >>> get_available_doe_algorithms()

    See also
    --------
    create_scenario
    execute_algo
    get_available_opt_algorithms
    get_algorithm_options_schema
    """
    from gemseo.algos.doe.doe_factory import DOEFactory

    return DOEFactory().algorithms


def get_available_surrogates() -> list[str]:
    """Return the names of the available surrogate disciplines.

    Returns:
        The names of the available surrogate disciplines.

    Examples
    --------
    >>> from gemseo.api import get_available_surrogates
    >>> print(get_available_surrogates())
    ['RBFRegressor', 'GaussianProcessRegressor', 'LinearRegressor', 'PCERegressor']

    See also
    --------
    create_surrogate
    get_surrogate_options_schema
    """
    from gemseo.mlearning.api import get_regression_models

    return get_regression_models()


def get_available_disciplines() -> list[str]:
    """Return the names of the available disciplines.

    Returns:
        The names of the available disciplines.

    Examples
    --------
    >>> from gemseo.api import get_available_disciplines
    >>> print(get_available_disciplines())
    ['RosenMF', 'SobieskiAerodynamics', 'ScalableKriging', 'DOEScenario',
    'MDOScenario', 'SobieskiMission', 'SobieskiDiscipline', 'Sellar1',
    'Sellar2', 'MDOChain', 'SobieskiStructure', 'AutoPyDiscipline',
    'Structure', 'SobieskiPropulsion', 'Scenario', 'AnalyticDiscipline',
    'MDOScenarioAdapter', 'ScalableDiscipline', 'SellarSystem', 'Aerodynamics',
    'Mission', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3',
    'PropaneReaction', 'MDOParallelChain']

    See also
    --------
    create_discipline
    import_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    return DisciplinesFactory().disciplines


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

    Examples
    --------
    >>> from gemseo.api import get_surrogate_options_schema
    >>> tmp = get_surrogate_options_schema('LinRegSurrogateDiscipline',
    >>>                                    pretty_print=True)

    See also
    --------
    create_surrogate
    get_available_surrogates
    """
    from gemseo.mlearning.api import get_regression_options

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

    Examples
    --------
    >>> from gemseo.api import get_algorithm_options_schema
    >>> schema = get_algorithm_options_schema('NLOPT_SLSQP', pretty_print=True)

    See also
    --------
    create_scenario
    execute_algo
    get_available_opt_algorithms
    get_available_doe_algorithms
    get_algorithm_options_schema
    """
    from gemseo.algos.doe.doe_factory import DOEFactory
    from gemseo.algos.opt.opt_factory import OptimizersFactory

    for factory in (DOEFactory(), OptimizersFactory()):
        if factory.is_available(algorithm_name):
            algo_lib = factory.create(algorithm_name)
            opts_gram = algo_lib.init_options_grammar(algorithm_name)
            schema = _get_schema(opts_gram, output_json, pretty_print)
            return schema
    raise ValueError(f"Algorithm named {algorithm_name} is not available.")


def get_discipline_inputs_schema(
    discipline: MDODiscipline,
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

    Examples
    --------
    >>> from gemseo.api import create_discipline, get_discipline_inputs_schema
    >>> discipline = create_discipline('Sellar1')
    >>> schema = get_discipline_inputs_schema(discipline, pretty_print=True)

    See also
    --------
    create_discipline
    import_discipline
    get_available_disciplines
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    return _get_schema(discipline.input_grammar, output_json, pretty_print)


def get_discipline_outputs_schema(
    discipline: MDODiscipline,
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

    Examples
    --------
    >>> from gemseo.api import get_discipline_outputs_schema, create_discipline
    >>> discipline = create_discipline('Sellar1')
    >>> get_discipline_outputs_schema(discipline, pretty_print=True)

    See also
    --------
    create_discipline
    import_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    return _get_schema(discipline.output_grammar, output_json, pretty_print)


def get_available_post_processings() -> list[str]:
    """Return the names of the available optimization post-processings.

    Returns:
        The names of the available post-processings.

    Examples
    --------
    >>> from gemseo.api import get_available_post_processings
    >>> print(get_available_post_processings())
    ['ScatterPlotMatrix', 'VariableInfluence', 'ConstraintsHistory',
    'RadarChart', 'Robustness', 'Correlations', 'SOM', 'KMeans',
    'ParallelCoordinates', 'GradientSensitivity', 'OptHistoryView',
    'BasicHistory', 'ObjConstrHist', 'QuadApprox']

    See also
    --------
    execute_post
    get_post_processing_options_schema
    """
    from gemseo.post.post_factory import PostFactory

    return PostFactory().posts


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

    Examples
    --------
    >>> from gemseo.api import get_post_processing_options_schema
    >>> schema = get_post_processing_options_schema('OptHistoryView',
    >>>                                             pretty_print=True)

    See also
    --------
    execute_post
    get_available_post_processings
    """
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.post.post_factory import PostFactory

    problem = OptimizationProblem(DesignSpace())
    problem.objective = MDOFunction(lambda x: x, "f")
    post_proc = PostFactory().create(problem, post_proc_name)
    return _get_schema(post_proc.opt_grammar, output_json, pretty_print)


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

    Examples
    --------
    >>> from gemseo.api import get_formulation_options_schema
    >>> schema = get_formulation_options_schema('MDF', pretty_print=True)

    See also
    --------
    create_scenario
    get_available_formulations
    get_formulation_sub_options_schema
    get_formulations_options_defaults
    get_formulations_sub_options_defaults
    """
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory

    factory = MDOFormulationsFactory().factory
    grammar = factory.get_options_grammar(formulation_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_formulation_sub_options_schema(
    formulation_name: str,
    output_json: bool = False,
    pretty_print: bool = False,
    **formulation_options: Any,
) -> str | dict[str, Any]:
    """Return the schema of the sub-options of a formulation.

    Args:
        formulation_name: The name of the formulation.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.
        **formulation_options: The options of the formulation
            required for its instantiation.

    Returns:
        The schema of the sub-options of the formulation.

    Examples
    --------
    >>> from gemseo.api import get_formulation_sub_options_schema
    >>> schema = get_formulation_sub_options_schema('MDF',
    >>>                                             main_mda_name='MDAJacobi',
    >>>                                             pretty_print=True)

    See also
    --------
    create_scenario
    get_available_formulations
    get_formulation_options_schema
    get_formulations_options_defaults
    get_formulations_sub_options_defaults
    """
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory

    factory = MDOFormulationsFactory().factory
    grammar = factory.get_sub_options_grammar(formulation_name, **formulation_options)
    return _get_schema(grammar, output_json, pretty_print)


def get_formulations_sub_options_defaults(
    formulation_name: str,
    **formulation_options: Any,
) -> dict[str, Any]:
    """Return the default values of the sub-options of a formulation.

    Args:
        formulation_name: The name of the formulation.
        **formulation_options: The options of the formulation
            required for its instantiation.

    Returns:
        The default values of the sub-options of the formulation.

    Examples
    --------
    >>> from gemseo.api import get_formulations_sub_options_defaults
    >>> get_formulations_sub_options_defaults('MDF',
    >>>                                       main_mda_name='MDAJacobi')

    See also
    --------
    create_scenario
    get_available_formulations
    get_formulation_options_schema
    get_formulation_sub_options_schema
    get_formulations_options_defaults
    """
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory

    factory = MDOFormulationsFactory().factory
    return factory.get_default_sub_options_values(
        formulation_name, **formulation_options
    )


def get_formulations_options_defaults(
    formulation_name: str,
) -> dict[str, Any]:
    """Return the default values of the options of a formulation.

    Args:
        formulation_name: The name of the formulation.

    Returns:
        The default values of the options of the formulation.

    Examples
    --------
    >>> from gemseo.api import get_formulations_options_defaults
    >>> get_formulations_options_defaults('MDF')
    {'main_mda_name': 'MDAChain',
     'maximize_objective': False,
     'inner_mda_name': 'MDAJacobi'}

    See also
    --------
    create_scenario
    get_available_formulations
    get_formulation_options_schema
    get_formulation_sub_options_schema
    get_formulations_sub_options_defaults
    """
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory

    factory = MDOFormulationsFactory().factory
    return factory.get_default_options_values(formulation_name)


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

    Examples
    --------
    >>> from gemseo.api import get_discipline_options_schema
    >>> schema = get_discipline_options_schema('Sellar1', pretty_print=True)

    See also
    --------
    create_discipline
    import_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    disc_fact = DisciplinesFactory()
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

    Examples
    --------
    >>> from gemseo.api import get_scenario_options_schema
    >>> get_scenario_options_schema('MDO')

    See also
    --------
    create_scenario
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differentiation_modes
    """
    if scenario_type not in get_available_scenario_types():
        raise ValueError(f"Unknown scenario type {scenario_type}")
    scenario_class = {"MDO": "MDOScenario", "DOE": "DOEScenario"}[scenario_type]
    return get_discipline_options_schema(scenario_class, output_json, pretty_print)


def get_scenario_inputs_schema(
    scenario: Scenario,
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

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario,
    get_scenario_inputs_schema
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> design_space = SellarDesignSpace()
    >>> disciplines = create_discipline(['Sellar1','Sellar2','SellarSystem'])
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space,
    'my_scenario', 'MDO')
    >>> get_scenario_inputs_schema(scenario)

    See also
    --------
    create_scenario
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_differentiation_modes
    """
    return get_discipline_inputs_schema(scenario, output_json, pretty_print)


def get_discipline_options_defaults(
    discipline_name: str,
) -> dict[str, Any]:
    """Return the default values of the options of a discipline.

    Args:
        discipline_name: The name of the discipline.

    Returns:
        The default values of the options of the discipline.

    Examples
    --------
    >>> from gemseo.api import get_discipline_options_defaults
    >>> get_discipline_options_defaults('Sellar1')

    See also
    --------
    create_discipline
    import_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    factory = DisciplinesFactory().factory
    return factory.get_default_options_values(discipline_name)


def get_scenario_differentiation_modes():
    """Return the names of the available differentiation modes of a scenario.

    Returns:
        The names of the available differentiation modes of a scenario.

    Examples
    --------
    >>> from gemseo.api import get_scenario_differentiation_modes
    >>> get_scenario_differentiation_modes()

    See also
    --------
    create_scenario
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    """
    from gemseo.algos.opt_problem import OptimizationProblem

    return OptimizationProblem.DIFFERENTIATION_METHODS


def get_available_scenario_types() -> list[str]:
    """Return the names of the available scenario types.

    Returns:
        The names of the available scenario types.

    Examples
    --------
    >>> from gemseo.api import get_available_scenario_types
    >>> get_available_scenario_types()

    See also
    --------
    create_scenario
    monitor_scenario
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differentiation_modes
    """
    return ["MDO", "DOE"]


def _get_schema(
    json_grammar: JSONGrammar | None,
    output_json: bool = False,
    pretty_print: bool = False,
) -> str | dict[str, Any] | None:
    """Return the schema of a JSON grammar.

    Args:
        json_grammar: The JSON grammar to be considered.
        output_json: Whether to apply the JSON format to the schema.
        pretty_print: Whether to print the schema in a tabular way.

    Returns:
        The schema of the JSON grammar if any.
    """
    from gemseo.third_party.prettytable import PrettyTable

    if json_grammar is None:
        return None

    dict_schema = json_grammar.schema

    if pretty_print:
        if "name" in dict_schema:
            title = dict_schema["name"].replace("_", " ")
        else:
            title = None
        table = PrettyTable(title=title, max_table_width=150)
        names = []
        descriptions = []
        types = []
        for name, value in dict_schema["properties"].items():
            names.append(name)
            descriptions.append(value.get("description"))
            description = descriptions[-1]
            tmp = []
            if descriptions[-1] is not None:
                descriptions[-1] = descriptions[-1].split(":type")[0]
                descriptions[-1] = descriptions[-1].capitalize()
                descriptions[-1] = descriptions[-1].replace("\n", " ")
                tmp = re.split(r":type ([*\w]+): (.*?)", description)
            if len(tmp) == 4:
                types.append(tmp[3].strip())
            else:
                types.append(value.get("type"))
        table.add_column("Name", names)
        table.add_column("Description", descriptions)
        table.add_column("Type", types)
        table.sortby = "Name"
        table.min_width = 25
        print(table)  # noqa: T201
        LOGGER.info("%s", table)

    if output_json:
        return json_grammar.to_json()

    return dict_schema


def get_available_mdas() -> list[str]:
    """Return the names of the available multidisciplinary analyses (MDAs).

    Returns:
        The names of the available MDAs.

    Examples
    --------
    >>> from gemseo.api import get_available_mdas
    >>> get_available_mdas()

    See also
    --------
    create_mda
    get_mda_options_schema
    """
    from gemseo.mda.mda_factory import MDAFactory

    return MDAFactory().mdas


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

    Examples
    --------
    >>> from gemseo.api import get_mda_options_schema
    >>> get_mda_options_schema('MDAJacobi')

    See also
    --------
    create_mda
    get_available_mdas
    """
    from gemseo.mda.mda_factory import MDAFactory

    factory = MDAFactory().factory
    grammar = factory.get_options_grammar(mda_name)
    return _get_schema(grammar, output_json, pretty_print)


def create_scenario(
    disciplines: Sequence[MDODiscipline],
    formulation: str,
    objective_name: str,
    design_space: DesignSpace | str | Path,
    name: str | None = None,
    scenario_type: str = "MDO",
    grammar_type: str = "JSONGrammar",
    maximize_objective: bool = False,
    **options: Any,
) -> Scenario:
    """Initialize a scenario.

    Args:
        disciplines: The disciplines
            used to compute the objective, constraints and observables
            from the design variables.
        formulation: The class name of the :class:`.MDOFormulation`,
            e.g. ``"MDF"``, ``"IDF"`` or ``"BiLevel"``.
        objective_name: The name(s) of the discipline output(s) used as objective.
            If multiple names are passed, the objective will be a vector.
        design_space: The search space including at least the design variables
            (some formulations requires additional variables,
            e.g. :class:`.IDF` with the coupling variables).
        name: The name to be given to this scenario.
            If ``None``, use the name of the class.
        scenario_type: The type of the scenario, e.g. ``"MDO"`` or ``"DOE"``.
        grammar_type: The type of grammar to declare the input and output variables
            either :attr:`~.MDODiscipline.JSON_GRAMMAR_TYPE`
            or :attr:`~.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.
        maximize_objective: Whether to maximize the objective.
        **options: The options of the :class:`.MDOFormulation`.

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> design_space = SellarDesignSpace()
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space,
    'SellarMDFScenario')

    See also
    --------
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differentiation_modes
    """
    from gemseo.core.doe_scenario import DOEScenario
    from gemseo.core.mdo_scenario import MDOScenario

    if not isinstance(disciplines, list):
        disciplines = [disciplines]

    if isinstance(design_space, (str, Path)):
        design_space = read_design_space(design_space)

    if scenario_type == "MDO":
        return MDOScenario(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name=name,
            grammar_type=grammar_type,
            maximize_objective=maximize_objective,
            **options,
        )

    if scenario_type == "DOE":
        return DOEScenario(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            grammar_type=grammar_type,
            maximize_objective=maximize_objective,
            **options,
        )

    raise ValueError(
        f"Unknown scenario type: {scenario_type}, use one of : 'MDO' or 'DOE'."
    )


def configure_logger(
    logger_name: str | None = None,
    level: str | int = logging.INFO,
    date_format: str = "%H:%M:%S",
    message_format: str = "%(levelname)8s - %(asctime)s: %(message)s",
    filename: str | Path | None = None,
    filemode: str = "a",
) -> Logger:
    """Configure |g| logging.

    Args:
        logger_name: The name of the logger to configure.
            If ``None``, return the root logger.
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

    Examples:
        >>> import logging
        >>> configure_logger(level=logging.WARNING)
    """
    from gemseo.utils.logging_tools import MultiLineFileHandler, MultiLineStreamHandler

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt=message_format, datefmt=date_format)

    # remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = MultiLineStreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = MultiLineFileHandler(
            filename, mode=filemode, delay=True, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_discipline(
    discipline_name: str | Iterable[str],
    **options: Any,
):
    """Instantiate one or more disciplines.

    Args:
        discipline_name: Either the name of a discipline
            or the names of several disciplines.
        **options: The options to be passed to the disciplines constructors.

    Returns:
        The disciplines.

    Examples
    --------
    >>> from gemseo.api import create_discipline
    >>> discipline = create_discipline('Sellar1')
    >>> discipline.execute()
    {'x_local': array([0.+0.j]),
     'x_shared': array([1.+0.j, 0.+0.j]),
     'y_0': array([0.89442719+0.j]),
     'y_1': array([1.+0.j])}

    See also
    --------
    import_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    factory = DisciplinesFactory()
    if isinstance(discipline_name, str):
        return factory.create(discipline_name, **options)

    return [factory.create(d_name, **options) for d_name in discipline_name]


def import_discipline(
    file_path: str | Path, cls: type[MDODiscipline] | None = None
) -> MDODiscipline:
    """Import a discipline from a pickle file.

    Args:
        file_path: The path to the file containing the discipline
            saved with the method :meth:`.MDODiscipline.serialize`.
        cls: A class of discipline.
            If ``None``, use ``MDODiscipline``.

    Returns:
        The discipline.

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    if cls is None:
        from gemseo.core.discipline import MDODiscipline

        cls = MDODiscipline

    return cls.deserialize(file_path)


def create_scalable(
    name: str,
    data: Dataset,
    sizes: Mapping[str, int] = None,
    **parameters: Any,
) -> ScalableDiscipline:
    """Create a scalable discipline from a dataset.

    Args:
        name: The name of the class of the scalable model.
        data: The learning dataset.
        sizes: The sizes of the input and output variables.
        **parameters: The parameters of the scalable model.

    Returns:
        The scalable discipline.
    """
    from gemseo.problems.scalable.data_driven.discipline import (  # noqa:F811
        ScalableDiscipline,
    )

    return ScalableDiscipline(name, data, sizes, **parameters)


def create_surrogate(
    surrogate: str | MLRegressionAlgo,
    data: Dataset | None = None,
    transformer: TransformerType = MLRegressionAlgo.DEFAULT_TRANSFORMER,
    disc_name: str | None = None,
    default_inputs: dict[str, ndarray] | None = None,
    input_names: Iterable[str] | None = None,
    output_names: Iterable[str] | None = None,
    **parameters: Any,
) -> SurrogateDiscipline:
    """Create a surrogate discipline, either from a dataset or a regression model.

    Args:
        surrogate: Either the class name
            or the instance of the :class:`.MLRegressionAlgo`.
        data: The learning dataset to train the regression model.
            If ``None``, the regression model is supposed to be trained.
        transformer: The strategies to transform the variables.
            The values are instances of :class:`.Transformer`
            while the keys are the names of
            either the variables
            or the groups of variables,
            e.g. "inputs" or "outputs" in the case of the regression algorithms.
            If a group is specified,
            the :class:`.Transformer` will be applied
            to all the variables of this group.
            If None, do not transform the variables.
            The :attr:`.MLRegressionAlgo.DEFAULT_TRANSFORMER` uses
            the :class:`.MinMaxScaler` strategy for both input and output variables.
        disc_name: The name to be given to the surrogate discipline.
            If ``None``, concatenate :attr:`.SHORT_ALGO_NAME` and ``data.name``.
        default_inputs: The default values of the inputs.
            If ``None``, use the center of the learning input space.
        input_names: The names of the input variables.
            If ``None``, consider all input variables mentioned in the learning dataset.
        output_names: The names of the output variables.
            If ``None``, consider all input variables mentioned in the learning dataset.
        **parameters: The parameters of the machine learning algorithm.

    See also
    --------
    get_available_surrogates
    get_surrogate_options_schema
    """
    from gemseo.disciplines.surrogate import SurrogateDiscipline  # noqa:F811

    return SurrogateDiscipline(
        surrogate,
        data,
        transformer,
        disc_name,
        default_inputs,
        input_names,
        output_names,
        **parameters,
    )


def create_mda(
    mda_name: str,
    disciplines: Sequence[MDODiscipline],
    **options: Any,
) -> MDA:
    """Create a multidisciplinary analysis (MDA).

    Args:
        mda_name: The name of the MDA.
        disciplines: The disciplines.
        **options: The options of the MDA.

    Returns:
        The MDA.

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_mda
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2'])
    >>> mda = create_mda('MDAGaussSeidel', disciplines)
    >>> mda.execute()
    {'x_local': array([0.+0.j]),
     'x_shared': array([1.+0.j, 0.+0.j]),
     'y_0': array([0.79999995+0.j]),
     'y_1': array([1.79999995+0.j])}

    See also
    --------
    create_mda
    get_available_mdas
    get_mda_options_schema
    """
    from gemseo.mda.mda_factory import MDAFactory

    factory = MDAFactory()
    return factory.create(mda_name=mda_name, disciplines=disciplines, **options)


def execute_post(
    to_post_proc: Scenario | OptimizationProblem | str | Path,
    post_name: str,
    **options: Any,
) -> dict[str, Figure]:
    """Post-process a result.

    Args:
        to_post_proc: The result to be post-processed,
            either a DOE scenario,
            a MDO scenario,
            an optimization problem
            or a path to an HDF file containing a saved optimization problem.
        post_name: The name of the post-processing.
        **options: The post-processing options.

    Returns:
        The figures, to be customized if not closed.

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario, execute_post
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> design_space = SellarDesignSpace()
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space,
    'SellarMDFScenario')
    >>> scenario.execute({'algo': 'NLOPT_SLSQP', 'max_iter': 100})
    >>> execute_post(scenario, 'OptHistoryView', show=False, save=True)

    See also
    --------
    get_available_post_processings
    get_post_processing_options_schema
    """
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.post.post_factory import PostFactory

    if hasattr(to_post_proc, "is_scenario") and to_post_proc.is_scenario():
        opt_problem = to_post_proc.formulation.opt_problem
    elif isinstance(to_post_proc, OptimizationProblem):
        opt_problem = to_post_proc
    elif isinstance(to_post_proc, str):
        opt_problem = OptimizationProblem.import_hdf(to_post_proc)
    else:
        raise TypeError(f"Cannot post process type: {type(to_post_proc)}")
    return PostFactory().execute(opt_problem, post_name, **options)


def execute_algo(
    opt_problem: OptimizationProblem,
    algo_name: str,
    algo_type: str = "opt",
    **options: Any,
) -> OptimizationResult:
    """Solve an optimization problem.

    Args:
        opt_problem: The optimization problem to be solved.
        algo_name: The name of the algorithm to be used to solve optimization problem.
        algo_type: The type of algorithm,
            either "opt" for optimization
            or "doe" for design of experiments.
        **options: The options of the algorithm.

    Examples
    --------
    >>> from gemseo.api import execute_algo
    >>> from gemseo.problems.analytical.rosenbrock import Rosenbrock
    >>> opt_problem = Rosenbrock()
    >>> opt_result = execute_algo(opt_problem, 'SLSQP')
    >>> opt_result
    Optimization result:
    |_ Design variables: [0.99999787 0.99999581]
    |_ Objective function: 5.054173713127532e-12
    |_ Feasible solution: True

    See also
    --------
    get_available_opt_algorithms
    get_available_doe_algorithms
    get_algorithm_options_schema
    """
    if algo_type == "opt":
        from gemseo.algos.opt.opt_factory import OptimizersFactory

        factory = OptimizersFactory()

    elif algo_type == "doe":
        from gemseo.algos.doe.doe_factory import DOEFactory

        factory = DOEFactory()
    else:
        raise ValueError(f"Unknown algo type: {algo_type}, please use 'doe' or 'opt' !")

    return factory.execute(opt_problem, algo_name, **options)


def monitor_scenario(
    scenario: Scenario,
    observer,
) -> None:
    """Add an observer to a scenario.

    The observer must have an ``update`` method
    that handles the execution status change of an atom.
    `update(atom)` is called everytime an atom execution changes.

    Args:
        scenario: The scenario to monitor.
        observer: The observer that handles an update of status.

    See also
    --------
    create_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differentiation_modes
    """
    from gemseo.core.monitoring import Monitoring

    # Monitoring object is a singleton
    monitor = Monitoring(scenario)
    monitor.add_observer(observer)


def print_configuration() -> None:
    """Print the current configuration.

    The log message contains the successfully loaded modules
    and failed imports with the reason.

    Examples
    --------
    >>> from gemseo.api import print_configuration
    >>> print_configuration()
    """
    from gemseo.algos.doe.doe_factory import DOEFactory
    from gemseo.algos.opt.opt_factory import OptimizersFactory
    from gemseo.formulations.formulations_factory import MDOFormulationsFactory
    from gemseo.mda.mda_factory import MDAFactory
    from gemseo.mlearning.regression.factory import RegressionModelFactory
    from gemseo.post.post_factory import PostFactory
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    settings = _log_settings()
    LOGGER.info("%s", settings)
    print(settings)  # noqa: T201
    for factory in (
        DisciplinesFactory,
        OptimizersFactory,
        DOEFactory,
        RegressionModelFactory,
        MDOFormulationsFactory,
        MDAFactory,
        PostFactory,
    ):
        factory_repr = repr(factory().factory)
        LOGGER.info("%s", factory_repr)
        print(factory_repr)  # noqa: T201


def read_design_space(
    file_path: str | Path,
    header: str | None = None,
) -> DesignSpace:
    """Read a design space from a file.

    The following columns must be in the file:
    "name", "lower_bound" and "upper_bound".

    Args:
        file_path: The path to the text file;
            the file shall contain space-separated values
            (the number of spaces is not important)
            with a row for each variable
            and at least the bounds of the variable.
        header: The names of the fields saved in the file.
            If ``None``, read them in fhe first row of the file.

    Returns:
        The design space.

    Examples
    --------
    >>> from gemseo.api import (create_design_space, export_design_space,
    >>>     read_design_space)
    >>> source_design_space = create_design_space()
    >>> source_design_space.add_variable('x', l_b=-1, value=0., u_b=1.)
    >>> export_design_space(source_design_space, 'file.txt')
    >>> read_design_space = read_design_space('file.txt')
    >>> print(read_design_space)
    Design Space:
    +------+-------------+-------+-------------+-------+
    | name | lower_bound | value | upper_bound | type  |
    +------+-------------+-------+-------------+-------+
    | x    |      -1     |   0   |      1      | float |
    +------+-------------+-------+-------------+-------+

    See also
    --------
    export_design_space
    create_design_space
    """
    from gemseo.algos.design_space import DesignSpace

    return DesignSpace.read_from_txt(file_path, header)


def export_design_space(
    design_space: DesignSpace,
    output_file: str | Path,
    export_hdf: bool = False,
    fields: Sequence[str] | None = None,
    header_char: str = "",
    **table_options: Any,
) -> None:
    """Save a design space to a text or HDF file.

    Args:
        design_space: The design space to be saved.
        output_file: The path to the file.
        export_hdf: If True, save to an HDF file.
            Otherwise, save to a text file.
        fields: The fields to be exported.
            If ``None``, export all fields.
        header_char: The header character.
        **table_options: The names and values of additional attributes
            for the :class:`.PrettyTable` view
            generated by :meth:`.DesignSpace.get_pretty_table`.

    Examples
    --------
    >>> from gemseo.api import create_design_space, export_design_space
    >>> design_space = create_design_space()
    >>> design_space.add_variable('x', l_b=-1, u_b=1, value=0.)
    >>> export_design_space(design_space, 'file.txt')

    See also
    --------
    read_design_space
    create_design_space
    """
    if export_hdf:
        design_space.export_hdf(output_file)
    else:
        design_space.export_to_txt(output_file, fields, header_char, **table_options)


def create_design_space() -> DesignSpace:
    """Create an empty design space.

    Returns:
        An empty design space.

    Examples
    --------
    >>> from gemseo.api import create_design_space
    >>> design_space = create_design_space()
    >>> design_space.add_variable('x', l_b=-1, u_b=1, value=0.)
    >>> print(design_space)
    Design Space:
    +------+-------------+-------+-------------+-------+
    | name | lower_bound | value | upper_bound | type  |
    +------+-------------+-------+-------------+-------+
    | x    |      -1     |   0   |      1      | float |
    +------+-------------+-------+-------------+-------+

    See also
    --------
    read_design_space
    export_design_space
    create_design_space
    gemseo.algos.design_space.DesignSpace
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

    Examples
    --------
    >>> from gemseo.api import get_available_caches
    >>> get_available_caches()
    ['AbstractFullCache', 'HDF5Cache', 'MemoryFullCache', 'SimpleCache']

    See also
    --------
    get_available_caches
    gemseo.core.discipline.MDODiscipline.set_cache_policy
    """
    from gemseo.caches.cache_factory import CacheFactory

    return CacheFactory().caches


def create_cache(
    cache_type: str,
    name: str | None = None,
    **options: Any,
) -> AbstractCache:
    """Return a cache.

    Args:
        cache_type: The type of the cache.
        name: The name to be given to the cache.
            If ``None``, use `cache_type`.
        **options: The options of the cache.

    Returns:
        The cache.

    Examples
    --------
    >>> from gemseo.api import create_cache
    >>> cache = create_cache('MemoryFullCache')
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

    See also
    --------
    get_available_caches
    gemseo.core.discipline.MDODiscipline.set_cache_policy
    """
    from gemseo.caches.cache_factory import CacheFactory

    return CacheFactory().create(cache_type, name=name, **options)


def create_dataset(
    name: str,
    data: ndarray | str | Path,
    variables: list[str] | None = None,
    sizes: dict[str, int] | None = None,
    groups: dict[str, str] | None = None,
    by_group: bool = True,
    delimiter: str = ",",
    header: bool = True,
    default_name: str | None = None,
) -> Dataset:
    """Create a dataset from a NumPy array or a data file.

    Args:
        name: The name to be given to the dataset.
        data: The data to be stored in the dataset,
            either a NumPy array or a file path.
        variables: The names of the variables.
            If ``None`` and `header` is True,
            read the names from the first line of the file.
            If ``None`` and `header` is False,
            use default names
            based on the patterns the :attr:`.Dataset.DEFAULT_NAMES`
            associated with the different groups.
        sizes: The sizes of the variables.
            If ``None``,
            assume that all the variables have a size equal to 1.
        groups: The groups of the variables.
            If ``None``,
            use :attr:`.Dataset.DEFAULT_GROUP` for all the variables.
        by_group: If True, store the data by group.
            Otherwise, store them by variables.
        delimiter: The field delimiter.
        header: If True and `data` is a string,
            read the variables names on the first line of the file.
        default_name: The name of the variable to be used as a pattern
            when variables is None.
            If ``None``,
            use the :attr:`.Dataset.DEFAULT_NAMES` for this group if it exists.
            Otherwise, use the group name.

    Returns:
        The dataset generated from the NumPy array or data file.

    See also
    --------
    load_dataset
    """
    from gemseo.core.dataset import Dataset

    dataset = Dataset(name, by_group)
    if isinstance(data, ndarray):
        dataset.set_from_array(data, variables, sizes, groups, default_name)
    else:
        dataset.set_from_file(data, variables, sizes, groups, delimiter, header)
    return dataset


def load_dataset(
    dataset: str,
    **options: Any,
) -> Dataset:
    """Instantiate a dataset.

    Typically, benchmark datasets can be found in :mod:`gemseo.core.dataset`.

    Args:
        dataset: The name of the dataset (its class name).
        **options: The options for creating the dataset.

    Returns:
        The dataset.

    See also
    --------
    create_dataset
    """
    from gemseo.problems.dataset.factory import DatasetFactory

    return DatasetFactory().create(dataset, **options)


def compute_doe(
    variables_space: DesignSpace,
    algo_name: str,
    size: int | None = None,
    unit_sampling: bool = False,
    **options: DOELibraryOptionType,
) -> ndarray:
    """Compute a design of experiments (DOE) in a variables space.

    Args:
        variables_space: The variables space to be sampled.
        size: The size of the DOE.
            If ``None``, the size is deduced from the ``options``.
        algo_name: The DOE algorithm.
        unit_sampling: Whether to sample in the unit hypercube.
        **options: The options of the DOE algorithm.

    Returns:
          The design of experiments
          whose rows are the samples and columns the variables.

    Examples
    --------
    >>> from gemseo.api import compute_doe, create_design_space
    >>> variables_space = create_design_space()
    >>> variables_space.add_variable("x", 2, l_b=-1.0, u_b=1.0)
    >>> doe = compute_doe(variables_space, algo_name="lhs", size=5)

    See also
    --------
    get_available_doe_algorithms
    get_algorithm_options_schema
    execute_algo
    """
    from gemseo.algos.doe.doe_factory import DOEFactory

    library = DOEFactory().create(algo_name)
    return library.compute_doe(
        variables_space, size=size, unit_sampling=unit_sampling, **options
    )


def _log_settings() -> str:
    from gemseo.algos.driver_lib import DriverLib
    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.utils.string_tools import MultiLineString

    add_de_prefix = lambda x: "" if x else "de"  # noqa: E731
    add_not_prefix = lambda x: "" if x else " not"  # noqa: E731
    text = MultiLineString()
    text.add("Settings")
    text.indent()
    text.add("MDODiscipline")
    text.indent()
    text.add(
        "The caches are {}activated.",
        add_de_prefix(MDODiscipline.activate_cache),
    )
    text.add(
        "The counters are {}activated.",
        add_de_prefix(MDODiscipline.activate_counters),
    )
    text.add(
        "The input data are{} checked before running the discipline.",
        add_not_prefix(MDODiscipline.activate_input_data_check),
    )
    text.add(
        "The output data are{} checked after running the discipline.",
        add_not_prefix(MDODiscipline.activate_output_data_check),
    )
    text.dedent()
    text.add("MDOFunction")
    text.indent()
    text.add(
        "The counters are {}activated.",
        add_de_prefix(MDOFunction.activate_counters),
    )
    text.dedent()
    text.add("DriverLib")
    text.indent()
    text.add(
        "The progress bar is {}activated.",
        add_de_prefix(DriverLib.activate_progress_bar),
    )
    return str(text)


AlgorithmFeatures = namedtuple(
    "AlgorithmFeature",
    [
        "algorithm_name",
        "library_name",
        "root_package_name",
        "handle_equality_constraints",
        "handle_inequality_constraints",
        "handle_float_variables",
        "handle_integer_variables",
        "handle_multiobjective",
        "require_gradient",
    ],
)


def get_algorithm_features(
    algorithm_name: str,
) -> AlgorithmFeatures:
    """Return the features of an optimization algorithm.

    Args:
        algorithm_name: The name of the optimization algorithm.

    Returns:
        The features of the optimization algorithm.

    Raises:
        ValueError: When the optimization algorithm does not exist.
    """
    from gemseo.algos.opt.opt_factory import OptimizersFactory

    factory = OptimizersFactory()
    if not factory.is_available(algorithm_name):
        raise ValueError(
            f"{algorithm_name} is not the name of an optimization algorithm."
        )

    driver = factory.create(algorithm_name)
    description = driver.descriptions[algorithm_name]
    return AlgorithmFeatures(
        algorithm_name=description.algorithm_name,
        library_name=description.library_name,
        root_package_name=factory.factory.get_library_name(driver.__class__.__name__),
        handle_equality_constraints=description.handle_equality_constraints,
        handle_inequality_constraints=description.handle_inequality_constraints,
        handle_float_variables=True,
        handle_integer_variables=description.handle_integer_variables,
        handle_multiobjective=description.handle_multiobjective,
        require_gradient=description.require_gradient,
    )


def configure(
    activate_discipline_counters: bool = True,
    activate_function_counters: bool = True,
    activate_progress_bar: bool = True,
    activate_discipline_cache: bool = True,
    check_input_data: bool = True,
    check_output_data: bool = True,
    check_desvars_bounds: bool = True,
) -> None:
    """Update the configuration of |g| if needed.

    This could be useful to speed up calculations in presence of cheap disciplines
    such as analytic formula and surrogate models.

    Warning:
        This function should be called before calling anything from |g|.

    Args:
        activate_discipline_counters: Whether to activate the counters
            attached to the disciplines,
            in charge of counting their execution time,
            number of evaluations
            and number of linearizations.
        activate_function_counters: Whether to activate the counters
            attached to the functions,
            in charge of counting their number of evaluations.
        activate_progress_bar: Whether to activate the progress bar
            attached to the drivers,
            in charge to log the execution of the process:
            iteration, execution time and objective value.
        activate_discipline_cache: Whether to activate the discipline cache.
        check_input_data: Whether to check the input data of a discipline
            before execution.
        check_output_data: Whether to check the output data of a discipline
            before execution.
        check_desvars_bounds: Whether to check the membership of design variables
            in the bounds when evaluating the functions in OptimizationProblem.
    """
    from gemseo.algos.driver_lib import DriverLib
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.mdofunctions.mdo_function import MDOFunction

    MDODiscipline.activate_counters = activate_discipline_counters
    MDOFunction.activate_counters = activate_function_counters
    DriverLib.activate_progress_bar = activate_progress_bar
    MDODiscipline.activate_input_data_check = check_input_data
    MDODiscipline.activate_output_data_check = check_output_data
    MDODiscipline.activate_cache = activate_discipline_cache
    OptimizationProblem.activate_bound_check = check_desvars_bounds
