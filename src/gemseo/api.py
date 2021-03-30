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
------------------------------

The aim of this API is to provide high level functions that are sufficient
to use |g| in most cases, without requiring a deep knowledge of |g|.

Besides, these functions shall change much less often than the internal
classes, which is key for backward compatibility,
which means ensuring that your current scripts using |g| will be usable
with the future versions of |g|.

Connect |g| to your favorite tools
-----------------------------------

The API also facilitates the interfacing of |g|
with a platform or other software.

To interface a simulation software with |g|,
please refer to: :ref:`software_connection`.

.. _extending-gemseo:

Extending |g|
--------------

|g| features can be extended with external python modules. All kinds of
additionnal features can be implemented: disciplines, algorithms, formulations,
post-processings, surrogate models, ... There are 2 ways to extend |g| with a
directory that contains the python modules:

- by adding the directory to the :envvar:`PYTHONPATH` if the directory name
  starts with :file:`gemseo_`,
- by setting the environment variable :envvar:`GEMSEO_PATH` to the directory
  path, multiple directories can be separated by :envvar:`:`.

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
- :meth:`~gemseo.api.get_all_inputs`
- :meth:`~gemseo.api.get_all_outputs`

Design space
------------

- :meth:`~gemseo.api.read_design_space`
- :meth:`~gemseo.api.export_design_space`
- :meth:`~gemseo.api.create_design_space`

Disciplines
-----------

- :meth:`~gemseo.api.create_discipline`
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
- :meth:`~gemseo.api.get_scenario_differenciation_modes`

Surrogates
----------

- :meth:`~gemseo.api.create_surrogate`
- :meth:`~gemseo.api.get_available_surrogates`
- :meth:`~gemseo.api.get_surrogate_options_schema`

API functions
*************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re

from future import standard_library
from six import string_types

from gemseo.third_party.prettytable import PrettyTable

# Most modules are imported directly in the methods, which adds a very small
# overhead, but prevents users from importing them from the API.
# All factories are Singletons which means that the scan for
# plugins is done once only
standard_library.install_aliases()

from gemseo import LOGGER

# pylint: disable=import-outside-toplevel


def generate_n2_plot(
    disciplines,
    file_path="n2.pdf",
    show_data_names=True,
    save=True,
    show=False,
    figsize=(15, 10),
):
    """
    Generate a N2 plot for the disciplines list.

    :param disciplines: List of disciplines to analyze.
    :type disciplines: list(MDODiscipline)
    :param file_path: File path of the figure.
    :type file_path: str
    :param show_data_names: If true, the names of the
        coupling data is shown
        otherwise, circles are drawn, which size depend on the
        number of coupling names.
    :type show_data_names: bool
    :param save: If True, saved the figure to file_path.
    :type save: bool
    :param show: If True, shows the plot.
    :type show: bool
    :param figsize: Size of the figure.
    :type figsize: tuple(float)

    Examples
    --------
    >>> from gemseo.api import create_discipline, generate_n2_plot
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> generate_n2_plot(disciplines)

    See also
    --------
    generate_coupling_graph
    get_all_inputs
    get_all_outputs
    """
    from gemseo.core.coupling_structure import MDOCouplingStructure

    coupling_structure = MDOCouplingStructure(disciplines)
    coupling_structure.plot_n2_chart(file_path, show_data_names, save, show, figsize)


def generate_coupling_graph(disciplines, file_path="coupling_graph.pdf"):
    """
    Generate a graph of the couplings for the disciplines list.

    :param disciplines: List of disciplines to analyze.
    :type disciplines: list(MDODiscipline)
    :param file_path: File path of the figure.
    :type file_path: str

    Examples
    --------
    >>> from gemseo.api import create_discipline, generate_coupling_graph
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> generate_coupling_graph(disciplines)

    See also
    --------
    generate_n2_plot
    get_all_inputs
    get_all_outputs
    """
    from gemseo.core.coupling_structure import MDOCouplingStructure

    coupling_structure = MDOCouplingStructure(disciplines)
    coupling_structure.graph.export_initial_graph(file_path)


def get_available_formulations():
    """
    List the available formulations in the current configuration.

    :return: list of available MDO formulations.
    :rtype: list(str)

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


def get_available_opt_algorithms():
    """
    List the available optimization algorithms
    names in the present configuration.

    :return: list of available optimization algorithms.
    :rtype: list(str)

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


def get_available_doe_algorithms():
    """
    List the available Design of Experiments algorithms
    names in the present configuration.

    :return: list of available DOE algorithms.
    :rtype: list(str)

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


def get_available_surrogates():
    """
    List the available surrogate model
    names in the present configuration

    :return: list of available surrogate disciplines.
    :rtype: list(str)

    Examples
    --------
    >>> from gemseo.api import get_available_surrogates
    >>> print get_available_surrogates()
    ['RBFRegression', 'GaussianProcessRegression', 'LinearRegression', 'PCERegression']

    See also
    --------
    create_surrogate
    get_surrogate_options_schema
    """
    from gemseo.mlearning.api import get_regression_models

    return get_regression_models()


def get_available_disciplines():
    """
    List the available disciplines names in the present configuration.

    :return: list of available disciplines.
    :rtype: list(str)

    Examples
    --------
    >>> from gemseo.api import get_available_disciplines
    >>> print get_available_disciplines()
    ['RosenMF', 'SobieskiAerodynamics', 'ScalableKriging', 'DOEScenario', 'MDOScenario', 'SobieskiMission', 'SobieskiBaseWrapper', 'Sellar1', 'Sellar2', 'MDOChain', 'SobieskiStructure', 'AutoPyDiscipline', 'Structure', 'SobieskiPropulsion', 'Scenario', 'AnalyticDiscipline', 'MDOScenarioAdapter', 'ScalableDiscipline', 'SellarSystem', 'Aerodynamics', 'Mission', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3', 'PropaneReaction', 'MDOParallelChain']

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    return DisciplinesFactory().disciplines


def get_surrogate_options_schema(surrogate_name, output_json=False, pretty_print=False):
    """
    Lists the available options for a surrogate discipline.

    :param surrogate_name: Name of the surrogate discipline
    :type surrogate_name: str
    :param output_json: Apply json format for the schema
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :returns: Option schema (string) of the surrogate discipline

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


def get_algorithm_options_schema(algorithm_name, output_json=False, pretty_print=False):
    """
    Get the options schema as a JSON Schema string or dictionary
    for a given algorithm.

    :param algorithm_name: Name of the algorithm
    :type algorithm_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: print the schema in a pretty table.
    :type pretty_print: bool
    :return: the option schema (string) of the algorithm

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
    raise ValueError("Algorithm named " + str(algorithm_name) + " is not available.")


def get_discipline_inputs_schema(discipline, output_json=False, pretty_print=False):
    """
    Get the inputs schema of a discipline.

    :param discipline: MDODiscipline instance.
    :type discipline: MDODiscipline
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :return: Option schema of the discipline inputs.
    :rtype: string

    Examples
    --------
    >>> from gemseo.api import create_discipline, get_discipline_inputs_schema
    >>> discipline = create_discipline('Sellar1')
    >>> schema = get_discipline_inputs_schema(discipline, pretty_print=True)

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    return _get_schema(discipline.input_grammar, output_json, pretty_print)


def get_discipline_outputs_schema(discipline, output_json=False, pretty_print=False):
    """
    Get the outputs schema of a discipline.

    :param discipline: MDODiscipline instance
    :type discipline: MDODiscipline
    :param output_json: If True, return a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :return: Schema of the discipline outputs
    :rtype: string

    Examples
    --------
    >>> from gemseo.api import get_discipline_outputs_schema, create_discipline
    >>> discipline = create_discipline('Sellar1')
    >>> get_discipline_outputs_schema(discipline, pretty_print=True)

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    return _get_schema(discipline.output_grammar, output_json, pretty_print)


def get_available_post_processings():
    """
    List the available optimization post-processings.

    :return: List of available post-processings.
    :rtype: list(str)

    Examples
    --------
    >>> from gemseo.api import get_available_post_processings
    >>> print get_available_post_processings()
    ['ScatterPlotMatrix', 'VariableInfluence', 'ConstraintsHistory', 'RadarChart', 'Robustness', 'Correlations', 'SOM', 'KMeans', 'ParallelCoordinates', 'GradientSensitivity', 'OptHistoryView', 'BasicHistory', 'ObjConstrHist', 'QuadApprox']

    See also
    --------
    execute_post
    get_post_processing_options_schema
    """
    from gemseo.post.post_factory import PostFactory

    return PostFactory().posts


def get_post_processing_options_schema(
    post_proc_name, output_json=False, pretty_print=False
):
    """
    List the options schema of a post-processing.

    :param post_proc_name: Post processing name.
    :type post_proc_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :type output_json: bool
    :return: Option schema of the post processing

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
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.post.post_factory import PostFactory

    post_proc = PostFactory().create(OptimizationProblem(None), post_proc_name)
    return _get_schema(post_proc.opt_grammar, output_json, pretty_print)


def get_formulation_options_schema(
    formulation_name, output_json=False, pretty_print=False
):
    """
    Get the options schema of a MDO formulation.

    :param formulation_name: Name of the MDO formulation.
    :type formulation_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :type output_json: bool
    :return: Option schema (string) of the MDO formulation.

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
    formulation_name, output_json=False, pretty_print=False, **formulation_options
):
    """
    Get the sub-options schema of a MDO formulation.

    :param formulation_name: Name of the MDO formulation
    :type formulation_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :param formulation_options: Options to be passed to the formulation;
        this is required to instantiate it.
    :return: Sub-option schema of the MDO formulation

    Examples
    --------
    >>> from gemseo.api import get_formulation_sub_options_schema
    >>> schema = get_formulation_sub_options_schema('MDF',
    >>>                                             main_mda_class='MDAJacobi',
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


def get_formulations_sub_options_defaults(formulation_name, **formulation_options):
    """
    Get the default values of the sub options of a formulation

    :param formulation_name: Name of the discipline.
    :type formulation_name: str
    :param formulation_options: Options to be passed to the formulation;
        this is required to instantiate it.
    :return: Default values of the sub-options of the MDO formulation.

    Examples
    --------
    >>> from gemseo.api import get_formulations_sub_options_defaults
    >>> get_formulations_sub_options_defaults('MDF',
    >>>                                       main_mda_class='MDAJacobi')

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


def get_formulations_options_defaults(formulation_name):
    """
    Get the default values of the options of a formulation

    :param formulation_name: Name of the discipline.
    :type formulation_name: str
    :return: Default values of the options of the MDO formulation.

    Examples
    --------
    >>> from gemseo.api import get_formulations_options_defaults
    >>> get_formulations_options_defaults('MDF')
    {'main_mda_class': 'MDAChain',
     'maximize_objective': False,
     'sub_mda_class': 'MDAJacobi'}

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
    discipline_name, output_json=False, pretty_print=False
):
    """
    Get the options schema of a discipline

    :param discipline_name: Name of the discipline.
    :type discipline_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: print the schema in a pretty table.
    :type pretty_print: bool
    :return: Options schema of the discipline

    Examples
    --------
    >>> from gemseo.api import get_discipline_options_schema
    >>> schema = get_discipline_options_schema('Sellar1', pretty_print=True)

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    disc_fact = DisciplinesFactory()
    grammar = disc_fact.get_options_grammar(discipline_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_scenario_options_schema(scenario_type, output_json=False, pretty_print=False):
    """
    Get the options schema of a scenario

    :param scenario_type: Type of scenario (DOE, MDO...)
    :type scenario_type: str
    :param output_json: True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :return: Options schema of the scenario

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
    get_scenario_differenciation_modes
    """
    if scenario_type not in get_available_scenario_types():
        raise ValueError("Unknown scenario type " + str(scenario_type))
    scenario_class = {"MDO": "MDOScenario", "DOE": "DOEScenario"}[scenario_type]
    return get_discipline_options_schema(scenario_class, output_json, pretty_print)


def get_scenario_inputs_schema(scenario, output_json=False, pretty_print=False):
    """
    Get the schema of the inputs of a scenario

    :param scenario: Scenario instance
    :type scenario: Scenario
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :return: Schema of the scenario inputs

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario, get_scenario_inputs_schema
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> design_space = SellarDesignSpace()
    >>> disciplines = create_discipline(['Sellar1','Sellar2','SellarSystem'])
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space, 'my_scenario', 'MDO')
    >>> get_scenario_inputs_schema(scenario)

    See also
    --------
    create_scenario
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_differenciation_modes
    """
    return get_discipline_inputs_schema(scenario, output_json, pretty_print)


def get_discipline_options_defaults(discipline_name):
    """
    Get the default values of the options of a discipline.

    :param discipline_name: Name of the discipline.
    :type discipline_name: str
    :return: Default option values of a discipline.

    Examples
    --------
    >>> from gemseo.api import get_discipline_options_defaults
    >>> get_discipline_options_defaults('Sellar1')

    See also
    --------
    create_discipline
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    factory = DisciplinesFactory().factory
    return factory.get_default_options_values(discipline_name)


def get_scenario_differenciation_modes():
    """
    List the available differenciation modes of a scenario

    :returns: List of differenciation modes.
    :rtype: list(str)

    Examples
    --------
    >>> from gemseo.api import get_scenario_differenciation_modes
    >>> get_scenario_differenciation_modes()

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


def get_available_scenario_types():
    """
    List the available scenario types.

    :return: list of available scenario types.
    :rtype: list(str)

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
    get_scenario_differenciation_modes
    """
    return ["MDO", "DOE"]


def _get_schema(json_grammar, output_json=False, pretty_print=False):
    """
    Get the schema of a JSON grammar

    :param json_grammar: a JSONGrammar instance
    :type json_grammar: JSONGrammar
    :param output_json: if True, returns a JSON string,
    :type output_json: bool
    :param pretty_print: print the schema in a pretty table.
    :type pretty_print: bool
    :return: a schema
    :rtype: dict or str
    """
    if json_grammar is None:
        return None
    schema = json_grammar.schema
    dict_schema = schema.to_dict()
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
    if output_json:
        return schema.to_json()
    return dict_schema


def get_available_mdas():
    """
    List the available MDAs.

    :return: list of available MDAs.
    :rtype: list(str)

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


def get_mda_options_schema(mda_name, output_json=False, pretty_print=False):
    """
    Get the options schema of a MDA.

    :param mda_name: Name of the MDA.
    :type mda_name: str
    :param output_json: If True, returns a JSON string,
        return a dict otherwise.
    :type output_json: bool
    :param pretty_print: Print the schema in a pretty table.
    :type pretty_print: bool
    :return: MDA options schema
    :rtype: dict or str

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


def get_all_inputs(disciplines, recursive=False):
    """
    List all the inputs of the disciplines.

    Merge the input data from the disciplines grammars.

    :param disciplines: List of disciplines to search
        for inputs.
    :type disciplines: list(MDODiscipline)
    :param recursive: If True, searches for the inputs of the
        sub disciplines (when some disciplines are scenarios).
    :type recursive: bool
    :returns: List of input data.

    Examples
    --------
    >>> from gemseo.api import create_discipline, get_all_inputs
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2'])
    >>> get_all_inputs(disciplines)
    ['y_0', 'x_shared', 'y_1', 'x_local']

    See also
    --------
    generate_n2_plot
    generate_coupling_graph
    get_all_outputs
    """
    from gemseo.utils.data_conversion import DataConversion

    return DataConversion.get_all_inputs(disciplines, recursive)


def get_all_outputs(disciplines, recursive=False):
    """
    List all the outputs of the disciplines.

    Merge the output data from the disicplines grammars.

    :param disciplines: List of disciplines to search
        for outputs.
    :type disciplines: list(MDODiscipline)
    :param recursive: If True, searches for the outputs of the
        sub disciplines (when some disciplines are scenarios).
    :type recursive: bool
    :returns: List of input data

    Examples
    --------
    >>> from gemseo.api import create_discipline, get_all_outputs
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2'])
    >>> get_all_outputs(disciplines)
    ['y_1', 'y_0']

    See also
    --------
    generate_n2_plot
    generate_coupling_graph
    get_all_inputs
    """
    from gemseo.utils.data_conversion import DataConversion

    return DataConversion.get_all_outputs(disciplines, recursive)


def create_scenario(
    disciplines,
    formulation,
    objective_name,
    design_space,
    name=None,
    scenario_type="MDO",
    maximize_objective=False,
    **options
):
    """
    Create a scenario.

    :param disciplines: Disciplines of the scenario.
    :type disciplines: list(MDODiscipline)
    :param formulation: Formulation name.
    :type formulation: str
    :param objective_name: Objective function name.
    :type objective_name: str
    :param design_space: Design space object or a file that contains
        the design space.
    :type design_space: DesignSpace or str
    :param name: scenario name
    :type name: str
    :param scenario_type: Type of scenario, e.g. "DOE" or "MDO".
    :type scenario_type: str
    :param maximize_objective: Maximize function objective.
    :type maximize_objective: bool
    :param options: Options passed to the MDO formulation.
    :returns: Scenario.
    :rtype: Scenario

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> design_space = SellarDesignSpace()
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space,
    >>>                            'SellarMDFScenario', 'MDO')

    See also
    --------
    monitor_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differenciation_modes
    """
    from gemseo.core.doe_scenario import DOEScenario
    from gemseo.core.mdo_scenario import MDOScenario

    if not isinstance(disciplines, list):
        disciplines = [disciplines]

    if isinstance(design_space, string_types):
        design_space = read_design_space(design_space)

    if scenario_type == "MDO":
        return MDOScenario(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            maximize_objective=maximize_objective,
            **options
        )

    if scenario_type == "DOE":
        return DOEScenario(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            maximize_objective=maximize_objective,
            **options
        )

    raise ValueError(
        "Unknown scenario type :"
        + str(scenario_type)
        + ", use one of : 'MDO' or 'DOE'."
    )


def configure_logger(
    logger_name=None,
    level=None,
    date_format=None,
    message_format=None,
    filename=None,
    filemode="a",
):
    """Set the logger configuration.

    :param logger_name: Name of the logger to configure, by default,
        the root logger.
    :type logger_name: str
    :param level: Logger print level, default 'INFO', can be:
        'DEBUG', 'INFO', 'WARNING' or 'CRITICAL'.
    :type level: str
    :param date_format: Date format. If None, use a default one.
    :type date_format: str
    :param message_format: Message format. If None, use a default one.
    :type message_format: str
    :param filename: File path if outputs must be written in a file.
        Default value: None.
    :type filename: str
    :param filemode: Write ('w') or append ('a') to the log file.
    :type filemode: str

    Examples
    --------
    >>> from gemseo.api import configure_logger
    >>> configure_logger('GEMSEO', 'WARNING')
    """
    from gemseo.core.logger_config import LoggerConfig

    logger = logging.getLogger(logger_name)
    config = LoggerConfig(logger)
    config.set_logger_config(level, date_format, message_format, filename, filemode)
    return logger


def create_discipline(discipline_name, **options):
    """
    Create disciplines that are known to |g|.

    |g| knows the disciplines located in the following directories:

    - the directories listed in the environment variable *GEMSEO_PATH*,
    - the directories located in the *problems* package,

    :param discipline_name: Name of the discipline, or list of names.
    :type discipline_name: MDODiscipline
    :param options: Additional options to be passed to the discipline
        constructor.
    :returns: Disciplines.
    :rtype: list(MDODiscipline)

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
    get_available_disciplines
    get_discipline_inputs_schema
    get_discipline_outputs_schema
    get_discipline_options_schema
    get_discipline_options_defaults
    """
    from gemseo.problems.disciplines_factory import DisciplinesFactory

    factory = DisciplinesFactory()
    if isinstance(discipline_name, list):
        return [factory.create(d_name, **options) for d_name in discipline_name]
    return factory.create(discipline_name, **options)


def create_scalable(name, data, sizes=None, **parameters):
    """Create a scalable discipline.

    :param str name: scalable model class name.
    :param AbstractFullCache data: learning dataset.
    :param dict sizes: sizes of input and output variables.
    :param parameters: model parameters
    """
    from gemseo.problems.scalable.discipline import ScalableDiscipline

    return ScalableDiscipline(name, data, sizes, **parameters)


def create_surrogate(
    surrogate,
    data=None,
    transformer=None,
    disc_name=None,
    default_inputs=None,
    input_names=None,
    output_names=None,
    **parameters
):
    """
    Create a surrogate discipline.

    :param surrogate: name of the surrogate model algorithm.
    :type surrogate: str or MLRegressionAlgo
    :param Dataset data: dataset to train the surrogate. If None,
        assumes that the surrogate is trained. Default: None.
    :param dict(str) transformer: transformation strategy for data groups.
        If None, do not transform data. Default: None.
    :param str disc_name: Surrogate discipline name.
    :param dict default_inputs: default inputs. If None, use the first
        sample from the dataset. Default: None.
    :param list(str) input_names: list of input names. If None, use all inputs.
        Default: None.
    :param list(str) output_names: list of output names. If None, use all
        outputs. Default: None.
    :param parameters: Additional parameters to be passed to the surrogate
        for its construction.
    :returns: Surrogate discipline instance.
    :rtype: SurrogateDiscipline

    See also
    --------
    get_available_surrogates
    get_surrogate_options_schema
    """
    from gemseo.core.surrogate_disc import SurrogateDiscipline

    return SurrogateDiscipline(
        surrogate,
        data,
        transformer,
        disc_name,
        default_inputs,
        input_names,
        output_names,
        **parameters
    )


def create_mda(mda_name, disciplines, **options):
    """
    Create an MDA.

    :param mda_name: Name of the MDA (its classname).
    :type mda_name: str
    :param disciplines: List of the disciplines.
    :type disciplines: list(MDODiscipline)
    :param options: Additional options specific to the MDA.
    :returns: MDA instance.
    :rtype: MDA

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


def execute_post(to_post_proc, post_name, **options):
    """
    Execute a post-processing method.

    :param to_post_proc: MDO or DOE scenario, or an optimization problem,
        or a path to a HDF file containing a saved OptimizationProblem.
    :type to_post_proc: MDOScenario, DOEScenario, OptimizationProblem,
        or str
    :param post_name: Post processing name.
    :type post_name: str
    :param options: Post-processing options.

    Examples
    --------
    >>> from gemseo.api import create_discipline, create_scenario, execute_post
    >>> from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    >>> disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
    >>> design_space = SellarDesignSpace()
    >>> scenario = create_scenario(disciplines, 'MDF', 'obj', design_space,
    >>>                            'SellarMDFScenario', 'MDO')
    >>> scenario.execute({'algo': 'NLOPT_SLSQP', 'max_iter': 100})
    >>> execute_post(scenario, 'OptHistoryView', show=False, save=True)

    See also
    --------
    get_available_post_processings
    get_post_processing_options_schema
    """
    from gemseo.algos import OptimizationProblem
    from gemseo.post.post_factory import PostFactory

    if hasattr(to_post_proc, "is_scenario") and to_post_proc.is_scenario():
        opt_problem = to_post_proc.formulation.opt_problem
    elif isinstance(to_post_proc, OptimizationProblem):
        opt_problem = to_post_proc
    elif isinstance(to_post_proc, string_types):
        opt_problem = OptimizationProblem.import_hdf(to_post_proc)
    else:
        raise TypeError("Cannot post process type : " + str(type(to_post_proc)))
    return PostFactory().execute(opt_problem, post_name, **options)


def execute_algo(opt_problem, algo_name, algo_type="opt", **options):
    """
    Solve an optimization problem using either a DOE or an
    Optimization algorithm.

    :param opt_problem: the OptimizationProblem to be solved.
    :type opt_problem: OptimizationProblem
    :param algo_name: Name of the algorithm to be used to solve
        the problem.
    :type algo_name: str
    :param algo_type: "opt" or "doe" to use an optimization or a
        Design of Experiments algorithm
    :type algo_type: str
    :param options: algorithm options

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
        raise ValueError(
            "Unknown algo type: " + str(algo_type) + ", please use 'doe' or 'opt' !"
        )

    return factory.execute(opt_problem, algo_name, **options)


def monitor_scenario(scenario, observer):
    """
    Adds an observer to a scenario
    The observer must have an "update(atom)" method that
    handles the execution status change of atom
    update(atom) is called everytime an atom execution changes

    :param scenario: Scenario to monitor
    :type scenario: Scenario
    :param observer: Observer that handles an update of status

    See also
    --------
    create_scenario
    get_available_scenario_types
    get_scenario_options_schema
    get_scenario_inputs_schema
    get_scenario_differenciation_modes
    """
    from gemseo.core.monitoring import Monitoring

    # Monitoring object is a singleton
    monitor = Monitoring(scenario)
    monitor.add_observer(observer)


def print_configuration():
    """
    Print the configuration with successfully loaded modules and
    failed imports with the reason

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

    for factory in (
        DisciplinesFactory,
        OptimizersFactory,
        DOEFactory,
        RegressionModelFactory,
        MDOFormulationsFactory,
        MDAFactory,
        PostFactory,
    ):
        LOGGER.info("%s", factory().factory)


def read_design_space(file_path, header=None):
    """
    Read a file containing a design space and return a DesignSpace object.

    :param file_path: Path to the text file
        shall contain a CSV with a row for each variable
        and at least the bounds of the variable.
    :type file_path: str
    :param header: Fields list, or by default, read in the file
    :type header: list(str)
    :returns:  Design space
    :rtype: DesignSpace

    Examples
    --------
    >>> from gemseo.api import create_design_space, export_design_space, read_design_space
    >>> source_design_space = create_design_space()
    >>> source_design_space.add_variable('x', l_b=-1, value=0., u_b=1.)
    >>> export_design_space(source_design_space, 'file.txt')
    >>> read_design_space = read_design_space('file.txt')
    >>> print read_design_space
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
    design_space,
    output_file,
    export_hdf=False,
    fields=None,
    header_char="",
    **table_options
):
    """Export a design space to a text or HDF file.

    :param design_space: Design space.
    :type design_space: DesignSpace
    :param export_hdf: Export to a HDF file (True, default)
        or a txt file (False).
    :type export_hdf: bool
    :param output_file: Output file path.
    :type output_file: str
    :param fields: List of fields to export, by default all.
    :type fields: list(str)
    :param header_char: Header to use when exporting to a text file.
        Default: "".
    :type header_char: str

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


def create_design_space():
    """
    Create an empty instance of a DesignSpace.

    :returns: Empty design space
    :rtype: DesignSpace

    Examples
    --------
    >>> from gemseo.api import create_design_space
    >>> design_space = create_design_space()
    >>> design_space.add_variable('x', l_b=-1, u_b=1, value=0.)
    >>> print design_space
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


def create_parameter_space():
    """
    Create an empty instance of a ParameterSpace.

    :returns: Empty parameter space
    :rtype: ParameterSpace
    """
    from gemseo.algos.parameter_space import ParameterSpace

    return ParameterSpace()


def get_available_caches():
    """Return available caches.

    :return: list of available caches.
    :rtype: list(str)

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


def create_cache(cache_type, name=None, **options):
    """Return a cache.

    :param str cache_type: type of cache.
    :param str name: name of the cache. If None, use cache_type.
    :param options: options specific to cache_type

    Examples
    --------
    >>> from gemseo.api import create_cache
    >>> cache = create_cache('MemoryFullCache')
    >>> print cache
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
    name,
    data,
    variables=None,
    sizes=None,
    groups=None,
    by_group=True,
    delimiter=",",
    header=True,
    default_name=None,
):
    """Create a dataset from a numpy array or a file.

    :param str name: dataset name.
    :param data: array dataset or file path.
    :type data: array or str
    :param list(str) variables: list of variables names.
    :param dict(int) sizes: list of variables sizes.
    :param dict(str) groups: list of variables groups.
    :param bool by_group: if True, store the data by group. Otherwise,
        store them by variables. Default: True.
    :param str delimiter: field delimiter. Default: ','.
    :param bool header: if True and data is a string, read the variables names
        on the first line of the file. Default: True.
    :param str default_name: default variable name.

    See also
    --------
    load_dataset
    """
    from gemseo.core.dataset import Dataset

    dataset = Dataset(name, by_group)
    if isinstance(data, str):
        dataset.set_from_file(data, variables, sizes, groups, delimiter, header)
    else:
        dataset.set_from_array(data, variables, sizes, groups, default_name)
    return dataset


def load_dataset(dataset, **options):
    """Create a dataset from an exsting subclass of Dataset.
    Typically, benchmark datasets can be found in gemseo.problems.dataset

    :param str dataset: dataset name (its classname).

    See also
    --------
    create_dataset
    """
    from gemseo.problems.dataset.factory import DatasetFactory

    return DatasetFactory().create(dataset, **options)
