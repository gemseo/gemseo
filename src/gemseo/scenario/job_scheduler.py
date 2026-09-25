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
"""Wrapping of a scenario in a job scheduler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from gemseo.util.constant import read_only_empty_dict
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from gemseo.discipline.wrapper.job_scheduler.discipline import (
        JobSchedulerDiscipline,
    )
    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import StrKeyMapping
    from gemseo.util.typing import StrPath

logger = logging.getLogger(__name__)


def wrap_scenario_in_job_scheduler(
    scenario: EvaluationScenario,
    scheduler_name: str,
    workdir_path: StrPath,
    adapter_settings: StrKeyMapping = read_only_empty_dict,
    **options: Any,
) -> JobSchedulerDiscipline:
    """Wrap a scenario in a job scheduler for remote execution on an HPC cluster.

    The scenario is first adapted into a discipline using
    [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
    when it solves an
    [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
    e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario],
    and
    [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
    otherwise,
    then wrapped with a
    [JobSchedulerDiscipline][gemseo.discipline.wrapper.job_scheduler.discipline.JobSchedulerDiscipline].

    The algorithm must be set on the scenario before calling this function,
    using
    [set_algorithm()][gemseo.scenario.evaluation.EvaluationScenario.set_algorithm].

    By default,
    when the scenario is driven by an optimization algorithm,
    the inputs of the wrapper are the design variables,
    whose values are used as the starting point of the optimization
    (the design variables that are not inputs of the wrapper
    keep their current value);
    when the scenario is driven by a DOE algorithm,
    which ignores the starting point,
    the wrapper has no input.
    Use `adapter_settings={"input_names": [...]}`
    to expose other inputs of the top-level disciplines of the scenario
    instead of the design variables,
    which are then no longer the starting point.
    By default,
    the outputs of the wrapper are the outputs of the objective, constraints and observables
    and the design variables at the end of the scenario execution,
    i.e. at the optimum for an optimization problem
    and at the last evaluated point otherwise.
    These are the values of the discipline outputs,
    not the standardized values used by the algorithm:
    a maximized objective comes back as `obj`, not `-obj`,
    and a constraint declared with a reference value comes back
    without that value subtracted.
    Only these output values are returned:
    the database and the result of the scenario remain in the remote process.
    The outputs of the functions
    that are not outputs of the top-level disciplines of the scenario,
    e.g. an aggregated constraint,
    cannot be returned by the wrapper and are skipped with a warning.
    Use `adapter_settings` to save the database to an HDF5 file
    in the job working directory,
    e.g. `adapter_settings={"save_databases": True}`.
    Only the outputs of the objective can be differentiated
    with respect to the inputs of the wrapper,
    and this post-optimal sensitivity is only meaningful
    when these inputs are parameters of the optimization problem
    rather than its design variables,
    as an optimum barely depends on its starting point.
    So,
    to linearize the wrapper of an optimization scenario,
    replace its inputs with such parameters
    and reduce its outputs to those of the objective,
    e.g. `adapter_settings={"input_names": ["alpha"], "output_names": ["obj"]}`.

    Args:
        scenario: The scenario to wrap.
            Its algorithm must have been set
            via [set_algorithm()][gemseo.scenario.evaluation.EvaluationScenario.set_algorithm].
        scheduler_name: The name of the job scheduler (for instance LSF, SLURM, PBS).
        workdir_path: The path to the workdir.
        adapter_settings: The settings of the scenario adapter
            overriding the default ones,
            e.g. `input_names`, `output_names` or `save_databases`.
            Note that `input_names` replaces the default inputs of the wrapper
            and that `output_names` replaces the default outputs of the wrapper,
            so the objective, the constraints, the observables
            and the design variables have to be repeated in it
            when they are still wanted.
        **options: The submission options.

    Returns:
        The job scheduler discipline wrapper around the scenario adapter.

    Raises:
        ValueError: If the algorithm has not been set on the scenario
            or if the objective of an optimization problem has not been set.

    Warning:
        This function serializes the scenario adapter,
        and so the scenario and all its disciplines,
        so they have to be serializable.
        All disciplines provided in GEMSEO are serializable but it is possible that
        custom ones are not and this will make the submission process fail.
        Also,
        see [Handling paths for cross-platforms][handling-paths-for-different-oses].
    """  # noqa: E501
    from gemseo.discipline.wrapper.job_scheduler.factory import (
        JobSchedulerDisciplineFactory,
    )
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.optimization.problem import OptimizationProblem
    from gemseo.scenario.adapter.evaluation import EvaluationScenarioAdapter
    from gemseo.scenario.adapter.mdo import MDOScenarioAdapter

    if scenario._algorithm_settings is None:
        msg = (
            "The algorithm must be set on the scenario before wrapping it "
            "in a job scheduler; use the set_algorithm method."
        )
        raise ValueError(msg)

    problem = scenario.formulation.problem
    if isinstance(problem, OptimizationProblem):
        if problem.objective is None:
            msg = (
                "The objective must be set on the scenario before wrapping it "
                "in a job scheduler; use the add_objective method."
            )
            raise ValueError(msg)

        adapter_class = MDOScenarioAdapter
    else:
        adapter_class = EvaluationScenarioAdapter

    design_space = scenario.design_space
    # A DOE algorithm ignores the starting point of the scenario,
    # so the design variables are not inputs of the wrapper in this case.
    is_doe = isinstance(scenario._algorithm_settings, BaseDOESettings)
    settings = {
        "input_names": () if is_doe else tuple(design_space.variables),
        "set_x0_before_exec": (
            not is_doe and not adapter_settings.get("reset_x0_before_exec", False)
        ),
    }
    if "output_names" not in adapter_settings:
        # The adapter can only return the outputs of the top-level disciplines
        # and the design variables.
        retrievable_names = set(design_space.variables)
        for discipline in scenario.formulation.get_top_level_disciplines():
            retrievable_names.update(discipline.io.output_grammar)

        function_output_names = dict.fromkeys(
            output_name
            for function in problem.functions
            for output_name in function.output_names
        )
        missing_output_names = function_output_names.keys() - retrievable_names
        if missing_output_names:
            logger.warning(
                "The function outputs %s of the scenario %s "
                "are not outputs of its top-level disciplines "
                "and cannot be returned by the job scheduler wrapper.",
                pretty_repr(missing_output_names),
                scenario.name,
            )

        output_names = [
            name for name in function_output_names if name in retrievable_names
        ]
        output_names += design_space.variables
        settings["output_names"] = tuple(dict.fromkeys(output_names))

    settings.update(adapter_settings)
    adapter = adapter_class(scenario, **settings)
    adapter.io.input_grammar.defaults.update({
        name: value
        for name, value in design_space.get_current_value(as_dict=True).items()
        if name in adapter.io.input_grammar
    })

    return JobSchedulerDisciplineFactory().wrap_discipline(
        discipline=adapter,
        scheduler_name=scheduler_name,
        workdir_path=workdir_path,
        **options,
    )
