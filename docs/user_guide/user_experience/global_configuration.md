---
reading_time: true
complexity: intermediate
description: "The global configuration of GEMSEO: what each option changes, the fast mode, and where the values come from."
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Global configuration { #concept-global-configuration }

GEMSEO offers options to change the behaviour of various features.
This global configuration can be accessed via `gemseo.configuration`,
an instance of [GlobalConfiguration][gemseo.util.global_configuration.GlobalConfiguration].

The following sections describe these options.

## Disciplines { #global-configuration-disciplines }

`enable_discipline_cache` sets the default cache type of the disciplines.
`True` gives a simple cache, `False` no cache.
A discipline chooses its cache when it is created.
A discipline created before the change keeps its cache.
This option does not apply to the MDAs.
They keep a simple cache by default.
See [Cache][concept-cache] for more details.

`validate_input_data` sets whether a discipline checks its input data
against its input grammar before execution.
`validate_output_data` does the same for the output data after execution.
They are read at each execution,
so they also apply to the disciplines already created.
Without validation,
an input of a wrong type reaches the discipline unchecked.
The discipline may then fail with a less explicit error.

`enable_discipline_status` sets whether the disciplines record their execution status.
`enable_discipline_statistics` sets whether they record their execution statistics:
number of executions, number of linearizations and execution time.
Both are read at each execution.

!!! how-to
    - [Count the executions of disciplines and the evaluations of functions][]

## Problems, functions and drivers { #global-configuration-problems }

`check_desvars_bounds` sets whether an
[EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem]
checks that the design variables are within their bounds
when it evaluates its functions.
It also applies to the problems already created.

`enable_working_database` sets whether a driver records the evaluations of a run a second time.
This second record uses the coordinates the algorithm works on.
It is stored in the `working_database` of the problem.
It is a debugging aid: nothing of GEMSEO reads it,
and a run gives the same results with or without it.
An algorithm building sub-problems and handing them to a sub-driver leaves it at `None`.
It is written by the process performing the evaluation,
so a parallel run records the evaluations of its own process only.
See [Database][concept-database] for more details.

`enable_function_statistics` sets whether the functions of a problem
count their evaluations.

`enable_progress_bar` sets whether the drivers log a progress bar.
The drivers include the optimizers and the samplers.
The progress bar is shown only when logging is enabled.
The setting `enable_progress_bar` of a driver takes precedence over this option.

!!! how-to
    - [Evaluate the functions outside the bounds of the design space][]
    - [Record the evaluations in the coordinates of the algorithm][]

## Parallel execution { #global-configuration-parallel-execution }

`enable_parallel_execution` sets whether GEMSEO uses
multi-processing or multi-threading by default.
When `False`,
the default number of threads or processes is 1 for the parallel MDAs,
[ParallelDisciplineChain][gemseo.discipline.chain.parallel_chain.ParallelDisciplineChain]
and
[CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution].
An MDA created before the change keeps its number of processes.

!!! how-to
    - [Disable the parallel execution][]

## Logging { #global-configuration-logging }

The `logging` option groups the settings of the GEMSEO loggers:
level, message and date formats, log file and file mode.
`logging.enable = False` disables the GEMSEO logs.

These settings apply to the logger `gemseo`
and to the loggers of its plugins, whose names start with `gemseo`.
A plugin logger gets them only if it exists when they are applied.
A plugin imported after `gemseo` gets them at the next change of `logging`.
By default,
they leave the root logger and the loggers of other libraries alone.
This follows the [Python recommendation for library logging](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library).
To see the logs of your own code,
configure your own loggers.
You can also set `logging.configure_root_logger` to `True`:
GEMSEO then configures the root logger with the same settings.
Once enabled,
it cannot be disabled for the rest of the process.

!!! how-to
    - [Configure the GEMSEO logs][]

## Directory manager { #global-configuration-directory-manager }

The `directory_manager` option groups the settings of the
[directory manager][the-directory-manager].
The directory manager is disabled by default.
Once enabled,
it cannot be disabled for the rest of the process.

!!! how-to
    - [Using the DirectoryManager][]

## Fast mode { #global-configuration-fast }

The method `enable_fast_mode()` configures GEMSEO for inexpensive disciplines.
It disables the options that cost time at every evaluation:

- `check_desvars_bounds`,
- `enable_discipline_cache`,
- `enable_discipline_statistics`,
- `enable_discipline_status`,
- `enable_function_statistics`,
- `enable_parallel_execution`,
- `enable_working_database`,
- `validate_input_data`,
- `validate_output_data`.

It does not disable the progress bar,
which is user feedback and not a per-evaluation overhead.
It does not disable the cache of the MDAs either.

An option set after `enable_fast_mode()` keeps the value set.
The method `disable_fast_mode()` resets these options
to their built-in default values.
This holds even for an option set from an environment variable or a `.env` file.

!!! how-to
    - [Speed up a process made of inexpensive disciplines][]

## Where the values come from { #global-configuration-sources }

The configuration is created when `gemseo` is imported.
Its values come from, by order of precedence:

1. the environment variables,
2. a file named `.env` in the current working directory,
3. the default values.

The environment variable of an option is named
`GEMSEO_` followed by the name of the option in upper case.
The variable of `logging.level` is `GEMSEO_LOGGING_LEVEL`.
An environment variable set after the import of `gemseo` has no effect.

After the import,
you can change an option at any time
by setting the attribute of `gemseo.configuration`.

!!! how-to
    - [Configure GEMSEO without changing the code][]

## Options { #global-configuration-options }

The list below gives the options,
their environment variables and their default values.

<!-- global-configuration-options -->
