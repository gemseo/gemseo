..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _global_configuration:

Global configuration
====================

Introduction
------------

Originally,
|g| has been designed for use cases
where the evaluation time of disciplines is a major part of the total computation time.
This is typically the case with disciplines that wrap numerical simulators
such as Computational Fluid Dynamics (CFD) and Computational Structural Mechanics (CSM) solvers.

For inexpensive disciplines,
such as the ones used for analytic formulas or surrogate models,
the computation time spent out of the disciplines is not negligible.

The variable :attr:`gemseo.configuration` allows
to disable data verification, data storage, and result display features
in order to minimize the time spent out of the disciplines.

The configuration of the logging applies to both |g| and its plugins (any logger which name starts with ``gemseo_``).

Default configuration
---------------------

The configuration settings and their default values are:

- ``check_desvars_bounds``, default: ``True`` - Whether to check the membership of design variables in the bounds when evaluating the functions in OptimizationProblem.
- ``enable_discipline_cache``, default: ``True`` - Whether to enable the discipline cache.
- ``enable_discipline_statistics``, default: ``False`` - Whether to record execution statistics of the disciplines such as the execution time, the number of executions and the number of linearizations.
- ``enable_discipline_status``, default: ``False`` - Whether to enable discipline statuses.
- ``enable_function_statistics``, default: ``False`` - Whether to record the statistics attached to the functions, in charge of counting their number of evaluations.
- ``enable_parallel_execution``, default: ``True`` - Whether to let GEMSEO use parallelism (multi-processing or multi-threading) by default.
- ``enable_progress_bar``, default: ``True`` - Whether to enable the progress bar attached to the drivers, in charge to log the execution of the process: iteration, execution time and objective value.
- ``logging``, default: enabled - The logging configuration of type :class:`.LoggingConfiguration`; use ``logging.enable = False`` to disable logging.
- ``validate_input_data``, default: ``True`` - Whether to validate the input data of a discipline before execution.
- ``validate_output_data``, default: ``True``  - Whether to validate the output data of a discipline after execution.

Changing configuration
----------------------

The global configuration can be changed in different ways that are detailed below.

Python
~~~~~~

The global configuration can be changed with the :attr:`gemseo.configuration` variable, e.g.,

.. code:: python

   from gemseo import configuration

   configuration.validate_input_data = False
   configuration.validate_output_data = False
   configuration.logging.enable = False

You can also use the ``fast`` option to use a global configuration suitable for inexpensive disciplines:

.. code:: python

   from gemseo import configuration

   configuration.fast = True

This global configuration disables the following options:

- ``check_desvars_bounds``,
- ``enable_discipline_cache``,
- ``enable_discipline_statistics``,
- ``enable_discipline_status``,
- ``enable_parallel_execution``,
- ``validate_input_data``,
- ``validate_output_data``.

Environment variables
~~~~~~~~~~~~~~~~~~~~~

The global configuration can also be changed using one of these environment variables:

- GEMSEO_CHECK_DESVARS_BOUNDS,
- GEMSEO_ENABLE_DISCIPLINE_CACHE,
- GEMSEO_ENABLE_DISCIPLINE_STATISTICS,
- GEMSEO_ENABLE_DISCIPLINE_STATUS,
- GEMSEO_ENABLE_FUNCTION_STATISTICS,
- GEMSEO_ENABLE_PARALLEL_EXECUTION,
- GEMSEO_ENABLE_PROGRESS_BAR,
- GEMSEO_LOGGING_DATE_FORMAT
- GEMSEO_LOGGING_ENABLE
- GEMSEO_LOGGING_LEVEL
- GEMSEO_LOGGING_FILE_PATH
- GEMSEO_LOGGING_FILE_MODE
- GEMSEO_LOGGING_MESSAGE_FORMAT
- GEMSEO_VALIDATE_INPUT_DATA,
- GEMSEO_VALIDATE_OUTPUT_DATA.

Dotenv (.env) support
~~~~~~~~~~~~~~~~~~~~~

The global configuration can also be defined in a dotenv file named ``.env``
looking like this:

.. code:: bash

   GEMSEO_CHECK_DESVARS_BOUNDS = True
   GEMSEO_ENABLE_DISCIPLINE_CACHE = True
   GEMSEO_ENABLE_DISCIPLINE_STATISTICS = False
   GEMSEO_ENABLE_DISCIPLINE_STATUS = False
   GEMSEO_ENABLE_FUNCTION_STATISTICS = False
   GEMSEO_ENABLE_PARALLEL_EXECUTION = True
   GEMSEO_ENABLE_PROGRESS_BAR = True
   GEMSEO_LOGGING_DATE_FORMAT = %H:%M:%S
   GEMSEO_LOGGING_ENABLE = False
   GEMSEO_LOGGING_LEVEL = 20
   GEMSEO_LOGGING_FILE_PATH =
   GEMSEO_LOGGING_FILE_MODE = a
   GEMSEO_LOGGING_MESSAGE_FORMAT = %(levelname)8s - %(asctime)s: %(message)s
   GEMSEO_VALIDATE_INPUT_DATA = True
   GEMSEO_VALIDATE_OUTPUT_DATA = True
