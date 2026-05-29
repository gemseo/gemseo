---
description: ""
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

## Introduction { #concept-introduction }

The logging configuration applies to GEMSEO and its plugins only (any logger whose name starts with `gemseo_`).
By default, it does not configure the root logger nor the loggers of client code or third-party libraries,
in line with the [Python recommendation for library logging](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library).
To see logs from your own code, configure your own loggers, for instance with [`logging.basicConfig`](https://docs.python.org/3/library/logging.html#logging.basicConfig),
or set `logging.configure_root_logger = True` to let GEMSEO configure the root logger as well.

## Default configuration { #concept-default-configuration }

<!-- TODO: replace by automatic print of pydantic model. -->

- `check_desvars_bounds`, default: `True` - Whether to check the membership of design variables in the bounds when evaluating the functions in OptimizationProblem.
- `enable_discipline_cache`, default: `True` - Whether to enable the discipline cache.
- `enable_discipline_statistics`, default: `False` - Whether to record execution statistics of the disciplines such as the execution time, the number of executions and the number of linearizations.
- `enable_discipline_status`, default: `False` - Whether to enable discipline statuses.
- `enable_function_statistics`, default: `False` - Whether to record the statistics attached to the functions, in charge of counting their number of evaluations.
- `enable_parallel_execution`, default: `True` - Whether to let GEMSEO use parallelism (multi-processing or multi-threading) by default.
- `enable_progress_bar`, default: `True` - Whether to enable the progress bar attached to the drivers, in charge to log the execution of the process: iteration, execution time and objective value.
- `logging`, default: enabled - The logging configuration of type [LoggingConfiguration][gemseo.utils.logging.LoggingConfiguration]; use `logging.enable = False` to disable logging.
- `validate_input_data`, default: `True` - Whether to validate the input data of a discipline before execution.
- `validate_output_data`, default: `True` - Whether to validate the output data of a discipline after execution.

## Changing configuration { #concept-changing-configuration }

<!-- TODO: make howtos -->

The global configuration can be changed in different ways that are detailed below.

### Python { #concept-python }

The global configuration can be changed with the `gemseo.configuration` variable, e.g.,

```python
from gemseo import configuration

configuration.validate_input_data = False
configuration.validate_output_data = False
configuration.logging.enable = False
```

You can also use the `fast` option to use a global configuration suitable for inexpensive disciplines:

```python
from gemseo import configuration

configuration.fast = True
```

This global configuration disables the following options:

- `check_desvars_bounds`,
- `enable_discipline_cache`,
- `enable_discipline_statistics`,
- `enable_discipline_status`,
- `enable_parallel_execution`,
- `validate_input_data`,
- `validate_output_data`.

### Environment variables { #concept-environment-variables }

The global configuration can also be changed using one of these environment variables:

- GEMSEO_CHECK_DESVARS_BOUNDS,
- GEMSEO_ENABLE_DISCIPLINE_CACHE,
- GEMSEO_ENABLE_DISCIPLINE_STATISTICS,
- GEMSEO_ENABLE_DISCIPLINE_STATUS,
- GEMSEO_ENABLE_FUNCTION_STATISTICS,
- GEMSEO_ENABLE_PARALLEL_EXECUTION,
- GEMSEO_ENABLE_PROGRESS_BAR,
- GEMSEO_LOGGING_CONFIGURE_ROOT_LOGGER
- GEMSEO_LOGGING_DATE_FORMAT
- GEMSEO_LOGGING_ENABLE
- GEMSEO_LOGGING_LEVEL
- GEMSEO_LOGGING_FILE_PATH
- GEMSEO_LOGGING_FILE_MODE
- GEMSEO_LOGGING_MESSAGE_FORMAT
- GEMSEO_VALIDATE_INPUT_DATA,
- GEMSEO_VALIDATE_OUTPUT_DATA.

### Dotenv (.env) support { #concept-dotenv-env-support }

The global configuration can also be defined in a dotenv file named `.env` looking like this:

```bash
GEMSEO_CHECK_DESVARS_BOUNDS = True
GEMSEO_ENABLE_DISCIPLINE_CACHE = True
GEMSEO_ENABLE_DISCIPLINE_STATISTICS = False
GEMSEO_ENABLE_DISCIPLINE_STATUS = False
GEMSEO_ENABLE_FUNCTION_STATISTICS = False
GEMSEO_ENABLE_PARALLEL_EXECUTION = True
GEMSEO_ENABLE_PROGRESS_BAR = True
GEMSEO_LOGGING_CONFIGURE_ROOT_LOGGER = False
GEMSEO_LOGGING_DATE_FORMAT = %H:%M:%S
GEMSEO_LOGGING_ENABLE = False
GEMSEO_LOGGING_LEVEL = 20
GEMSEO_LOGGING_FILE_PATH =
GEMSEO_LOGGING_FILE_MODE = a
GEMSEO_LOGGING_MESSAGE_FORMAT = %(levelname)8s - %(asctime)s: %(message)s
GEMSEO_VALIDATE_INPUT_DATA = True
GEMSEO_VALIDATE_OUTPUT_DATA = True
```
