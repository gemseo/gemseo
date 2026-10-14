# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Configure the GEMSEO logs

## Problem

You want to change the level of the GEMSEO logs or write them in a file.
You also want to see the logs of your own code with the same format.

## Solution

Change the option `logging` of the
[global configuration][concept-global-configuration].

## Step-by-step guide
"""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from numpy import cos
from numpy import sin

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace

# %%
# ### Prerequisites
#
# An optimization problem whose resolution logs messages:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=-0.5)
problem = OptimizationProblem(design_space)
problem.objective = ArrayFunction(sin, name="f", jac=cos)
settings = L_BFGS_B_Settings(max_iter=3)
_ = execute_algo(problem, settings_model=settings)

# %%
# and a temporary directory to store the log file:
temporary_directory = TemporaryDirectory()
directory = Path(temporary_directory.name)

# %%
# ### 1. Change the level
#
# Here, only the warnings and errors are logged,
# so solving the problem again logs nothing:
configuration.logging.level = "WARNING"

problem.reset()
_ = execute_algo(problem, settings_model=settings)

# %%
# ### 2. Write the logs in a file
#
# Here, the level is set back to its default value:
log_file_path = directory / "gemseo.log"
configuration.logging.file_path = log_file_path
configuration.logging.level = "INFO"

problem.reset()
_ = execute_algo(problem, settings_model=settings)
log_file_path.read_text()

# %%
# ### 3. Log the messages of your own code
#
# By default, only the GEMSEO loggers are configured.
# Let GEMSEO configure the root logger too:
configuration.logging.configure_root_logger = True

logging.getLogger("my_module").info("A message from my code.")

# %%
# ### 4. Disable the logs
configuration.logging.enable = False
problem.reset()

# %%
# Then, solving the problem again logs nothing:
_ = execute_algo(problem, settings_model=settings)

# %%
# ## Summary
#
# Set the fields of `configuration.logging`
# to change the level, the format and the destination of the GEMSEO logs.
# Set `configuration.logging.configure_root_logger` to `True`
# to apply the same settings to the logs of your own code.
