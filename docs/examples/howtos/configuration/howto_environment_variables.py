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
"""# Configure GEMSEO without changing the code

## Problem

You want to change the
[global configuration of `gemseo`][concept-global-configuration]
of a script without editing it.

## Solution

Set the environment variables named `GEMSEO_<OPTION>`
or write them in a file named `.env`
in the directory where you run the script.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import configuration

# %%
# ### 1. Name the environment variable of an option
#
# The name of the environment variable associated with the option `[<domain>].name`
# is `GEMSEO_[<DOMAIN>_]NAME`:
#
# | Option                     | Environment variable              |
# | -------------------------- | --------------------------------- |
# | `enable_progress_bar`      | `GEMSEO_ENABLE_PROGRESS_BAR`      |
# | `logging.level`            | `GEMSEO_LOGGING_LEVEL`            |
# | `directory_manager.enable` | `GEMSEO_DIRECTORY_MANAGER_ENABLE` |
#
# Read the explanations on the
# [global configuration of `gemseo`][global-configuration-options]
# to find out about the name of every option.
#
# ### 2. Set environment variables
#
# Set the environment variables in the shell,
# then run the script:
#
# ```console
# $ export GEMSEO_ENABLE_PROGRESS_BAR=false
# $ export GEMSEO_LOGGING_LEVEL=WARNING
# $ python my_script.py
# ```
#
# In `my_script.py`,
# the configuration has these values:
configuration.enable_progress_bar, configuration.logging.level

# %%
# Out:
#
# ```
# (False, 'WARNING')
# ```
#
# !!! note
#     Give the logging level by its name.
#     A numerical value like `20` is rejected,
#     see [issue #2025](https://gitlab.com/gemseo/dev/gemseo/-/work_items/2025).
#
# ### 3. Use a `.env` file
#
# Write the environment variables in a file named `.env`
# in the directory where you run the script:
#
# ```bash
# GEMSEO_ENABLE_PROGRESS_BAR=false
# GEMSEO_LOGGING_LEVEL=WARNING
# ```
#
# Only the `.env` file of the current working directory is read,
# not the ones of its parent directories.
# An environment variable, for example one passed on the command line,
# takes precedence over the `.env` file:
#
# ```console
# $ GEMSEO_LOGGING_LEVEL=ERROR python my_script.py
# ```
#
# runs the script with the logging level `ERROR`
# and the progress bar disabled.

# %%
# ## Summary
#
# Set the environment variables `GEMSEO_<OPTION>` before starting Python.
# You can also write them in a `.env` file of the current working directory.
# An environment variable set after the import of `gemseo` has no effect.
