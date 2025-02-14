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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Jean-François Figué
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Create a retry discipline
=========================

Sometimes,
the execution of a discipline can fail and work after several repetitions.
The :class:`.RetryDiscipline` facilitates the management of these failures and repetitions.
This class illustrates this feature.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

from numpy import array

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo.core.discipline import Discipline
from gemseo.disciplines.wrappers.retry_discipline import RetryDiscipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

LOGGER = configure_logger()

# %%
# Toy discipline
# --------------
# For that example,
# we create an :class:`.AnalyticDiscipline` to evaluate the expression :math:`y=1/x`:
analytic_disc = create_discipline("AnalyticDiscipline", expressions={"y": "1/x"})

# %%
# This discipline will raise a ``ZeroDivisionError`` when :math:`x=0`.
#
# Execution without failure
# -------------------------
# Let's wrap this toy discipline in a :class:`.RetryDiscipline`
# parametrized by a maximum number of 3 execution attempts:
retry_disc = RetryDiscipline(analytic_disc, n_trials=3)

# %%
# We can execute this :class:`.RetryDiscipline` at :math:`x=2`:
retry_disc.execute({"x": array([2.0])})
retry_disc.io.data

# %%
# and verify that the computation is correctly performed, :math:`y=0.5`,
# with only one execution attempt:
retry_disc.n_executions

# %%
# Execution with failure
# ----------------------
# If an exception like a ``ZeroDivisionError`` occurs,
# we do not want to retry the execution and just do something else.
# To do this,
# we need to define the fatal exceptions for which the execution is not retried.
# It means that if that error is raised,
# then the discipline :class:`.RetryDiscipline` will stop execution
# rather than retrying an attempt.
retry_disc = RetryDiscipline(
    analytic_disc, n_trials=3, fatal_exceptions=[ZeroDivisionError]
)

try:
    retry_disc.execute()
except ZeroDivisionError:
    LOGGER.info("Manage this fatal exception.")

# %%
# We can verify the number of attempts is only :math:`1`:
retry_disc.n_executions

# %%
# To highlight the use of ``n_trials`` parameter, let's try another toy discipline,
# which will crash the first 2 executions and finally succeed at the third attempt.


class FictiveDiscipline(Discipline):
    """Discipline to be executed several times.

    - The first 2 times, raise a RuntimeError,
    - and finally succeed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempt = 0

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        self.attempt += 1
        LOGGER.info("attempt: %s", self.attempt)
        if self.attempt < 3:
            raise RuntimeError
        return {}


# %%
# We can then illustrate the use of ``n_trials`` parameter. Here we intentionally set
# this value to 4, knowing the discipline will complete before at the third trial:


test_n_trials = FictiveDiscipline()
retry_disc = RetryDiscipline(test_n_trials, n_trials=4)

retry_disc.execute()

# %%
# and verify the calculation has been tried 3 times to succeed:
retry_disc.n_executions

# %%
# Limit the execution time
# ------------------------
# If you want to limit the duration of the wrapped discipline,
# use the ``timeout`` option.
# Here is an example of a discipline
# whose execution does nothing except sleep for 5 seconds:


class DisciplineLongTimeRunning(Discipline):
    """A discipline that could run for a while, to test the timeout feature."""

    def _run(self, input_data: StrKeyMapping) -> None:
        time.sleep(5.0)


# %%
# Now we wrap it in :class:`.RetryDiscipline`,
# set the ``timeout`` argument to 2 seconds
# and execute this new discipline:

retry_disc = RetryDiscipline(DisciplineLongTimeRunning(), n_trials=1, timeout=2.0)

sys.tracebacklimit = 0
try:
    LOGGER.info("Running discipline...")
    retry_disc.execute({})
    LOGGER.info("Discipline completed without reaching the time limit.")
except TimeoutError:
    LOGGER.info("Discipline stopped, due to a TimeoutError.")

# %%
# In the log,
# we can see the initial and final times of the discipline execution.
# We can also read that the timeout is reached.
#
# In some cases,
# this option could be very useful.
# For example if you wrap an SSH discipline
# (see `gemseo-ssh plugin <https://gemseo.gitlab.io/dev/gemseo-ssh>`__)
# in :class:`.RetryDiscipline`.
# In that context,
# it can be important to limit the duration when an ssh connexion is too slow.
#

# %%
# .. note::
#
#    The user can build his :class:`.RetryDiscipline` with a combination of all the
#    available parameters.
#    Some attributes of the discipline are public and can be modified after
#    instantiation (``fatal_exceptions``, ``n_trials``, ...)
#
# .. note::
#
#    In the previous example, we added ``sys.tracebacklimit = 0`` to
#    limit message output by exception, just in order the
#    output is only focused on what we aim to demonstrate with that example.
#    Please don't put this statement in normal use, otherwise you could miss some
#    important messages in the output.
#
