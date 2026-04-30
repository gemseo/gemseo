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
"""# Retry the discipline execution

## Problem

Sometimes,
the execution of a discipline can fail or be stuck and work after several repetitions.
How can you retry until success?

## Solution

The [RetryDiscipline][gemseo.disciplines.wrappers.retry_discipline.RetryDiscipline]
facilitates the management of these repetitions.

## Step-by-step guide

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import LOGGER
from gemseo.core.discipline import Discipline
from gemseo.disciplines.wrappers.retry_discipline import RetryDiscipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


# %%
# ### 1. Create a discipline
#
# This discipline will crash the first 2 executions
# and finally succeed at the third attempt.
class FictiveDiscipline(Discipline):
    """Discipline to be executed several times.

    - The first 2 times, raise a `RuntimeError`,
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


discipline = FictiveDiscipline()

# %%
# ### 2. Multiple trials
#
# Wrap your discipline with
# [RetryDiscipline][gemseo.disciplines.wrappers.retry_discipline.RetryDiscipline].
retry_discipline = RetryDiscipline(discipline, n_trials=4)
retry_discipline.execute()

# %%
# and verify the calculation has been tried 3 times to succeed:
retry_discipline.n_executions

# %%
# !!! tips
#     The discipline can also be retried if taking too long to be executed.
#
# ## Summary
#
# Wrap your discipline with the
# [RetryDiscipline][gemseo.disciplines.wrappers.retry_discipline.RetryDiscipline]
# discipline wrapper.
#
# The execution of your discipline can be retried when:
#
# - there is a non-fatal error (fatal exeptions can be given to the wrapper),
# - the execution takes too much time to run.
