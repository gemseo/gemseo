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
"""The multiprocessing start method used by GEMSEO."""

from __future__ import annotations

from multiprocessing import get_start_method

from strenum import StrEnum

from gemseo.util.platform import PLATFORM_IS_LINUX


class MultiProcessingStartMethod(StrEnum):
    """The multiprocessing start method."""

    FORK = "fork"
    SPAWN = "spawn"
    FORKSERVER = "forkserver"


MULTI_PROCESSING_START_METHOD: MultiProcessingStartMethod = (
    MultiProcessingStartMethod.FORK
    if PLATFORM_IS_LINUX
    else MultiProcessingStartMethod(get_start_method())
)
"""The start method used by GEMSEO to create the worker processes.

On Linux, `fork` is used instead of the default start method of the interpreter,
which is `forkserver` since Python 3.14. Unlike `fork`, `forkserver` requires a
new interpreter to be started and GEMSEO to be imported in every worker, and it
does not preserve the shared memory of the parent process, such as the counters
of the execution statistics.

Change this value to use another start method:

```python
from gemseo.util.multiprocessing import start_method
from gemseo.util.multiprocessing.start_method import MultiProcessingStartMethod

start_method.MULTI_PROCESSING_START_METHOD = MultiProcessingStartMethod.SPAWN
```
"""
