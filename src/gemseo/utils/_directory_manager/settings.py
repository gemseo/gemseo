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

"""Settings for the directory manager."""

from __future__ import annotations

import sys
from multiprocessing import current_process
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import field_validator
from pydantic import model_validator
from strenum import StrEnum

from gemseo.scenarios.backup_settings import BaseBackupSettings
from gemseo.utils.base_multiton import BaseMultiton

if TYPE_CHECKING:
    from typing_extensions import Self

_KEEP_ALL: Final[str] = "KEEP_ALL"
_KEEP_LAST_ONLY: Final[str] = "KEEP_LAST_ONLY"


class CleanUpPolicy(StrEnum):
    """Cleanup policy for scenario execution directories."""

    KEEP_ALL = _KEEP_ALL
    """Keep all generated files and directories."""

    KEEP_LAST_ONLY = _KEEP_LAST_ONLY
    """Keep only the last directory."""

    KEEP_SOLUTION_ONLY = "KEEP_SOLUTION_ONLY"
    """Keep only the solution directory and files."""

    KEEP_BASELINE_AND_SOLUTION = "KEEP_BASELINE_AND_SOLUTION"
    """Keep only the baseline and the solution directories and files."""


class MDACleanUpPolicy(StrEnum):
    """Cleanup policy for MDA solver iteration directories."""

    KEEP_ALL = _KEEP_ALL
    """Keep all generated files and directories."""

    KEEP_LAST_ONLY = _KEEP_LAST_ONLY
    """Keep only the last directory."""


class Settings(
    BaseModel,
    extra="forbid",
    validate_assignment=True,
    validate_default=True,
):
    """Configuration settings for the directory manager.

    These settings control directory creation, cleanup policies, backup settings,
    and the ability to track execution history for GEMSEO workflows.
    """

    __execution_root_path: Path = Path()
    """Store the last execution root path created here.

    Lets the validator skip a path it already created when it re-runs
    (validate_assignment fires it on every change). The default (the relative
    current directory) matches the default ``execution_root_path``, so the
    current directory is never (re)created.
    """

    enable: bool = Field(
        default=False,
        description="Whether to enable the directory management feature.",
    )

    _enabled_once: bool = PrivateAttr(default=False)
    """Whether the manager has been enabled at least once.

    Once enabled, the manager cannot be disabled: classes are decorated
    in place when first observed, so turning observation off afterwards
    would leave them half-instrumented.
    """

    clean_up_policy: CleanUpPolicy = Field(
        default=CleanUpPolicy.KEEP_ALL,
        description=CleanUpPolicy.__doc__,
    )

    mda_clean_up_policy: MDACleanUpPolicy = Field(
        default=MDACleanUpPolicy.KEEP_ALL,
        description=MDACleanUpPolicy.__doc__,
    )

    # The default is the relative current directory, NOT Path.cwd():
    # evaluating cwd here would capture it at import time of gemseo,
    # while the relative path resolves to the current directory at use time.
    execution_root_path: Path = Field(
        default=Path(),
        description="""The path to the root directory,
        where the directory manager will create the directories,
        if empty then use the current directory.""",
    )

    save_history_backup: bool = Field(
        default=False,
        description="Whether to save the history backup.",
    )

    backup_settings: BaseBackupSettings = Field(
        default=BaseBackupSettings(),
        description=BaseBackupSettings.__doc__,
    )

    save_mda_residuals: bool = Field(
        default=False,
        description="Whether to save the mda residuals.",
    )

    keep_failed_executions: bool = Field(
        default=False,
        description="Whether to keep failed executions.",
    )

    @field_validator("enable")
    @classmethod
    def __reset_directory_manager(cls, value: bool) -> bool:
        # This will force the directory manager singleton to be reset.
        # Only its cache entry is evicted: the cache is shared with the other
        # multitons (e.g. all the factories), which shall not be reset.
        # The manager module cannot be imported here (circular import via
        # global_configuration, where this validator runs at import time):
        # if it has not been imported yet, there is no instance to evict.
        manager_module = sys.modules.get("gemseo.utils._directory_manager.manager")
        if manager_module is not None:
            BaseMultiton.clear_cache(manager_module.DirectoryManager)
        return value

    @model_validator(mode="after")
    def __forbid_disabling_once_enabled(self) -> Self:
        # Observed classes are decorated in place the first time they are
        # instantiated while enabled, and that decoration is permanent.
        # Disabling afterwards would leave those classes instrumented but
        # without an observer, so it is forbidden: a fresh Settings instance
        # must be created to start from a disabled state.
        if self.enable:
            self._enabled_once = True
        elif self._enabled_once:
            msg = "The directory manager cannot be disabled once it is enabled."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def __create_execution_root_path(self) -> Self:
        if self.enable:
            if not self.execution_root_path.is_absolute():
                # Resolve once against the current directory: the manager
                # chdirs the process afterwards, so a relative path would
                # drift (in particular when passed to a worker process).
                self.execution_root_path = self.execution_root_path.resolve()
            # The current directory (the default root) and a path already
            # created by a previous validation need no (re)creation.
            if self.execution_root_path not in {
                Path.cwd(),
                self.__execution_root_path,
            }:
                # In a worker process the root was created by the parent and
                # already exists, so it is reused. In the main process the root
                # must not exist beforehand, like the execution subdirectories
                # created at run time, so creating an existing one raises.
                # The branch is not seen by coverage: the worker side only
                # runs in subprocesses, which the coverage tracer does not
                # record.
                if not hasattr(current_process(), "parent_path"):  # pragma: no branch
                    self.execution_root_path.mkdir(parents=True)
                self.__execution_root_path = self.execution_root_path
        return self
