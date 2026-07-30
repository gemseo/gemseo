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

"""Tools for the management of directories."""

from __future__ import annotations

import operator
import shutil
from multiprocessing import current_process
from multiprocessing import parent_process
from os import chdir
from os import getpid
from os import walk
from pathlib import Path
from sys import maxsize
from threading import current_thread
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.util._directory_manager.settings import _KEEP_ALL
from gemseo.util._directory_manager.settings import _KEEP_LAST_ONLY
from gemseo.util._directory_manager.settings import CleanUpPolicy
from gemseo.util._filename_sanitizer import secure_filename
from gemseo.util._workflow_observer.mda import MDAExecutionWorkflowObserver
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver
from gemseo.util.base_multiton import BaseMultiton
from gemseo.util.global_configuration import _configuration

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
    from gemseo.util._workflow_observer.mda import MDAWorkflowObserver


class DirectoryManager(metaclass=BaseMultiton):
    """Manages execution directories throughout workflow execution.

    This class implements a multiton (resettable singleton) that creates and manages
    execution directories, handles homonymic directory naming via indexed suffixes,
    and applies cleanup policies based on configuration. It also handles multi-threading
    and multiprocessing contexts by tracking thread-local working directories.

    The directory manager is automatically reset when enabled via global
    configuration through the `BaseMultiton` metaclass; it cannot be disabled
    once enabled.
    """

    suffix_separator: ClassVar[str] = "#"
    """The separator used when suffixing homonymic directories."""

    _MARKER_FILE_NAME: ClassVar[str] = ".gemseo_directory_manager"
    """The name of the marker file identifying directories created by the manager.

    The marker allows recognizing managed directories across processes,
    since the path-to-observer mapping is local to each process.
    """

    __path_to_observer: dict[Path, BaseWorkflowObserver]
    """The mapping from directory path to observer."""

    __main_scenario_observer: BaseWorkflowObserver | None
    """The observer for the main scenario."""

    # __failed_exec_directories: list[Path]
    # """The directories where executions failed."""

    def __init__(self) -> None:  # noqa: D107
        self.__path_to_observer = {}
        self.__main_scenario_observer = None

        if parent_path := getattr(current_process(), "parent_path", None):
            # In a worker process created with a non-fork start method,
            # a new instance of this class is created and the starting path is
            # no longer the root path: it is the parent working directory set
            # on the process object (see _rebuild_directory_manager and
            # _init_process_worker).
            # Not seen by coverage: this line only runs in worker
            # subprocesses, which the coverage tracer does not record.
            chdir(parent_path)  # pragma: no cover
        else:
            # The root path is created by default when validating
            # the configuration settings.
            chdir(_configuration.directory_manager.execution_root_path)

    def __reduce__(
        self,
    ) -> tuple[Callable[..., DirectoryManager], tuple[int, Path, dict[str, Any]]]:
        # Route unpickling through _rebuild_directory_manager so the receiving
        # process always resolves to its own singleton.
        #
        # Context: with a non-fork multiprocessing start method (the default
        # on Windows and on macOS/Python 3.14+), workers receive their state
        # via pickle rather than via fork. Processors hold a reference to this
        # manager (BaseDMProcessor.__dm); without __reduce__, that reference
        # would be unpickled as a standalone instance carrying the parent's
        # __path_to_observer snapshot, and __init__ would never run in the
        # child. The symptom was concurrent BiLevel sub-scenario workers
        # all generating the same Optimizer_iteration_1 path under a shared
        # parent and racing on mkdir (FileExistsError on Windows).
        #
        # Moreover, such workers rebuild the global configuration from scratch
        # (the directory manager is disabled there), and they unpickle the
        # pool initargs BEFORE the pool initializer tags the process with the
        # parent metadata. Both the metadata and the settings are therefore
        # pickled here and applied in _rebuild_directory_manager before the
        # singleton is constructed. Co-pickled references collapse to that
        # same singleton, keeping state consistent across all processors in
        # the worker.
        return (
            _rebuild_directory_manager,
            (
                getpid(),
                _get_cwd(),
                _configuration.directory_manager.model_dump(),
            ),
        )

    def start_directory(
        self,
        observer: BaseWorkflowObserver,
        name: str,
    ) -> None:
        """Start using a new directory.

        Args:
            observer: The current observer.
            name: The name of the processor.
        """
        directory_path = self.__get_directory_path(name)
        self.__path_to_observer[directory_path] = observer

        observer_is_scenario = isinstance(observer, ScenarioWorkflowObserver)

        if self.__main_scenario_observer is None and observer_is_scenario:
            self.__main_scenario_observer = observer

        directory_path.mkdir()
        (directory_path / self._MARKER_FILE_NAME).touch()
        self.__set_cwd(directory_path)

        if (
            _configuration.directory_manager.save_history_backup
            and observer_is_scenario
        ):
            # Do not pass plot: EvaluationScenario.set_backup_settings does not
            # have this argument (only the MDOScenario override does) and the
            # history view is written by end_directory anyway.
            observer._object.set_backup_settings(
                file_path=directory_path
                / _configuration.directory_manager.backup_settings.file_path,
                at_each_iteration=_configuration.directory_manager.backup_settings.at_each_iteration,
                at_each_function_call=_configuration.directory_manager.backup_settings.at_each_function_call,
                erase=False,
                load=False,
            )

    @staticmethod
    def __set_cwd(path: Path) -> None:
        """Set the current working directory in a thread safe way.

        When multi-threading, the current working directory is shared among
        the threads, thus no longer reliable: we explicitly store it in the
        thread object such that it can be easily obtained.

        Args:
            path: The path to be the current working directory.
        """
        chdir(path)
        thread = current_thread()
        # Store the cwd if we are in a thread spawned from gemseo.
        if hasattr(thread, "parent_path"):
            thread.cwd = path

    def __get_directory_path(self, name: str) -> Path:
        """Return the path to a new directory.

        It handles homonymic directories by renaming them with an indexed suffix.

        Args:
            name: The name of the processor.

        Returns:
            The path of the new directory.
        """
        # Unless in multi-threading, we make sure that the parent directory
        # is the current working directory (see end_directory where we chdir
        # to the parent).
        # When multi-threading, the current working directory is shared among
        # the threads, thus no longer reliable: we use the explicitly stored value,
        # i.e. the last directory used by this thread if any (see __set_cwd),
        # the starting directory of the worker thread otherwise.
        thread = current_thread()
        parent_path = (
            getattr(thread, "cwd", getattr(thread, "parent_path", None)) or Path.cwd()
        )

        # Ensure that name can be a filename.
        # secure_filename can return an empty string (e.g. for a name made of
        # non-ASCII characters only); such a name cannot be turned into a
        # directory name, so reject it instead of silently using a fallback.
        filename = secure_filename(name)
        if not filename:
            msg = (
                f"The name {name!r} cannot be used to create a directory; "
                "please use a name with at least one letter or digit."
            )
            raise ValueError(msg)

        directory_path = parent_path / filename

        # Go reverse since a potential homonymic directory could be the last one.
        # Iterate on a copy because we could modify the data structure during iteration.
        for path, observer_ in reversed(tuple(self.__path_to_observer.items())):
            if path == directory_path:
                # Add an indexed suffix to the previous homonymic
                # unsuffixed directory.
                previous_suffix = 0
                new_path = path.with_name(
                    path.name + self.suffix_separator + str(previous_suffix)
                )
                self.__path_to_observer.pop(path)
                self.__path_to_observer[new_path] = observer_

                # Get all the subdirectories with the current name.
                old_sub_directory_paths = [Path(x[0]) for x in walk(path)]
                old_sub_directory_paths.pop(0)

                # Rename the directory.
                # (On Windows the current working directory cannot be renamed.)
                chdir(path.parent)
                path.rename(new_path)
                chdir(new_path)

                # Get the subdirectories under the renamed directory.
                new_sub_directory_paths = [Path(x[0]) for x in walk(new_path)]
                new_sub_directory_paths.pop(0)

                # Update the paths to observers with the new paths.
                # The walk may also return directories that are not tracked
                # by the manager (e.g. created by an executed command line):
                # those have no observer and shall be left untouched.
                for old_sub_folder_path, new_sub_folder_path in zip(
                    old_sub_directory_paths, new_sub_directory_paths, strict=False
                ):
                    sub_observer = self.__path_to_observer.pop(
                        old_sub_folder_path, None
                    )
                    if sub_observer is not None:
                        self.__path_to_observer[new_sub_folder_path] = sub_observer
            elif (
                # Is it a homonymic directory that has been suffixed?
                str(path).startswith(str(directory_path))
                # Yes, then is it suffixed with the separator and an index?
                and (previous_suffix := path.name.rsplit(self.suffix_separator, 1)[-1])
                != path.name
            ):
                previous_suffix = int(previous_suffix)
            else:
                continue

            return directory_path.with_name(
                directory_path.name + self.suffix_separator + str(previous_suffix + 1)
            )

        return directory_path

    def end_directory(self, observer: BaseWorkflowObserver) -> None:
        """Finish using a directory.

        Args:
            observer: The current observer.
        """
        directory_path = self.__get_observer_path(observer)
        self.__set_cwd(directory_path)

        try:
            if (
                _configuration.directory_manager.save_history_backup
                and _configuration.directory_manager.backup_settings.plot
                and isinstance(observer, ScenarioWorkflowObserver)
            ):
                self.__write_history_view(observer)

            if _configuration.directory_manager.save_mda_residuals and isinstance(
                observer, MDAExecutionWorkflowObserver
            ):
                self.__write_mda_residuals(observer)

            for dir_path in self.__get_directories_to_remove(observer, directory_path):
                shutil.rmtree(dir_path)

            # Path(directory_path / "log").write_text(...)
            # TODO: Fix logging handling
        finally:
            # Always restore the parent as the current working directory, even
            # when a plot or a removal fails; this allows to easily determine
            # the parent path of children at the beginning of start_directory.
            self.__set_cwd(directory_path.parent)

    def __get_directories_to_remove(
        self, observer: BaseWorkflowObserver, directory_path: Path
    ) -> set[Path]:
        """Return the path of the directories to remove.

        Args:
            observer: The current observer.
            directory_path: The path of the directory of the observer.

        Returns:
            The paths of the directories to remove.
        """
        observer_is_mda = isinstance(observer, MDAExecutionWorkflowObserver)
        observer_is_scenario = isinstance(observer, ScenarioWorkflowObserver)

        if not (observer_is_mda or observer_is_scenario):
            return set()

        policy = (
            _configuration.directory_manager.mda_clean_up_policy
            if observer_is_mda
            else _configuration.directory_manager.clean_up_policy
        )

        # Subdirectories directly under the directory of the observer.
        # Only the directories created by the manager are candidates for removal:
        # directories created by other means (e.g. by an executed command line
        # or by the user) shall never be removed.
        sub_dir_paths = {
            path
            for path in directory_path.iterdir()
            if path.is_dir() and self.__is_managed(path)
        }

        if policy == _KEEP_ALL:
            return self.__get_removals_keep_all(observer_is_mda, sub_dir_paths)

        if policy == _KEEP_LAST_ONLY:
            return self.__get_removals_keep_last(sub_dir_paths)

        return self.__get_removals_solution(
            observer,
            sub_dir_paths,
            keep_baseline=policy == CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION,
        )

    def __get_removals_keep_all(
        self, observer_is_mda: bool, sub_dir_paths: set[Path]
    ) -> set[Path]:
        """Return removals for the KEEP_ALL policy.

        Args:
            observer_is_mda: Whether the observer is an MDA execution observer.
            sub_dir_paths: The subdirectories of the current working directory.

        Returns:
            The paths of the directories to remove.
        """
        if observer_is_mda:
            return set()
        return sub_dir_paths - self.__filter_paths_with_managed_subdirs(sub_dir_paths)

    def __get_removals_keep_last(self, sub_dir_paths: set[Path]) -> set[Path]:
        """Return removals for the KEEP_LAST_ONLY policy.

        Args:
            sub_dir_paths: The subdirectories of the current working directory.

        Returns:
            The paths of the directories to remove.
        """
        managed = self.__filter_paths_with_managed_subdirs(sub_dir_paths)
        suffixed = [
            (suffix, path)
            for path in managed
            if (suffix := self.__get_iteration_suffix(path)) is not None
        ]
        if not suffixed:
            # No managed iteration dir to keep: drop unmanaged dirs only.
            return sub_dir_paths - managed
        last_path = max(suffixed, key=operator.itemgetter(0))[1]
        return sub_dir_paths - {last_path}

    def __get_removals_solution(
        self,
        observer: BaseWorkflowObserver,
        sub_dir_paths: set[Path],
        keep_baseline: bool,
    ) -> set[Path]:
        """Return removals for the solution-based policies.

        Args:
            observer: The current observer.
            sub_dir_paths: The subdirectories of the current working directory.
            keep_baseline: Whether to also keep the baseline (first) iteration.

        Returns:
            The paths of the directories to remove.
        """
        problem = observer._object.formulation.problem
        try:
            optimum_iteration = problem.database.get_iteration(problem.optimum[1])
        except ValueError:
            # The execution failed before any complete evaluation (e.g. the
            # database is empty): keep everything rather than mask the
            # exception being propagated.
            return set()

        dir_paths_to_keep: set[Path] = set()
        baseline_suffix = maxsize
        baseline_path: Path | None = None

        for path in sub_dir_paths:
            suffix = self.__get_iteration_suffix(path)
            if suffix is None:
                continue
            if keep_baseline and suffix < baseline_suffix:
                baseline_suffix = suffix
                baseline_path = path
            if suffix == optimum_iteration:
                dir_paths_to_keep.add(path)

        if baseline_path is not None:
            dir_paths_to_keep.add(baseline_path)

        return sub_dir_paths - dir_paths_to_keep

    @staticmethod
    def __get_iteration_suffix(path: Path) -> int | None:
        """Return the trailing `_<int>` suffix of a path, or `None` if absent.

        Args:
            path: The path to inspect.

        Returns:
            The integer suffix, or `None` if the path does not end with `_<int>`.
        """
        tail = path.name.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return None

    @classmethod
    def __is_managed(cls, path: Path) -> bool:
        """Return whether a directory was created by the manager.

        Args:
            path: The path of the directory to inspect.

        Returns:
            Whether the directory was created by the manager.
        """
        return (path / cls._MARKER_FILE_NAME).exists()

    @classmethod
    def __filter_paths_with_managed_subdirs(cls, paths: Iterable[Path]) -> set[Path]:
        """Return the paths that contain at least one directory of the manager.

        Args:
            paths: The paths to inspect.

        Returns:
            The paths that contain at least one directory created by the manager.
        """
        return {
            path
            for path in paths
            if any(cls.__is_managed(child) for child in path.iterdir())
        }

    @staticmethod
    def __write_history_view(observer: ScenarioWorkflowObserver) -> None:
        """Write optimization history visualization to the execution directory.

        Generates and saves an OptHistoryView plot if the database contains
        sufficient data points (more than 2 entries).

        Args:
            observer: The scenario observer.
        """
        # Imported here to avoid a module-level dependency of gemseo.util on
        # gemseo.post.
        from gemseo.post import OptHistoryView_Settings

        scenario = observer._object
        if len(scenario.formulation.problem.database) > 2:
            scenario.post_process(
                OptHistoryView_Settings(
                    save=True,
                    show=False,
                    file_path=_configuration.directory_manager.backup_settings.file_path.stem,
                )
            )

    @staticmethod
    def __write_mda_residuals(observer: MDAWorkflowObserver) -> None:
        """Write MDA residual convergence plot to the execution directory.

        Generates and saves a visualization of residuals across MDA iterations.

        Args:
            observer: The MDA workflow observer.
        """
        mda = observer._object
        mda.plot_residual_history(
            save=True, filename=f"{mda.name}_residuals_history.pdf"
        )

    def __get_observer_path(self, observer: BaseWorkflowObserver) -> Path:
        """Return an observer's corresponding path.

        Args:
            observer: The workflow observer.

        Returns:
            The observer's corresponding path.

        Raises:
            RuntimeError: If the observer has no corresponding path.
        """
        for path, observer_ in reversed(self.__path_to_observer.items()):
            if id(observer_) == id(observer):
                return path.resolve()
        msg = f"No directory path found for observer {observer}"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover


def _get_cwd() -> Path:
    """Return the current working directory for the calling context.

    In multi-threaded contexts, returns the thread-local working directory stored
    on the current thread. Otherwise, returns the OS current working directory.

    Returns:
        The current working directory path.
    """
    # When multi-threading, the current working directory is shared among
    # the threads, thus no longer reliable: we use the explicitly stored value.
    return getattr(current_thread(), "cwd", Path.cwd())


def _rebuild_directory_manager(
    parent_id: int,
    parent_path: Path,
    settings_dump: dict[str, Any],
) -> DirectoryManager:
    """Recreate the directory manager singleton when unpickling.

    In a worker process created with a non-fork start method, the global
    configuration is rebuilt from scratch (so the directory manager is
    disabled), and the manager is recreated while unpickling the pool initargs,
    BEFORE the worker initializer tags the process with the parent metadata.
    The metadata and the directory manager settings pickled with the manager
    are therefore applied here, before constructing the singleton.

    When unpickling in a non-worker process (e.g. loading a pickled
    discipline in a later session), nothing is applied: the manager of the
    current process is simply returned.

    Args:
        parent_id: The process id of the parent process at pickling time.
        parent_path: The working directory of the parent at pickling time.
        settings_dump: The directory manager settings at pickling time.

    Returns:
        The directory manager singleton of the current process.
    """
    process = current_process()
    # A worker process either has a parent process (when unpickling task
    # arguments) or is inheriting its state (when unpickling the pool initargs
    # during the bootstrap of a spawned process, where parent_process() is not
    # set yet and _inheriting is the stdlib marker of that phase).
    in_worker = parent_process() is not None or getattr(process, "_inheriting", False)
    # Not seen by coverage: this block only runs in worker subprocesses,
    # which the coverage tracer does not record.
    if (  # pragma: no cover
        in_worker and getpid() != parent_id and not hasattr(process, "parent_path")
    ):
        process.parent_id = parent_id  # type: ignore[attr-defined]
        process.parent_path = parent_path  # type: ignore[attr-defined]
        # The parent metadata is set on the process above before the settings
        # are applied: the settings validator detects the worker from it and
        # reuses the parent's (already existing) execution root instead of
        # recreating it (which would raise in the main process).
        _configuration.directory_manager = type(_configuration.directory_manager)(
            **settings_dump
        )
    return DirectoryManager()
