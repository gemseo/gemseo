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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Abstraction of processes."""
from __future__ import annotations

import collections
import logging
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count
from multiprocessing import Manager
from multiprocessing import Value
from pathlib import Path
from timeit import default_timer as timer
from typing import Any
from typing import ClassVar
from typing import Generator
from typing import Iterable
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import TYPE_CHECKING

from numpy import concatenate
from numpy import empty
from numpy import ndarray
from numpy import zeros
from numpy.typing import NDArray
from strenum import StrEnum

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.cache import AbstractCache
from gemseo.core.data_processor import DataProcessor
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.discipline_data import DisciplineData
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.defaults import Defaults
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.namespaces import remove_prefix_from_list
from gemseo.core.serializable import Serializable
from gemseo.disciplines.utils import get_sub_disciplines
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.utils.derivatives.derivatives_approx import EPSILON
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.multiprocessing import get_multi_processing_manager
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.core.execution_sequence import SerialExecSequence

LOGGER = logging.getLogger(__name__)


def default_dict_factory() -> dict:
    """Instantiate a defaultdict(None) object."""
    return defaultdict(None)


class MDODiscipline(Serializable):
    """A software integrated in the workflow.

    To be used,
    subclass :class:`.MDODiscipline`
    and implement the :meth:`._run` method which defines the execution of the software.
    Typically,
    :meth:`._run` gets the input data stored in the :attr:`.local_data`,
    passes them to the callable computing the output data, e.g. a software,
    and stores these output data in the :attr:`.local_data`.

    Then,
    the end-user calls the :meth:`.execute` method
    with optional ``input_data``;
    if not,
    :attr:`.default_inputs` are used.

    This :meth:`.execute` method uses name grammars
    to check the variable names and types of
    both the passed input data before calling :meth:`.run`
    and the returned output data before they are stored in the :attr:`.cache`.
    A grammar can be either a :class:`.SimpleGrammar` or a :class:`.JSONGrammar`,
    or your own which derives from :class:`.BaseGrammar`.
    """

    class ExecutionStatus(StrEnum):
        """The execution statuses of a discipline."""

        VIRTUAL = "VIRTUAL"
        PENDING = "PENDING"
        DONE = "DONE"
        RUNNING = "RUNNING"
        FAILED = "FAILED"
        LINEARIZE = "LINEARIZE"

    class GrammarType(StrEnum):
        """The name of the grammar class."""

        JSON = "JSONGrammar"
        SIMPLE = "SimpleGrammar"

    class CacheType(StrEnum):
        """The name of the cache class."""

        SIMPLE = "SimpleCache"
        HDF5 = "HDF5Cache"
        MEMORY_FULL = "MemoryFullCache"
        NONE = ""
        """No cache is used."""

    ApproximationMode = ApproximationMode

    LinearizationMode = merge_enums(
        "LinearizationMode",
        StrEnum,
        DerivationMode,
        ApproximationMode,
    )

    class ReExecutionPolicy(StrEnum):
        """The re-execution policy of a discipline."""

        DONE = "RE_EXEC_DONE"
        NEVER = "RE_EXEC_NEVER"

    input_grammar: BaseGrammar
    """The input grammar."""

    output_grammar: BaseGrammar
    """The output grammar."""

    data_processor: DataProcessor
    """A tool to pre- and post-process discipline data."""

    residual_variables: Mapping[str, str]
    """The output variables mapping to their inputs, to be considered as residuals; they
    shall be equal to zero."""

    run_solves_residuals: bool
    """Whether the run method shall solve the residuals."""

    jac: dict[str, dict[str, ndarray]]
    """The Jacobians of the outputs wrt inputs.

    The structure is ``{output: {input: matrix}}``.
    """

    exec_for_lin: bool
    """Whether the last execution was due to a linearization."""

    name: str
    """The name of the discipline."""

    cache: AbstractCache | None
    """The cache containing one or several executions of the discipline according to the
    cache policy."""

    activate_counters: ClassVar[bool] = True
    """Whether to activate the counters (execution time, calls and linearizations)."""

    activate_input_data_check: ClassVar[bool] = True
    """Whether to check the input data respect the input grammar."""

    activate_output_data_check: ClassVar[bool] = True
    """Whether to check the output data respect the output grammar."""

    activate_cache: bool = True
    """Whether to cache the discipline evaluations by default."""

    re_exec_policy: ReExecutionPolicy
    """The policy to re-execute the same discipline."""

    GRAMMAR_DIRECTORY: ClassVar[str | None] = None
    """The directory in which to search for the grammar files if not the class one."""

    virtual_execution: ClassVar[bool] = False
    """Whether to skip the :meth:`._run` method during execution and return the
    default_outputs, whatever the inputs."""

    N_CPUS = cpu_count()

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = {
        "_status_observers",
    }

    __mp_manager: Manager = None
    time_stamps = None

    def __init__(
        self,
        name: str | None = None,
        input_grammar_file: str | Path | None = None,
        output_grammar_file: str | Path | None = None,
        auto_detect_grammar_files: bool = False,
        grammar_type: GrammarType = GrammarType.JSON,
        cache_type: CacheType = CacheType.SIMPLE,
        cache_file_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            name: The name of the discipline.
                If None, use the class name.
            input_grammar_file: The input grammar file path.
                If ``None`` and ``auto_detect_grammar_files=True``,
                look for ``"ClassName_input.json"``
                in the :attr:`.GRAMMAR_DIRECTORY` if any
                or in the directory of the discipline class module.
                If ``None`` and ``auto_detect_grammar_files=False``,
                do not initialize the input grammar from a schema file.
            output_grammar_file: The output grammar file path.
                If ``None`` and ``auto_detect_grammar_files=True``,
                look for ``"ClassName_output.json"``
                in the :attr:`.GRAMMAR_DIRECTORY` if any
                or in the directory of the discipline class module.
                If ``None`` and ``auto_detect_grammar_files=False``,
                do not initialize the output grammar from a schema file.
            auto_detect_grammar_files: Whether to
                look for ``"ClassName_{input,output}.json"``
                in the :attr:`.GRAMMAR_DIRECTORY` if any
                or in the directory of the discipline class module
                when ``{input,output}_grammar_file`` is ``None``.
            grammar_type: The type of the input and output grammars.
            cache_type: The type of cache.
            cache_file_path: The HDF file path
                when ``grammar_type`` is :attr:`.MDODiscipline.CacheType.HDF5`.
        """  # noqa: D205, D212, D415
        self.data_processor = None
        self.input_grammar = None
        self.output_grammar = None

        # Allow to re-execute the same discipline twice, only if did not fail
        # and not running
        self.re_exec_policy = self.ReExecutionPolicy.DONE
        # : list of outputs that shall be null, to be considered as residuals
        self.residual_variables = {}

        self._disciplines = []

        self.run_solves_residuals = False

        self._differentiated_inputs = []  # : inputs to differentiate
        # : outputs to be used for differentiation
        self._differentiated_outputs = []
        self._n_calls = None  # : number of calls to execute()
        self._exec_time = None  # : cumulated execution time
        # : number of calls to linearize()
        self._n_calls_linearize = None
        self._in_data_hash_dict = {}
        self.jac = {}  # : Jacobians of outputs wrt inputs dictionary
        # : True if linearize() has already been called
        self._is_linearized = False
        self._jac_approx = None  # Jacobian's approximation object
        self._linearize_on_last_state = False  # If true, the linearization
        # is performed on the state computed by the disciplines
        # (MDAs for instance) otherwise, the inputs that are also outputs
        # are reset at the previous inputs value be fore calling
        # _compute_jacobian

        # the last execution was due to a linearization
        self.exec_for_lin = False
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.cache = None
        if not self.activate_cache:
            cache_type = self.CacheType.NONE

        if cache_type is not self.CacheType.NONE:
            self.cache = self.__create_new_cache(
                cache_type, hdf_file_path=cache_file_path, hdf_node_path=self.name
            )

        self._cache_was_loaded = False

        self._linearization_mode = self.LinearizationMode.AUTO

        self._grammar_type = grammar_type

        if auto_detect_grammar_files:
            if input_grammar_file is None:
                input_grammar_file = self.auto_get_grammar_file()

            if output_grammar_file is None:
                output_grammar_file = self.auto_get_grammar_file(is_input=False)

        self._instantiate_grammars(
            input_grammar_file, output_grammar_file, self._grammar_type
        )

        self._local_data = None
        self.__set_local_data({})

        self._status = self.ExecutionStatus.PENDING
        if self.activate_counters:
            self._init_shared_memory_attrs()

        self._status_observers = []

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("Inputs: {}", pretty_str(self.get_input_data_names()))
        msg.add("Outputs: {}", pretty_str(self.get_output_data_names()))
        return str(msg)

    def _init_shared_memory_attrs(self) -> None:
        self._n_calls = Value("i", 0)
        self._exec_time = Value("d", 0.0)
        self._n_calls_linearize = Value("i", 0)

    @property
    def disciplines(self) -> list[MDODiscipline]:
        """The sub-disciplines, if any."""
        return self._disciplines

    @property
    def local_data(self) -> DisciplineData:
        """The current input and output data."""
        return self._local_data

    @local_data.setter
    def local_data(self, data: MutableMapping[str, Any]) -> None:
        self.__set_local_data(data)

    def __set_local_data(self, data: MutableMapping[str, Any]) -> None:
        self._local_data = DisciplineData(
            data,
            input_to_namespaced=self.input_grammar.to_namespaced,
            output_to_namespaced=self.output_grammar.to_namespaced,
        )

    @property
    def n_calls(self) -> int | None:
        """The number of times the discipline was executed.

        This property is multiprocessing safe.

        Raises:
            RuntimeError: When the discipline counters are disabled.
        """
        if self.activate_counters:
            return self._n_calls.value

    @n_calls.setter
    def n_calls(
        self,
        value: int,
    ) -> None:
        if not self.activate_counters:
            raise RuntimeError("The discipline counters are disabled.")

        self._n_calls.value = value

    @property
    def exec_time(self) -> float | None:
        """The cumulated execution time of the discipline.

        This property is multiprocessing safe.

        Raises:
            RuntimeError: When the discipline counters are disabled.
        """
        if self.activate_counters:
            return self._exec_time.value

    @exec_time.setter
    def exec_time(
        self,
        value: float,
    ) -> None:
        if not self.activate_counters:
            raise RuntimeError("The discipline counters are disabled.")

        self._exec_time.value = value

    @property
    def n_calls_linearize(self) -> int | None:
        """The number of times the discipline was linearized.

        This property is multiprocessing safe.

        Raises:
            RuntimeError: When the discipline counters are disabled.
        """
        if self.activate_counters:
            return self._n_calls_linearize.value

    @n_calls_linearize.setter
    def n_calls_linearize(
        self,
        value: int,
    ) -> None | NoReturn:
        if not self.activate_counters:
            raise RuntimeError("The discipline counters are disabled.")

        self._n_calls_linearize.value = value

    @property
    def grammar_type(self) -> GrammarType:
        """The type of grammar to be used for inputs and outputs declaration."""
        return self._grammar_type

    def auto_get_grammar_file(
        self,
        is_input: bool = True,
        name: str | None = None,
        comp_dir: str | Path | None = None,
    ) -> str:
        """Use a naming convention to associate a grammar file to the discipline.

        Search in the directory ``comp_dir`` for
        either an input grammar file named ``name + "_input.json"``
        or an output grammar file named ``name + "_output.json"``.

        Args:
            is_input: Whether to search for an input or output grammar file.
            name: The name to be searched in the file names.
                If ``None``,
                use the name of the discipline class.
            comp_dir: The directory in which to search the grammar file.
                If None,
                use the :attr:`.GRAMMAR_DIRECTORY` if any,
                or the directory of the discipline class module.

        Returns:
            The grammar file path.
        """
        cls = self.__class__
        initial_name = name or cls.__name__
        classes = [cls] + [
            base for base in cls.__bases__ if issubclass(base, MDODiscipline)
        ]
        names = [initial_name] + [cls.__name__ for cls in classes[1:]]

        in_or_out = "in" if is_input else "out"
        for cls, name in zip(classes, names):
            grammar_file_path = self.__get_grammar_file_path(
                cls, comp_dir, in_or_out, name
            )
            if grammar_file_path.is_file():
                return grammar_file_path

        file_name = f"{initial_name}_{in_or_out}put.json"
        raise FileNotFoundError(f"The grammar file {file_name} is missing.")

    @staticmethod
    def __get_grammar_file_path(
        cls: type, comp_dir: str | Path | None, in_or_out: str, name: str
    ) -> Path:
        """Return the grammar file path.

        Args:
            cls: The class for which the grammar file is searched.
            comp_dir: The initial directory path if any.
            in_or_out: The suffix to look for in the file name, either "in" or "out".
            name: The name to be searched in the file names.

        Returns:
            The grammar file path.
        """
        grammar_directory = comp_dir
        if grammar_directory is None:
            grammar_directory = cls.GRAMMAR_DIRECTORY

        if grammar_directory is None:
            class_module = sys.modules[cls.__module__]
            grammar_directory = Path(class_module.__file__).parent.absolute()
        else:
            grammar_directory = Path(grammar_directory)

        return grammar_directory / f"{name}_{in_or_out}put.json"

    def add_differentiated_inputs(
        self,
        inputs: Iterable[str] | None = None,
    ) -> None:
        """Add the inputs against which to differentiate the outputs.

        If the discipline grammar type is :attr:`.MDODiscipline.GrammarType.JSON` and
        an input is either a non-numeric array or not an array, it will be ignored.
        If an input is declared as an array but the type of its items is not defined, it
        is assumed as a numeric array.

        If the discipline grammar type is :attr:`.MDODiscipline.GrammarType.SIMPLE` and
        an input is not an array, it will be ignored. Keep in mind that in this case
        the array subtype is not checked.

        Args:
            inputs: The input variables against which to differentiate the outputs.
                If None, all the inputs of the discipline are used.

        Raises:
            ValueError: When the inputs wrt which differentiate the discipline
                are not inputs of the latter.
        """
        if (inputs is not None) and (not self.is_all_inputs_existing(inputs)):
            raise ValueError(
                f"Cannot differentiate the discipline {self.name} w.r.t. the inputs "
                "that are not among the discipline inputs: "
                f"{self.get_input_data_names()}."
            )

        if inputs is None:
            inputs = self.get_input_data_names()

        inputs = [
            input_ for input_ in inputs if self.input_grammar.is_array(input_, True)
        ]

        in_diff = self._differentiated_inputs
        self._differentiated_inputs = list(set(in_diff).union(inputs))

    def add_differentiated_outputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> None:
        """Add the outputs to be differentiated.

        If the discipline grammar type is :attr:`.MDODiscipline.GrammarType.JSON` and
        an output is either a non-numeric array or not an array, it will be ignored.
        If an output is declared as an array but the type of its items is not defined,
        it is assumed as a numeric array.

        If the discipline grammar type is :attr:`.MDODiscipline.GrammarType.SIMPLE` and
        an output is not an array, it will be ignored. Keep in mind that in this case
        the array subtype is not checked.

        Args:
            outputs: The output variables to be differentiated.
                If None, all the outputs of the discipline are used.

        Raises:
            ValueError: When the outputs to differentiate are not discipline outputs.
        """
        if (outputs is not None) and (not self.is_all_outputs_existing(outputs)):
            raise ValueError(
                f"Cannot differentiate the discipline {self.name} w.r.t. the outputs "
                "that are not among the discipline outputs: "
                f"{self.get_output_data_names()}."
            )

        out_diff = self._differentiated_outputs
        if outputs is None:
            outputs = self.get_output_data_names()

        outputs = [
            output for output in outputs if self.output_grammar.is_array(output, True)
        ]

        self._differentiated_outputs = list(set(out_diff).union(outputs))

    def __create_new_cache(
        self,
        class_name: str,
        **kwargs: bool | float | str,
    ) -> AbstractCache:
        """Create a cache object.

        Args:
            class_name: The name of the cache class.
            **kwargs: The arguments to instantiate a cache object.

        Returns:
            The cache object.
        """
        if class_name != self.CacheType.HDF5:
            for key in ("hdf_file_path", "hdf_node_path"):
                if key in kwargs:
                    del kwargs[key]

        if class_name != self.CacheType.MEMORY_FULL:
            key = "is_memory_shared"
            if key in kwargs:
                del kwargs[key]

        return CacheFactory().create(class_name, name=self.name, **kwargs)

    def set_cache_policy(
        self,
        cache_type: CacheType = CacheType.SIMPLE,
        cache_tolerance: float = 0.0,
        cache_hdf_file: str | Path | None = None,
        cache_hdf_node_path: str | None = None,
        is_memory_shared: bool = True,
    ) -> None:
        """Set the type of cache to use and the tolerance level.

        This method defines when the output data have to be cached
        according to the distance between the corresponding input data
        and the input data already cached for which output data are also cached.

        The cache can be either a :class:`.SimpleCache` recording the last execution
        or a cache storing all executions,
        e.g. :class:`.MemoryFullCache` and :class:`.HDF5Cache`.
        Caching data can be either in-memory,
        e.g. :class:`.SimpleCache` and :class:`.MemoryFullCache`,
        or on the disk,
        e.g. :class:`.HDF5Cache`.

        The attribute :attr:`.CacheFactory.caches` provides the available caches types.

        Args:
            cache_type: The type of cache.
            cache_tolerance: The maximum relative norm
                of the difference between two input arrays
                to consider that two input arrays are equal.
            cache_hdf_file: The path to the HDF file to store the data;
                this argument is mandatory when the
                :attr:`.MDODiscipline.CacheType.HDF5` policy is used.
            cache_hdf_node_path: The name of the HDF file node
                to store the discipline data,
                possibly passed as a path ``root_name/.../group_name/.../node_name``.
                If None, :attr:`.MDODiscipline.name` is used.
            is_memory_shared: Whether to store the data with a shared memory dictionary,
                which makes the cache compatible with multiprocessing.

                .. warning:

                   If set to False,
                   and multiple disciplines point
                   to the same cache or the process is multiprocessed,
                   there may be duplicate computations
                   because the cache will not be shared among the processes.
        """
        if self.cache.__class__.__name__ != cache_type or not (
            cache_type == self.CacheType.HDF5
            and cache_hdf_file == self.cache.hdf_file.hdf_file_path
            and cache_hdf_node_path == self.cache.hdf_node_path
        ):
            self.cache = self.__create_new_cache(
                cache_type,
                tolerance=cache_tolerance,
                hdf_file_path=cache_hdf_file,
                hdf_node_path=cache_hdf_node_path or self.name,
                is_memory_shared=is_memory_shared,
            )
        else:
            LOGGER.warning(
                (
                    "The cache policy is already set to %s "
                    "with the file path %r and node name %r; "
                    "call discipline.cache.clear() to clear the cache."
                ),
                cache_type,
                cache_hdf_file,
                cache_hdf_node_path,
            )

    def get_sub_disciplines(self, recursive: bool = False) -> list[MDODiscipline]:
        """Determine the sub-disciplines.

        This method lists the sub-disciplines' disciplines. It will list up to one level
        of disciplines contained inside another one unless the ``recursive`` argument is
        set to ``True``.

        Args:
            recursive: If ``True``, the method will look inside any discipline that has
                other disciplines inside until it reaches a discipline without
                sub-disciplines, in this case the return value will not include any
                discipline that has sub-disciplines. If ``False``, the method will list
                up to one level of disciplines contained inside another one, in this
                case the return value may include disciplines that contain
                sub-disciplines.

        Returns:
            The sub-disciplines.
        """
        return get_sub_disciplines(self._disciplines, recursive)

    def get_expected_workflow(self) -> SerialExecSequence:
        """Return the expected execution sequence.

        This method is used for the XDSM representation.

        The default expected execution sequence
        is the execution of the discipline itself.

        .. seealso::

           MDOFormulation.get_expected_workflow

        Returns:
            The expected execution sequence.
        """
        # avoid circular dependency
        from gemseo.core.execution_sequence import ExecutionSequenceFactory

        return ExecutionSequenceFactory.serial(self)

    def get_expected_dataflow(
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        """Return the expected data exchange sequence.

        This method is used for the XDSM representation.

        The default expected data exchange sequence is an empty list.

        .. seealso::

           MDOFormulation.get_expected_dataflow

        Returns:
            The data exchange arcs.
        """
        return []

    def get_disciplines_in_dataflow_chain(self) -> list[MDODiscipline]:
        """Return the disciplines that must be shown as blocks in the XDSM.

        By default, only the discipline itself is shown.
        This function can be differently implemented for any type of inherited discipline.

        Returns:
            The disciplines shown in the XDSM chain.
        """
        return [self]

    def _instantiate_grammars(
        self,
        input_grammar_file: str | Path | None,
        output_grammar_file: str | Path | None,
        grammar_type: GrammarType = GrammarType.JSON,
    ) -> None:
        """Create the input and output grammars.

        Args:
            input_grammar_file: The input grammar file path.
                If None, do not initialize the input grammar from a schema file.
            output_grammar_file: The output grammar file path.
                If None, do not initialize the output grammar from a schema file.
            grammar_type: The type of the input and output grammars.
        """
        factory = GrammarFactory()
        self.input_grammar = factory.create(
            grammar_type,
            name=f"{self.name}_input",
            file_path=input_grammar_file,
        )
        self.output_grammar = factory.create(
            grammar_type,
            name=f"{self.name}_output",
            file_path=output_grammar_file,
        )

    def _run(self) -> None:
        """Define the execution of the process, given that data has been checked.

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def _filter_inputs(
        self, input_data: dict[str, Any] | None = None
    ) -> MutableMapping[str, Any]:
        """Filter data with the discipline inputs and use the default values if missing.

        Args:
            input_data: The data to be filtered.

        Returns:
            The values of the input variables based on the provided data.

        Raises:
            TypeError: When the input data are not passed as a dictionary.
        """
        if input_data is None:
            return deepcopy(self.default_inputs)

        if not isinstance(input_data, collections.abc.Mapping):
            raise TypeError(
                "Input data must be of dict type, "
                "got {} instead.".format(type(input_data))
            )

        full_input_data = DisciplineData({})
        for input_name in self.input_grammar:
            input_value = input_data.get(input_name)
            if input_value is not None:
                full_input_data[input_name] = input_value
            else:
                input_value = self.input_grammar.defaults.get(input_name)
                if input_value is not None:
                    full_input_data[input_name] = input_value

        return full_input_data

    def _filter_local_data(self) -> None:
        """Filter the local data after execution.

        This method removes data that are neither inputs nor outputs.
        """
        for key in self._local_data.keys() - self.get_input_output_data_names():
            del self._local_data[key]

    def _check_status_before_run(self) -> None:
        """Check the status of the discipline.

        Check the status of the discipline depending on
        :attr:`.MDODiscipline.re_execute_policy`.

        If ``re_exec_policy == ReExecutionPolicy.NEVER``,
        the status shall be either :attr:`.MDODiscipline.ExecutionStatus.PENDING`
        or :attr:`.MDODiscipline.VIRTUAL`.

        If ``self.re_exec_policy == ReExecutionPolicy.NEVER``,

        - if status is :attr:`.MDODiscipline.ExecutionStatus.DONE`,
          :meth:`.MDODiscipline.reset_statuses_for_run`.
        - otherwise status must be :attr:`.MDODiscipline.VIRTUAL`
          or :attr:`.MDODiscipline.ExecutionStatus.PENDING`.

        Raises:
            ValueError:
                When the discipline status and the re-execution policy are no consistent.
        """
        if self.status not in [
            self.ExecutionStatus.PENDING,
            self.ExecutionStatus.VIRTUAL,
            self.ExecutionStatus.DONE,
        ] or (
            self.status == self.ExecutionStatus.DONE
            and self.re_exec_policy == self.ReExecutionPolicy.NEVER
        ):
            raise ValueError(
                f"Trying to run a discipline {type(self)} with status: {self.status} "
                f"while re_exec_policy is {self.re_exec_policy}."
            )

    def __get_input_data_for_cache(
        self,
        input_data: dict[str, Any],
        in_names: Iterable[str],
    ) -> dict[str, Any]:
        """Prepare the input data for caching.

        Args:
            input_data: The values of the inputs.
            in_names: The names of the inputs.

        Returns:
            The input data to be cached.
        """
        in_and_out = set(in_names) & set(self.get_output_data_names())

        cached_inputs = input_data.copy()

        for key in in_and_out:
            val = input_data.get(key)
            if val is not None:
                # If also an output, keeps a copy of the original input value
                cached_inputs[key] = deepcopy(val)

        return cached_inputs

    def execute(
        self,
        input_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the discipline.

        This method executes the discipline:

        * Adds the default inputs to the ``input_data``
          if some inputs are not defined in input_data
          but exist in :attr:`.MDODiscipline.default_inputs`.
        * Checks whether the last execution of the discipline was called
          with identical inputs, i.e. cached in :attr:`.MDODiscipline.cache`;
          if so, directly returns ``self.cache.get_output_cache(inputs)``.
        * Caches the inputs.
        * Checks the input data against :attr:`.MDODiscipline.input_grammar`.
        * If :attr:`.MDODiscipline.data_processor` is not None, runs the preprocessor.
        * Updates the status to :attr:`.MDODiscipline.ExecutionStatus.RUNNING`.
        * Calls the :meth:`.MDODiscipline._run` method, that shall be defined.
        * If :attr:`.MDODiscipline.data_processor` is not None, runs the postprocessor.
        * Checks the output data.
        * Caches the outputs.
        * Updates the status to :attr:`.MDODiscipline.ExecutionStatus.DONE`
          or :attr:`.MDODiscipline.ExecutionStatus.FAILED`.
        * Updates summed execution time.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.MDODiscipline.default_inputs`.

        Returns:
            The discipline local data after execution.

        Raises:
            RuntimeError: When residual_variables are declared but
                self.run_solves_residuals is False. This is not supported yet.
        """
        if self.residual_variables and not self.run_solves_residuals:
            raise RuntimeError(
                "Disciplines that do not solve their residuals are not supported yet."
            )
        # Load the default_inputs if the user did not provide all required data
        input_data = self._filter_inputs(input_data)

        if self.cache is not None:
            # Check if the cache already contains the outputs associated to these inputs
            in_names = self.get_input_data_names()
            _, out_cached, out_jac = self.cache[input_data]
            if out_cached:
                self.__update_local_data_from_cache(input_data, out_cached, out_jac)
                return self._local_data

        # Cache was not loaded, see self.linearize
        self._cache_was_loaded = False

        if self.cache is not None:
            # Save the state of the inputs
            cached_inputs = self.__get_input_data_for_cache(input_data, in_names)

        if self.activate_input_data_check:
            self.check_input_data(input_data)

        processor = self.data_processor
        if processor is not None:
            self.__set_local_data(processor.pre_process_data(input_data))
        else:
            self.__set_local_data(input_data)

        self._is_linearized = False
        if self.activate_counters:
            self.__increment_n_calls()

        t_0 = timer()

        self._check_status_before_run()
        self.status = self.ExecutionStatus.RUNNING

        if not self.virtual_execution:
            try:
                # Effectively run the discipline, the _run method has to be
                # Defined by the subclasses
                self._run()
            except Exception:
                self.status = self.ExecutionStatus.FAILED
                # Update the status but
                # raise the same exception
                raise
        else:
            self.store_local_data(**self.default_outputs)

        self.status = self.ExecutionStatus.DONE

        if self.activate_counters:
            self.__increment_exec_time(t_0)

        # If the data processor is set, post process the data after _run
        # See gemseo.core.data_processor module
        if processor is not None:
            self.__set_local_data(processor.post_process_data(self._local_data))

        # Filter data that is neither outputs nor inputs
        self._filter_local_data()

        if self.activate_output_data_check:
            self.check_output_data()

        if self.cache is not None:
            # Caches output data in case the discipline is called twice in a row
            # with the same inputs
            out_names = self.get_output_data_names()
            if out_names:
                self.cache.cache_outputs(
                    cached_inputs, self._local_data.copy(keys=out_names)
                )
            # Some disciplines are always linearized during execution, cache the
            # jac in this case
            if self._is_linearized:
                self.cache.cache_jacobian(cached_inputs, self.jac)

        return self._local_data

    def __update_local_data_from_cache(
        self,
        input_data: dict[str, Any],
        out_cached: dict[str, Any],
        out_jac: dict[str, ndarray],
    ) -> None:
        """Update the local data from the cache.

        Args:
            input_data: The input data.
            out_cached: The output data retrieved from the cache.
            out_jac: The Jacobian data retrieved from the cache.
        """
        self.__set_local_data(input_data)
        self._local_data.update(out_cached)

        if out_jac is not None:
            self.jac = out_jac
            self._is_linearized = True
        else:  # Erase jacobian which is unknown
            self.jac.clear()
            self._is_linearized = False

        self.check_output_data()
        self._cache_was_loaded = True

    def __increment_n_calls(self) -> None:
        """Increment by 1 the number of executions.."""
        with self._n_calls.get_lock():
            self._n_calls.value += 1

    def __increment_n_calls_lin(self) -> None:
        """Increment by 1 the number of linearizations."""
        with self._n_calls_linearize.get_lock():
            self._n_calls_linearize.value += 1

    def __increment_exec_time(
        self,
        t_0: float,
        linearize: bool = False,
    ) -> None:
        """Increment the execution time of the discipline.

        The execution can be either an evaluation or a linearization.

        Args:
            t_0: The time of the execution start.
            linearize: Whether it is a linearization.
        """
        curr_t = timer()
        with self._exec_time.get_lock():
            self._exec_time.value += curr_t - t_0

            time_stamps = MDODiscipline.time_stamps
            if time_stamps is not None:
                disc_stamps = time_stamps.get(self.name, self.__mp_manager.list())
                stamp = (t_0, curr_t, linearize)
                disc_stamps.append(stamp)
                time_stamps[self.name] = disc_stamps

    def _retrieve_diff_inouts(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Get the inputs and outputs used in the differentiation of the discipline.

        Args:
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables against which to differentiate the output ones
                with :meth:`.add_differentiated_inputs`
                and set these output variables to differentiate
                with :meth:`.add_differentiated_outputs`.
        """
        if compute_all_jacobians:
            return self.get_input_data_names(), self.get_output_data_names()

        return self._differentiated_inputs, self._differentiated_outputs

    @classmethod
    def activate_time_stamps(cls) -> None:
        """Activate the time stamps.

        For storing start and end times of execution and linearizations.
        """
        MDODiscipline.__mp_manager = manager = get_multi_processing_manager()
        MDODiscipline.time_stamps = manager.dict()

    @classmethod
    def deactivate_time_stamps(cls) -> None:
        """Deactivate the time stamps.

        For storing start and end times of execution and linearizations.
        """
        MDODiscipline.time_stamps = None

    def linearize(
        self,
        input_data: Mapping[str, Any] | None = None,
        compute_all_jacobians: bool = False,
        execute: bool = True,
    ) -> dict[str, dict[str, NDArray[float]]]:
        """Compute the Jacobians of some outputs with respect to some inputs.

        Args:
            input_data: The input data for which to compute the Jacobian.
                If ``None``, use the :attr:`.MDODiscipline.default_inputs`.
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables against which to differentiate the output ones
                with :meth:`.add_differentiated_inputs`
                and set these output variables to differentiate
                with :meth:`.add_differentiated_outputs`.
            execute: Whether to start by executing the discipline
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.

        Returns:
            The Jacobian of the discipline
            shaped as ``{output_name: {input_name: jacobian_array}}`` where
            ``jacobian_array[i, j]`` is the partial derivative of ``output_name[i]``
            with respect to ``input_name[j]``.

        Raises:
            ValueError: When either the inputs
                for which to differentiate the outputs
                or the outputs to differentiate are missing.
        """
        # TODO: remove the execution when no option exec_before_lin
        # is set to True
        inputs, outputs = self._retrieve_diff_inouts(compute_all_jacobians)
        if not outputs or not inputs:
            input_data = self._filter_inputs(input_data)
            _, out_cached, out_jac = self.cache[input_data]
            if out_cached:
                self.jac = out_jac
            else:
                self.jac = {}
            # self.jac.clear()  # this aint work
            # self.jac = {}  # this aint work
            # return {}#this aint work
            return self.jac  # this aint work

        full_input_data = self._filter_inputs(input_data)
        if execute:
            self.reset_statuses_for_run()
            self.exec_for_lin = True
            self.execute(full_input_data)
            self.exec_for_lin = False

        # The local_data shall be reset to their original values
        # in case an input is also an output,
        # if we don't want to keep the computed state (as in MDAs).
        if not self._linearize_on_last_state:
            self._local_data.update(full_input_data)

        # If the caching was triggered,
        # check if the jacobian was loaded,
        # or the discipline._run method also linearizes the discipline.
        if self._cache_was_loaded or self._is_linearized:
            if self.jac:
                # For cases when linearization is called twice with different I/O
                # while cache_was_loaded=True,
                # the check_jacobian_shape raises a KeyError.
                try:
                    self._check_jacobian_shape(inputs, outputs)
                    return self.jac
                except KeyError:
                    # In this case,
                    # another computation of Jacobian is triggered.
                    pass

        self.status = self.ExecutionStatus.LINEARIZE
        t_0 = timer()
        if self._linearization_mode in set(self.ApproximationMode):
            # Time already counted in execute()
            self.jac = self._jac_approx.compute_approx_jac(outputs, inputs)
        else:
            self._compute_jacobian(inputs, outputs)
            if self.activate_counters:
                self.__increment_exec_time(t_0, linearize=True)

        if self.activate_counters:
            self.__increment_n_calls_lin()

        if not compute_all_jacobians:
            for output_name in list(self.jac.keys()):
                if output_name not in outputs:
                    del self.jac[output_name]
                else:
                    jac = self.jac[output_name]
                    for input_name in list(jac.keys()):
                        if input_name not in inputs:
                            del jac[input_name]

        self._check_jacobian_shape(inputs, outputs)
        if self.cache is not None:
            self.cache.cache_jacobian(full_input_data, self.jac)

        self.status = self.ExecutionStatus.DONE
        return self.jac

    def set_jacobian_approximation(
        self,
        jac_approx_type: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        jax_approx_step: float = 1e-7,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        jac_approx_wait_time: float = 0,
    ) -> None:
        """Set the Jacobian approximation method.

        Sets the linearization mode to approx_method,
        sets the parameters of the approximation for further use
        when calling :meth:`.MDODiscipline.linearize`.

        Args:
            jac_approx_type: The approximation method,
                either "complex_step" or "finite_differences".
            jax_approx_step: The differentiation step.
            jac_approx_n_processes: The maximum simultaneous number of threads,
                if ``jac_approx_use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            jac_approx_use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            jac_approx_wait_time: The time waited between two forks
                of the process / thread.
        """
        appx = DisciplineJacApprox(
            self,
            approx_method=jac_approx_type,
            step=jax_approx_step,
            parallel=jac_approx_n_processes > 1,
            n_processes=jac_approx_n_processes,
            use_threading=jac_approx_use_threading,
            wait_time_between_fork=jac_approx_wait_time,
        )
        self._jac_approx = appx
        self.linearization_mode = jac_approx_type

    def set_optimal_fd_step(
        self,
        outputs: Iterable[str] | None = None,
        inputs: Iterable[str] | None = None,
        compute_all_jacobians: bool = False,
        print_errors: bool = False,
        numerical_error: float = EPSILON,
    ):
        """Compute the optimal finite-difference step.

        Compute the optimal step
        for a forward first order finite differences gradient approximation.
        Requires a first evaluation of the perturbed functions values.
        The optimal step is reached when the truncation error
        (cut in the Taylor development),
        and the numerical cancellation errors
        (round-off when doing f(x+step)-f(x))
        are approximately equal.

        .. warning::

           This calls the discipline execution twice per input variables.

        .. seealso::

           https://en.wikipedia.org/wiki/Numerical_differentiation
           and
           "Numerical Algorithms and Digital Representation", Knut Morken ,
           Chapter 11, "Numerical Differentiation"

        Args:
            inputs: The inputs wrt which the outputs are linearized.
                If None, use the :attr:`.MDODiscipline._differentiated_inputs`.
            outputs: The outputs to be linearized.
                If None, use the :attr:`.MDODiscipline._differentiated_outputs`.
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables against which to differentiate the output ones
                with :meth:`.add_differentiated_inputs`
                and set these output variables to differentiate
                with :meth:`.add_differentiated_outputs`.
            print_errors: Whether to display the estimated errors.
            numerical_error: The numerical error associated to the calculation of f.
                By default, this is the machine epsilon (appx 1e-16),
                but can be higher
                when the calculation of f requires a numerical resolution.

        Returns:
            The estimated errors of truncation and cancellation error.

        Raises:
            ValueError: When the Jacobian approximation method has not been set.
        """
        if self._jac_approx is None:
            raise ValueError(
                "set_jacobian_approximation must be called "
                "before setting an optimal step."
            )
        diff_inputs, diff_outputs = self._retrieve_diff_inouts(
            compute_all_jacobians=compute_all_jacobians
        )
        if outputs is None or compute_all_jacobians:
            outputs = diff_outputs
        if inputs is None or compute_all_jacobians:
            inputs = diff_inputs
        return self._jac_approx.auto_set_step(
            outputs, inputs, print_errors, numerical_error=numerical_error
        )

    @staticmethod
    def __get_len(container) -> int:
        """Measure the length of a container."""
        if container is None:
            return -1
        try:
            return len(container)
        except TypeError:
            return 1

    def _check_jacobian_shape(
        self,
        inputs: Iterable[str],
        outputs: Iterable[str],
    ) -> None:
        """Check that the Jacobian is a dictionary of dictionaries of 2D NumPy arrays.

        Args:
            inputs: The inputs wrt the outputs are linearized.
            outputs: The outputs to be linearized.

        Raises:
            ValueError:
                When the discipline was not linearized.
                When the Jacobian is not of the right shape.
            KeyError:
                When outputs are missing in the Jacobian of the discipline.
                When inputs are missing for an output in the Jacobian of the discipline.
        """
        if not self.jac:
            raise ValueError(f"The discipline {self.name} was not linearized.")
        out_set = set(outputs)
        in_set = set(inputs)
        out_jac_set = self.jac.keys()

        if not out_set.issubset(out_jac_set):
            msg = "Missing outputs in Jacobian of discipline {}: {}."
            missing_outputs = out_set.difference(out_jac_set)
            raise KeyError(msg.format(self.name, missing_outputs))

        for j_o in outputs:
            j_out = self.jac[j_o]
            out_dv_set = j_out.keys()
            output_vals = self._local_data.get(j_o)
            n_out_j = self.__get_len(output_vals)

            if not in_set.issubset(out_dv_set):
                msg = "Missing inputs {} in Jacobian of discipline {}, for output: {}."
                missing_inputs = in_set.difference(out_dv_set)
                raise KeyError(msg.format(missing_inputs, self.name, j_o))

            for j_i in inputs:
                input_vals = self._local_data.get(j_i)
                n_in_j = self.__get_len(input_vals)
                j_mat = j_out[j_i]
                expected_shape = (n_out_j, n_in_j)

                if -1 in expected_shape:
                    # At least one of the dimensions is unknown
                    # Don't check shape
                    continue

                if j_mat.shape != expected_shape:
                    raise ValueError(
                        f"The shape {j_mat.shape} "
                        f"of the Jacobian matrix d{j_o}/d{j_i} "
                        f"of the discipline {self.name} "
                        "does not match "
                        f"(output_size, input_size)=({n_out_j}, {n_in_j})."
                    )

        # Discard imaginary part of Jacobian
        for output_jacobian in self.jac.values():
            for input_name, input_output_jacobian in output_jacobian.items():
                output_jacobian[input_name] = input_output_jacobian.real

    @property
    def cache_tol(self) -> float:
        """The cache input tolerance.

        This is the tolerance for equality of the inputs in the cache.
        If norm(stored_input_data-input_data) <= cache_tol * norm(stored_input_data),
        the cached data for ``stored_input_data`` is returned
        when calling ``self.execute(input_data)``.

        Raises:
            ValueError: When the discipline does not have a cache.
        """
        if self.cache is None:
            raise ValueError(f"The discipline {self.name} does not have a cache.")

        return self.cache.tolerance

    @cache_tol.setter
    def cache_tol(
        self,
        cache_tol: float,
    ) -> None:
        if self.cache is None:
            raise ValueError(f"The discipline {self.name} does not have a cache.")

        self._set_cache_tol(cache_tol)

    def _set_cache_tol(
        self,
        cache_tol: float,
    ) -> None:
        """Set to the cache input tolerance.

        To be overloaded by subclasses.

        Args:
            cache_tol: The cache tolerance.
        """
        self.cache.tolerance = cache_tol or 0.0

    @property
    def default_inputs(self) -> Defaults:
        """The default inputs."""
        return self.input_grammar.defaults

    @default_inputs.setter
    def default_inputs(self, data: Mapping[str, Any]) -> None:
        self.input_grammar.defaults = data

    @property
    def default_outputs(self) -> Defaults:
        """The default outputs."""
        return self.output_grammar.defaults

    @default_outputs.setter
    def default_outputs(self, data: Mapping[str, Any]) -> None:
        self.output_grammar.defaults = data

    def add_namespace_to_input(self, name: str, namespace: str) -> None:
        """Add a namespace prefix to an existing input grammar element.

        The updated input grammar element name will be
        ``namespace`` + :data:`~gemseo.core.namespaces.namespaces_separator` + ``name``.

        Args:
            name: The element name to rename.
            namespace: The name of the namespace.
        """
        self.input_grammar.add_namespace(name, namespace)

    def add_namespace_to_output(self, name: str, namespace: str) -> None:
        """Add a namespace prefix to an existing output grammar element.

        The updated output grammar element name will be
        ``namespace`` + :data:`~gemseo.core.namespaces.namespaces_separator` + ``name``.

        Args:
            name: The element name to rename.
            namespace: The name of the namespace.
        """
        self.output_grammar.add_namespace(name, namespace)

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        """Compute the Jacobian matrix.

        To be overloaded by subclasses, actual computation of the Jacobian matrix.

        Args:
            inputs: The inputs wrt the outputs are linearized.
                If None,
                the linearization should be performed wrt all inputs.
            outputs: The outputs to be linearized.
                If None,
                the linearization should be performed on all outputs.
        """

    def _init_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
        with_zeros: bool = False,
        fill_missing_keys: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Initialize the Jacobian dictionary of the form ``{input: {output: matrix}}``.

        Args:
            inputs: The inputs wrt to which the outputs are linearized.
                If None,
                the linearization should be performed wrt all inputs.
            outputs: The outputs to be linearized.
                If None,
                the linearization should be performed on all outputs declared differentiable.
                fill_missing_keys: if True, just fill the missing items
            with_zeros: If True,
                the matrices are set to zero
                otherwise,
                they are empty matrices.
            fill_missing_keys: If True,
                just fill the missing items with zeros/empty
                but do not override the existing data.

        Returns:
            The names of the input variables
            against which to differentiate the output ones,
            and these output variables.
        """
        output_names = self._differentiated_outputs if outputs is None else outputs
        output_values = [self.get_outputs_by_name(name) for name in output_names]
        input_names = self._differentiated_inputs if inputs is None else inputs
        input_values = [self.get_inputs_by_name(name) for name in input_names]
        default_matrix = zeros if with_zeros else empty
        if fill_missing_keys:
            jac = self.jac
            # Only fill the missing sub jacobians
            for output_name, output_value in zip(output_names, output_values):
                jac_loc = jac.get(output_name, defaultdict(None))
                for input_name, input_value in zip(input_names, input_values):
                    sub_jac = jac_loc.get(input_name)
                    if sub_jac is None:
                        jac_loc[input_name] = default_matrix(
                            (len(output_value), len(input_value))
                        )
        else:
            # When a key is not in the default dict, ie a function is not in
            # the Jacobian; return an empty defaultdict(None)
            jac = defaultdict(default_dict_factory)
            if input_names:
                for output_name, output_value in zip(output_names, output_values):
                    jac_loc = jac[output_name]
                    for input_name, input_value in zip(input_names, input_values):
                        jac_loc[input_name] = default_matrix(
                            (len(output_value), len(input_value))
                        )
            self.jac = jac

        return input_names, output_names

    @property
    def linearization_mode(self) -> LinearizationMode:
        """The linearization mode among :attr:`.MDODiscipline.LinearizationMode`.

        Raises:
            ValueError: When the linearization mode is unknown.
        """
        return self._linearization_mode

    @linearization_mode.setter
    def linearization_mode(
        self,
        linearization_mode: LinearizationMode,
    ) -> None:
        self._linearization_mode = linearization_mode

        if (
            linearization_mode in set(self.ApproximationMode)
            and self._jac_approx is None
        ):
            self.set_jacobian_approximation(linearization_mode)

    def check_jacobian(
        self,
        input_data: dict[str, ndarray] | None = None,
        derr_approx: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-7,
        threshold: float = 1e-8,
        linearization_mode: str = "auto",
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
        parallel: bool = False,
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0,
        auto_set_step: bool = False,
        plot_result: bool = False,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10,
        fig_size_y: float = 10,
        reference_jacobian_path: str | Path | None = None,
        save_reference_jacobian: bool = False,
        indices: Iterable[int] | None = None,
    ) -> bool:
        """Check if the analytical Jacobian is correct with respect to a reference one.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `True`,
        compute the reference Jacobian with the approximation method
        and save it in `reference_jacobian_path`.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `False`,
        do not compute the reference Jacobian
        but read it from `reference_jacobian_path`.

        If `reference_jacobian_path` is `None`,
        compute the reference Jacobian without saving it.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.MDODiscipline.default_inputs`.
            derr_approx: The approximation method,
                either "complex_step" or "finite_differences".
            threshold: The acceptance threshold for the Jacobian error.
            linearization_mode: the mode of linearization: direct, adjoint
                or automated switch depending on dimensions
                of inputs and outputs (Default value = 'auto')
            inputs: The names of the inputs wrt which to differentiate the outputs.
            outputs: The names of the outputs to be differentiated.
            step: The differentiation step.
            parallel: Whether to differentiate the discipline in parallel.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            wait_time_between_fork: The time waited between two forks
                of the process / thread.
            auto_set_step: Whether to compute the optimal step
                for a forward first order finite differences gradient approximation.
            plot_result: Whether to plot the result of the validation
                (computed vs approximated Jacobians).
            file_path: The path to the output file if ``plot_result`` is ``True``.
            show: Whether to open the figure.
            fig_size_x: The x-size of the figure in inches.
            fig_size_y: The y-size of the figure in inches.
            reference_jacobian_path: The path of the reference Jacobian file.
            save_reference_jacobian: Whether to save the reference Jacobian.
            indices: The indices of the inputs and outputs
                for the different sub-Jacobian matrices,
                formatted as ``{variable_name: variable_components}``
                where ``variable_components`` can be either
                an integer, e.g. `2`
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0,3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
                If None,
                consider all the components of all the ``inputs`` and ``outputs``.

        Returns:
            Whether the analytical Jacobian is correct
            with respect to the reference one.
        """
        # Do not use self._jac_approx because we may want to check  complex
        # step approximation with the finite differences for instance
        approx = DisciplineJacApprox(
            self,
            derr_approx,
            step,
            parallel,
            n_processes,
            use_threading,
            wait_time_between_fork,
        )
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        if auto_set_step:
            approx.auto_set_step(outputs, inputs)

        # Differentiate analytically
        self.add_differentiated_inputs(inputs)
        self.add_differentiated_outputs(outputs)
        self.linearization_mode = linearization_mode
        self.reset_statuses_for_run()
        # Linearize performs execute() if needed
        self.linearize(input_data)

        return approx.check_jacobian(
            self.jac,
            outputs,
            inputs,
            self,
            threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            fig_size_x=fig_size_x,
            fig_size_y=fig_size_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )

    @property
    def status(self) -> ExecutionStatus:
        """The status of the discipline.

        The status aims at monitoring the process and give the user a simplified view on
        the state (the process state = execution or linearize or done) of the
        disciplines. The core part of the execution is _run, the core part of linearize
        is _compute_jacobian or approximate jacobian computation.
        """
        return self._status

    def set_disciplines_statuses(
        self,
        status: str,
    ) -> None:
        """Set the sub-disciplines statuses.

        To be implemented in subclasses.

        Args:
            status: The status.
        """

    def is_output_existing(
        self,
        data_name: str,
    ) -> bool:
        """Test if a variable is a discipline output.

        Args:
            data_name: The name of the variable.

        Returns:
            Whether the variable is a discipline output.
        """
        return data_name in self.output_grammar

    def is_all_outputs_existing(
        self,
        data_names: Iterable[str],
    ) -> bool:
        """Test if several variables are discipline outputs.

        Args:
            data_names: The names of the variables.

        Returns:
            Whether all the variables are discipline outputs.
        """
        output_names = self.output_grammar.keys()
        for data_name in data_names:
            if data_name not in output_names:
                return False
        return True

    def is_all_inputs_existing(
        self,
        data_names: Iterable[str],
    ) -> bool:
        """Test if several variables are discipline inputs.

        Args:
            data_names: The names of the variables.

        Returns:
            Whether all the variables are discipline inputs.
        """
        input_names = self.input_grammar.keys()
        for data_name in data_names:
            if data_name not in input_names:
                return False
        return True

    def is_input_existing(
        self,
        data_name: str,
    ) -> bool:
        """Test if a variable is a discipline input.

        Args:
            data_name: The name of the variable.

        Returns:
            Whether the variable is a discipline input.
        """
        return data_name in self.input_grammar

    def _is_status_ok_for_run_again(
        self,
        status: str,
    ) -> bool:
        """Check if the discipline can be run again.

        Args:
            status: The status of the discipline.

        Returns:
            Whether the discipline can be run again.
        """
        return status != self.ExecutionStatus.RUNNING

    def reset_statuses_for_run(self) -> None:
        """Set all the statuses to :attr:`.MDODiscipline.ExecutionStatus.PENDING`.

        Raises:
            ValueError: When the discipline cannot be run because of its status.
        """
        if not self._is_status_ok_for_run_again(self.status):
            raise ValueError(
                "Cannot run discipline {} with status {}.".format(
                    self.name, self.status
                )
            )
        self.status = self.ExecutionStatus.PENDING

    @status.setter
    def status(
        self,
        status: ExecutionStatus,
    ) -> None:
        self._status = status
        self.notify_status_observers()

    def add_status_observer(
        self,
        obs: Any,
    ) -> None:
        """Add an observer for the status.

        Add an observer for the status
        to be notified when self changes of status.

        Args:
            obs: The observer to add.
        """
        if obs not in self._status_observers:
            self._status_observers.append(obs)

    def remove_status_observer(
        self,
        obs: Any,
    ) -> None:
        """Remove an observer for the status.

        Args:
            obs: The observer to remove.
        """
        if obs in self._status_observers:
            self._status_observers.remove(obs)

    def notify_status_observers(self) -> None:
        """Notify all status observers that the status has changed."""
        for obs in self._status_observers[:]:
            obs.update_status(self)

    def store_local_data(self, **kwargs: Any) -> None:
        """Store discipline data in local data.

        Args:
            **kwargs: The data to be stored in :attr:`.MDODiscipline.local_data`.
        """
        out_ns = self.output_grammar.to_namespaced
        if not out_ns:
            self._local_data.update(kwargs)
        else:
            out_names = self.output_grammar.keys()
            for key, value in kwargs.items():
                if key in out_names:
                    self._local_data[key] = value
                else:
                    key_with_ns = out_ns.get(key)
                    if key_with_ns is not None:
                        self._local_data[key_with_ns] = value
                    # else out data will be cleared

    def check_input_data(
        self,
        input_data: dict[str, Any],
        raise_exception: bool = True,
    ) -> None:
        """Check the input data validity.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
            raise_exception: Whether to raise on error.
        """
        try:
            self.input_grammar.validate(input_data, raise_exception)
        except InvalidDataError as err:
            err.args = (
                err.args[0].replace("Invalid data", "Invalid input data")
                + f" in discipline {self.name}",
            )
            raise

    def check_output_data(
        self,
        raise_exception: bool = True,
    ) -> None:
        """Check the output data validity.

        Args:
            raise_exception: Whether to raise an exception when the data is invalid.
        """
        try:
            self.output_grammar.validate(self._local_data, raise_exception)
        except InvalidDataError as err:
            err.args = (
                err.args[0].replace("Invalid data", "Invalid output data")
                + f" in discipline {self.name}",
            )
            raise

    def get_outputs_asarray(self) -> ndarray:
        """Return the local input data as a large NumPy array.

        The order is the one of :meth:`.MDODiscipline.get_all_inputs`.

        Returns:
            The local input data.
        """
        return concatenate(list(self.get_all_outputs()))

    def get_inputs_asarray(self) -> ndarray:
        """Return the local output data as a large NumPy array.

        The order is the one of :meth:`.MDODiscipline.get_all_outputs`.

        Returns:
            The local output data.
        """
        return concatenate(list(self.get_all_inputs()))

    def get_inputs_by_name(
        self,
        data_names: Iterable[str],
    ) -> list[Any]:
        """Return the local data associated with input variables.

        Args:
            data_names: The names of the input variables.

        Returns:
            The local data for the given input variables.

        Raises:
            ValueError: When a variable is not an input of the discipline.
        """
        try:
            return self.get_data_list_from_dict(data_names, self._local_data)
        except KeyError as err:
            raise ValueError(f"Discipline {self.name} has no input named {err}.")

    def get_outputs_by_name(
        self,
        data_names: Iterable[str],
    ) -> list[Any]:
        """Return the local data associated with output variables.

        Args:
            data_names: The names of the output variables.

        Returns:
            The local data for the given output variables.

        Raises:
            ValueError: When a variable is not an output of the discipline.
        """
        try:
            return self.get_data_list_from_dict(data_names, self._local_data)
        except KeyError as err:
            raise ValueError(f"Discipline {self.name} has no output named {err}.")

    def get_input_data_names(self, with_namespaces: bool = True) -> list[str]:
        """Return the names of the input variables.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The names of the input variables.
        """
        if with_namespaces:
            return list(self.input_grammar.keys())
        else:
            return remove_prefix_from_list(self.input_grammar.keys())

    def get_output_data_names(self, with_namespaces: bool = True) -> list[str]:
        """Return the names of the output variables.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The names of the output variables.
        """
        if with_namespaces:
            return list(self.output_grammar.keys())
        else:
            return remove_prefix_from_list(self.output_grammar.keys())

    def get_input_output_data_names(self, with_namespaces: bool = True) -> list[str]:
        """Return the names of the input and output variables.

         Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The name of the input and output variables.
        """
        in_outs = self.output_grammar.keys() | self.input_grammar.keys()
        if with_namespaces:
            return list(in_outs)
        else:
            return remove_prefix_from_list(in_outs)

    def get_all_inputs(self) -> list[Any]:
        """Return the local input data as a list.

        The order is given by :meth:`.MDODiscipline.get_input_data_names`.

        Returns:
            The local input data.
        """
        return self.get_inputs_by_name(self.get_input_data_names())

    def get_all_outputs(self) -> list[Any]:
        """Return the local output data as a list.

        The order is given by :meth:`.MDODiscipline.get_output_data_names`.

        Returns:
            The local output data.
        """
        return self.get_outputs_by_name(self.get_output_data_names())

    def get_output_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the local output data as a dictionary.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The local output data.
        """
        return self._local_data.copy(
            keys=self.output_grammar.keys(),
            with_namespace=with_namespaces or not self.output_grammar.to_namespaced,
        )

    def get_input_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the local input data as a dictionary.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The local input data.
        """
        return self._local_data.copy(
            keys=self.input_grammar.keys(),
            with_namespace=with_namespaces or not self.input_grammar.to_namespaced,
        )

    def to_pickle(self, file_path: str | Path) -> None:
        """Serialize the discipline and store it in a file.

        Args:
            file_path: The path to the file to store the discipline.
        """
        with Path(file_path).open("wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def from_pickle(file_path: str | Path) -> MDODiscipline:
        """Deserialize a discipline from a file.

        Args:
            file_path: The path to the file containing the discipline.

        Returns:
            The discipline instance.
        """
        with Path(file_path).open("rb") as file_:
            pickler = pickle.Unpickler(file_)
            obj = pickler.load()
        return obj

    def __setstate__(
        self,
        state: Mapping[str, Any],
    ) -> None:
        super().__setstate__(state)
        # Initialize the attributes that are not serializable nor Synchronized last.
        self._status_observers = []

    def get_local_data_by_name(
        self,
        data_names: Iterable[str],
    ) -> Generator[Any]:
        """Return the local data of the discipline associated with variables names.

        Args:
            data_names: The names of the variables.

        Returns:
            The local data associated with the variables names.

        Raises:
            ValueError: When a name is not a discipline input name.
        """
        try:
            return self.get_data_list_from_dict(data_names, self._local_data)
        except KeyError as err:
            raise ValueError(f"Discipline {self.name} has no local_data named {err}.")

    @staticmethod
    def is_scenario() -> bool:
        """Whether the discipline is a scenario."""
        return False

    @staticmethod
    def get_data_list_from_dict(
        keys: str | Iterable,
        data_dict: dict[str, Any],
    ) -> Any | Generator[Any]:
        """Filter the dict from a list of keys or a single key.

        If keys is a string, then the method return the value associated to the
        key.
        If keys is a list of strings, then the method returns a generator of
        value corresponding to the keys which can be iterated.

        Args:
            keys: One or several names.
            data_dict: The mapping from which to get the data.

        Returns:
            Either a data or a generator of data.
        """
        if isinstance(keys, str):
            return data_dict[keys]
        return (data_dict[name] for name in keys)
