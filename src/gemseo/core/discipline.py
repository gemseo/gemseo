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
from multiprocessing.sharedctypes import Synchronized
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

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import concatenate
from numpy import empty
from numpy import ndarray
from numpy import zeros

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.cache import AbstractCache
from gemseo.core.data_processor import DataProcessor
from gemseo.core.derivatives import derivation_modes
from gemseo.core.discipline_data import DisciplineData
from gemseo.core.discipline_data import MutableData
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.namespaces import remove_prefix_from_dict
from gemseo.core.namespaces import remove_prefix_from_list
from gemseo.disciplines.utils import get_sub_disciplines
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.utils.derivatives.derivatives_approx import EPSILON
from gemseo.utils.multiprocessing import get_multi_processing_manager
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.core.execution_sequence import SerialExecSequence

LOGGER = logging.getLogger(__name__)


def default_dict_factory() -> dict:
    """Instantiate a defaultdict(None) object."""
    return defaultdict(None)


class MDODiscipline(metaclass=GoogleDocstringInheritanceMeta):
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

    input_grammar: BaseGrammar
    """The input grammar."""

    output_grammar: BaseGrammar
    """The output grammar."""

    data_processor: DataProcessor
    """A tool to pre- and post-process discipline data."""

    re_exec_policy: str
    """The policy to re-execute the same discipline."""

    residual_variables: Mapping[str, str]
    """The output variables mapping to their inputs,
    to be considered as residuals; they shall be equal to zero.
    """

    run_solves_residuals: bool
    """If True, the run method shall solve the residuals."""

    jac: dict[str, dict[str, ndarray]]
    """The Jacobians of the outputs wrt inputs
    of the form ``{output: {input: matrix}}``."""

    exec_for_lin: bool
    """Whether the last execution was due to a linearization."""

    name: str
    """The name of the discipline."""

    cache: AbstractCache | None
    """The cache containing one or several executions of the discipline
    according to the cache policy."""

    STATUS_VIRTUAL = "VIRTUAL"
    STATUS_PENDING = "PENDING"
    STATUS_DONE = "DONE"
    STATUS_RUNNING = "RUNNING"
    STATUS_FAILED = "FAILED"
    AVAILABLE_STATUSES = [
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_VIRTUAL,
    ]

    JSON_GRAMMAR_TYPE = "JSONGrammar"
    SIMPLE_GRAMMAR_TYPE = "SimpleGrammar"

    GRAMMAR_DIRECTORY: ClassVar[str | None] = None
    """The directory in which to search for the grammar files if not the class one."""

    COMPLEX_STEP = derivation_modes.COMPLEX_STEP
    FINITE_DIFFERENCES = derivation_modes.FINITE_DIFFERENCES

    SIMPLE_CACHE = "SimpleCache"
    HDF5_CACHE = "HDF5Cache"
    MEMORY_FULL_CACHE = "MemoryFullCache"

    activate_cache: bool = True
    """Whether to cache the discipline evaluations by default."""

    APPROX_MODES = [FINITE_DIFFERENCES, COMPLEX_STEP]
    AVAILABLE_MODES = (
        derivation_modes.AUTO_MODE,
        derivation_modes.DIRECT_MODE,
        derivation_modes.ADJOINT_MODE,
        derivation_modes.REVERSE_MODE,
        derivation_modes.FINITE_DIFFERENCES,
        derivation_modes.COMPLEX_STEP,
    )

    RE_EXECUTE_DONE_POLICY = "RE_EXEC_DONE"
    RE_EXECUTE_NEVER_POLICY = "RE_EXEC_NEVER"
    N_CPUS = cpu_count()

    activate_counters: ClassVar[bool] = True
    """Whether to activate the counters (execution time, calls and linearizations)."""

    activate_input_data_check: ClassVar[bool] = True
    """Whether to check the input data respect the input grammar."""

    activate_output_data_check: ClassVar[bool] = True
    """Whether to check the output data respect the output grammar."""

    _ATTR_TO_SERIALIZE = (
        "_cache_was_loaded",
        "_default_inputs",
        "_differentiated_inputs",
        "_differentiated_outputs",
        "_in_data_hash_dict",
        "_is_linearized",
        "_jac_approx",
        "_linearization_mode",
        "_linearize_on_last_state",
        "_grammar_type",
        "_local_data",
        "_status",
        "cache",
        "data_processor",
        "_disciplines",
        "exec_for_lin",
        "exec_time",
        "input_grammar",
        "jac",
        "n_calls",
        "n_calls_linearize",
        "name",
        "output_grammar",
        "re_exec_policy",
        "residual_variables",
        "run_solves_residuals",
    )

    __mp_manager: Manager = None
    time_stamps = None

    def __init__(
        self,
        name: str | None = None,
        input_grammar_file: str | Path | None = None,
        output_grammar_file: str | Path | None = None,
        auto_detect_grammar_files: bool = False,
        grammar_type: str = JSON_GRAMMAR_TYPE,
        cache_type: str | None = SIMPLE_CACHE,
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
            grammar_type: The type of grammar to define the input and output variables,
                e.g. :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE`
                or :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.
            cache_type: The type of policy to cache the discipline evaluations,
                e.g. :attr:`.MDODiscipline.SIMPLE_CACHE` to cache the last one,
                :attr:`.MDODiscipline.HDF5_CACHE` to cache them in an HDF file,
                or :attr:`.MDODiscipline.MEMORY_FULL_CACHE` to cache them in memory.
                If ``None`` or if :attr:`.activate_cache` is ``True``,
                do not cache the discipline evaluations.
            cache_file_path: The HDF file path
                when ``grammar_type`` is :attr:`.MDODiscipline.HDF5_CACHE`.
        """  # noqa: D205, D212, D415
        self.data_processor = None
        self._default_inputs = None
        self.input_grammar = None
        self.output_grammar = None
        self.__set_default_inputs({})
        # Allow to re-execute the same discipline twice, only if did not fail
        # and not running
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        # : list of outputs that shall be null, to be considered as residuals
        self.residual_variables = {}

        self._disciplines = []

        self.run_solves_residuals = False

        self._differentiated_inputs = []  # : outputs to differentiate
        # : inputs to be used for differentiation
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
            cache_type = None

        if cache_type is not None:
            self.cache = self.__create_new_cache(
                cache_type, hdf_file_path=cache_file_path, hdf_node_path=self.name
            )

        self._cache_was_loaded = False

        # linearize mode :auto, adjoint, direct
        self._linearization_mode = derivation_modes.AUTO_MODE

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

        # : The current status of execution
        self._status = self.STATUS_PENDING
        if self.activate_counters:
            self._init_shared_attrs()

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

    def _init_shared_attrs(self) -> None:
        """Initialize the shared attributes in multiprocessing."""
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
    def grammar_type(self) -> BaseGrammar:
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

        If the discipline grammar type is :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE` and
        an input is either a non-numeric array or not an array, it will be ignored.
        If an input is declared as an array but the type of its items is not defined, it
        is assumed as a numeric array.

        If the discipline grammar type is :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE` and
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

        If the discipline grammar type is :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE` and
        an output is either a non-numeric array or not an array, it will be ignored.
        If an output is declared as an array but the type of its items is not defined,
        it is assumed as a numeric array.

        If the discipline grammar type is :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE` and
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
        if class_name != self.HDF5_CACHE:
            for key in ("hdf_file_path", "hdf_node_path"):
                if key in kwargs:
                    del kwargs[key]

        if class_name != self.MEMORY_FULL_CACHE:
            key = "is_memory_shared"
            if key in kwargs:
                del kwargs[key]

        return CacheFactory().create(class_name, name=self.name, **kwargs)

    def set_cache_policy(
        self,
        cache_type: str = SIMPLE_CACHE,
        cache_tolerance: float = 0.0,
        cache_hdf_file: str | Path | None = None,
        cache_hdf_node_name: str | None = None,
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
                :attr:`.MDODiscipline.HDF5_CACHE` policy is used.
            cache_hdf_node_name: The name of the HDF file node
                to store the discipline data.
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
            cache_type == self.HDF5_CACHE
            and cache_hdf_file == self.cache.hdf_file.hdf_file_path
            and cache_hdf_node_name == self.cache.node_path
        ):
            self.cache = self.__create_new_cache(
                cache_type,
                tolerance=cache_tolerance,
                hdf_file_path=cache_hdf_file,
                hdf_node_path=cache_hdf_node_name or self.name,
                is_memory_shared=is_memory_shared,
            )
        else:
            LOGGER.warning(
                "Cache policy is set to %s: call clear() to clear a discipline cache",
                cache_type,
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
        grammar_type: str = JSON_GRAMMAR_TYPE,
    ) -> None:
        """Create the input and output grammars.

        Args:
            input_grammar_file: The input grammar file path.
                If None, do not initialize the input grammar from a schema file.
            output_grammar_file: The output grammar file path.
                If None, do not initialize the output grammar from a schema file.
            grammar_type: The type of grammar,
                e.g. :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE`
                or :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.
        """
        factory = GrammarFactory()
        self.input_grammar = factory.create(
            grammar_type,
            name=f"{self.name}_input",
            schema_path=input_grammar_file,
        )
        self.output_grammar = factory.create(
            grammar_type,
            name=f"{self.name}_output",
            schema_path=output_grammar_file,
        )

    def _run(self) -> None:
        """Define the execution of the process, given that data has been checked.

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def _filter_inputs(
        self,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        for key in self.input_grammar.keys():
            val = input_data.get(key)
            if val is not None:
                full_input_data[key] = val
            else:
                val = self._default_inputs.get(key)
                if val is not None:
                    full_input_data[key] = val

        return full_input_data

    def _filter_local_data(self) -> None:
        """Filter the local data after execution.

        This method removes data that are neither inputs nor outputs.
        """
        all_data_names = self.get_input_output_data_names()

        for key in tuple(self._local_data.keys()):
            if key not in all_data_names:
                del self._local_data[key]

    def _check_status_before_run(self) -> None:
        """Check the status of the discipline.

        Check the status of the discipline depending on
        :attr:`.MDODiscipline.re_execute_policy`.

        If ``re_exec_policy == RE_EXECUTE_NEVER_POLICY``,
        the status shall be either :attr:`.MDODiscipline.STATUS_PENDING`
        or :attr:`.MDODiscipline.VIRTUAL`.

        If ``self.re_exec_policy == RE_EXECUTE_NEVER_POLICY``,

        - if status is :attr:`.MDODiscipline.STATUS_DONE`,
          :meth:`.MDODiscipline.reset_statuses_for_run`.
        - otherwise status must be :attr:`.MDODiscipline.VIRTUAL`
          or :attr:`.MDODiscipline.STATUS_PENDING`.

        Raises:
            ValueError:
                When the re-execution policy is unknown.
                When the discipline status and the re-execution policy
                are no consistent.
        """
        status_ok = True
        if self.status == self.STATUS_RUNNING:
            status_ok = False
        if self.re_exec_policy == self.RE_EXECUTE_NEVER_POLICY:
            if self.status not in [self.STATUS_PENDING, self.STATUS_VIRTUAL]:
                status_ok = False
        elif self.re_exec_policy == self.RE_EXECUTE_DONE_POLICY:
            if self.status == self.STATUS_DONE:
                self.reset_statuses_for_run()
                status_ok = True
            elif self.status not in [self.STATUS_PENDING, self.STATUS_VIRTUAL]:
                status_ok = False
        else:
            raise ValueError(f"Unknown re_exec_policy: {self.re_exec_policy}.")
        if not status_ok:
            raise ValueError(
                "Trying to run a discipline {} with status: {} "
                "while re_exec_policy is {}.".format(
                    type(self), self.status, self.re_exec_policy
                )
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
        * Updates the status to :attr:`.MDODiscipline.STATUS_RUNNING`.
        * Calls the :meth:`.MDODiscipline._run` method, that shall be defined.
        * If :attr:`.MDODiscipline.data_processor` is not None, runs the postprocessor.
        * Checks the output data.
        * Caches the outputs.
        * Updates the status to :attr:`.MDODiscipline.STATUS_DONE`
          or :attr:`.MDODiscipline.STATUS_FAILED`.
        * Updates summed execution time.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.MDODiscipline.default_inputs`.

        Returns:
            The discipline local data after execution.

        Raises:
            RuntimeError: When residual_variables are declared but
                self.run_solves_residuals is False. This is not suported yet.
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

        self._check_status_before_run()

        if self.activate_input_data_check:
            self.check_input_data(input_data)

        processor = self.data_processor
        if processor is not None:
            self.__set_local_data(processor.pre_process_data(input_data))
        else:
            self.__set_local_data(input_data)

        self.status = self.STATUS_RUNNING
        self._is_linearized = False
        if self.activate_counters:
            self.__increment_n_calls()

        t_0 = timer()

        try:
            # Effectively run the discipline, the _run method has to be
            # Defined by the subclasses
            self._run()
        except Exception:
            self.status = self.STATUS_FAILED
            # Update the status but
            # raise the same exception
            raise

        if self.activate_counters:
            self.__increment_exec_time(t_0)

        self.status = self.STATUS_DONE

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
            self.cache.cache_outputs(
                cached_inputs, {name: self._local_data[name] for name in out_names}
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
        force_all: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Get the inputs and outputs used in the differentiation of the discipline.

        Args:
            force_all: If True,
                consider all the inputs and outputs of the discipline;
                otherwise,
                consider :attr:`.MDODiscipline._differentiated_inputs`
                and :attr:`.MDODiscipline._differentiated_outputs`.
        """
        if force_all:
            inputs = self.get_input_data_names()
            outputs = self.get_output_data_names()
        else:
            inputs = self._differentiated_inputs
            outputs = self._differentiated_outputs
        return inputs, outputs

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
        input_data: dict[str, Any] | None = None,
        force_all: bool = False,
        force_no_exec: bool = False,
    ) -> dict[str, dict[str, ndarray]]:
        """Execute the linearized version of the code.

        Args:
            input_data: The input data needed to linearize the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.MDODiscipline.default_inputs`.
            force_all: If False,
                :attr:`.MDODiscipline._differentiated_inputs` and
                :attr:`.MDODiscipline._differentiated_outputs`
                are used to filter the differentiated variables.
                otherwise, all outputs are differentiated wrt all inputs.
            force_no_exec: If True,
                the discipline is not re-executed, cache is loaded anyway.

        Returns:
            The Jacobian of the discipline.
        """
        # TODO: remove the execution when no option exec_before_lin
        # is set to True
        inputs, outputs = self._retrieve_diff_inouts(force_all)
        if not outputs:
            self.jac.clear()
            return self.jac
        # Save inputs dict for caching
        input_data = self._filter_inputs(input_data)

        # if force_no_exec, we do not re-execute the discipline
        # otherwise, we ensure that the discipline was executed
        # with the right input_data.
        # This may trigger caching (see self.execute()
        if not force_no_exec:
            self.reset_statuses_for_run()
            self.exec_for_lin = True
            self.execute(input_data)
            self.exec_for_lin = False

        # The local_data shall be reset to their original values in case
        # an input is also an output, if we don't want to keep the computed
        # state (as in MDAs)
        if not self._linearize_on_last_state:
            self._local_data.update(input_data)

        # If the caching was triggered, check if the jacobian
        # was loaded,
        # Or the discipline._run method also linearizes the discipline
        if self._cache_was_loaded or self._is_linearized:
            if self.jac:
                # for cases when linearization is called
                # twice with different i/o
                # while cache_was_loaded=True, the check_jacobian_shape raises
                # a KeyError.
                try:
                    self._check_jacobian_shape(inputs, outputs)
                    return self.jac
                except KeyError:
                    # in this case, another computation of jacobian is
                    # triggered.
                    pass

        t_0 = timer()
        if self._linearization_mode in self.APPROX_MODES:
            # Time already counted in execute()
            self.jac = self._jac_approx.compute_approx_jac(outputs, inputs)
        else:
            self._compute_jacobian(inputs, outputs)
            if self.activate_counters:
                self.__increment_exec_time(t_0, linearize=True)

        if self.activate_counters:
            self.__increment_n_calls_lin()

        self._check_jacobian_shape(inputs, outputs)

        if self.cache is not None:
            # Cache the Jacobian matrix
            self.cache.cache_jacobian(input_data, self.jac)

        return self.jac

    def set_jacobian_approximation(
        self,
        jac_approx_type: str = FINITE_DIFFERENCES,
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
        force_all: bool = False,
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
            force_all: Whether to consider all the inputs and outputs of the discipline;
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
        inpts, outps = self._retrieve_diff_inouts(force_all=force_all)
        if outputs is None or force_all:
            outputs = outps
        if inputs is None or force_all:
            inputs = inpts
        errors, steps = self._jac_approx.auto_set_step(
            outputs, inputs, print_errors, numerical_error=numerical_error
        )
        return errors, steps

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
        out_jac_set = set(self.jac.keys())

        if not out_set.issubset(out_jac_set):
            msg = "Missing outputs in Jacobian of discipline {}: {}."
            missing_outputs = out_set.difference(out_jac_set)
            raise KeyError(msg.format(self.name, missing_outputs))

        for j_o in outputs:
            j_out = self.jac[j_o]
            out_dv_set = set(j_out.keys())
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
                    msg = (
                        "Jacobian matrix of discipline {} d{}/d{}"
                        "is not of the right shape.\n "
                        "Expected: ({},{}), got: {}"
                    )
                    data = [self.name, j_o, j_i, n_out_j, n_in_j, j_mat.shape]
                    raise ValueError(msg.format(*data))

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
    def default_inputs(self) -> dict[str, Any]:
        """The default inputs.

        Raises:
            TypeError: When the default inputs are not passed as a dictionary.
        """
        return self._default_inputs

    @default_inputs.setter
    def default_inputs(self, default_inputs: dict[str, Any]) -> None:
        if not isinstance(default_inputs, collections.abc.Mapping):
            raise TypeError(
                "MDODiscipline default_inputs must be of dict-like type, "
                "got {} instead.".format(type(default_inputs))
            )
        self.__set_default_inputs(default_inputs, with_namespace=True)

    def __set_default_inputs(self, data: MutableData, with_namespace=False) -> None:
        """Set the default inputs.

        Args:
            data: The mapping containing default values.
            with_namespace: Whether to add the input grammar namespaces prefixes
                to the keys of the default inputs.
        """
        if with_namespace:
            to_ns = self.input_grammar.to_namespaced
            in_names = self.input_grammar.keys()
            data = {
                (to_ns[k] if (k not in in_names and k in to_ns) else k): v
                for k, v in data.items()
            }
        if self.input_grammar is not None:
            disc_data = DisciplineData(
                data, input_to_namespaced=self.input_grammar.to_namespaced
            )
        else:
            disc_data = DisciplineData(data)
        self._default_inputs = disc_data

    def add_namespace_to_input(self, name: str, namespace: str):
        """Add a namespace prefix to an existing input grammar element.

        The updated input grammar element name will be
        ``namespace`` + :data:`~gemseo.core.namespaces.namespaces_separator` + ``name``.

        Args:
            name: The element name to rename.
            namespace: The name of the namespace.
        """
        self.input_grammar.add_namespace(name, namespace)
        default_value = self.default_inputs.get(name)
        if default_value is not None:
            del self.default_inputs[name]
            self.default_inputs[self.input_grammar.to_namespaced[name]] = default_value

    def add_namespace_to_output(self, name: str, namespace: str):
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
    ) -> None:
        """Initialize the Jacobian dictionary of the form ``{input: {output: matrix}}``.

        Args:
            inputs: The inputs wrt the outputs are linearized.
                If None,
                the linearization should be performed wrt all inputs.
            outputs: The outputs to be linearized.
                If None,
                the linearization should be performed on all outputs.
                fill_missing_keys: if True, just fill the missing items
            with_zeros: If True,
                the matrices are set to zero
                otherwise,
                they are empty matrices.
            fill_missing_keys: If True,
                just fill the missing items with zeros/empty
                but do not override the existing data.
        """
        if inputs is None:
            inputs_names = self._differentiated_inputs
        else:
            inputs_names = inputs
        inputs_vals = []
        for diff_name in inputs_names:
            inputs_vals.append(self.get_inputs_by_name(diff_name))

        if outputs is None:
            outputs_names = self._differentiated_outputs
        else:
            outputs_names = outputs
        outputs_vals = []
        for diff_name in outputs_names:
            outputs_vals.append(self.get_outputs_by_name(diff_name))
        if with_zeros:
            np_method = zeros
        else:
            np_method = empty
        if not fill_missing_keys:
            # When a key is not in the default dict, ie a function is not in
            # the Jacobian; return an empty defaultdict(None)
            jac = defaultdict(default_dict_factory)
            for out_n, out_v in zip(outputs_names, outputs_vals):
                jac_loc = jac[out_n]
                for in_n, in_v in zip(inputs_names, inputs_vals):
                    jac_loc[in_n] = np_method((len(out_v), len(in_v)))
            self.jac = jac
        else:
            jac = self.jac
            # Only fill the missing sub jacobians
            for out_n, out_v in zip(outputs_names, outputs_vals):
                jac_loc = jac.get(out_n, defaultdict(None))
                for in_n, in_v in zip(inputs_names, inputs_vals):
                    sub_jac = jac_loc.get(in_n)
                    if sub_jac is None:
                        jac_loc[in_n] = np_method((len(out_v), len(in_v)))

    @property
    def linearization_mode(self) -> str:
        """The linearization mode among :attr:`.MDODiscipline.AVAILABLE_MODES`.

        Raises:
            ValueError: When the linearization mode is unknown.
        """
        return self._linearization_mode

    @linearization_mode.setter
    def linearization_mode(
        self,
        linearization_mode: str,
    ) -> None:
        if linearization_mode not in self.AVAILABLE_MODES:
            msg = "Linearization mode '{}' is unknown; it must be one of {}."
            raise ValueError(msg.format(linearization_mode, self.AVAILABLE_MODES))

        self._linearization_mode = linearization_mode

        if linearization_mode in self.APPROX_MODES and self._jac_approx is None:
            self.set_jacobian_approximation(linearization_mode)

    def check_jacobian(
        self,
        input_data: dict[str, ndarray] | None = None,
        derr_approx: str = FINITE_DIFFERENCES,
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
    ):
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
    def status(self) -> str:
        """The status of the discipline."""
        return self._status

    def _check_status(
        self,
        status: str,
    ) -> None:
        """Check the status according to possible statuses.

        Raises:
            ValueError: When the status is unknown.
        """
        if status not in [
            self.STATUS_PENDING,
            self.STATUS_VIRTUAL,
            self.STATUS_DONE,
            self.STATUS_RUNNING,
            self.STATUS_FAILED,
        ]:
            raise ValueError(f"Unknown status: {status}.")

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
        return status not in [self.STATUS_RUNNING]

    def reset_statuses_for_run(self) -> None:
        """Set all the statuses to :attr:`.MDODiscipline.STATUS_PENDING`.

        Raises:
            ValueError: When the discipline cannot be run because of its status.
        """
        if not self._is_status_ok_for_run_again(self.status):
            raise ValueError(
                "Cannot run discipline {} with status {}.".format(
                    self.name, self.status
                )
            )
        self.status = self.STATUS_PENDING

    @status.setter
    def status(
        self,
        status: str,
    ) -> None:
        self._check_status(status)
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
        except InvalidDataException as err:
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
        except InvalidDataException as err:
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

    def get_input_data_names(self, with_namespaces=True) -> list[str]:
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

    def get_output_data_names(self, with_namespaces=True) -> list[str]:
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

    def get_input_output_data_names(self, with_namespaces=True) -> list[str]:
        """Return the names of the input and output variables.

         Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The name of the input and output variables.
        """
        in_outs = set(self.output_grammar.keys()).union(self.input_grammar.keys())
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

    def get_output_data(self, with_namespaces=True) -> dict[str, Any]:
        """Return the local output data as a dictionary.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The local output data.
        """
        if with_namespaces or not self.output_grammar.to_namespaced:
            output_names = self.output_grammar.keys()
            return {k: v for k, v in self._local_data.items() if k in output_names}
        else:
            return remove_prefix_from_dict(self.get_output_data())

    def get_input_data(self, with_namespaces=True) -> dict[str, Any]:
        """Return the local input data as a dictionary.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The local input data.
        """
        if with_namespaces or not self.input_grammar.to_namespaced:
            input_names = self.input_grammar.keys()
            return {k: v for k, v in self._local_data.items() if k in input_names}
        else:
            return remove_prefix_from_dict(self.get_input_data())

    def serialize(
        self,
        file_path: str | Path,
    ) -> None:
        """Serialize the discipline and store it in a file.

        Args:
            file_path: The path to the file to store the discipline.
        """
        with Path(file_path).open("wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def deserialize(
        file_path: str | Path,
    ) -> MDODiscipline:
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

    def get_attributes_to_serialize(self) -> list[str]:  # pylint: disable=R0201
        """Define the names of the attributes to be serialized.

        Shall be overloaded by disciplines

        Returns:
            The names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overridden,
        # it is a function==> self is required
        return list(self._ATTR_TO_SERIALIZE)

    def __getstate__(self) -> dict[str, Any]:
        """Used by pickle to define what to serialize.

        Returns:
            The attributes to be serialized.

        Raises:
            AttributeError: When an attribute to be serialized is undefined.
            TypeError: When an attribute has an undefined type.
        """
        state = {}
        for attribute_name in self.get_attributes_to_serialize():
            if attribute_name not in self.__dict__:
                if "_" + attribute_name not in self.__dict__ and not hasattr(
                    self, attribute_name
                ):
                    msg = (
                        "The discipline {} cannot be serialized "
                        "because its attribute {} does not exist."
                    ).format(self.name, attribute_name)
                    raise AttributeError(msg)

                prop = self.__dict__.get("_" + attribute_name)
                # May appear for properties that overload public attrs of super class
                if hasattr(self, attribute_name) and not isinstance(prop, Synchronized):
                    continue

                if not isinstance(prop, Synchronized):
                    raise TypeError(
                        "The discipline {} cannot be serialized "
                        "because its attribute {} has an undefined type.".format(
                            self.name, attribute_name
                        )
                    )
                # Don´t serialize shared memory object,
                # this is meaningless, save the value instead
                state[attribute_name] = prop.value
            else:
                state[attribute_name] = self.__dict__[attribute_name]

        return state

    def __setstate__(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._init_shared_attrs()
        self._status_observers = []
        __dict__ = self.__dict__
        for key, val in state.items():
            _key = f"_{key}"
            if _key not in __dict__.keys():
                __dict__[key] = val
            else:
                __dict__[_key].value = val

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
