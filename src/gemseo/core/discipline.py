# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import inspect
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Manager, Value, cpu_count
from multiprocessing.sharedctypes import Synchronized
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import six
from custom_inherit import DocInheritMeta
from numpy import concatenate, empty, ndarray, zeros

if TYPE_CHECKING:
    from gemseo.core.execution_sequence import SerialExecSequence

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.grammar import AbstractGrammar, InvalidDataException
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.utils.derivatives_approx import EPSILON, DisciplineJacApprox
from gemseo.utils.py23_compat import Path
from gemseo.utils.string_tools import MultiLineString, pretty_repr

# TODO: remove try except when py2 is no longer supported
try:
    import cPickle as pickle  # noqa: N813
except ImportError:
    import pickle


LOGGER = logging.getLogger(__name__)


def default_dict_factory():  # type: (...) -> Dict
    """Instantiate a defaultdict(None) object."""
    return defaultdict(None)


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class MDODiscipline(object):
    """A software integrated in the workflow.

    The inputs and outputs are defined in a grammar, which can be
    either a SimpleGrammar or a JSONGrammar, or your own which
    derives from the Grammar abstract class.

    To be used, use a subclass and implement the _run method
    which defined the execution of the software.
    Typically, in the _run method, get the inputs from the
    input grammar, call your software, and write the outputs
    to the output grammar.

    The JSONGrammar files are automatically detected when in the same
    folder as your subclass module and named "CLASSNAME_input.json"
    use ``auto_detect_grammar_files=True`` to activate this option.

    Attributes:
        input_grammar (AbstractGrammar): The input grammar.
        output_grammar (AbstractGrammar): The output grammar.
        grammar_type (str): The type of grammar
            to be used for inputs and outputs declaration.
        comp_dir (str): The path to the directory of the discipline module file if any.
        data_processor (DataProcessor): A tool to pre- and post-process discipline data.
        re_exec_policy (str): The policy to re-execute the same discipline.
        residual_variables (List[str]): The output variables
            to be considered as residuals; they shall be equal to zero.
        jac (Dict[str, Dict[str, ndarray]]): The Jacobians of the outputs wrt inputs
            of the form ``{output: {input: matrix}}``.
        exec_for_lin (bool): Whether the last execution was due to a linearization.
        name (str): The name of the discipline.
        cache (AbstractCache): The cache
            containing one or several executions of the discipline
            according to the cache policy.
        local_data (Dict[str, Any]): The last input and output data.
    """

    STATUS_VIRTUAL = "VIRTUAL"
    STATUS_PENDING = "PENDING"
    STATUS_DONE = "DONE"
    STATUS_RUNNING = "RUNNING"
    STATUS_FAILED = "FAILED"

    __DEPRECATED_GRAMMAR_TYPES = {"JSON": "JSONGrammar", "Simple": "SimpleGrammar"}
    JSON_GRAMMAR_TYPE = "JSONGrammar"
    SIMPLE_GRAMMAR_TYPE = "SimpleGrammar"

    COMPLEX_STEP = "complex_step"
    FINITE_DIFFERENCES = "finite_differences"

    SIMPLE_CACHE = "SimpleCache"
    HDF5_CACHE = "HDF5Cache"
    MEMORY_FULL_CACHE = "MemoryFullCache"

    APPROX_MODES = [FINITE_DIFFERENCES, COMPLEX_STEP]
    AVAILABLE_MODES = (
        JacobianAssembly.AUTO_MODE,
        JacobianAssembly.DIRECT_MODE,
        JacobianAssembly.ADJOINT_MODE,
        JacobianAssembly.REVERSE_MODE,
        FINITE_DIFFERENCES,
        COMPLEX_STEP,
    )

    RE_EXECUTE_DONE_POLICY = "RE_EXEC_DONE"
    RE_EXECUTE_NEVER_POLICY = "RE_EXEC_NEVER"
    N_CPUS = cpu_count()

    _ATTR_TO_SERIALIZE = (
        "residual_variables",
        "output_grammar",
        "name",
        "local_data",
        "jac",
        "input_grammar",
        "_status",
        "cache",
        "n_calls",
        "n_calls_linearize",
        "_differentiated_inputs",
        "_differentiated_outputs",
        "data_processor",
        "_is_linearized",
        "_linearization_mode",
        "_default_inputs",
        "re_exec_policy",
        "exec_time",
        "_cache_type",
        "_cache_file_path",
        "_cache_tolerance",
        "_cache_hdf_node_name",
        "_linearize_on_last_state",
        "_cache_was_loaded",
        "_grammar_type",
        "comp_dir",
        "exec_for_lin",
        "_in_data_hash_dict",
        "_jac_approx",
    )

    __time_stamps_mp_manager = None
    time_stamps = None

    def __init__(
        self,
        name=None,  # type: Optional[str]
        input_grammar_file=None,  # type: Optional[Union[str, Path]]
        output_grammar_file=None,  # type: Optional[Union[str, Path]]
        auto_detect_grammar_files=False,  # type: bool
        grammar_type=JSON_GRAMMAR_TYPE,  # type: str
        cache_type=SIMPLE_CACHE,  # type: str
        cache_file_path=None,  # type: Optional[Union[str, Path]]
    ):  # type: (...) -> None
        """
        Args:
            name: The name of the discipline.
                If None, use the class name.
            input_grammar_file: The input grammar file path.
                If None and ``auto_detect_grammar_files=True``,
                use the file naming convention ``name + "_input.json"``
                and look for it in the directory containing the discipline source file
                If None and ``auto_detect_grammar_files=False``,
                do not initialize the input grammar from a schema file.
            output_grammar_file: The output grammar file path.
                If None and ``auto_detect_grammar_files=True``,
                use the file naming convention ``name + "_output.json"``
                and look for it in the directory containing the discipline source file
                If None and ``auto_detect_grammar_files=False``,
                do not initialize the output grammar from a schema file.
            auto_detect_grammar_files: Whether to use the naming convention
                when ``input_grammar_file`` and ``output_grammar_file`` are None.
                If so,
                search for these file names in the directory
                containing the discipline source file.
            grammar_type: The type of grammar to use for inputs and outputs declaration,
                e.g. :attr:`.JSON_GRAMMAR_TYPE` or :attr:`.SIMPLE_GRAMMAR_TYPE`.
            cache_type: The type of policy to cache the discipline evaluations,
                e.g. :attr:`.SIMPLE_CACHE` or :attr:`.HDF5_CACHE`.
            cache_file_path: The HDF file path
                when ``grammar_type`` is :attr:`.HDF5_CACHE`.
        """
        self.input_grammar = None  # : input grammar
        self.output_grammar = None  # : output grammar
        self._grammar_type = grammar_type
        self.comp_dir = None

        # : data converters between execute and _run
        self.data_processor = None
        self._default_inputs = {}  # : default data to be set if not passed
        # Allow to re execute the same discipline twice, only if did not fail
        # and not running
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        # : list of outputs that shall be null, to be considered as residuals
        self.residual_variables = []
        self._differentiated_inputs = []  # : outputs to differentiate
        # : inputs to be used for differentiation
        self._differentiated_outputs = []
        self._n_calls = None  # : number of calls to execute()
        self._exec_time = None  # : cumulated execution time
        # : number of calls to linearize()
        self._n_calls_linearize = None
        self._in_data_hash_dict = {}
        self.jac = None  # : Jacobians of outputs wrt inputs dictionary
        # : True if linearize() has already been called
        self._is_linearized = False
        self._jac_approx = None  # Jacobian approximation object
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
        self._cache_type = cache_type
        self._cache_file_path = cache_file_path
        self._cache_tolerance = 0.0
        self._cache_hdf_node_name = None
        # By default, dont use approximate cache
        # It is up to the user to choose to optimize CPU time with this or not
        self.set_cache_policy(cache_type=cache_type, cache_hdf_file=cache_file_path)
        # linearize mode :auto, adjoint, direct
        self._linearization_mode = JacobianAssembly.AUTO_MODE

        self_module = sys.modules.get(self.__class__.__module__)
        has_file = hasattr(self_module, "__file__")
        if has_file:
            self.comp_dir = str(Path(inspect.getfile(self.__class__)).parent.absolute())

        if input_grammar_file is None and auto_detect_grammar_files:
            input_grammar_file = self.auto_get_grammar_file(True)
        if output_grammar_file is None and auto_detect_grammar_files:
            output_grammar_file = self.auto_get_grammar_file(False)

        self._instantiate_grammars(
            input_grammar_file, output_grammar_file, self._grammar_type
        )

        self.local_data = {}  # : the inputs and outputs data
        # : The current status of execution
        self._status = self.STATUS_PENDING
        self._cache_was_loaded = False
        self._init_shared_attrs()
        self._status_observers = []

    def __str__(self):  # type: (...) -> str
        return self.name

    def __repr__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        inputs = sorted(self.get_input_data_names())
        outputs = sorted(self.get_output_data_names())
        msg.add("Inputs: {}", pretty_repr(inputs))
        msg.add("Outputs: {}", pretty_repr(outputs))
        return str(msg)

    def _init_shared_attrs(self):  # type: (...) -> None
        """Initialize the shared attributes in multiprocessing."""
        self._n_calls = Value("i", 0)
        self._exec_time = Value("d", 0.0)
        self._n_calls_linearize = Value("i", 0)

    def __init_cache_attr(self):  # type: (...) -> None
        """Initialize the cache attributes."""
        if self._cache_type == self.HDF5_CACHE:
            self.cache = None
            self.set_cache_policy(
                self.HDF5_CACHE,
                self._cache_tolerance,
                self._cache_file_path,
                self._cache_hdf_node_name,
            )

    @property
    def n_calls(self):  # type: (...) -> int
        """The number of times the discipline was executed.

        .. note::

            This property is multiprocessing safe.
        """
        return self._n_calls.value

    @n_calls.setter
    def n_calls(
        self,
        value,  # type: int
    ):  # type: (...) -> None
        self._n_calls.value = value

    @property
    def exec_time(self):  # type: (...) -> float
        """The cumulated execution time of the discipline.

        .. note::

            This property is multiprocessing safe.
        """
        return self._exec_time.value

    @exec_time.setter
    def exec_time(
        self,
        value,  # type: float
    ):  # type: (...) -> None
        self._exec_time.value = value

    @property
    def n_calls_linearize(self):  # type: (...) -> int
        """The number of times the discipline was linearized.

        .. note::

            This property is multiprocessing safe.
        """
        return self._n_calls_linearize.value

    @n_calls_linearize.setter
    def n_calls_linearize(
        self,
        value,  # type: int
    ):  # type: (...) -> None
        self._n_calls_linearize.value = value

    @property
    def grammar_type(self):  # type: (...) -> AbstractGrammar
        """The grammar type."""
        return self._grammar_type

    def auto_get_grammar_file(
        self,
        is_input=True,  # type: bool
        name=None,  # type: Optional[str]
        comp_dir=None,  # type: Optional[Union[str, Path]]
    ):  # type: (...) -> Path
        """Use a naming convention to associate a grammar file to a discipline.

        This method searches in a directory for
        either an input grammar file named ``name + "_input.json"``
        or an output grammar file named``name + "_output.json"``.

        Args:
            is_input: If True,
                autodetect the input grammar file;
                otherwise,
                autodetect the output grammar file.
            name: The name to be searched in the file names.
                If None,
                use the :attr:`.name` name of the discipline.
            comp_dir: The directory in which to search the grammar file.
                If None, use :attr:`.comp_dir`.

        Returns:
            The grammar file path.
        """
        if comp_dir is None:
            comp_dir = self.comp_dir
        if name is None:
            name = self.name
        if is_input:
            suffix = "input"
        else:
            suffix = "output"
        return Path(comp_dir) / "{}_{}.json".format(name, suffix)

    def add_differentiated_inputs(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        """Add inputs against which to differentiate the outputs.

        This method updates :attr:`._differentiated_inputs` with ``inputs``.

        Args:
            inputs: The input variables against which to differentiate the outputs.
                If None, all the inputs of the discipline are used.

        Raises:
            ValueError: When the inputs wrt which differentiate the discipline
                are not inputs of the latter.
        """
        if (inputs is not None) and (not self.is_all_inputs_existing(inputs)):
            raise ValueError(
                "Cannot differentiate the discipline {} wrt the inputs "
                "that are not among the discipline inputs: {}.".format(
                    self.name, self.get_input_data_names()
                )
            )

        if inputs is None:
            inputs = self.get_input_data_names()
        in_diff = self._differentiated_inputs
        self._differentiated_inputs = list(set(in_diff) | set(inputs))

    def add_differentiated_outputs(
        self,
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        """Add outputs to be differentiated.

        This method updates :attr:`._differentiated_outputs` with ``outputs``.

        Args:
            outputs: The output variables to be differentiated.
                If None, all the outputs of the discipline are used.

        Raises:
            ValueError: When the outputs to differentiate are not discipline outputs.
        """
        if (outputs is not None) and (not self.is_all_outputs_existing(outputs)):
            raise ValueError(
                "Cannot differentiate {} "
                "that are not among the discipline outputs {}.".format(
                    self.name, self.get_output_data_names()
                )
            )

        out_diff = self._differentiated_outputs
        if outputs is None:
            outputs = self.get_output_data_names()
        self._differentiated_outputs = list(set(out_diff) | set(outputs))

    def set_cache_policy(
        self,
        cache_type=SIMPLE_CACHE,  # type: str
        cache_tolerance=0.0,  # type: float
        cache_hdf_file=None,  # type: Optional[Union[str, Path]]
        cache_hdf_node_name=None,  # type: Optional[str]
        is_memory_shared=True,  # type: bool
    ):  # type: (...) -> None
        """Set the type of cache to use and the tolerance level.

        This method defines when the output data have to be cached
        according to the distance between the corresponding input data
        and the input data already cached for which output data are also cached.

        The cache can be either a :class:`SimpleCache` recording the last execution
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
                this argument is mandatory when the :attr:`.HDF5Cache` policy is used.
            cache_hdf_node_name: The name of the HDF file node
                to store the discipline data.
                If None, :attr:`.name` is used.
            is_memory_shared: Whether to store the data with a shared memory dictionary,
                which makes the cache compatible with multiprocessing.

                .. warning:

                   If set to False,
                   and multiple disciplines point
                   to the same cache or the process is multiprocessed,
                   there may be duplicate computations
                   because the cache will not be shared among the processes.
        """

        create_cache = CacheFactory().create

        if cache_type == self.HDF5_CACHE:
            not_same_file = cache_hdf_file != self._cache_file_path
            not_same_node = cache_hdf_node_name != self._cache_hdf_node_name
            cache_none = self.cache is None
            already_hdf = self._cache_type == self.HDF5_CACHE
            not_already_hdf = self._cache_type != self.HDF5_CACHE
            if cache_none or (
                (already_hdf and (not_same_file or not_same_node)) or not_already_hdf
            ):
                node_path = cache_hdf_node_name or self.name
                self.cache = create_cache(
                    self.HDF5_CACHE,
                    hdf_file_path=cache_hdf_file,
                    hdf_node_path=node_path,
                    tolerance=cache_tolerance,
                    name=self.name,
                )
                self._cache_hdf_node_name = cache_hdf_node_name
                self._cache_file_path = cache_hdf_file

        elif cache_type != self._cache_type or self.cache is None:
            if cache_type == self.MEMORY_FULL_CACHE:
                self.cache = create_cache(
                    cache_type,
                    tolerance=cache_tolerance,
                    name=self.name,
                    is_memory_shared=is_memory_shared,
                )
            else:
                self.cache = create_cache(
                    cache_type, tolerance=cache_tolerance, name=self.name
                )
        else:
            LOGGER.warning(
                "Cache policy is already set to %s. To clear the"
                " discipline cache, use its clear() method.",
                cache_type,
            )

        self._cache_type = cache_type
        self._cache_tolerance = cache_tolerance

    def get_sub_disciplines(
        self,
    ):  # type: (...) -> List[MDODiscipline]  # pylint: disable=R0201
        """Return the sub-disciplines if any.

        Returns:
            The sub-disciplines.
        """
        return []

    def get_expected_workflow(self):  # type: (...) -> SerialExecSequence
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
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        """Return the expected data exchange sequence.

        This method is used for the XDSM representation.

        The default expected data exchange sequence is an empty list.

        .. seealso::

           MDOFormulation.get_expected_dataflow

        Returns:
            The data exchange arcs.
        """
        return []

    def _instantiate_grammars(
        self,
        input_grammar_file,  # type: Optional[Union[str, Path]]
        output_grammar_file,  # type: Optional[Union[str, Path]]
        grammar_type=JSON_GRAMMAR_TYPE,  # type: str
    ):  # type: (...) -> None
        """Create the input and output grammars.

        Args:
            input_grammar_file: The input grammar file path.
                If None, do not initialize the input grammar from a schema file.
            output_grammar_file: The output grammar file path.
                If None, do not initialize the output grammar from a schema file.
            grammar_type: The type of grammar,
                e.g. :attr:`.JSONGrammar` or :attr:`.SimpleGrammar`.
        """
        factory = GrammarFactory()
        grammar_type = self.__DEPRECATED_GRAMMAR_TYPES.get(grammar_type, grammar_type)
        # TODO: deprecate this at some point.

        self.input_grammar = factory.create(
            grammar_type,
            name="{}_input".format(self.name),
            schema_file=input_grammar_file,
        )
        self.output_grammar = factory.create(
            grammar_type,
            name="{}_output".format(self.name),
            schema_file=output_grammar_file,
        )

    def _run(self):  # type: (...) -> None
        """Define the execution of the process, given that data has been checked.

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def _filter_inputs(
        self,
        input_data=None,  # type: Optional[Dict[str, Any]]
    ):  # type: (...) -> Dict[str, Any]
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
        if not isinstance(input_data, dict):
            raise TypeError(
                "Input data must be of dict type, "
                "got {} instead.".format(type(input_data))
            )

        # Take default inputs if not in input_data
        filt_inputs = self._default_inputs.copy()  # Shallow copy
        filt_inputs.update(input_data)

        # Remove inputs that should not be there
        in_names = self.get_input_data_names()
        filt_inputs = {key: val for key, val in filt_inputs.items() if key in in_names}

        return filt_inputs

    def _filter_local_data(self):  # type: (...) -> None
        """Filter the local data after execution.

        This method removes data that are neither inputs nor outputs.
        """
        all_data_names = self.get_input_output_data_names()

        self.local_data = {
            key: val for key, val in self.local_data.items() if key in all_data_names
        }

    def _check_status_before_run(self):  # type: (...) -> None
        """Check the status of the discipline before calling :meth:`._run`.

        Check the status of the discipline depending on :attr:`.re_execute_policy`.

        If ``re_exec_policy == RE_EXECUTE_NEVER_POLICY``,
        the status shall be either :attr:`.PENDING` or :attr:`.VIRTUAL`.

        If ``self.re_exec_policy == RE_EXECUTE_NEVER_POLICY``,

        - if status is :attr:`.DONE`,
          :meth:`.reset_statuses_for_run` is called prior :meth:`._run`,
        - otherwise status must be :attr:`.VIRTUAL` or :attr:`.PENDING`.

        Raises:
            ValueError:
                * When the re-execution policy is unknown.
                * When the discipline status and the re-execution policy
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
            raise ValueError("Unknown re_exec_policy: {}.".format(self.re_exec_policy))
        if not status_ok:
            raise ValueError(
                "Trying to run a discipline {} with status: {} "
                "while re_exec_policy is {}.".format(
                    type(self), self.status, self.re_exec_policy
                )
            )

    def __get_input_data_for_cache(
        self,
        input_data,  # type: Dict[str, Any]
        in_names,  # type: Iterable[str]
    ):  # type: (...) -> Dict[str, Any]
        """Prepare the input data for caching.

        Args:
            input_data: The values of the inputs.
            in_names: The names of the inputs.

        Returns:
            The input data to be cached.
        """
        in_and_out = set(in_names) & set(self.get_output_data_names())

        cached_inputs = dict(input_data.items())
        for key in in_and_out:
            val = input_data.get(key)
            if val is not None:
                # If also an output, keeps a copy of the original input value
                cached_inputs[key] = deepcopy(val)

        return cached_inputs

    def execute(
        self,
        input_data=None,  # type:Optional[Dict[str, Any]]
    ):  # type: (...) -> Dict[str, Any]
        """Execute the discipline.

        This method executes the discipline:

        * Adds the default inputs to the ``input_data``
          if some inputs are not defined in input_data
          but exist in :attr:`._default_inputs`.
        * Checks whether the last execution of the discipline was called
          with identical inputs, ie. cached in :attr:`.cache`;
          if so, directly returns ``self.cache.get_output_cache(inputs)``.
        * Caches the inputs.
        * Checks the input data against :attr:`.input_grammar`.
        * If :attr:`.data_processor` is not None, runs the preprocessor.
        * Updates the status to :attr:`.RUNNING`.
        * Calls the :meth:`._run` method, that shall be defined.
        * If :attr:`.data_processor` is not None, runs the postprocessor.
        * Checks the output data.
        * Caches the outputs.
        * Updates the status to :attr:`.DONE` or :attr:`.FAILED`.
        * Updates summed execution time.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.default_inputs`.

        Returns:
            The discipline local data after execution.
        """
        # Load the default_inputs if the user did not provide all required data
        input_data = self._filter_inputs(input_data)

        # Check if the cache already the contains outputs associated to these
        # inputs
        in_names = self.get_input_data_names()
        out_cached, out_jac = self.cache.get_outputs(input_data, in_names)

        if out_cached is not None:
            self.__update_local_data_from_cache(input_data, out_cached, out_jac)
            return self.local_data

        # Cache was not loaded, see self.linearize
        self._cache_was_loaded = False

        # Save the state of the inputs
        cached_inputs = self.__get_input_data_for_cache(input_data, in_names)
        self._check_status_before_run()

        self.check_input_data(input_data)
        self.local_data = {}
        self.local_data.update(input_data)

        processor = self.data_processor
        # If the data processor is set, pre-process the data before _run
        # See gemseo.core.data_processor module
        if processor is not None:
            self.local_data = processor.pre_process_data(input_data)
        self.status = self.STATUS_RUNNING
        self._is_linearized = False
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
        self.__increment_exec_time(t_0)
        self.status = self.STATUS_DONE

        # If the data processor is set, post process the data after _run
        # See gemseo.core.data_processor module
        if processor is not None:
            self.local_data = processor.post_process_data(self.local_data)

        # Filter data that is neither outputs nor inputs
        self._filter_local_data()

        self.check_output_data()

        # Caches output data in case the discipline is called twice in a row
        # with the same inputs
        out_names = self.get_output_data_names()
        self.cache.cache_outputs(cached_inputs, in_names, self.local_data, out_names)
        # Some disciplines are always linearized during execution, cache the
        # jac in this case
        if self._is_linearized:
            self.cache.cache_jacobian(cached_inputs, in_names, self.jac)
        return self.local_data

    def __update_local_data_from_cache(
        self,
        input_data,  # type: Dict[str, Any]
        out_cached,  # type: Dict[str, Any]
        out_jac,  # type: Dict[str, ndarray]
    ):  # type: (...) -> None
        """Update the local data from the cache.

        Args:
            input_data: The input data.
            out_cached: The output data retrieved from the cache.
            out_jac: The Jacobian data retrieved from the cache.
        """
        self.local_data = {}
        self.local_data.update(input_data)
        self.local_data.update(out_cached)

        if out_jac is not None:
            self.jac = out_jac
            self._is_linearized = True
        else:  # Erase jacobian which is unknown
            self.jac = None
            self._is_linearized = False

        self.check_output_data()
        self._cache_was_loaded = True

    def __increment_n_calls(self):  # type: (...) -> None
        """Increment by 1 the number of executions.."""
        with self._n_calls.get_lock():
            self._n_calls.value += 1

    def __increment_n_calls_lin(self):  # type: (...) -> None
        """Increment by 1 the number of linearizations."""
        with self._n_calls_linearize.get_lock():
            self._n_calls_linearize.value += 1

    def __increment_exec_time(
        self,
        t_0,  # type: float
        linearize=False,  # type: bool
    ):  # type: (...) -> None
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
                disc_stamps = time_stamps.get(self.name)
                if disc_stamps is None:
                    if os.name == "nt":
                        disc_stamps = []
                    else:
                        disc_stamps = MDODiscipline.__time_stamps_mp_manager.list()
                stamp = (t_0, curr_t, linearize)
                disc_stamps.append(stamp)
                time_stamps[self.name] = disc_stamps

    def _retrieve_diff_inouts(
        self,
        force_all=False,  # type: bool
    ):  # type: (...) -> Tuple[List[str], List[str]]
        """Get the inputs and outputs used in the differentiation of the discipline.

        Args:
            force_all: If True,
                consider all the inputs and outputs of the discipline;
                otherwise,
                consider :attr:`_differentiated_inputs`
                and :attr:`_differentiated_outputs`.
        """
        if force_all:
            inputs = self.get_input_data_names()
            outputs = self.get_output_data_names()
        else:
            inputs = self._differentiated_inputs
            outputs = self._differentiated_outputs
        return inputs, outputs

    _retreive_diff_inouts = _retrieve_diff_inouts
    # TODO: deprecate it at some point

    @classmethod
    def activate_time_stamps(cls):  # type: (...) -> None
        """Activate the time stamps.

        For storing start and end times of execution and linearizations.
        """
        if os.name == "nt":  # No multiprocessing under windows
            MDODiscipline.time_stamps = {}
        else:
            manager = Manager()
            MDODiscipline.__time_stamps_mp_manager = manager
            MDODiscipline.time_stamps = manager.dict()

    @classmethod
    def deactivate_time_stamps(cls):  # type: (...) -> None
        """Deactivate the time stamps.

        For storing start and end times of execution and linearizations.
        """
        MDODiscipline.time_stamps = None
        MDODiscipline.__time_stamps_mp_manager = None

    def linearize(
        self,
        input_data=None,  # type: Optional[Dict[str, Any]]
        force_all=False,  # type: bool
        force_no_exec=False,  # type: bool
    ):  # type: (...) -> Dict[str, Dict[str, ndarray]]
        """Execute the linearized version of the code.

        Args:
            input_data: The input data needed to linearize the discipline
                according to the discipline input grammar.
                If None, use the :attr:`.default_inputs`.
            force_all: If False,
                :attr:`._differentiated_inputs` and :attr:`.differentiated_output`
                are used to filter the differentiated variables.
                otherwise, all outputs are differentiated wrt all inputs.
            force_no_exec: If True,
                the discipline is not re executed, cache is loaded anyway.

        Returns:
            The Jacobian of the discipline.
        """
        # TODO: remove the execution when no option exec_before_lin
        # is set to True
        inputs, outputs = self._retrieve_diff_inouts(force_all)
        if not outputs:
            self.jac = {}
            return self.jac
        # Save inputs dict for caching
        input_data = self._filter_inputs(input_data)

        # if force_no_exec, we do not re execute the discipline
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
            self.local_data.update(input_data)

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
        approximate_jac = self.linearization_mode in self.APPROX_MODES

        if approximate_jac:  # Time already counted in execute()
            self.jac = self._jac_approx.compute_approx_jac(outputs, inputs)
        else:
            self._compute_jacobian(inputs, outputs)
            self.__increment_exec_time(t_0, linearize=True)

        self.__increment_n_calls_lin()

        self._check_jacobian_shape(inputs, outputs)
        # Cache the Jacobian matrix
        self.cache.cache_jacobian(input_data, self.get_input_data_names(), self.jac)

        return self.jac

    def set_jacobian_approximation(
        self,
        jac_approx_type=FINITE_DIFFERENCES,  # type: str
        jax_approx_step=1e-7,  # type: float
        jac_approx_n_processes=1,  # type: int
        jac_approx_use_threading=False,  # type: bool
        jac_approx_wait_time=0,  # type: float
    ):  # type: (...) -> None
        """Set the Jacobian approximation method.

        Sets the linearization mode to approx_method,
        sets the parameters of the approximation for further use
        when calling :meth:`.linearize`.

        Args:
            jac_approx_type: The approximation method,
                either "complex_step" or "finite_differences".
            jax_approx_step: The differentiation step.
            jac_approx_n_processes: The maximum number of processors on which to run.
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
        outputs=None,  # type: Optional[Iterable[str]]
        inputs=None,  # type: Optional[Iterable[str]]
        force_all=False,  # type: bool
        print_errors=False,  # type: bool
        numerical_error=EPSILON,  # type: float
    ):
        """Compute the optimal finite-difference step.

        Compute the optimal step
        for a forward first order finite differences gradient approximation.
        Requires a first evaluation of the perturbed functions values.
        The optimal step is reached when the truncation error
        (cut in the Taylor development),
        and the numerical cancellation errors
        (roundoff when doing f(x+step)-f(x))
         are approximately equal.

        .. warning::

           This calls the discipline execution twice per input variables.

        .. seealso::

           https://en.wikipedia.org/wiki/Numerical_differentiation
           and
           "Numerical Algorithms and Digital Representation", Knut Morken ,
           Chapter 11, "Numerical Differenciation"

        Args:
            inputs: The inputs wrt which the outputs are linearized.
                If None, use the :attr:`_differentiated_inputs`.
            outputs: The outputs to be linearized.
                If None, use the :attr:`_differentiated_outputs`.
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
    def __get_len(container):  # type: (...) -> int
        """Measure the length of a container."""
        if container is None:
            return -1
        try:
            return len(container)
        except TypeError:
            return 1

    def _check_jacobian_shape(
        self,
        inputs,  # type: Iterable[str]
        outputs,  # type: Iterable[str]
    ):  # type: (...) -> None
        """Check that the Jacobian is a dictionary of dictionaries of 2D NumPy arrays.

        Args:
            inputs: The inputs wrt the outputs are linearized.
            outputs: The outputs to be linearized.

        Raises:
            ValueError:
            * When the discipline was not linearized.
            * When the Jacobian is not of the right shape.
            KeyError:
            * When outputs are missing in the Jacobian of the discipline.
            * When inputs are missing for an output in the Jacobian of the discipline.
        """
        if self.jac is None:
            raise ValueError("The discipline {} was not linearized.".format(self.name))
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
            output_vals = self.local_data.get(j_o)
            n_out_j = self.__get_len(output_vals)

            if not in_set.issubset(out_dv_set):
                msg = "Missing inputs {} in Jacobian of discipline {}, for output: {}."
                missing_inputs = in_set.difference(out_dv_set)
                raise KeyError(msg.format(missing_inputs, self.name, j_o))

            for j_i in inputs:
                input_vals = self.local_data.get(j_i)
                n_in_j = self.__get_len(input_vals)
                j_mat = j_out[j_i]
                expected_shape = (n_out_j, n_in_j)

                if -1 in expected_shape:
                    # At least one of the dimensions is unknown
                    # Dont check shape
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
        for jac_loc in self.jac.values():
            for desv, jac_out in jac_loc.items():
                jac_loc[desv] = jac_out.real

    @property
    def cache_tol(self):  # type: (...) -> float
        """The cache input tolerance.

        This is the tolerance for equality of the inputs in the cache.
        If norm(stored_input_data-input_data) <= cache_tol * norm(stored_input_data),
        the cached data for ``stored_input_data`` is returned
        when calling ``self.execute(input_data)``.
        """
        return self.cache.tolerance

    @cache_tol.setter
    def cache_tol(
        self,
        cache_tol,  # type: float
    ):  # type: (...) -> None
        self._set_cache_tol(cache_tol)

    def _set_cache_tol(
        self,
        cache_tol,  # type: float
    ):  # type: (...) -> None
        """Set to the cache input tolerance.

        To be overloaded by subclasses.

        Args:
            cache_tol: The cache tolerance.
        """
        self.cache.tolerance = cache_tol or 0.0

    @property
    def default_inputs(self):  # type: (...) -> Dict[str, Any]
        """The default inputs.

        Raises:
            TypeError: When the default inputs are not passed as a dictionary.
        """
        return self._default_inputs

    @default_inputs.setter
    def default_inputs(
        self, default_inputs  # type: Dict[str,Any]
    ):  # type: (...) -> None
        if not isinstance(default_inputs, dict):
            raise TypeError(
                "MDODiscipline default inputs must be of dict type, "
                "got {} instead.".format(type(default_inputs))
            )
        self._default_inputs = default_inputs

    def _compute_jacobian(
        self,
        inputs=None,  # type:Optional[Iterable[str]]
        outputs=None,  # type:Optional[Iterable[str]]
    ):  # type: (...)-> None
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
        inputs=None,  # type:Optional[Iterable[str]]
        outputs=None,  # type:Optional[Iterable[str]]
        with_zeros=False,  # type: bool
        fill_missing_keys=False,  # type: bool
    ):  # type: (...) -> None
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
                jac_loc = defaultdict(None)
                jac[out_n] = jac_loc
                # When a key is not in the default dict,
                # ie a variable is not in the
                # Jacobian; return a defaultdict(None)
                for in_n, in_v in zip(inputs_names, inputs_vals):
                    jac_loc[in_n] = np_method((len(out_v), len(in_v)))
            self.jac = jac
        else:
            jac = self.jac
            # Only fill the missing sub jacobians
            for out_n, out_v in zip(outputs_names, outputs_vals):
                jac_loc = jac.get(out_n)
                if jac_loc is None:
                    jac_loc = defaultdict(None)
                    jac[out_n] = jac_loc

                for in_n, in_v in zip(inputs_names, inputs_vals):
                    sub_jac = jac_loc.get(in_n)
                    if sub_jac is None:
                        jac_loc[in_n] = np_method((len(out_v), len(in_v)))

    @property
    def linearization_mode(self):  # type: (...) -> str
        """The linearization mode among :attr:`.LINEARIZE_MODE_LIST`.

        Raises:
            ValueError: When the linearization mode is unknown.
        """
        return self._linearization_mode

    @linearization_mode.setter
    def linearization_mode(
        self,
        linearization_mode,  # type: str
    ):  # type: (...) -> None
        if linearization_mode not in self.AVAILABLE_MODES:
            msg = "Linearization mode '{}' is unknown; it must be one of {}."
            raise ValueError(msg.format(linearization_mode, self.AVAILABLE_MODES))

        self._linearization_mode = linearization_mode

        if linearization_mode in self.APPROX_MODES and self._jac_approx is None:
            self.set_jacobian_approximation(linearization_mode)

    def check_jacobian(
        self,
        input_data=None,  # type: Optional[Dict[str, ndarray]]
        derr_approx=FINITE_DIFFERENCES,  # type: str
        step=1e-7,  # type: float
        threshold=1e-8,  # type: float
        linearization_mode="auto",  # type: str
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
        parallel=False,  # type: bool
        n_processes=N_CPUS,  # type: int
        use_threading=False,  # type: bool
        wait_time_between_fork=0,  # type: float
        auto_set_step=False,  # type: bool
        plot_result=False,  # type: bool
        file_path="jacobian_errors.pdf",  # type: Union[str, Path]
        show=False,  # type: bool
        figsize_x=10,  # type: float
        figsize_y=10,  # type: float
        reference_jacobian_path=None,  # type: Optional[str, Path]
        save_reference_jacobian=False,  # type: bool
        indices=None,  # type: Optional[Iterable[int]]
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
                If None, use the :attr:`.default_inputs`.
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
            n_processes: The maximum number of processors on which to run.
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
            figsize_x: The x-size of the figure in inches.
            figsize_y: The y-size of the figure in inches.
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
            approx.auto_set_step(outputs, inputs, print_errors=True)

        # Differentiate analytically
        self.add_differentiated_inputs(inputs)
        self.add_differentiated_outputs(outputs)
        self.linearization_mode = linearization_mode
        self.reset_statuses_for_run()
        # Linearize performs execute() if needed
        self.linearize(input_data)
        o_k = approx.check_jacobian(
            self.jac,
            outputs,
            inputs,
            self,
            threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            figsize_x=figsize_x,
            figsize_y=figsize_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )
        return o_k

    @property
    def status(self):  # type: (...) -> str
        """The status of the discipline."""
        return self._status

    def _check_status(
        self,
        status,  # type: str
    ):  # type: (...) -> None
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
            raise ValueError("Unknown status: {}.".format(status))

    def set_disciplines_statuses(
        self,
        status,  # type: str
    ):  # type: (...) -> None
        """Set the sub-disciplines statuses.

        To be implemented in subclasses.

        Args:
            status: The status.
        """

    def is_output_existing(
        self,
        data_name,  # type: str
    ):  # type: (...) -> bool
        """Test if a variable is a discipline output.

        Args:
            data_name: The name of the variable.

        Returns:
            Whether the variable is a discipline output.
        """
        return self.output_grammar.is_data_name_existing(data_name)

    def is_all_outputs_existing(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> bool
        """Test if several variables are discipline outputs.

        Args:
            data_names: The names of the variables.

        Returns:
            Whether all the variables are discipline outputs.
        """
        return self.output_grammar.is_all_data_names_existing(data_names)

    def is_all_inputs_existing(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> bool
        """Test if several variables are discipline inputs.

        Args:
            data_names: The names of the variables.

        Returns:
            Whether all the variables are discipline inputs.
        """
        return self.input_grammar.is_all_data_names_existing(data_names)

    def is_input_existing(
        self,
        data_name,  # type: str
    ):  # type: (...) -> bool
        """Test if a variable is a discipline input.

        Args:
            data_name: The name of the variable.

        Returns:
            Whether the variable is a discipline input.
        """
        return self.input_grammar.is_data_name_existing(data_name)

    def _is_status_ok_for_run_again(
        self,
        status,  # type: str
    ):  # type: (...) -> bool
        """Check if the discipline can be run again.

        Args:
            status: The status of the discipline.

        Returns:
            Whether the discipline can be run again.
        """
        return status not in [self.STATUS_RUNNING]

    def reset_statuses_for_run(self):  # type: (...) -> None
        """Set all the statuses to :attr:`.PENDING`.

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
        status,  # type: str
    ):  # type: (...) -> None
        self._check_status(status)
        self._status = status
        self.notify_status_observers()

    def add_status_observer(
        self,
        obs,  # type: Any
    ):  # type: (...) -> None
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
        obs,  # type: Any
    ):  # type: (...) -> None
        """Remove an observer for the status.

        Args:
            obs: The observer to remove.
        """
        if obs in self._status_observers:
            self._status_observers.remove(obs)

    def notify_status_observers(self):  # type: (...) -> None
        """Notify all status observers that the status has changed."""
        for obs in self._status_observers[:]:
            obs.update_status(self)

    def store_local_data(
        self, **kwargs  # type:Any
    ):  # type: (...) -> None
        """Store discipline data in local data.

        Args:
            kwargs: The data to be stored in :attr:`.local_data`.
        """
        self.local_data.update(kwargs)

    def check_input_data(
        self,
        input_data,  # type: Dict[str,Any]
        raise_exception=True,  # type: bool
    ):  # type: (...) -> None
        """Check the input data validity.

        Args:
            input_data: The input data needed to execute the discipline
                according to the discipline input grammar.
        """
        try:
            self.input_grammar.load_data(input_data, raise_exception)
        except InvalidDataException as err:
            err.args = (
                err.args[0].replace("Invalid data", "Invalid input data")
                + " in discipline {}".format(self.name),
            )
            raise

    def check_output_data(
        self,
        raise_exception=True,  # type: bool
    ):  # type: (...) -> None
        """Check the output data validity.

        Args:
            raise_exception: Whether to raise an exception when the data is invalid.
        """
        try:
            self.output_grammar.load_data(self.local_data, raise_exception)
        except InvalidDataException as err:
            err.args = (
                err.args[0].replace("Invalid data", "Invalid output data")
                + " in discipline {}".format(self.name),
            )
            raise

    def get_outputs_asarray(self):  # type: (...) -> ndarray
        """Return the local input data as a large NumPy array.

        The order is the one of :meth:`.get_all_inputs`.

        Returns:
            The local input data.
        """
        return concatenate(list(self.get_all_outputs()))

    def get_inputs_asarray(self):  # type: (...) -> ndarray
        """Return the local output data as a large NumPy array.

        The order is the one of :meth:`.get_all_outputs`.

        Returns:
            The local output data.
        """
        return concatenate(list(self.get_all_inputs()))

    def get_inputs_by_name(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> List[Any]
        """Return the local data associated with input variables.

        Args:
            data_names: The names of the input variables.

        Returns:
            The local data for the given input variables.

        Raises:
            ValueError: When a variable is not an input of the discipline.
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline {} has no input named {}.".format(self.name, err)
            )

    def get_outputs_by_name(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> List[Any]
        """Return the local data associated with output variables.

        Args:
            data_names: The names of the output variables.

        Returns:
            The local data for the given output variables.

        Raises:
            ValueError: When a variable is not an output of the discipline.
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline {} has no output named {}.".format(self.name, err)
            )

    def get_input_data_names(self):  # type: (...) -> List[str]
        """Return the names of the input variables.

        Returns:
            The names of the input variables.
        """
        return self.input_grammar.get_data_names()

    def get_output_data_names(self):  # type: (...) -> List[str]
        """Return the names of the output variables.

        Returns:
            The names of the output variables.
        """
        return self.output_grammar.get_data_names()

    def get_input_output_data_names(self):  # type: (...) -> List[str]
        """Return the names of the input and output variables.

        Returns:
            The name of the input and output variables.
        """
        outpt = self.output_grammar.get_data_names()
        inpt = self.input_grammar.get_data_names()
        return list(set(outpt) | set(inpt))

    def get_all_inputs(self):  # type: (...) -> List[Any]
        """Return the local input data as a list.

        The order is given by :meth:`.get_input_data_names`.

        Returns:
            The local input data.
        """
        return self.get_inputs_by_name(self.get_input_data_names())

    def get_all_outputs(self):  # type: (...) -> List[Any]
        """Return the local output data as a list.

        The order is given by :meth:`.get_output_data_names`.

        Returns:
            The local output data.
        """
        return self.get_outputs_by_name(self.get_output_data_names())

    def get_output_data(self):  # type: (...) -> Dict[str, Any]
        """Return the local output data as a dictionary.

        Returns:
            The local output data.
        """
        return dict(
            (k, v) for k, v in self.local_data.items() if self.is_output_existing(k)
        )

    def get_input_data(self):  # type: (...) -> Dict[str, Any]
        """Return the local input data as a dictionary.

        Returns:
            The local input data.
        """
        return dict(
            (k, v) for k, v in self.local_data.items() if self.is_input_existing(k)
        )

    def serialize(
        self,
        out_file,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Serialize the discipline and store it in a file.

        Args:
            out_file: The path to the file to store the discipline.
        """
        out_file = Path(out_file)
        with out_file.open("wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def deserialize(
        in_file,  # type: Union[str, Path]
    ):  # type: (...) -> MDODiscipline
        """Deserialize a discipline from a file.

        Args:
            in_file: The path to the file containing the discipline.

        Returns:
            The discipline instance.
        """
        in_file = Path(in_file)
        with in_file.open("rb") as in_fobj:
            pickler = pickle.Unpickler(in_fobj)
            return pickler.load()

    def get_attributes_to_serialize(self):  # pylint: disable=R0201
        """Define the names of the attributes to be serialized.

        Shall be overloaded by disciplines

        Returns:
            The names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overriden,
        # it is a function==> self is required
        return list(self._ATTR_TO_SERIALIZE)

    def __getstate__(self):  # type: (...) -> Dict[str, Any]
        """Used by pickle to define what to serialize.

        Returns:
            The attributes to be serialized.

        Raises:
            AttributeError: When an attribute to be serialized is undefined.
            TypeError: When an attribute has an undefined type.
        """
        out_d = {}
        for keep_name in self.get_attributes_to_serialize():
            if keep_name not in self.__dict__:
                if "_" + keep_name not in self.__dict__ and not hasattr(
                    self, keep_name
                ):
                    msg = (
                        "Discipline {} defined attribute '{}' "
                        "as required for serialization, "
                        "but it appears to "
                        "be undefined.".format(self.name, keep_name)
                    )
                    raise AttributeError(msg)

                prop = self.__dict__.get("_" + keep_name)
                # May appear for properties that overload public attrs of super class
                if hasattr(self, keep_name) and not isinstance(prop, Synchronized):
                    continue

                if not isinstance(prop, Synchronized):
                    raise TypeError(
                        "Cant handle attribute {} serialization "
                        "of undefined type.".format(keep_name)
                    )
                # Dont serialize shared memory object,
                # this is meaningless, save the value instead
                out_d[keep_name] = prop.value
            else:
                out_d[keep_name] = self.__dict__[keep_name]

        if self._cache_type == self.HDF5_CACHE:
            out_d.pop("cache")
        return out_d

    def __setstate__(
        self,
        state_dict,  # type: Dict[str, Any]
    ):  # type: (...) -> None
        """Used by pickle to define what to deserialize.

        Args:
            state_dict: Update ``self.__dict__`` from ``state_dict``
                to deserialize the discipline.
        """
        self._init_shared_attrs()
        self._status_observers = []
        out_d = self.__dict__
        shared_attrs = list(out_d.keys())
        for key, val in state_dict.items():
            if "_" + key not in shared_attrs:
                out_d[key] = val
            else:
                out_d["_" + key].value = val

        self.__init_cache_attr()

    def get_local_data_by_name(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> Generator[Any]
        """Return the local data of the discipline associated with variables names.

        Args:
            data_names: The names of the variables.

        Returns:
            The local data associated with the variables names.

        Raises:
            ValueError: When a name is not not a discipline input name.
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline {} has no local_data named {}.".format(self.name, err)
            )

    @staticmethod
    def is_scenario():  # type: (...) -> bool
        """Whether the discipline is a scenario."""
        return False

    @staticmethod
    def get_data_list_from_dict(
        keys,  # type: Union[str, Iterable]
        data_dict,  # type: Dict[str, Any]
    ):  # type: (...) -> Union[Any, Generator[Any]]
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
        if isinstance(keys, six.string_types):
            return data_dict[keys]
        return (data_dict[name] for name in keys)
