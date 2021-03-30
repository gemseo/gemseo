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
"""
Abstraction of processes
************************
"""
from __future__ import absolute_import, division, unicode_literals

import inspect
import sys
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Value, cpu_count
from multiprocessing.sharedctypes import Synchronized
from os.path import abspath, dirname, join
from timeit import default_timer as timer

from future import standard_library
from numpy import concatenate, empty, zeros
from six import string_types

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.grammar import InvalidDataException, SimpleGrammar
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.core.json_grammar import JSONGrammar
from gemseo.utils.derivatives_approx import EPSILON, DisciplineJacApprox

try:
    import cPickle as pickle
except ImportError:
    import pickle

standard_library.install_aliases()


from gemseo import LOGGER


def default_dict_factory():
    """Instantiate a defaultdict(None) object."""
    return defaultdict(None)


class MDODiscipline(object):
    """A software integrated in the workflow.

    The inputs and outputs are defined in a grammar, which can be
    either a SimpleGrammar or a JSONGrammar, or your own which
    derives from the Grammar abstract class

    To be used, use a subclass and implement the _run method
    which defined the execution of the software.
    Typically, in the _run method, get the inputs from the
    input grammar, call your software, and write the outputs
    to the output grammar.

    The JSON Grammar are automatically detected when in the same
    folder as your subclass module and named "CLASSNAME_input.json"
    use auto_detect_grammar_files=True to activate this option


    """

    STATUS_VIRTUAL = "VIRTUAL"
    STATUS_PENDING = "PENDING"
    STATUS_DONE = "DONE"
    STATUS_RUNNING = "RUNNING"
    STATUS_FAILED = "FAILED"

    JSON_GRAMMAR_TYPE = "JSON"
    SIMPLE_GRAMMAR_TYPE = "Simple"

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

    def __init__(
        self,
        name=None,
        input_grammar_file=None,
        output_grammar_file=None,
        auto_detect_grammar_files=False,
        grammar_type=JSON_GRAMMAR_TYPE,
        cache_type=SIMPLE_CACHE,
        cache_file_path=None,
    ):
        """Constructor.

        :param name: the name of the discipline
        :param input_grammar_file: the file for input grammar description,
            if None, name + "_input.json" is used
        :param output_grammar_file: the file for output grammar description,
            if None, name + "_output.json" is used
        :param auto_detect_grammar_files: if no input and output grammar files
            are provided, auto_detect_grammar_files uses a naming convention
            to associate a
            grammar file to a discipline:
            searches in the "comp_dir" directory containing the
            discipline source file for files basenames
            self.name _input.json and self.name _output.json
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :param cache_type: type of cache policy, SIMPLE_CACHE
            or HDF5_CACHE
        :param cache_file_path: the file to store the data,
            mandatory when HDF caching is used
        """

        self.input_grammar = None  # : input grammar
        self.output_grammar = None  # : output grammar
        self.grammar_type = grammar_type
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
        # : inputs to be used for differenciation
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
        self._cache_factory = CacheFactory()
        # By default, dont use approximate cache
        # It is up to the user to choose to optimize CPU time with this or not
        self.set_cache_policy(cache_type=cache_type, cache_hdf_file=cache_file_path)
        # linearize mode :auto, adjoint, direct
        self._linearization_mode = JacobianAssembly.AUTO_MODE

        self_module = sys.modules.get(self.__class__.__module__)
        has_file = hasattr(self_module, "__file__")
        if has_file:
            f_class = inspect.getfile(self.__class__)
            self.comp_dir = abspath(dirname(f_class))

        if input_grammar_file is None and auto_detect_grammar_files:
            input_grammar_file = self.auto_get_grammar_file(True)
        if output_grammar_file is None and auto_detect_grammar_files:
            output_grammar_file = self.auto_get_grammar_file(False)

        self._instantiate_grammars(
            input_grammar_file, output_grammar_file, grammar_type
        )

        self.local_data = {}  # : the inputs and outputs data
        # : The current status of execution
        self._status = self.STATUS_PENDING
        self._cache_was_loaded = False
        self._init_shared_attrs()
        self._status_observers = []

    def _init_shared_attrs(self):
        """Initialize the shared attributes in multiprocessing."""
        self._n_calls = Value("i", 0)
        self._exec_time = Value("d", 0.0)
        self._n_calls_linearize = Value("i", 0)

    def __init_cache_attr(self):
        """Initialize cache attributes after deserialization."""
        if self._cache_type == self.HDF5_CACHE:
            self.cache = None
            self.set_cache_policy(
                self.HDF5_CACHE,
                self._cache_tolerance,
                self._cache_file_path,
                self._cache_hdf_node_name,
            )

    @property
    def n_calls(self):
        """Return the number of calls to execute() which triggered the _run().

        Multiprocessing safe.
        """
        return self._n_calls.value

    @n_calls.setter
    def n_calls(self, value):
        """Set the number of calls to execute() which triggered the _run().

        Multiprocessing safe
        :param value: the value of n_calls
        """
        self._n_calls.value = value

    @property
    def exec_time(self):
        """Return the cumulated execution time.

        Multiprocessing safe.
        """
        return self._exec_time.value

    @exec_time.setter
    def exec_time(self, value):
        """Set the cumulated execution time.

        Multiprocessing safe
        :param value: the value of exec_time
        """
        self._exec_time.value = value

    @property
    def n_calls_linearize(self):
        """Return the number of calls to linearize() which triggered the
        _compute_jacobian() method.

        Multiprocessing safe.
        """
        return self._n_calls_linearize.value

    @n_calls_linearize.setter
    def n_calls_linearize(self, value):
        """Set the number of calls to linearize() which triggered the
        _compute_jacobian() method

        Multiprocessing safe
        :param value: the value of n_calls_linearize
        """
        self._n_calls_linearize.value = value

    def auto_get_grammar_file(self, is_input=True, name=None, comp_dir=None):
        """Use a naming convention to associate a grammar file to a discipline.

        This method searches in the "comp_dir" directory containing the
        discipline source file for files basenames self.name _input.json and
        self.name _output.json

        :param is_input: if True, searches for _input.json,
            otherwise _output.json (Default value = True)
        :param name: the name of the discipline (Default value = None)
        :param comp_dir: the containing directory
            if None, use self.comp_dir (Default value = None)
        :returns: path to the grammar file
        :rtype: string
        """
        if comp_dir is None:
            comp_dir = self.comp_dir
        if name is None:
            name = self.name
        if is_input:
            endf = "_input.json"
        else:
            endf = "_output.json"
        return join(comp_dir, name + endf)

    def add_differentiated_inputs(self, inputs=None):
        """Add inputs to the differentiation list.

        This method updates self._differentiated_inputs with inputs

        :param inputs: list of inputs variables to differentiate
            if None, all inputs of discipline are used (Default value = None)

        """
        if (inputs is not None) and (not self.is_all_inputs_existing(inputs)):
            raise ValueError(
                "Cannot differentiate discipline "
                + self.name
                + " wrt inputs that are not"
                + " among discipline inputs: "
                + str(self.get_input_data_names())
            )

        if inputs is None:
            inputs = self.get_input_data_names()
        in_diff = self._differentiated_inputs
        self._differentiated_inputs = list(set(in_diff) | set(inputs))

    def add_differentiated_outputs(self, outputs=None):
        """Add outputs to the differentiation list.

        Update self._differentiated_inputs with inputs.

        :param outputs: list of output variables to differentiate
            if None, all outputs of discipline are used
        """
        if (outputs is not None) and (not self.is_all_outputs_existing(outputs)):
            raise ValueError(
                "Cannot differentiate discipline "
                + self.name
                + " outputs that are not"
                + " among discipline outputs: "
                + str(self.get_output_data_names())
            )

        out_diff = self._differentiated_outputs
        if outputs is None:
            outputs = self.get_output_data_names()
        self._differentiated_outputs = list(set(out_diff) | set(outputs))

    def set_cache_policy(
        self,
        cache_type=SIMPLE_CACHE,
        cache_tolerance=0.0,
        cache_hdf_file=None,
        cache_hdf_node_name=None,
    ):
        """Set the type of cache to use and the tolerance level.

        This method set the cache policy to cache data whose inputs are close
        to inputs whose outputs are already cached. The cache can be either a
        simple cache recording the last execution or a full cache storing all
        executions.  Caching data can be either in-memory, e.g.
        :class:`.SimpleCache` and
        :class:`.MemoryFullCache` ,
        or on the disk, e.g.
        :class:`.HDF5Cache` .
        :attr:`.CacheFactory.caches`
        provides the list of available types of caches.

        :param str cache_type: type of cache to use.
        :param float cache_tolerance: tolerance for the approximate cache
            maximal relative norm difference to consider that
            two input arrays are equal
        :param str cache_hdf_file: the file to store the data,
            mandatory when HDF caching is used
        :param str cache_hdf_node_name: name of the HDF
            dataset to store the discipline
            data. If None, self.name is used
        """
        create_cache = self._cache_factory.create

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

    def get_sub_disciplines(self):  # pylint: disable=R0201
        """Gets the sub disciplines of self
        By default, empty

        :returns: the list of disciplines
        """
        return []

    def get_expected_workflow(self):
        """Return the expected execution sequence.

        This method is used for XDSM representation
        Default to the execution of the discipline itself
        See MDOFormulation.get_expected_workflow
        """
        # avoid circular dependency
        from gemseo.core.execution_sequence import ExecutionSequenceFactory

        return ExecutionSequenceFactory.serial(self)

    def get_expected_dataflow(self):  # pylint: disable=R0201
        """Return the expected data exchange sequence.

        This method is used for the XDSM representation.

        Default to empty list
        See MDOFormulation.get_expected_dataflow

        :returns: a list representing the data exchange arcs
        """
        return []

    def _instantiate_grammars(
        self, input_grammar_file, output_grammar_file, grammar_type=JSON_GRAMMAR_TYPE
    ):
        """Create the input and output grammars.

        :param input_grammar_file: the input file of the grammar
        :param output_grammar_file: the output file of the grammar
        :param grammar_type: the type of grammar to use, JSON,
            Simple or yours ! (Default value = JSON_GRAMMAR_TYPE)
        """
        if grammar_type == self.JSON_GRAMMAR_TYPE:
            self.input_grammar = JSONGrammar(
                name=self.name + "_input",
                schema_file=input_grammar_file,
                grammar_type="input",
            )
            self.output_grammar = JSONGrammar(
                name=self.name + "_output",
                schema_file=output_grammar_file,
                grammar_type="output",
            )
        elif grammar_type == self.SIMPLE_GRAMMAR_TYPE:
            self.input_grammar = SimpleGrammar(name=self.name + "_input")
            self.output_grammar = SimpleGrammar(name=self.name + "_output")
        else:
            raise ValueError("Unknown grammar type: " + str(grammar_type))

    def _run(self):
        """Define the execution of the process, given that data has been checked.

        To be overloaded by sub classes.
        """
        raise NotImplementedError()

    def _filter_inputs(self, input_data=None):
        """Load the input data and adds default data when not present.

        This method filters the inputs that shall not be used by the
        discipline.

        :param input_data: a data dictionary (Default value = None)
        :returns: the filtered data dictionary
        """
        if input_data is None:
            return deepcopy(self.default_inputs)
        if not isinstance(input_data, dict):
            raise TypeError(
                "Input data must be a dict, got " + str(type(input_data)) + " instead"
            )

        # Take default inputs if not in input_data
        filt_inputs = self._default_inputs.copy()  # Shallow copy
        filt_inputs.update(input_data)

        # Remove inputs that should not be there
        in_names = self.get_input_data_names()
        filt_inputs = {key: val for key, val in filt_inputs.items() if key in in_names}

        return filt_inputs

    def _filter_local_data(self):
        """Filter the local data after execution.

        This method removes data that are neither inputs nor outputs.
        """
        all_data_names = self.get_input_output_data_names()

        self.local_data = {
            key: val for key, val in self.local_data.items() if key in all_data_names
        }

    def _check_status_before_run(self):
        """Check the status of the discipline before calling _run.

        Check the status of the discipline depending on self.re_execute_policy.

        if self.re_exec_policy == RE_EXECUTE_NEVER_POLICY:
            status shall be either PENDING or VIRTUAL

        if self.re_exec_policy == RE_EXECUTE_NEVER_POLICY:
            if status is DONE, self.reset_statuses_for_run is called prior run
            otherwise status must be VIRTUAL or PENDING
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
            raise ValueError("Unknown re_exec_policy :" + str(self.re_exec_policy))
        if not status_ok:
            raise ValueError(
                "Trying to run a discipline "
                + str(type(self))
                + " with status: "
                + str(self.status)
                + " while re_exec_policy is : "
                + str(self.re_exec_policy)
            )

    def __get_input_data_for_cache(self, input_data, in_names):
        """
        Prepares the input data dict for caching

        :param input_data: input data dict
        :param in_names: input data names
        :returns: a data dict
        """
        in_and_out = set(in_names) & set(self.get_output_data_names())

        cached_inputs = dict(input_data.items())
        for key in in_and_out:
            val = input_data.get(key)
            if val is not None:
                # If also an output, keeps a copy of the original input value
                cached_inputs[key] = deepcopy(val)

        return cached_inputs

    def execute(self, input_data=None):
        """Execute the discipline.

        This method executes the discipline:

        * Adds default inputs to the input_data if some inputs are not defined
           in input_data but exist in self._default_data
        * Checks if the last execution of the discipline wan not called with
           identical inputs, cached in self.cache, if yes, directly
           return self.cache.get_output_cache(inputs)
        * Caches the inputs
        * Checks the input data against self.input_grammar
        * if self.data_processor is not None: runs the preprocessor
        * updates the status to RUNNING
        * calls the _run() method, that shall be defined
        * if self.data_processor is not None: runs the postprocessor
        * checks the output data
        * Caches the outputs
        * updates the status to DONE or FAILED
        * updates summed execution time

        :param input_data: the input data dict needed to execute the
            disciplines according to the discipline input grammar
            (Default value = None)
        :type input_data: dict
        :returns: the discipline local data after execution
        :rtype: dict
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

    def __update_local_data_from_cache(self, input_data, out_cached, out_jac):
        """Update the local data from the cache.

        :param input_data: dict of inputs
        :param out_cached: outputs retrieved from the cache
        :param out_jac: jacobian retreived from the cache
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

    def __increment_n_calls(self):
        """Increment by 1 the number of calls."""
        with self._n_calls.get_lock():
            self._n_calls.value += 1

    def __increment_n_calls_lin(self):
        """Increment by 1 the number of calls."""
        with self._n_calls_linearize.get_lock():
            self._n_calls_linearize.value += 1

    def __increment_exec_time(self, t_0):
        """Increment the execution time."""
        curr_t = timer()
        with self._exec_time.get_lock():
            self._exec_time.value += curr_t - t_0

    def _retreive_diff_inouts(self, force_all=False):
        """Get the list of outputs to be differentiated wrt inputs.

        Get the list of outputs to be differentiated, depending on the
        self._differentiated_inputs and self._differentiated_inputs attributes,
        and the force_all option
        """
        if force_all:
            inputs = self.get_input_data_names()
            outputs = self.get_output_data_names()
        else:
            inputs = self._differentiated_inputs
            outputs = self._differentiated_outputs
        return inputs, outputs

    def linearize(self, input_data=None, force_all=False, force_no_exec=False):
        """Execute the linearized version of the code.

        :param input_data: the input data dict needed to execute the
            disciplines according to the discipline input grammar
        :param force_all: if False, self._differentiated_inputs and
            self.differentiated_output are used to filter
            the differentiated variables
            otherwise, all outputs are differentiated wrt all
            inputs (Default value = False)
        :param force_no_exec: if True, the discipline is not
            re executed, cache is loaded anyway
        """
        # TODO: remove the execution when no option exec_before_lin
        # is set to True
        inputs, outputs = self._retreive_diff_inouts(force_all)
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

        if approximate_jac:
            self.jac = self._jac_approx.compute_approx_jac(outputs, inputs)
        else:
            self._compute_jacobian(inputs, outputs)

        self.__increment_n_calls_lin()
        if not approximate_jac:  # Time already counted in execute()
            self.__increment_exec_time(t_0)

        self._check_jacobian_shape(inputs, outputs)
        # Cache the Jacobian matrix
        self.cache.cache_jacobian(input_data, self.get_input_data_names(), self.jac)

        return self.jac

    def set_jacobian_approximation(
        self,
        jac_approx_type=FINITE_DIFFERENCES,
        jax_approx_step=1e-7,
        jac_approx_n_processes=1,
        jac_approx_use_threading=False,
        jac_approx_wait_time=0,
    ):
        """Set the jacobian approximation method.

        Sets the linearization mode to approx_method, sets the parameters of
        the approximation for further use when calling self.linearize

        :param jac_approx_type: "complex_step" or "finite_differences"
        :param jax_approx_step: the step for finite differences or complex step
        :param jac_approx_n_processes: maximum number of processors
            on which to run
        :param jac_approx_use_threading: if True, use Threads
            instead of processes
            to parallelize the execution
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note if you want to execute the same
            discipline multiple times, you shall use multiprocessing
        :param jac_approx_wait_time: time waited between two forks of the
            process /Thread
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
        outputs=None,
        inputs=None,
        force_all=False,
        print_errors=False,
        numerical_error=EPSILON,
    ):
        """Compute the optimal finite-difference step.

        Compute the optimal step for a forward first order finite differences
        gradient approximation.
        Requires a first evaluation of perturbed functions values.
        The optimal step is reached when the truncation error
        (cut in the Taylor development),
        and the numerical cancellation errors
        (roundoff when doing f(x+step)-f(x)) are approximately equal.

        Warning: this calls the discipline execution two times
        per input variables.

        See:
        https://en.wikipedia.org/wiki/Numerical_differentiation
        and
        "Numerical Algorithms and Digital Representation", Knut Morken ,
        Chapter 11, "Numerical Differenciation"

        :param inputs: inputs wrt the linearization is made.
            If None, use differentiated inputs
        :param outputs: outputs of the linearization is made.
            If None, use differentiated outputs
        :param force_all: if True, all inputs and outputs are used
        :param print_errors: if True, displays the estimated errors
        :param numerical_error: numerical error associated to the calculation
            of f. By default Machine epsilon (appx 1e-16),
            but can be higher when
            the calculation of f requires a numerical resolution
        :returns: the estimated errors of truncation and cancelation error.
        """
        if self._jac_approx is None:
            raise ValueError(
                "set_jacobian_approximation must be called "
                + "before setting an optimal step"
            )
        inpts, outps = self._retreive_diff_inouts(force_all=force_all)
        if outputs is None or force_all:
            outputs = outps
        if inputs is None or force_all:
            inputs = inpts
        errors, steps = self._jac_approx.auto_set_step(
            outputs, inputs, print_errors, numerical_error=numerical_error
        )
        return errors, steps

    @staticmethod
    def __get_len(container):
        """Return a measure of the length of an container."""
        if container is None:
            return -1
        try:
            return len(container)
        except TypeError:
            return 1

    def _check_jacobian_shape(self, inputs, outputs):
        """Check that the jacobian is a dict of dict of 2d numpy arrays.

        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
        """
        if self.jac is None:
            raise ValueError("The discipline " + self.name + " was not linearized")
        out_set = set(outputs)
        in_set = set(inputs)
        out_jac_set = set(self.jac.keys())

        if not out_set.issubset(out_jac_set):
            msg = "Missing outputs in Jacobian of discipline "
            msg += self.name
            msg += ": " + str(out_set.difference(out_jac_set))
            raise KeyError(msg)

        for j_o in outputs:
            j_out = self.jac[j_o]
            out_dv_set = set(j_out.keys())
            output_vals = self.local_data.get(j_o)
            n_out_j = self.__get_len(output_vals)

            if not in_set.issubset(out_dv_set):
                msg = "Missing inputs " + str(in_set.difference(out_dv_set))
                msg += " in Jacobian of discipline "
                msg += self.name + ", for output : " + str(j_o)
                raise KeyError(msg)

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
                    msg = "Jacobian matrix of discipline " + str(self.name)
                    msg += " d " + j_o + "/d " + j_i
                    msg += " is not of the right shape !"
                    msg += "\nExpected : " + str((n_out_j, n_in_j))
                    msg += " got : " + str(j_mat.shape)
                    raise ValueError(msg)

        # Discard imaginary part of Jacobian
        for jac_loc in self.jac.values():
            for desv, jac_out in jac_loc.items():
                jac_loc[desv] = jac_out.real

    @property
    def cache_tol(self):
        """Accessor to the cache input tolerance."""
        return self.cache.tolerance

    @cache_tol.setter
    def cache_tol(self, cache_tol):
        """Set to the cache input tolerance.

        :param cache_tol: float, tolerance for equality
            of the inputs in the cache.
            If norm(inpt1-input2)<=cache_tol * norm(inpt1),
            the cached data for inpt1 is returned when calling
            self.execute(input2)
        """
        self._set_cache_tol(cache_tol)

    def _set_cache_tol(self, cache_tol):
        """Set to the cache input tolerance.

        To be overloaded by subclasses

        :param cache_tol: float, cache tolerance
        """
        self.cache.tolerance = cache_tol or 0.0

    @property
    def default_inputs(self):
        """Accessor to the default inputs."""
        return self._default_inputs

    @default_inputs.setter
    def default_inputs(self, default_inputs):
        """Set the default_inputs dict.

        :param default_inputs: the default_inputs of the disciplines
            it must be a dict
        """
        if not isinstance(default_inputs, dict):
            raise TypeError(
                "MDODiscipline default inputs must be a dict,"
                + " got "
                + str(type(default_inputs))
                + " instead"
            )
        self._default_inputs = default_inputs

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the jacobian matrix.

        To be overloaded by subclasses, actual computation of the jacobians

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

    def _init_jacobian(
        self, inputs=None, outputs=None, with_zeros=False, fill_missing_keys=False
    ):
        """Initialize the jacobian dict.

        :param with_zeros: if True, the matrices are set to zero
             otherwise, they are empty matrices (Default value = False)
        :type with_zeros: logical
        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        :param fill_missing_keys: if True, just fill the missing items
            with zeros/empty but dont override existing data
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
    def linearization_mode(self):
        """Accessor to the linearization mode."""
        return self._linearization_mode

    @linearization_mode.setter
    def linearization_mode(self, linearization_mode):
        """Set the linearization_mode.

        :param linearization_mode: among self.LINEARIZE_MODE_LIST
        """
        if linearization_mode not in self.AVAILABLE_MODES:
            msg = "Linearize '" + str(linearization_mode)
            msg += "'mode unknown. Must be in "
            msg += str(self.AVAILABLE_MODES)
            raise ValueError(msg)

        self._linearization_mode = linearization_mode

        if linearization_mode in self.APPROX_MODES and self._jac_approx is None:
            self.set_jacobian_approximation(linearization_mode)

    def check_jacobian(
        self,
        input_data=None,
        derr_approx=FINITE_DIFFERENCES,
        step=1e-7,
        threshold=1e-8,
        linearization_mode="auto",
        inputs=None,
        outputs=None,
        parallel=False,
        n_processes=N_CPUS,
        use_threading=False,
        wait_time_between_fork=0,
        auto_set_step=False,
        plot_result=False,
        file_path="jacobian_errors.pdf",
        show=False,
        figsize_x=10,
        figsize_y=10,
    ):
        """Check if the jacobian provided by the linearize() method is correct.

        :param input_data: input data dict (Default value = None)
        :param derr_approx: derivative approximation method: COMPLEX_STEP
            (Default value = COMPLEX_STEP)
        :param threshold: acceptance threshold for the jacobian error
            (Default value = 1e-8)
        :param linearization_mode: the mode of linearization: direct, adjoint
            or automated switch depending on dimensions
            of inputs and outputs (Default value = 'auto')
        :param inputs: list of inputs wrt which to differentiate
            (Default value = None)
        :param outputs: list of outputs to differentiate (Default value = None)
        :param step: the step for finite differences or complex step
        :param parallel: if True, executes in parallel
        :param n_processes: maximum number of processors on which to run
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note if you want to execute the same
            discipline multiple times, you shall use multiprocessing
        :param wait_time_between_fork: time waited between two forks of the
            process /Thread
        :param auto_set_step: Compute optimal step for a forward first
            order finite differences gradient approximation
        :param plot_result: plot the result of the validation (computed
            and approximate jacobians)
        :param file_path: path to the output file if plot_result is True
        :param show: if True, open the figure
        :param figsize_x: x size of the figure in inches
        :param figsize_y: y size of the figure in inches
        :returns: True if the check is accepted, False otherwise
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
        )
        return o_k

    @property
    def status(self):
        """Status accessor."""
        return self._status

    def _check_status(self, status):
        """Check the status according to possible statuses.

        Raises an exception if status is invalid

        :param status: the status to check
        :type status: string
        """
        if status not in [
            self.STATUS_PENDING,
            self.STATUS_VIRTUAL,
            self.STATUS_DONE,
            self.STATUS_RUNNING,
            self.STATUS_FAILED,
        ]:
            raise ValueError("Unknown status: " + str(status))

    def set_disciplines_statuses(self, status):
        """Set the sub disciplines statuses.

        To be implemented in subclasses.
        :param status: the status
        """

    def is_output_existing(self, data_name):
        """Test if output named  data_name is an output of the discipline.

        :param data_name: the name of the output
        :returns: True if data_name is in output grammar
        :rtype: logical
        """
        return self.output_grammar.is_data_name_existing(data_name)

    def is_all_outputs_existing(self, data_names):
        """Test if all the names in data_names are outputs of the discipline.

        :param data_names: the names of the outputs
        :returns: True if data_names are all in output grammar
        :rtype: logical
        """
        return self.output_grammar.is_all_data_names_existing(data_names)

    def is_all_inputs_existing(self, data_names):
        """Test if all the names in data_names are inputs of the discipline.

        :param data_names: the names of the inputs
        :returns: True if data_names are all in input grammar
        :rtype: logical
        """
        return self.input_grammar.is_all_data_names_existing(data_names)

    def is_input_existing(self, data_name):
        """Test if input named  data_name is an input of the discipline.

        :param data_name: the name of the output
        :returns: True if data_name is in input grammar
        :rtype: logical

        """
        return self.input_grammar.is_data_name_existing(data_name)

    def _is_status_ok_for_run_again(self, status):
        """Checks if the discipline can be run again.

        :param status: the status
        """
        return status not in [self.STATUS_RUNNING]

    def reset_statuses_for_run(self):
        """Sets all the statuses to PENDING"""
        if not self._is_status_ok_for_run_again(self.status):
            raise ValueError(
                "Cannot run discipline "
                + self.name
                + " with status "
                + str(self.status)
            )
        self.status = self.STATUS_PENDING

    @status.setter
    def status(self, status):
        """Set the statuses.

        :param status: the status
        """
        self._check_status(status)
        self._status = status
        self.notify_status_observers()

    def add_status_observer(self, obs):
        """Add an observer for the status

        Add an observer for the status to be notified when self changes of
        status.

        :param obs: the observer to add
        """
        if obs not in self._status_observers:
            self._status_observers.append(obs)

    def remove_status_observer(self, obs):
        """Remove an observer for the status.

        :param obs: the observer to remove
        """
        if obs in self._status_observers:
            self._status_observers.remove(obs)

    def notify_status_observers(self):
        """Notify all status observers that the status has changed."""
        for obs in self._status_observers[:]:
            obs.update_status(self)

    def store_local_data(self, **kwargs):
        """Store discipline data in local data.

        :param kwargs: the data as key value pairs

        """
        self.local_data.update(kwargs)

    def check_input_data(self, input_data, raise_exception=True):
        """Check the input data validity.

        :param input_data: the input data dict
        :param raise_exception: Default value = True)
        """
        try:
            self.input_grammar.load_data(input_data, raise_exception)
        except InvalidDataException:
            raise InvalidDataException("Invalid input data for: " + self.name)

    def check_output_data(self, raise_exception=True):
        """Check the output data validity.

        :param raise_exception: if true, an exception is raised
            when data is invalid (Default value = True)
        """
        try:
            self.output_grammar.load_data(self.local_data, raise_exception)
        except InvalidDataException:
            raise InvalidDataException("Invalid output data for: " + self.name)

    def get_outputs_asarray(self):
        """Accessor for the outputs as a large numpy array.

        The order is the one of self.get_all_outputs()

        :returns: the outputs array
        :rtype: ndarray
        """
        return concatenate(list(self.get_all_outputs()))

    def get_inputs_asarray(self):
        """Accessor for the outputs as a large numpy array.

        The order is the one of self.get_all_outputs().

        :returns: the outputs array
        :rtype: ndarray
        """
        return concatenate(list(self.get_all_inputs()))

    def get_inputs_by_name(self, data_names):
        """Accessor for the inputs as a list.

        :param data_names: the data names list
        :returns: the data list
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline " + str(self.name) + " has no input named: " + str(err)
            )

    def get_outputs_by_name(self, data_names):
        """Accessor for the outputs as a list.

        :param data_names: the data names list
        :returns: the data list
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline " + str(self.name) + " has no output named: " + str(err)
            )

    def get_input_data_names(self):
        """Accessor for the input names as a list.

        :returns: the data names list
        """
        return self.input_grammar.get_data_names()

    def get_output_data_names(self):
        """Accessor for the output names as a list.

        :returns: the data names list
        """
        return self.output_grammar.get_data_names()

    def get_input_output_data_names(self):
        """Accessor for the input and output names as a list.

        :returns: the data names list
        """
        outpt = self.output_grammar.get_data_names()
        inpt = self.input_grammar.get_data_names()
        return list(set(outpt) | set(inpt))

    def get_all_inputs(self):
        """Accessor for the input data as a list of values.

        The order is given by self.get_input_data_names().

        :returns: the data
        """
        return self.get_inputs_by_name(self.get_input_data_names())

    def get_all_outputs(self):
        """Accessor for the output data as a list of values.

        The order is given by self.get_output_data_names().

        :returns: the data
        """
        return self.get_outputs_by_name(self.get_output_data_names())

    def get_output_data(self):
        """Accessor for the output data as a dict of values.

        :returns: the data dict
        """
        return dict(
            (k, v) for k, v in self.local_data.items() if self.is_output_existing(k)
        )

    def get_input_data(self):
        """Accessor for the input data as a dict of values.

        :returns: the data dict
        """
        return dict(
            (k, v) for k, v in self.local_data.items() if self.is_input_existing(k)
        )

    def serialize(self, out_file):
        """Serialize the discipline.

        :param out_file: destination file for serialization
        """
        with open(out_file, "wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def deserialize(in_file):
        """Derialize the discipline from a file.

        :param in_file: input file for serialization
        :returns: a discipline instance
        """
        with open(in_file, "rb") as in_fobj:
            pickler = pickle.Unpickler(in_fobj)
            return pickler.load()

    def get_attributes_to_serialize(self):  # pylint: disable=R0201
        """Define the attributes to be serialized.

        Shall be overloaded by disciplines

        :returns: the list of attributes names
        :rtype: list
        """
        # pylint warning ==> method could be a function but when overriden,
        # it is a function==> self is required
        return [
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
            "_cache_factory",
            "_cache_tolerance",
            "_cache_hdf_node_name",
            "_linearize_on_last_state",
            "_cache_was_loaded",
            "grammar_type",
            "comp_dir",
            "exec_for_lin",
            "_in_data_hash_dict",
            "_jac_approx",
        ]

    def __getstate__(self):
        """Used by pickle to define what to serialize.

        :returns: the dict to serialize
        """
        out_d = {}
        for keep_name in self.get_attributes_to_serialize():
            if keep_name not in self.__dict__:
                if "_" + keep_name not in self.__dict__:
                    msg = "Discipline " + str(self.name)
                    msg += " defined attribute '" + str(keep_name)
                    msg += "' as required for serialization, but "
                    msg += " it appears to be undefined !"
                    raise AttributeError(msg)

                prop = self.__dict__["_" + keep_name]
                if not isinstance(prop, Synchronized):
                    raise TypeError(
                        "Cant handle attribute "
                        + str(keep_name)
                        + " serialization of undefined type"
                    )
                # Dont serialize shared memory object,
                # this is meaningless, save the value instead
                out_d[keep_name] = prop.value
            else:
                out_d[keep_name] = self.__dict__[keep_name]

        if self._cache_type == self.HDF5_CACHE:
            out_d.pop("cache")
        return out_d

    def __setstate__(self, state_dict):
        """Used by pickle to define what to deserialize.

        :param state_dict: update self dict from state_dict to deserialize
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

    def get_local_data_by_name(self, data_names):
        """Accessor for the local data of the discipline as a dict of values.

        :param data_names: the names of the data which will be the keys of the
            dictionary
        :returns: the data list
        """
        try:
            return self.get_data_list_from_dict(data_names, self.local_data)
        except KeyError as err:
            raise ValueError(
                "Discipline " + str(self.name) + " has no local_data named: " + str(err)
            )

    @staticmethod
    def is_scenario():
        """Return True if self is a scenario.

        :returns: True if self is a scenario
        """
        return False

    @staticmethod
    def get_data_list_from_dict(keys, data_dict):
        """Filter the dict from a list of keys or a single key.

        If keys is a string, then the method return the value associated to the
        key. If keys is a list of string, then the method return a generator of
        value corresponding to the keys which can be iterated.

        :param keys: a sting key or a list of keys
        :param data_dict: the dict to get the data from
        :returns: a data or a generator of data
        """
        if isinstance(keys, string_types):
            return data_dict[keys]
        return (data_dict[name] for name in keys)
