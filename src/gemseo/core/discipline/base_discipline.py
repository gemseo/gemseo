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
"""The base class defining the concept of discipline."""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from strenum import StrEnum

from gemseo.caches.cache_entry import CacheEntry
from gemseo.caches.factory import CacheFactory
from gemseo.caches.simple_cache import SimpleCache
from gemseo.core._base_monitored_process import BaseMonitoredProcess
from gemseo.core._process_flow.base_flow import BaseFlow
from gemseo.core.discipline.io import IO
from gemseo.core.grammars.factory import GrammarType as _GrammarType
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.caches.base_cache import BaseCache
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.core.grammars.base_grammar import BaseGrammar
    from gemseo.core.grammars.grammar_properties import GrammarProperties
    from gemseo.typing import MutableStrKeyMapping
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)

_CACHE_FACTORY = CacheFactory()


class CacheType(StrEnum):
    """The types of cache."""

    SIMPLE = "SimpleCache"
    """Store the last execution data."""

    HDF5 = "HDF5Cache"
    """Store all the execution data on the disk."""

    MEMORY_FULL = "MemoryFullCache"
    """Store all the execution data in memory."""

    NONE = ""
    """Store nothing."""


class BaseDiscipline(BaseMonitoredProcess):
    """The base class defining the concept of discipline.

    A discipline computes output data from input data
    using its :meth:`.execute` method.
    These data are in dictionary form,
    i.e. ``{variable_name: variable_value, ...}``.
    The input-output data resulting from an execution
    can be accessed via :attr:`.local_data`
    or separately via :meth:`.get_input_data` and :meth:`.get_output_data`.

    For both input and output variables,
    default values can be provided
    using the mappings :attr:`.default_input_data` and :attr:`.default_output_data`.
    In this case,
    the discipline will use these default input values at execution
    when an input value is not provided
    and these default output values in the case of :attr:`.virtual_execution`.

    In other aspects,
    the :attr:`.cache` can store zero, one or more discipline evaluations
    depending on the :attr:`.CacheType`.
    This cache is set at instantiation
    and can be changed with the :meth:`.set_cache` method.

    Lastly,
    a discipline is equipped with
    an :attr:`.input_grammar` to check the input data
    and an :attr:`.output_grammar` to check the output data.
    This validation depends on the :class:`.GrammarType`,
    e.g. name verification, data type verification, etc.
    """

    GrammarType: ClassVar[type[_GrammarType]] = _GrammarType
    """The types of grammar."""

    GRAMMAR_DIRECTORY: ClassVar[str | Path] = ""
    """The directory in which to search for the grammar files if not the class one."""

    default_grammar_type: ClassVar[_GrammarType] = GrammarType.JSON
    """The default type of grammar."""

    auto_detect_grammar_files: ClassVar[bool] = False
    """Whether to find the grammar files automatically."""

    validate_input_data: ClassVar[bool] = True
    """Whether to validate the input data."""

    validate_output_data: ClassVar[bool] = True
    """Whether to validate the output data."""

    virtual_execution: ClassVar[bool] = False
    """Whether :meth:`.execute` returns the :attr:`.default_output_data`.

    A virtual execution mocks the input-output process without performing the true
    execution.
    """

    CacheType: ClassVar[type[CacheType]] = CacheType
    """The type of cache."""

    default_cache_type: ClassVar[CacheType] = CacheType.SIMPLE
    """The default type of cache."""

    cache: BaseCache | None
    """The execution and linearization data saved according to the cache type."""

    _process_flow_class: ClassVar[type[BaseFlow]] = BaseFlow
    """The class used to create the process flow."""

    def __init__(
        self,
        name: str = "",
    ) -> None:
        """
        Args:
            name: The name of the discipline.
                If empty, use the name of the class.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        self.cache = None
        self.set_cache(self.default_cache_type)
        self.io = IO(
            self.__class__,
            self.name,
            self.default_grammar_type,
            self.auto_detect_grammar_files,
            self.GRAMMAR_DIRECTORY,
        )

    def _get_string_representation(self) -> MultiLineString:
        mls = MultiLineString()
        mls.add(self.name)
        mls.indent()
        mls.add(
            "Inputs: {}",
            pretty_str(self.io.input_grammar.keys()),
        )
        mls.add(
            "Outputs: {}",
            pretty_str(self.io.output_grammar.keys()),
        )
        return mls

    def _store_cache(self, input_data: StrKeyMapping) -> None:
        """Store the output data in the cache.

        Args:
            input_data: The input data.
        """
        output_grammar = self.io.output_grammar

        if not output_grammar:
            return

        output_data = self.io.data.copy()
        for name in output_data.keys() - output_grammar:
            del output_data[name]

        # Non simple caches require NumPy arrays.
        if not isinstance(self.cache, SimpleCache):
            to_array = output_grammar.data_converter.convert_value_to_array
            for name, value in output_data.items():
                output_data[name] = to_array(name, value)

        self.cache.cache_outputs(input_data, output_data)  # type: ignore[union-attr]  # because cache is checked to be not None in the caller

    def __create_input_data_for_cache(
        self,
        input_data: StrKeyMapping,
    ) -> StrKeyMapping:
        """Prepare the input data for caching.

        Args:
            input_data: The original input data.

        Returns:
            The input data to be cached.
        """
        input_data_ = input_data.copy()

        # Deepcopy the auto coupled data.
        auto_coupled_names = set(self.io.input_grammar).intersection(
            self.io.output_grammar
        )

        for auto_coupled_name in auto_coupled_names:
            value = input_data.get(auto_coupled_name)
            if value is not None:
                input_data_[auto_coupled_name] = deepcopy(value)

        # Non simple caches require NumPy arrays.
        if not isinstance(self.cache, SimpleCache):
            to_array = self.io.input_grammar.data_converter.convert_value_to_array
            for input_name, value in input_data_.items():
                input_data_[input_name] = to_array(input_name, value)

        return input_data_

    def _set_data_from_cache(self, cache_entry: CacheEntry) -> None:
        """Update the local data from a cache entry.

        Args:
            cache_entry: The cache entry.
        """
        self.io.data = cache_entry.inputs
        self.io.data.update(cache_entry.outputs)

    def _can_load_cache(self, input_data: StrKeyMapping) -> bool:
        """Search and load the cached output data from input data.

        On cache hit, the local data are restored from the cached output data.

        Args:
            input_data: The input data.

        Returns:
            Whether the output data was in the cache.
        """
        cache_entry = self.cache[input_data]

        if not cache_entry.outputs:
            return False

        # Non simple caches require NumPy arrays.
        if not isinstance(self.cache, SimpleCache):
            # Do not modify the cache entry which is mutable.
            cache_output = cache_entry.outputs.copy()
            to_value = self.io.output_grammar.data_converter.convert_array_to_value
            for output_name, value in cache_output.items():
                cache_output[output_name] = to_value(output_name, value)
        else:
            cache_output = cache_entry.outputs

        # TODO: Fix this workaround for input_data that does not match strictly
        #  the cache one.
        cache_entry = CacheEntry(input_data, cache_output, cache_entry.jacobian)

        self._set_data_from_cache(cache_entry)

        return True

    def set_cache(
        self,
        cache_type: CacheType,
        tolerance: float = 0.0,
        **kwargs: Any,
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

        Args:
            cache_type: The type of cache.
            tolerance: The cache tolerance.
            **kwargs: The other arguments passed to :meth:`.CacheFactory.create`

        .. warning:

           If is_memory_shared is set to False,
           and multiple disciplines point
           to the same cache or the process is multiprocessed,
           there may be duplicate computations
           because the cache will not be shared among the processes.
        """
        if cache_type == self.CacheType.NONE:
            self.cache = None
            return

        if (
            self.cache is None
            or self.cache.__class__.__name__ != cache_type
            or not (
                cache_type == self.CacheType.HDF5
                and kwargs.get("hdf_file_path", "") == self.cache.hdf_file.hdf_file_path
                and kwargs.get("hdf_node_path", "") == self.cache.hdf_node_path
            )
        ):
            if cache_type == self.CacheType.HDF5:
                kwargs.setdefault("hdf_node_path", self.name)
            self.cache = _CACHE_FACTORY.create(
                cache_type, tolerance=tolerance, **kwargs
            )
        else:
            LOGGER.warning(
                (
                    "The cache policy is already set to %s "
                    "with the file path %r and node name %r; "
                    "call discipline.cache.clear() to clear the cache."
                ),
                cache_type,
                kwargs.get("hdf_file_path", ""),
                kwargs.get("hdf_node_path", ""),
            )

    def execute(
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> DisciplineData:
        """Execute the discipline, i.e. compute output data from input data.

        If :attr:`.virtual_execution` is ``True``,
        this method returns the :attr:`.default_output_data`.
        Otherwise,
        it calls the :meth:`._run` method performing the true execution
        and returns the corresponding output data.
        This :meth:`._run` method must be implemented in subclasses.

        Args:
            input_data: The input data.
                Complete this dictionary with the :attr:`.default_input_data`.

        Returns:
            The input and output data.
        """
        input_data = self.io.prepare_input_data(input_data)

        if self.cache is not None:
            if self._can_load_cache(input_data):
                if self.validate_output_data:
                    self.io.output_grammar.validate(self.io.data)
                return self.io.data

            # Keep a pristine copy of the input data before it is eventually changed.
            input_data_for_cache = self.__create_input_data_for_cache(input_data)

        self.io.initialize(input_data, self.validate_input_data)

        if self.virtual_execution:
            self.io.update_output_data(self.io.output_grammar.defaults)
        else:
            self._execute_monitored()

        self.io.finalize(self.validate_output_data)

        if self.cache is not None:
            self._store_cache(input_data_for_cache)

        return self.io.data

    def _execute(self) -> None:
        if self.io.input_grammar.to_namespaced:
            input_data = self.io.get_input_data(with_namespaces=False)
        else:
            # No namespaces, avoid useless processing.
            input_data = self.io.data

        data_processor = self.io.data_processor
        if data_processor is not None:
            input_data = data_processor.pre_process_data(input_data)

        output_data = self._run(input_data=input_data)

        if output_data is not None:
            if data_processor is not None:
                output_data = data_processor.post_process_data(output_data)

            self.io.update_output_data(output_data)

    @abstractmethod
    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        """Compute the outputs from the inputs.

        This method shall be implemented in derived classes.

        The ``input_data`` are the discipline inputs completed with the default inputs.
        This method may return the output data.

        These input and output data are dictionaries
        of the form ``{variable_name_without_namespace: variable_value, ...}``.

        Using the provided ``input_data`` and also returning the output data
        will ensure that the discipline can be used with namespaces.
        This approach, which appeared in the version 6 of |g|, is preferable.

        As in the |g| versions prior to 6,
        you can also avoid using ``input_data`` and return output data,
        and thus leave the body ``_run`` unchanged.
        But in that case
        the discipline does not automatically support the use of namespaces.
        For this reason,
        it is preferable to use the first approach.

        Args:
            input_data: The input data without namespace prefixes.

        Returns:
            Eventually the output data.
        """

    # The following methods provide easier access to attributes of sub-objects.

    @property
    def input_grammar(self) -> BaseGrammar:
        """The input grammar."""
        return self.io.input_grammar

    @input_grammar.setter
    def input_grammar(self, grammar: BaseGrammar) -> None:
        self.io.input_grammar = grammar

    @property
    def default_input_data(self) -> GrammarProperties:
        """The default input data."""
        return self.io.input_grammar.defaults

    @default_input_data.setter
    def default_input_data(self, data: StrKeyMapping) -> None:
        self.io.input_grammar.defaults = data

    def add_namespace_to_input(self, input_name: str, namespace: str) -> None:
        """Rename an input name with a namespace prefix.

        The updated input name will be
        ``namespace``
        + :data:`~gemseo.core.namespaces.namespaces_separator`
        + ``input_name``.

        Args:
            input_name: The input name to rename.
            namespace: The name of the namespace.
        """
        self.io.input_grammar.add_namespace(input_name, namespace)

    def get_input_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the input data of the last execution.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The input data of the last execution.
        """
        return self.io.get_input_data(with_namespaces)

    @property
    def output_grammar(self) -> BaseGrammar:
        """The output grammar."""
        return self.io.output_grammar

    @output_grammar.setter
    def output_grammar(self, grammar: BaseGrammar) -> None:
        self.io.output_grammar = grammar

    @property
    def default_output_data(self) -> GrammarProperties:
        """The default output data used when :attr:`.virtual_execution` is ``True``."""
        return self.io.output_grammar.defaults

    @default_output_data.setter
    def default_output_data(self, data: MutableStrKeyMapping) -> None:
        self.io.output_grammar.defaults = data

    def add_namespace_to_output(self, output_name: str, namespace: str) -> None:
        """Rename an output name with a namespace prefix.

        The updated output name will be
        ``namespace``
        + :data:`~gemseo.core.namespaces.namespaces_separator`
        + ``output_name``.

        Args:
            output_name: The output name to rename.
            namespace: The name of the namespace.
        """
        self.io.output_grammar.add_namespace(output_name, namespace)

    def get_output_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the output data of the last execution.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The output data of the last execution.
        """
        return self.io.get_output_data(with_namespaces)

    @property
    def local_data(self) -> DisciplineData:
        """The current input and output data."""
        return self.io.data

    @local_data.setter
    def local_data(self, data: MutableStrKeyMapping) -> None:
        self.io.data = data
