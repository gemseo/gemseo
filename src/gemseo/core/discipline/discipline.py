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
"""The discipline class."""

from __future__ import annotations

from collections import defaultdict
from multiprocessing import cpu_count
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import empty
from numpy import ndarray
from numpy import zeros
from scipy.sparse import csr_array
from strenum import StrEnum

from gemseo.core._discipline_class_injector import ClassInjector
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.approximation_modes import HybridApproximationMode
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.utils.derivatives.error_estimators import EPSILON
from gemseo.utils.enumeration import merge_enums

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from enum import EnumType
    from pathlib import Path

    from gemseo.caches.cache_entry import CacheEntry
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping


def _default_dict_factory() -> dict:
    """Instantiate a ``defaultdict(None)`` object.

    Returns:
        A ``defaultdict(None)`` object.
    """
    return defaultdict(None)


class Discipline(BaseDiscipline, metaclass=ClassInjector):
    """The base class for disciplines.

    The :meth:`.execute` method is used to do compute output data from input data.
    The :meth:`.linearize` method can be used
    to compute the Jacobian of the differentiable outputs
    with respect to differentiated inputs.
    The :attr:`.jac` stores this Jacobian.
    This method can evaluate the true derivatives (default)
    if the :meth:`_compute_jacobian` method is implemented
    or approximated the derivatives,
    depending on the :attr:`.linearization_mode`
    (one of :attr:`.LinearizationMode`).
    """

    class InitJacobianType(StrEnum):
        """The way to initialize the Jacobian matrices."""

        EMPTY = "empty"
        """Initialized as empty NumPy arrays."""

        DENSE = "dense"
        """Initialized as NumPy arrays filled with zeros."""

        SPARSE = "sparse"
        """Initialized as SciPy CSR arrays filled with zeros."""

    ApproximationMode: EnumType = merge_enums(
        "ApproximationMode", StrEnum, ApproximationMode, HybridApproximationMode
    )

    LinearizationMode: EnumType = merge_enums(
        "LinearizationMode",
        StrEnum,
        DerivationMode,
        ApproximationMode,
    )

    _linearize_on_last_state: ClassVar[bool] = False
    """Whether to update the local data from the input data before linearizing."""

    jac: JacobianData
    """The Jacobian matrices of the outputs.

    The structure is ``{output_name: {input_name: jacobian_matrix}}``.
    """

    _has_jacobian: bool
    """Whether the jacobian has been set either by :meth:`_run` or from the cache."""

    _differentiated_input_names: list[str]
    """The names of the inputs to differentiate the outputs."""

    _differentiated_output_names: list[str]
    """The names of the outputs to differentiate."""

    _jac_approx: DisciplineJacApprox | None
    """The jacobian approximator."""

    _linearization_mode: LinearizationMode
    """The linearization mode."""

    __input_names: Iterable[str]
    """The input names used for handling execution status and statistics."""

    __output_names: Iterable[str]
    """The output names used for handling execution status and statistics."""

    __hybrid_approximation_name_to_mode: ClassVar[
        Mapping[ApproximationMode, ApproximationMode]
    ] = {
        ApproximationMode.HYBRID_FINITE_DIFFERENCES: ApproximationMode.FINITE_DIFFERENCES,  # noqa: E501
        ApproximationMode.HYBRID_CENTERED_DIFFERENCES: ApproximationMode.CENTERED_DIFFERENCES,  # noqa: E501
        ApproximationMode.HYBRID_COMPLEX_STEP: ApproximationMode.COMPLEX_STEP,
    }

    def __init__(  # noqa: D107
        self,
        name: str = "",
    ) -> None:
        super().__init__(name)
        self._differentiated_input_names = []
        self._differentiated_output_names = []
        self._jac_approx = None
        self._linearization_mode = self.LinearizationMode.AUTO
        self._has_jacobian = False
        self.jac = {}

    @property
    def linearization_mode(self) -> LinearizationMode:
        """The differentiation mode."""
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

    def linearize(
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        compute_all_jacobians: bool = False,
        execute: bool = True,
    ) -> JacobianData:
        """Compute the Jacobians of some outputs with respect to some inputs.

        Args:
            input_data: The input data.
                If empty, use the :attr:`.`default_input_data`.
            compute_all_jacobians: Whether to compute the Jacobians of
                all the outputs with respect to all the inputs.
                Otherwise,
                set the output variables to differentiate
                with :meth:`.add_differentiated_outputs`
                and the input variables with respect to which to differentiate them
                with :meth:`.add_differentiated_inputs`.
            execute: Whether to start by executing the discipline
                to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.

        Returns:
            The Jacobian matrices
            in the dictionary form ``{output_name: {input_name: jacobian_matrix}}``
            where ``jacobian_matrix[i, j]`` is
            the partial derivative of ``output_name[i]`` wrt ``input_name[j]``.

        Raises:
            ValueError: When either the inputs
                for which to differentiate the outputs
                or the outputs to differentiate are missing.
        """
        input_data = self.io.prepare_input_data(input_data)

        input_names, output_names = self._get_differentiated_io(compute_all_jacobians)

        if self.cache is not None and not (input_names and output_names):
            self.jac = self.cache[input_data].jacobian
            return self.jac

        if execute:
            self.execute(input_data)

        if not self._linearize_on_last_state:
            # The data shall be reset to their original values
            # in case an input is also an output,
            # if we don't want to keep the computed state (as in MDAs).
            self.io.data.update(input_data)

        # TODO: that should be before the previous bloc,
        # but a test_parallel_chain_combinatorial_thread fails,
        # copy above bloc into MDOParallelChain linearization?
        if self._has_jacobian and self.jac:
            # For cases when linearization is called twice with different I/O
            # while cache_was_loaded=True,
            # the check_jacobian_shape raises a KeyError.
            try:
                self._check_jacobian_shape(input_names, output_names)
            except KeyError:
                # In this case, another computation of Jacobian is triggered.
                pass
            else:
                return self.jac

        self.__input_names = input_names
        self.__output_names = output_names
        self.execution_status.handle(
            self.execution_status.Status.LINEARIZING,
            self.execution_statistics.record_linearization,
            self.__compute_jacobian,
        )

        if not compute_all_jacobians:
            for output_name in tuple(self.jac.keys()):
                if output_name not in output_names:
                    del self.jac[output_name]
                else:
                    jac = self.jac[output_name]
                    for input_name in list(jac.keys()):
                        if input_name not in input_names:
                            del jac[input_name]

        # The check of the jacobian shape is required only when some of its
        # components are requested.
        if input_names and output_names:
            self._check_jacobian_shape(input_names, output_names)

        if self.cache is not None:
            self.cache.cache_jacobian(input_data, self.jac)

        return self.jac

    def __compute_jacobian(self):
        """Callable used for handling execution status and statistics."""
        if self._linearization_mode in set(self.ApproximationMode):
            if self._linearization_mode in set(HybridApproximationMode):
                self._compose_hybrid_jacobian(self.__output_names, self.__input_names)
            else:
                self.jac = self._jac_approx.compute_approx_jac(
                    self.__output_names, self.__input_names
                )
        else:
            self._compute_jacobian(self.__input_names, self.__output_names)

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
        when calling :meth:`.Discipline.linearize`.

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
        approx_method = (
            self.__hybrid_approximation_name_to_mode[jac_approx_type]
            if jac_approx_type in set(HybridApproximationMode)
            else jac_approx_type
        )
        self._jac_approx = DisciplineJacApprox(
            # TODO: pass the bare minimum instead of self.
            self,
            approx_method=approx_method,
            step=jax_approx_step,
            parallel=jac_approx_n_processes > 1,
            n_processes=jac_approx_n_processes,
            use_threading=jac_approx_use_threading,
            wait_time_between_fork=jac_approx_wait_time,
        )
        self._linearization_mode = jac_approx_type

    def set_optimal_fd_step(
        self,
        output_names: Iterable[str] = (),
        input_names: Iterable[str] = (),
        compute_all_jacobians: bool = False,
        print_errors: bool = False,
        numerical_error: float = EPSILON,
    ) -> tuple[ndarray, dict[str, ndarray]]:
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
            input_names: The inputs with respect to which the outputs are linearized.
                If empty, use the differentiated inputs defined by
                :meth:`.add_differentiated_inputs`.
            output_names: The outputs to be linearized.
                If empty, use the outputs defined by
                :meth:`.add_differentiated_outputs`.
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables
                with respect to which to differentiate the output ones
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
            msg = (
                "set_jacobian_approximation must be called "
                "before setting an optimal step."
            )
            raise ValueError(msg)
        diff_inputs, diff_outputs = self._get_differentiated_io(
            compute_all_jacobians=compute_all_jacobians
        )
        if not output_names or compute_all_jacobians:
            output_names = diff_outputs
        if not input_names or compute_all_jacobians:
            input_names = diff_inputs
        return self._jac_approx.auto_set_step(
            output_names, input_names, print_errors, numerical_error=numerical_error
        )

    def _check_jacobian_shape(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> None:
        """Check that the Jacobian is a dictionary of dictionaries of 2D NumPy arrays.

        Args:
            input_names: The input names wrt the output names are linearized.
            output_names: The output names to be linearized.

        Raises:
            ValueError:
                When the discipline was not linearized.
                When the Jacobian is not of the right shape.
            KeyError:
                When output names are missing in the Jacobian of the discipline.
                When input names are missing for an output in the Jacobian of the
                discipline.
        """
        if not self.jac:
            msg = f"The discipline {self.name} was not linearized."
            raise ValueError(msg)

        unique_output_names = set(output_names)
        unique_input_names = set(input_names)
        jac_output_names = self.jac.keys()

        if not unique_output_names.issubset(jac_output_names):
            msg = (
                f"Missing output names in Jacobian of discipline {self.name}: "
                f"{unique_output_names.difference(jac_output_names)}."
            )
            raise KeyError(msg)

        get_input_size = self.io.input_grammar.data_converter.get_value_size
        get_output_size = self.io.output_grammar.data_converter.get_value_size

        for output_name in output_names:
            jac_output = self.jac[output_name]

            if not unique_input_names.issubset(jac_output.keys()):
                msg = (
                    f"Missing input names {unique_input_names.difference(jac_output)} "
                    f"in Jacobian of discipline {self.name}, "
                    f"for output: {output_name}."
                )
                raise KeyError(msg)

            output_value = self.io.data.get(output_name)
            if output_value is None:
                # Unknown dimension, don't check the shape.
                continue

            output_size = get_output_size(output_name, output_value)

            for input_name in input_names:
                input_value = self.io.data.get(input_name)
                if input_value is None:
                    # Unknown dimension, don't check the shape.
                    continue

                input_size = get_input_size(input_name, input_value)

                if jac_output[input_name].shape != (output_size, input_size):
                    msg = (
                        f"The shape {jac_output[input_name].shape} "
                        f"of the Jacobian matrix d{output_name}/d{input_name} "
                        f"of the discipline {self.name} "
                        "does not match "
                        f"(output_size, input_size)=({output_size}, {input_size})."
                    )
                    raise ValueError(msg)

        # Discard imaginary part of Jacobian
        for output_jacobian in self.jac.values():
            for input_name, input_output_jacobian in output_jacobian.items():
                output_jacobian[input_name] = input_output_jacobian.real

    def _init_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        init_type: InitJacobianType = InitJacobianType.DENSE,
        fill_missing_keys: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Initialize the Jacobian dictionary :attr:`.jac`.

        Args:
            input_names: The inputs with respect to which to differentiate the outputs.
                If empty, use all the inputs.
            output_names: The outputs to be differentiated.
                If empty, use all the outputs.
            init_type: The type used to initialize the Jacobian matrices.
            fill_missing_keys: Whether to just fill the missing items with zeros/empty
                but do not override the existing data.

        Returns:
            The names of the input variables
            with respect to which to differentiate the output ones,
            and these output variables.
        """
        if init_type == self.InitJacobianType.EMPTY:
            default_matrix = empty
        elif init_type == self.InitJacobianType.DENSE:
            default_matrix = zeros
        elif init_type == self.InitJacobianType.SPARSE:
            default_matrix = csr_array
        else:
            # Cast the argument so that the enum class raise an explicit error.
            self.InitJacobianType(init_type)

        input_names = input_names or self._differentiated_input_names
        input_names_to_sizes = (
            self.io.input_grammar.data_converter.compute_names_to_sizes(
                input_names,
                self.io.data,
            )
        )

        output_names = output_names or self._differentiated_output_names
        output_names_to_sizes = (
            self.io.output_grammar.data_converter.compute_names_to_sizes(
                output_names,
                self.io.data,
            )
        )

        if fill_missing_keys:
            jac = self.jac
            # Only fill the missing sub jacobians
            for output_name, output_size in output_names_to_sizes.items():
                jac_loc = jac.setdefault(output_name, defaultdict(None))
                for input_name, input_size in input_names_to_sizes.items():
                    sub_jac = jac_loc.get(input_name)
                    if sub_jac is None:
                        jac_loc[input_name] = default_matrix((output_size, input_size))
        else:
            # When a key is not in the default dict, ie a function is not in
            # the Jacobian; return an empty defaultdict(None)
            jac = defaultdict(_default_dict_factory)
            if input_names:
                for output_name, output_size in output_names_to_sizes.items():
                    jac_loc = jac[output_name]
                    for input_name, input_size in input_names_to_sizes.items():
                        jac_loc[input_name] = default_matrix((output_size, input_size))
            self.jac = jac

        return input_names, output_names

    def _prepare_io_for_check_jacobian(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> tuple[Iterable[str], Iterable[str]]:
        if not input_names:
            input_names = self.io.input_grammar
        if not output_names:
            output_names = self.io.output_grammar
        return input_names, output_names

    def check_jacobian(
        self,
        input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
        derr_approx: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-7,
        threshold: float = 1e-8,
        linearization_mode: LinearizationMode = LinearizationMode.AUTO,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        parallel: bool = False,
        n_processes: int = cpu_count(),
        use_threading: bool = False,
        wait_time_between_fork: float = 0,
        auto_set_step: bool = False,
        plot_result: bool = False,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10,
        fig_size_y: float = 10,
        reference_jacobian_path: str | Path = "",
        save_reference_jacobian: bool = False,
        indices: Mapping[
            str, int | Sequence[int] | Ellipsis | slice
        ] = READ_ONLY_EMPTY_DICT,
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
                If ``None``, use the :attr:`.Discipline.io.input_grammar.defaults`.
            derr_approx: The approximation method,
                either "complex_step" or "finite_differences".
            threshold: The acceptance threshold for the Jacobian error.
            linearization_mode: the mode of linearization: direct, adjoint
                or automated switch depending on dimensions
                of inputs and outputs (Default value = 'auto')
            input_names: The names of the inputs wrt which to differentiate the outputs.
            output_names: The names of the outputs to be differentiated.
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
                If ``None``,
                consider all the components of all the ``inputs`` and ``outputs``.

        Returns:
            Whether the analytical Jacobian is correct
            with respect to the reference one.
        """
        # Do not use self._jac_approx because we may want to check  complex
        # step approximation with the finite differences for instance
        input_names, output_names = self._prepare_io_for_check_jacobian(
            input_names, output_names
        )

        # Differentiate analytically
        self.add_differentiated_inputs(input_names)
        self.add_differentiated_outputs(output_names)
        self.linearization_mode = linearization_mode

        approx = DisciplineJacApprox(
            self,
            derr_approx,
            step,
            parallel,
            n_processes,
            use_threading,
            wait_time_between_fork,
        )

        if auto_set_step:
            approx.auto_set_step(output_names, input_names)

        self.linearize(input_data)

        return approx.check_jacobian(
            output_names,
            input_names,
            threshold=threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            fig_size_x=fig_size_x,
            fig_size_y=fig_size_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
            input_data=input_data,
        )

    def _get_differentiated_io(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the inputs and outputs used in the differentiation of the discipline.

        Args:
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables
                with respect to which to differentiate the output ones
                with :meth:`.add_differentiated_inputs`
                and set these output variables to differentiate
                with :meth:`.add_differentiated_outputs`.
        """
        if compute_all_jacobians:
            return (
                tuple(self.io.input_grammar),
                tuple(self.io.output_grammar),
            )

        return tuple(self._differentiated_input_names), tuple(
            self._differentiated_output_names
        )

    def add_differentiated_inputs(
        self,
        input_names: Iterable[str] = (),
    ) -> None:
        """Add the inputs with respect to which to differentiate the outputs.

        The inputs that do not represent continuous numbers are filtered out.

        Args:
            input_names: The input variables
                with respect to which to differentiate the outputs.
                If empty, use all the inputs.

        Raises:
            ValueError: When an input name is not the name of a discipline input.
        """
        input_grammar = self.io.input_grammar

        if input_names and not self.io.input_grammar.has_names(input_names):
            msg = (
                f"Cannot differentiate the discipline {self.name} w.r.t. the inputs "
                "that are not among the discipline inputs: "
                f"{list(input_grammar)}."
            )
            raise ValueError(msg)

        if not input_names:
            input_names = input_grammar

        self._differentiated_input_names = list(
            set(self._differentiated_input_names).union(
                filter(input_grammar.data_converter.is_continuous, input_names)
            )
        )

    def add_differentiated_outputs(
        self,
        output_names: Iterable[str] = (),
    ) -> None:
        """Add the outputs to be differentiated.

        The outputs that do not represent continuous numbers are filtered out.

        Args:
            output_names: The outputs to be differentiated.
                If empty, use all the outputs.

        Raises:
            ValueError: When an output name is not the name of a discipline output.
        """
        output_grammar = self.io.output_grammar

        if output_names and not self.io.output_grammar.has_names(output_names):
            msg = (
                f"Cannot differentiate the discipline {self.name} for variables "
                "that are not among the discipline outputs: "
                f"{list(output_grammar)}."
            )
            raise ValueError(msg)

        if not output_names:
            output_names = output_grammar

        self._differentiated_output_names = list(
            set(self._differentiated_output_names).union(
                filter(output_grammar.data_converter.is_continuous, output_names)
            )
        )

    def _store_cache(self, input_data: DisciplineData) -> None:
        super()._store_cache(input_data)
        if self._has_jacobian:
            self.cache.cache_jacobian(input_data, self.jac)

    def _set_data_from_cache(self, cache_entry: CacheEntry) -> None:
        super()._set_data_from_cache(cache_entry)
        self._has_jacobian = True
        if cache_entry.jacobian:
            self.jac = cache_entry.jacobian
        else:
            # TODO: This is required to pass all the tests instead of self.jac.clear(),
            #  there is an implicit side effect in how this attr is used,
            #  this should be made explicit.
            self.jac = {}

    def execute(  # noqa: D102
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> DisciplineData:
        # TODO: investigate the side effects in linearize that prevents clearing jac.
        # self.jac.clear()
        self._has_jacobian = False
        return super().execute(input_data)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute one Jacobian matrix per input-output pair.

        Store the result in :attr:`.jac`
        as a dictionary ``{output_name: {input_name: jacobian_matrix}}``.

        Args:
            input_names: The names of the inputs
                with respect to which to differentiate the outputs.
                If empty, use all the inputs.
            output_names: The names of the outputs to be differentiated.
                If empty, use all the outputs.
        """

    def _compose_hybrid_jacobian(
        self,
        output_names: Iterable[str] = (),
        input_names: Iterable[str] = (),
    ) -> None:
        """Compose a hybrid Jacobian using analytical and approximated expressions.

        This method allows to complete a given Jacobian for which not all
        inputs-outputs have been defined by using approximation methods.

        Args:
            input_names: The names of the input wrt the ``output_names`` are
                linearized.
            output_names: The names  of the output to be linearized.
        """
        analytical_jacobian = self.jac

        self.set_jacobian_approximation(self._linearization_mode)

        approximated_jac = {}
        outputs_names_to_approximate = []
        input_names_to_approximate = []

        for output_name in output_names:
            if output_name not in analytical_jacobian:
                analytical_jacobian[output_name] = {}
            for input_name in input_names:
                if input_name not in analytical_jacobian[output_name]:
                    approximated_jac[output_name] = {}
                    # Map the outputs to be differentiated wrt the corresponding inputs.
                    outputs_names_to_approximate.append(output_name)
                    input_names_to_approximate.append(input_name)

        # Compute approximated Jacobian elements.
        for output_name, input_name in zip(
            outputs_names_to_approximate, input_names_to_approximate
        ):
            jac_input_output = self._jac_approx.compute_approx_jac(
                [output_name], [input_name]
            )
            approximated_jac[output_name][input_name] = jac_input_output[output_name][
                input_name
            ]

        # Fill in missing inputs of the Jacobian.
        for output_name in output_names:
            analytical_jacobian_out = analytical_jacobian[output_name]
            approximated_jac_out = approximated_jac[output_name]
            for input_name in input_names:
                if input_name not in analytical_jacobian_out:
                    analytical_jacobian_out[input_name] = approximated_jac_out[
                        input_name
                    ]

        # Recover analytical Jacobian data.
        self.jac = analytical_jacobian
