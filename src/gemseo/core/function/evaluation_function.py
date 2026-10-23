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

"""A function recording its evaluations in a database."""

from __future__ import annotations

from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.function._nan import check_blocks_for_nan
from gemseo.core.function._nan import check_for_nan
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.base_problem_function import BaseProblemFunction
from gemseo.core.problem.database import Database
from gemseo.core.serializable import Serializable
from gemseo.util._compatibility.scipy import sparse_classes
from gemseo.util.constant import _enable_function_statistics
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.derivative.approximator.factory import GradientApproximatorFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from gemseo.space.design import DesignSpace
    from gemseo.space.transformation.base import BaseSpaceTransformation
    from gemseo.util.derivative.approximation_mode import ApproximationMode
    from gemseo.util.typing import NumberArray


class EvaluationFunction(BaseProblemFunction, Serializable):
    """A function evaluating another one for a problem.

    This half evaluates a function of a problem,
    and it works in the **original** coordinates:
    the values it looks up and stores are the ones the user declared,
    and so are the keys of the database.
    Adapting a point of the working coordinates is the business of the other half.

    It is also where an approximated Jacobian is computed,
    because its result must reach the database.
    The perturbations it evaluates must **not**:
    the approximator is built on the callable that does not record,
    so an approximated Jacobian adds one gradient entry to the database
    and its perturbations add none.
    Once
    [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
    hands this half a transformation,
    the perturbations are taken in the working coordinates instead,
    the step is the one the algorithm works with,
    and each perturbed point is mapped back through the transformation
    before reaching the wrapped function,
    so it is rounded when that backward map rounds.

    A `None` database disables the lookup and the store, and nothing else.
    Counting the calls and approximating the derivatives are unaffected by it.
    """  # noqa: E501

    enable_statistics: ClassVar[bool] = _enable_function_statistics
    """Whether to count the calls of the functions."""

    pre_compute_at_new_point: Callable[[], None] | None
    """The callable to execute when reaching a point absent from the database."""

    _database: Database | None
    """The database recording the evaluations, if any."""

    __store_jacobian: bool
    """Whether to record the Jacobian matrices."""

    __transformation: BaseSpaceTransformation | None
    """The map whose working coordinates an approximated Jacobian perturbs in.

    `None` when the perturbations are taken in the original coordinates,
    either because the Jacobian is not approximated
    or because
    [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
    was not told one.
    """  # noqa: E501

    __differentiation_method: ApproximationMode | None
    """The method approximating the Jacobian.

    `None` to use the derivatives of the wrapped function instead.
    """

    __differentiation_method_options: dict[str, Any]
    """The options of the differentiation method."""

    __design_space: DesignSpace | None
    """The design space bounding the perturbations of an approximated Jacobian.

    `None` when the input space is not a design space, e.g. a `RandomSpace`, in
    which case the perturbations are not bounded.
    Bounds them while no transformation is set;
    once
    [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
    is told one, the working space it maps to bounds them instead.
    """  # noqa: E501

    __approximate_jacobian: Callable[[NumberArray], NumberArray] | None
    """The approximator of the Jacobian, cached after it is first used.

    `None` before it is built,
    and forever when the Jacobian is not approximated,
    since it is then never asked for.
    Invalidated by
    [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation],
    since the callable it perturbs and the space it checks bounds against
    depend on the transformation.
    """  # noqa: E501

    _stop_if_nan: bool
    """Whether the evaluation stops when the function returns a `NaN`."""

    _owner: Any
    """The problem that installed this wrapper.

    It is the problem itself,
    not its database,
    so that it is still recognized after
    [database][gemseo.core.problem.evaluation.EvaluationProblem.database]
    is replaced with a new one:
    the problem rebuilds past the wrapper it installed itself
    rather than past one a caller passed.
    """

    def __init__(
        self,
        function: ArrayFunction,
        database: Database | None = None,
        store_jacobian: bool = True,
        support_sparse_jacobian: bool = True,
        vectorize: bool = False,
        design_space: DesignSpace | None = None,
        stop_if_nan: bool = True,
        owner: Any = None,
        differentiation_method: ApproximationMode | None = None,
        differentiation_method_options: Mapping[str, Any] = read_only_empty_dict,
    ) -> None:
        """
        Args:
            function: The function whose evaluations are recorded.
            database: The database to record the evaluations in.
                If `None`, neither look up nor store;
                this is what a driver disabling the database asks for.
            store_jacobian: Whether to record the Jacobian matrices.
            support_sparse_jacobian: Whether the driver supports a sparse Jacobian.
                When it does not, a sparse Jacobian is densified
                before being recorded,
                so the database holds what the driver reads.
            vectorize: Whether the function is evaluated on a matrix of samples
                rather than on a single point,
                which a vectorized DOE asks for.
            design_space: The design space bounding the perturbations of an
                approximated Jacobian while
                [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
                is told no transformation.
                `None` when the input space is not a design space,
                e.g. a `RandomSpace`,
                in which case the perturbations are not bounded.
            stop_if_nan: Whether the evaluation stops
                when the function returns a `NaN`.
            owner: The problem installing this wrapper.
                If `None`, use `database`;
                a problem passes itself
                even when it asks for no lookup,
                so that it recognizes its wrapper
                whatever its database is set to afterward.
            differentiation_method: The method approximating the Jacobian.
                If `None`, use the derivatives of the wrapped function.
            differentiation_method_options: The options of that method.
        """  # noqa: D205, D212, E501
        self._init_shared_memory_attrs_before()
        self._database = database
        self.__store_jacobian = store_jacobian
        self.__support_sparse_jacobian = support_sparse_jacobian
        self._vectorize = vectorize
        self.stop_if_nan = stop_if_nan
        self._owner = database if owner is None else owner
        self.pre_compute_at_new_point = None
        self._gradient_name = Database.get_gradient_name(function.name)

        self.__design_space = design_space
        self.__differentiation_method = differentiation_method
        if differentiation_method is not None:
            # Checked eagerly, unlike the approximator itself:
            # an invalid name must fail the moment a driver binds its
            # functions, not wait for the first Jacobian it never asks for.
            GradientApproximatorFactory().get_class(differentiation_method)
        # Copied, so a caller passing a mapping that changes afterward, e.g. a
        # dict it keeps building on, cannot reach back into this wrapper.
        self.__differentiation_method_options = dict(differentiation_method_options)
        self.__transformation = None
        # Built lazily, on the first Jacobian approximation,
        # rather than here:
        # a driver installs a transformation and releases it again
        # before ever asking for a Jacobian,
        # and building one for each would be wasted work.
        self.__approximate_jacobian = None

        super().__init__(function)
        # The function this one wraps, not `function.original`:
        # it may be a wrapper another problem installed,
        # e.g. for a sub-problem built on the functions of that problem,
        # and a sub-algorithm rebuilding a problem from `original`
        # must keep it, so that its evaluations reach that problem's database.
        self.original = function

    @property
    def stop_if_nan(self) -> bool:
        """Whether the evaluation stops when the function returns a `NaN`."""
        return self._stop_if_nan

    @stop_if_nan.setter
    def stop_if_nan(self, value: bool) -> None:
        self._stop_if_nan = value

    def _evaluate(self, input_value: NumberArray) -> NumberArray:
        """Evaluate the wrapped function without recording anything.

        This is what an approximated Jacobian perturbs,
        so that the perturbations never reach the database.

        Args:
            input_value: The input value, in the original coordinates.

        Returns:
            The output value.
        """
        return self._wrapped_function.func(input_value)

    def _evaluate_in_working_coordinates(self, input_value: NumberArray) -> NumberArray:
        """Evaluate the wrapped function at a point mapped back from the working space.

        This is what an approximated Jacobian perturbs
        once
        [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
        told this half a transformation,
        so that a perturbed point of the working space
        reaches the wrapped function mapped back to the coordinates it works in,
        rounded when the backward map rounds.

        The name is not mangled,
        unlike the other private helpers here,
        because the approximator holds this method
        and pickle reduces a bound method through the name it carries:
        a mangled one is not found on the class
        when a process pool loads the function back.

        Args:
            input_value: The input value, in the working coordinates.

        Returns:
            The output value.
        """
        return self._evaluate(
            self.__transformation.inverse_transform_value(input_value, no_check=True)
        )

    def set_perturbation_transformation(
        self, transformation: BaseSpaceTransformation | None
    ) -> None:
        """Set the map whose working coordinates an approximated Jacobian perturbs in.

        The Jacobian is approximated at the image of the point under the map:
        each perturbed point is mapped back through the transformation
        before reaching the wrapped function,
        so it is rounded when the backward map rounds
        and left as it is when the map only relaxes,
        and the Jacobian is mapped back before being recorded.
        The step therefore lives in the working coordinates,
        and the bounds checked are the ones of the working space.

        Without effect when the Jacobian is not approximated.
        Only stores the transformation here;
        the approximator itself is rebuilt lazily,
        the next time the Jacobian is approximated,
        so that setting and releasing a transformation
        without ever approximating a Jacobian in between
        builds none.

        Args:
            transformation: The map from the original coordinates to the
                working ones the perturbations are taken in.
                `None` to perturb in the original coordinates,
                bounded by `design_space`.
        """  # noqa: E501
        self.__transformation = transformation
        # Invalidated rather than rebuilt here:
        # see `__get_jacobian_approximator`.
        self.__approximate_jacobian = None

    def __get_jacobian_approximator(
        self,
    ) -> Callable[[NumberArray], NumberArray]:
        """Return the approximator of the Jacobian, building it on first use.

        Only called when the Jacobian is approximated,
        i.e. when `__differentiation_method` is not `None`.
        Cached afterward,
        until
        [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]
        invalidates it,
        since the callable it perturbs and the space it checks bounds against
        depend on the transformation.

        Returns:
            The approximator.
        """  # noqa: E501
        if self.__approximate_jacobian is None:
            self.__approximate_jacobian = self.__create_jacobian_approximator()

        return self.__approximate_jacobian

    def __create_jacobian_approximator(
        self,
    ) -> Callable[[NumberArray], NumberArray]:
        """Build the approximator of the Jacobian.

        Only called through the lazy cache above,
        i.e. when the Jacobian is approximated.

        Returns:
            The approximator.
        """
        transformation = self.__transformation
        perturbation_space = self.__design_space
        if transformation is None:
            f_pointer = self._evaluate
        else:
            # The perturbations are taken in the working coordinates,
            # so the callable perturbed is the one mapping a perturbed point
            # back before it reaches the wrapped function,
            # and the bounds checked are the ones of the working space.
            f_pointer = self._evaluate_in_working_coordinates
            # A transformation keeps the kind of its space,
            # so the working space is a design space exactly when
            # `__design_space` is one; when it is not, e.g. the problem is
            # defined on a `RandomSpace`, the approximators take the
            # perturbations without bounds (see
            # util/derivative/approximator/forward_differences.py and
            # centered_differences.py).
            if perturbation_space is not None:
                perturbation_space = transformation.working_space

        # The approximator is deliberately built on a callable that does not
        # record:
        # the perturbations it evaluates must stay out of the database,
        # while the Jacobian it returns is recorded by `_compute_jacobian`.
        approximator = GradientApproximatorFactory().create(
            self.__differentiation_method,
            f_pointer,
            design_space=perturbation_space,
            **self.__differentiation_method_options,
        )
        return approximator.f_gradient

    def _compute_output(self, input_value: NumberArray) -> NumberArray:
        """Compute an output value, recording it.

        Args:
            input_value: The input value, in the original coordinates.

        Returns:
            The output value.
        """
        # Checked here
        # because this half is the one every driver goes through,
        # including a driver building no working problem,
        # such as a linear one or a meta-algorithm,
        # so a NaN point stops it too,
        # before it reaches the user's function and the history.
        check_for_nan(input_value)
        if self._database is None:
            output_value = self._evaluate(input_value)
        elif self._vectorize:
            output_value = self.__compute_output_of_samples(input_value)
        else:
            hashed_input_value, output_value = self.__read(self.name, input_value)
            if output_value is None:
                output_value = self._evaluate(input_value)
                # The value is recorded
                # even when it holds a NaN,
                # so that the point the evaluation stopped at is visible in the
                # history.
                self._database.store(hashed_input_value, {self.name: output_value})

        # Checked here
        # because this half is the one every driver goes through,
        # including a driver building no working problem,
        # such as a linear one or a meta-algorithm,
        # so it is stopped too.
        check_for_nan(output_value, self.stop_if_nan, self.name, input_value)
        return output_value

    def __compute_output_of_samples(self, input_values: NumberArray) -> NumberArray:
        """Compute the output values of a matrix of samples, recording each one.

        Args:
            input_values: The samples, of shape `(n_samples, input_dimension)`.

        Returns:
            The output values.
        """
        output_values = self._evaluate(input_values)
        self._database.store_batch(input_values, output_values, self.name)
        return output_values

    def __compute_jacobian_of_samples(self, input_values: NumberArray) -> NumberArray:
        """Compute the Jacobian of a matrix of samples, recording each block.

        The blocks are recorded only when there is a database to record them in;
        they are checked for a `NaN` either way.

        Args:
            input_values: The samples, of shape `(n_samples, input_dimension)`.

        Returns:
            The block diagonal Jacobian.
        """
        jacobians = self.__evaluate_jacobian(input_values)
        # Both sizes are read from what this call handles
        # rather than from a space declared at build time:
        # the samples matrix is the one thing that is always there,
        # so a function built without an input space is checked
        # the same as one built with it.
        # The NaN check below needs them either way.
        output_dimension = jacobians.shape[0] // len(input_values)
        input_dimension = input_values.shape[1]
        if self.__store_jacobian and self._database is not None:
            self._database.store_batch(
                input_values, jacobians, self._gradient_name, is_jacobian=True
            )

        # Checked block by block
        # rather than over the whole matrix,
        # so that the message names the sample the NaN belongs to,
        # and after the recording,
        # as in `_compute_output`:
        # the whole matrix is computed by then,
        # so every block is visible in the history,
        # including the one the evaluation stops at.
        check_blocks_for_nan(
            jacobians,
            input_values,
            output_dimension,
            input_dimension,
            self.name,
            self.stop_if_nan,
        )
        return jacobians

    def _compute_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Compute a Jacobian, recording it but not the perturbations behind it.

        Args:
            input_value: The input value, in the original coordinates.

        Returns:
            The Jacobian.
        """
        # Checked here
        # for the same reason as in `_compute_output`:
        # this half is the one every driver goes through,
        # including a driver building no working problem,
        # such as a linear one or a meta-algorithm,
        # so a NaN point stops it too,
        # before it reaches the user's function and the history.
        check_for_nan(input_value)
        if self._vectorize:
            # That path checks the Jacobian itself,
            # block by block,
            # rather than the block diagonal matrix it returns as a whole,
            # which is what names the sample a NaN belongs to.
            # It is taken whether or not there is a database to record in,
            # since a driver recording nothing deserves the same message.
            return self.__compute_jacobian_of_samples(input_value)

        if self._database is None:
            jacobian = self.__evaluate_jacobian(input_value)
        else:
            hashed_input_value, jacobian = self.__read(self._gradient_name, input_value)
            if jacobian is None:
                jacobian = self.__evaluate_jacobian(input_value)
                if self.__store_jacobian:
                    # The full matrix is recorded,
                    # whatever is returned below.
                    self._database.store(
                        hashed_input_value, {self._gradient_name: jacobian}
                    )

        check_for_nan(jacobian, self.stop_if_nan, self.name, input_value)
        if (
            self.dim == 1
            and not self._vectorize
            and not isinstance(jacobian, sparse_classes)
        ):
            # A vectorized call returns the block-diagonal matrix of the samples,
            # whose rows a caller splits back into one block per sample,
            # so it is returned as it is even for a function of dimension 1.
            # A sparse Jacobian is returned as it is too:
            # a driver declaring it supports one reads the matrix,
            # and flattening it would change its type.
            return jacobian.ravel()

        return jacobian

    def __evaluate_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Evaluate the Jacobian, approximating it when asked to.

        Args:
            input_value: The input value, in the original coordinates.

        Returns:
            The Jacobian.
        """
        if self.__differentiation_method is None:
            jacobian = self._wrapped_function.jac(input_value).real
        elif self.__transformation is None:
            jacobian = self.__get_jacobian_approximator()(input_value).real
        else:
            # Approximated at the image of the point under the map,
            # since that is where the perturbations are taken;
            # mapped back at that same point,
            # in the original coordinates,
            # because the tangent map of a non-affine transformation needs it.
            # The adaptation half maps it forward again
            # before handing it to the algorithm,
            # the round trip amounting to the chain rule.
            transformation = self.__transformation
            working = self.__get_jacobian_approximator()(
                transformation.transform_value(input_value)
            )
            jacobian = transformation.inverse_transform_jacobian(
                working, input_value
            ).real

        if not self.__support_sparse_jacobian and isinstance(jacobian, sparse_classes):
            # Densified before the recording,
            # so the database holds what the driver reads.
            return jacobian.todense()

        return jacobian

    def __read(
        self, name: str, input_value: NumberArray
    ) -> tuple[Any, NumberArray | None]:
        """Read a recorded value, signalling a point absent from the database.

        Args:
            name: The name under which the value is recorded.
            input_value: The input value, in the original coordinates.

        Returns:
            The hashed input value, and the recorded value if there is one.
        """
        database = self._database
        hashed_input_value = database.get_hashable_ndarray(input_value)
        recorded_values = database.get(hashed_input_value)
        if not recorded_values and self.pre_compute_at_new_point is not None:
            self.pre_compute_at_new_point()

        value = None if recorded_values is None else recorded_values.get(name)
        return hashed_input_value, value

    @ArrayFunction.func.setter
    def func(self, f_pointer: Callable[[NumberArray], NumberArray]) -> None:  # noqa: D102
        if self.enable_statistics:
            self.__get_counter().value = 0

        super(__class__, self.__class__).func.fset(self, f_pointer)

    def evaluate(self, x_vect: NumberArray) -> NumberArray:  # noqa: D102
        value = super().evaluate(x_vect)
        if self.enable_statistics:
            # One count per evaluation asked of this function:
            # a vectorized call counts once for the whole matrix of samples,
            # and the perturbations of an approximated Jacobian count not at all,
            # going through `_evaluate`.
            # This is both multiprocess- and multithread-safe,
            # thanks to a lock.
            counter = self.__get_counter()
            with counter.get_lock():
                counter.value += 1

        return value

    @property
    def n_calls(self) -> int:
        """The number of times the function has been evaluated.

        Zero when the counters are disabled.
        """
        if self.enable_statistics:
            return self.__get_counter().value

        return 0

    @n_calls.setter
    def n_calls(self, value: int) -> None:
        if not self.enable_statistics:
            msg = "The function counters are disabled."
            raise RuntimeError(msg)

        counter = self.__get_counter()
        with counter.get_lock():
            counter.value = value

    def __get_counter(self) -> Synchronized:
        """Return the counter of the evaluations, building it when it is missing.

        The counter is a piece of shared memory with a lock of its own,
        which the counting alone needs,
        so it is built only for a function counting its calls.
        A function is left without one
        when the counters were disabled when it was built
        and are enabled afterwards,
        and when it is unpickled in that state,
        `__setstate__` restoring the plain count instead.
        Either way it is built here,
        carrying on from the count it is handed.

        Returns:
            The counter of the evaluations.
        """
        counter = self.__dict__.get("_n_calls")
        if not isinstance(counter, Synchronized):
            counter = Value("i", 0 if counter is None else counter)
            self._n_calls = counter

        return counter

    def _init_shared_memory_attrs_before(self) -> None:
        """Initialize the shared attributes in multiprocessing."""
        # Allocated only for a function counting its calls,
        # which is not the default:
        # this runs for every function of a problem,
        # every time a driver wraps the functions,
        # and the shared memory and the lock behind a counter are not free.
        if self.enable_statistics:
            self._n_calls = Value("i", 0)
