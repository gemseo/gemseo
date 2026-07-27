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
"""Injection and instrumentation of workflow observers into target classes."""

from __future__ import annotations

import logging
from functools import cache
from functools import wraps
from inspect import isabstract
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.utils._workflow_observers.discipline import (
    DisciplineWorkflowObserver as _DisciplineWorkflowObserver,
)
from gemseo.utils._workflow_observers.doe import (
    DOEWorkflowObserver as _DOEWorkflowObserver,
)
from gemseo.utils._workflow_observers.interface import CallArguments
from gemseo.utils._workflow_observers.interface import CallSpec
from gemseo.utils._workflow_observers.mda import (
    MDAWorkflowObserver as _MDAWorkflowObserver,
)
from gemseo.utils._workflow_observers.optimizer import (
    OptimizerWorkflowObserver as _OptimizerWorkflowObserver,
)
from gemseo.utils._workflow_observers.scenario import (
    ScenarioWorkflowObserver as _ScenarioWorkflowObserver,
)
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from gemseo.utils._workflow_observers.base_observer import InjectableObserver
    from gemseo.utils._workflow_observers.base_observer import ObservationSpec
    from gemseo.utils._workflow_observers.interface import WorkflowObserverInterface

LOGGER = logging.getLogger(__name__)

_WRAPPER_ATTRIBUTE: Final[str] = "__workflow_observer_wrapper__"
"""The attribute marking the method wrappers created by this module.

When a concrete subclass of an observed concrete class is decorated,
the methods it inherits are already wrapped; the marker, together with a
wrapper-identity check at call time, prevents double observation.
"""


class _WorkflowObserverInjector:
    """A class for injecting workflow observers into target classes."""

    __observer_classes: ClassVar[set[type[InjectableObserver]]] = set()
    """The workflow observer classes."""

    __observed_classes: ClassVar[set[type[Any]]] = set()
    """The classes already observed."""

    @classmethod
    def register(cls, observer_class: type[InjectableObserver]) -> None:
        """Register a workflow observer class.

        Args:
            observer_class: The workflow observers class.
        """
        cls.__observer_classes.add(observer_class)

    @classmethod
    def accept(cls, class_: type[Any]) -> bool:
        """Return whether a class can be observed by any registered observer.

        A class can be observed if:
        - The directory manager is enabled
        - At least one observer class is registered
        - The class is not abstract
        - The class has not already been observed
        - A registered observer can observe the class

        Args:
            class_: The class to check.

        Returns:
            True if the class can be observed, False otherwise.
        """
        from gemseo.utils.global_configuration import _configuration

        if (
            not _configuration.directory_manager.enable
            or not cls.__observer_classes
            or isabstract(class_)
            or class_ in cls.__observed_classes
        ):
            return False

        for observer_class in cls.__observer_classes:
            # TODO: check that there is a unique observer for the given class?
            if _match_spec(observer_class._spec, class_):
                cls.__observed_classes.add(class_)
                return True

        return False

    @classmethod
    def inject(cls, class_: type[Any]) -> None:
        """Inject an observer into a class by decorating its methods.

        This method finds the first registered observer that can observe the class,
        then decorates the class's methods according to the observer specification.

        Args:
            class_: The class to inject the observer into.

        Raises:
            RuntimeError: If no observer can be found to inject into the class.
        """
        for observer_class in cls.__observer_classes:
            if _match_spec(observer_class._spec, class_):
                _decorate_class(observer_class._spec, observer_class, class_)
                return
        msg = f"Cannot find an observer to inject into the class {class_}."
        raise RuntimeError(msg)


@cache
def _get_base_classes_full_names(class_: type[Any]) -> frozenset[str]:
    """Return the fully qualified names of all base classes in the MRO.

    Args:
        class_: The class whose MRO to inspect.

    Returns:
        The fully qualified names of all base classes.
    """
    names = set()
    for base_class in class_.__mro__:
        module = base_class.__module__
        name = base_class.__name__
        names.add(f"{module}.{name}")
    return frozenset(names)


def _match_spec(
    spec: ObservationSpec,
    class_: type,
) -> bool:
    """Return whether the given class matches an observation specification.

    A class matches if its MRO contains the spec's base class
    and it does not inherit from any excluded subclasses.

    Args:
        spec: The observation specification to match against.
        class_: The class to check.

    Returns:
        True if the class matches the specification, False otherwise.
    """
    observee_base_classes = _get_base_classes_full_names(class_)
    return (
        spec.base_class in observee_base_classes
        and observee_base_classes.isdisjoint(spec.excluded_sub_classes)
    )


def _decorate_class(
    spec: ObservationSpec,
    observer_class: type[WorkflowObserverInterface],
    class_: type[Any],
) -> None:
    """Decorate the methods of the given class to enable observation.

    This wraps the class's ``__init__`` method and specified methods
    with observer decorators based on the observation specification.

    Args:
        spec: The observation specification.
        observer_class: The workflow observer class to instantiate.
        class_: The class to decorate.
    """
    if not getattr(class_.__init__, _WRAPPER_ATTRIBUTE, False):
        class_.__init__ = _decorate_init(class_.__init__, observer_class)

    _decorate_methods(class_, spec.method_names_for_start, _decorate_with_start)
    _decorate_methods(class_, spec.method_names_for_finish, _decorate_with_finish)
    _decorate_methods(class_, spec.method_names_for_both, _decorate_with_both)


def _decorate_methods(
    class_: type[Any],
    method_names: Iterable[str],
    decorator: Callable[[Callable], Callable],
) -> None:
    """Decorate methods of a class, skipping the already decorated ones.

    A method inherited from an observed concrete base class is already
    decorated: wrapping it again would notify the observer twice.

    Args:
        class_: The class to decorate.
        method_names: The names of the methods to decorate.
        decorator: The decorator to apply.
    """
    for method_name in method_names:
        method = getattr(class_, method_name)
        if not getattr(method, _WRAPPER_ATTRIBUTE, False):
            setattr(class_, method_name, decorator(method))


def _decorate_init(
    callable_: Callable,
    observer_class: type[WorkflowObserverInterface],
) -> Callable:
    """Create a decorator to initialize a workflow observer on the decorated object.

    This decorator wraps an `__init__` method to attach a `_workflow_observer`
    attribute to the object after initialization. The observer instance is
    created with the object and the initialization call arguments.

    Args:
        callable_: The `__init__` method to decorate.
        observer_class: The workflow observer class to instantiate.

    Returns:
        The decorated `__init__` method that initializes the observer.
    """

    @wraps(callable_)
    def _wrapper(self, *args: Any, **kwargs: Any) -> Any:
        callable_(self, *args, **kwargs)
        # Only the outermost wrapper (the one bound to the instance's class)
        # creates the observer: when a subclass defines its own decorated
        # __init__ calling super().__init__, the inherited wrapper must not
        # create a second observer on a partially initialized object.
        # No observer is created while the observation is disabled: objects
        # created at that time are not observed.
        if type(self).__init__ is _wrapper and _is_observation_enabled():
            self._workflow_observer = observer_class(
                self, CallArguments(args=args, kwargs=kwargs)
            )

    setattr(_wrapper, _WRAPPER_ATTRIBUTE, True)
    return _wrapper


def _decorate_with_start(callable_: Callable) -> Callable:
    """Create a decorator that notifies the observer when the method starts.

    The decorated method will call `observer.start()` before executing the
    underlying callable.

    Args:
        callable_: The method to decorate.

    Returns:
        The decorated method that notifies the observer of the start event.
    """

    @wraps(callable_)
    def _wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if getattr(type(self), callable_.__name__, None) is not _wrapper:
            # A wrapper closer to the instance's class owns the observation.
            return callable_(self, *args, **kwargs)
        observer = _get_observer(self)
        if observer is None:
            return callable_(self, *args, **kwargs)
        observer.start(CallSpec(callable_=callable_, args=args, kwargs=kwargs))
        return callable_(self, *args, **kwargs)

    setattr(_wrapper, _WRAPPER_ATTRIBUTE, True)
    return _wrapper


def _decorate_with_finish(callable_: Callable) -> Callable:
    """Create a decorator that notifies the observer when the method finishes.

    The decorated method will call `observer.end()` after executing the
    underlying callable, even if an exception is raised. The return value
    (or None if an exception occurred) is passed to the observer.

    Args:
        callable_: The method to decorate.

    Returns:
        The decorated method that notifies the observer of the end event.
    """

    @wraps(callable_)
    def _wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if getattr(type(self), callable_.__name__, None) is not _wrapper:
            # A wrapper closer to the instance's class owns the observation.
            return callable_(self, *args, **kwargs)
        observer = _get_observer(self)
        if observer is None:
            return callable_(self, *args, **kwargs)
        call_spec = CallSpec(callable_=callable_, args=args, kwargs=kwargs)
        try:
            returned_data = callable_(self, *args, **kwargs)
        except BaseException:
            _end_observation_safely(observer, call_spec)
            raise
        observer.end(call_spec, returned_data)
        return returned_data

    setattr(_wrapper, _WRAPPER_ATTRIBUTE, True)
    return _wrapper


def _decorate_with_both(callable_: Callable) -> Callable:
    """Create a decorator to notify the observer of both start and finish events.

    The decorated method will call `observer.start()` before executing and
    `observer.end()` after executing the underlying callable. The observer
    is notified of both events even if an exception is raised.

    Args:
        callable_: The method to decorate.

    Returns:
        The decorated method that notifies the observer of start and end events.
    """

    @wraps(callable_)
    def _wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if getattr(type(self), callable_.__name__, None) is not _wrapper:
            # A wrapper closer to the instance's class owns the observation.
            return callable_(self, *args, **kwargs)
        observer = _get_observer(self)
        if observer is None:
            return callable_(self, *args, **kwargs)
        call_spec = CallSpec(callable_=callable_, args=args, kwargs=kwargs)
        observer.start(call_spec)
        try:
            returned_data = callable_(self, *args, **kwargs)
        except BaseException:
            _end_observation_safely(observer, call_spec)
            raise
        observer.end(call_spec, returned_data)
        return returned_data

    setattr(_wrapper, _WRAPPER_ATTRIBUTE, True)
    return _wrapper


def _is_observation_enabled() -> bool:
    """Return whether the workflow observation is enabled.

    Returns:
        Whether the workflow observation is enabled.
    """
    # This call-time check exists mostly for the test suite. In production the
    # manager is enabled once and never disabled, so the check is always true
    # when a wrapper runs. But a class is decorated in place permanently the
    # first time it is observed, and that decoration cannot be undone; the test
    # suite resets the global configuration to a disabled Settings instance
    # between tests, and this check is what keeps an already-decorated class
    # inert under such a disabled configuration (avoiding stray directories and
    # chdir in tests that do not use the manager).
    # Avoid a circular import.
    from gemseo.utils.global_configuration import _configuration

    return _configuration.directory_manager.enable


def _get_observer(object_: object) -> WorkflowObserverInterface | None:
    """Return the observer of an object if it shall be notified.

    Args:
        object_: The observed object.

    Returns:
        The observer, or None when the observation is disabled.
    """
    if not _is_observation_enabled():
        return None
    # The attribute is accessed directly rather than with a default: an
    # observed object is expected to carry its observer, set by the wrapped
    # __init__ while observation is enabled. The only way to reach this point
    # without the attribute is an object whose class is decorated but that
    # never ran the wrapped __init__ -- e.g. an instance unpickled via __new__
    # in a worker process, or loaded from a pickle in a later enabled session.
    # That path is not exercised by the test suite; should it occur, the
    # resulting AttributeError is preferable to silently skipping observation.
    return object_._workflow_observer


def _end_observation_safely(
    observer: WorkflowObserverInterface,
    call_spec: CallSpec,
) -> None:
    """End an observation while an exception is being propagated.

    An error raised while ending the observation is logged instead of being
    raised, so that it cannot mask the exception of the observed method.

    Args:
        observer: The observer to end.
        call_spec: The call specification of the observed method.
    """
    try:
        observer.end(call_spec, None)
    except Exception:
        LOGGER.exception(
            "The observation of %s could not be ended.",
            call_spec.callable_.__qualname__,
        )


def inject_observer(class_: type[Any]) -> None:
    """Inject an observer into a class if the class applies.

    Args:
        class_: The class to inject the observer into.

    Raises:
        RuntimeError: If no observer can be found to inject into the class.
    """
    if _WorkflowObserverInjector.accept(class_):
        _WorkflowObserverInjector.inject(class_)


class WorkflowObserverMeta(ABCGoogleDocstringInheritanceMeta):
    """Metaclass that automatically injects workflow observers into class instantiation.

    When a class uses this metaclass, the injector checks if the class can be observed.
    If so, it decorates the class's methods before instantiation proceeds. This ensures
    that workflow observers are transparently applied to observed GEMSEO classes without
    requiring explicit observer setup.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa:D102
        inject_observer(self)
        return super().__call__(*args, **kwargs)


_WorkflowObserverInjector.register(_ScenarioWorkflowObserver)
_WorkflowObserverInjector.register(_DisciplineWorkflowObserver)
_WorkflowObserverInjector.register(_MDAWorkflowObserver)
_WorkflowObserverInjector.register(_OptimizerWorkflowObserver)
_WorkflowObserverInjector.register(_DOEWorkflowObserver)
