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
"""Interface for workflow observer classes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from inspect import Parameter
from inspect import Signature
from inspect import signature
from typing import TYPE_CHECKING
from typing import Any

from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.util.typing import StrKeyMapping


@cache
def _get_signature_without_self(callable_: Callable[..., Any]) -> Signature:
    """Return the signature of a method, without its first parameter.

    The observed methods are decorated as plain functions, so their signature
    includes `self`, while the arguments captured by the wrappers do not.
    Computing a signature is expensive relative to an observed call, e.g. an
    MDA iteration, hence the cache; it is keyed by the undecorated function,
    of which there is one per observed method.

    Args:
        callable_: The method, as an undecorated function.

    Returns:
        The signature of the method without its first parameter.
    """
    signature_ = signature(callable_)
    parameters = list(signature_.parameters.values())[1:]
    return signature_.replace(parameters=parameters)


def normalize_arguments_safely(
    callable_: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind the arguments of a call to the parameters of the method called.

    The parameters that the caller omitted are set to their default value, so
    that an argument can be read by its parameter name whatever the way it was
    passed. A positional-only parameter is returned as a keyword argument even
    though it cannot be passed that way.

    The extra positional arguments captured by a `*args` parameter are kept
    as a tuple under the name of that parameter prefixed with `*`, e.g.
    `"*args"`, so that a trace shows which argument is variadic. The extra
    keyword arguments captured by a `**kwargs` parameter are flattened into
    the returned mapping instead, so each of them can be read by name like
    any other argument.

    `bind` routes a keyword argument matching a named parameter to that
    parameter, so flattening overwrites no argument, with two exceptions
    that Python accepts: a keyword argument named after a positional-only
    parameter, e.g. `f(self, x, /, **kwargs)` called as `f(1, x=2)`, and a
    key such as `"*args"` passed by dictionary unpacking. In these cases
    only, the `**kwargs` entries are left nested under the name of that
    parameter prefixed with `**`, e.g. `"**kwargs"`, rather than flattened.
    This is not raised on, since the normalization must never break a call
    that Python itself accepts.

    Args:
        callable_: The method, as an undecorated function.
        args: The positional arguments passed to the method, without the
            instance the method is bound to.
        kwargs: The keyword arguments passed to the method.

    Returns:
        The arguments, by parameter name, the extra positional ones under
        the `*`-prefixed name of the `*args` parameter. The entries of a
        `**kwargs` parameter are flattened into the returned mapping, unless
        one of them clashes with another key, in which case they are left
        nested under the `**`-prefixed name of that parameter.

    Raises:
        TypeError: When the arguments do not match the signature of the
            method, which would also make the observed call fail.
    """
    signature_ = _get_signature_without_self(callable_)
    try:
        bound_arguments = signature_.bind(*args, **kwargs)
    except TypeError as error:
        # The name of the method is part of the message raised by an
        # un-instrumented call; binding here would otherwise lose it.
        msg = f"{callable_.__qualname__}() {error}"
        raise TypeError(msg) from error
    bound_arguments.apply_defaults()
    arguments: dict[str, Any] = {}
    for name, value in bound_arguments.arguments.items():
        kind = signature_.parameters[name].kind
        if kind is Parameter.VAR_POSITIONAL:
            arguments[f"*{name}"] = value
        elif kind is Parameter.VAR_KEYWORD:
            # A `**kwargs` parameter is the last one of a signature, so all the
            # other arguments are already in `arguments` at this point.
            if arguments.keys() & value.keys():
                arguments[f"**{name}"] = value
            else:
                arguments.update(value)
        else:
            arguments[name] = value

    return arguments


@dataclass
class CallSpec:
    """Complete specification of a callable invocation.

    It provides complete information about a method call, including what was
    called and with what arguments.
    """

    kwargs: dict[str, Any]
    """The arguments, by parameter name.

    The arguments are normalized, see `create_safely`, so an argument passed
    positionally is stored under the name of its parameter.
    """

    callable_: Callable[..., Any]
    """The callable."""

    @classmethod
    def create_safely(
        cls,
        callable_: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CallSpec:
        """Return the specification of a call to a method.

        Args:
            callable_: The method, as an undecorated function.
            args: The positional arguments passed to the method, without
                the instance the method is bound to.
            kwargs: The keyword arguments passed to the method.

        Returns:
            The call specification, with its arguments normalized by
            parameter name, see `normalize_arguments_safely`.
        """
        return cls(
            normalize_arguments_safely(callable_, args, kwargs), callable_=callable_
        )


class WorkflowObserverInterface(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Interface for workflow observer implementations.

    A workflow observer tracks the lifecycle of object execution by notifying
    about start and end events of observed methods. Implementations should handle
    these events to perform custom actions like logging, monitoring, or state tracking.
    """

    @abstractmethod
    def __init__(
        self,
        object_: object,
        init_arguments: StrKeyMapping,
    ) -> None:
        """
        Args:
            object_: The object to observe.
            init_arguments: The normalized arguments used when instantiating the
                object to observe, by parameter name.
        """  # noqa: D205, D212

    @abstractmethod
    def start(self, call_spec: CallSpec) -> None:
        """Start the observation.

        Args:
            call_spec: The call specification of the method to observe.
        """

    @abstractmethod
    def end(self, call_spec: CallSpec, returned_data: Any) -> None:
        """Finish the observation.

        Args:
            call_spec: The call specification of the method to observe.
            returned_data: The data returned by the method to observe.
        """
