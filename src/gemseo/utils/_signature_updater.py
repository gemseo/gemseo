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
"""Inject arguments in a method from a pydantic model."""

from __future__ import annotations

from inspect import Parameter
from inspect import Signature
from inspect import signature
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final

from docstring_inheritance import GoogleDocstringInheritor
from makefun import create_function

if TYPE_CHECKING:
    from gemseo.core.grammars.pydantic_grammar import ModelType


class SignatureUpdater(type):
    """A metaclass to update the arguments of a method signature from a pydantic model.

    The method to be updated must only have the arguments self and **kwargs.
    """

    model_attr_name: ClassVar[str] = "Model"
    """The name of the attribute of the target class that holds the pydantic model."""

    method_name: ClassVar[str] = "method"
    """The name of the method for which to update the arguments."""

    __MISSING_DESCRIPTION: Final[str] = "The description is missing."
    """The fallback docstring for arguments that have no description."""

    def __new__(
        cls,
        class_name: str,
        class_bases: tuple[type],
        class_dict: dict[str, Any],
    ) -> SignatureUpdater:
        if cls.model_attr_name in class_dict:
            method = cls.get_method(class_bases, class_dict)
            cls.check_method(method)
            new_signature = cls.create_signature(
                class_dict[cls.model_attr_name], method
            )
            new_docstring = cls.create_docstring(
                class_dict[cls.model_attr_name], method
            )
            class_dict[cls.method_name] = create_function(
                new_signature,
                method,
                func_name=cls.method_name,
                doc=new_docstring,
                qualname=class_dict["__qualname__"],
                module_name=class_dict["__module__"],
            )
        return type.__new__(cls, class_name, class_bases, class_dict)

    @staticmethod
    def check_method(method: Callable) -> None:
        """Check that the method can be updated.

        Args:
            method: The method to check.

        Raises:
            RuntimeError: If the method cannot be updated.
        """
        parameters = signature(method).parameters
        error_msg = (
            f"The method {method.__name__} must only have the argument "
            "**kwargs beside self."
        )
        if len(parameters) > 2:
            raise RuntimeError(error_msg)
        second_parameter = list(parameters.values())[1]
        if second_parameter.kind != second_parameter.VAR_KEYWORD:
            raise RuntimeError(error_msg)

    @classmethod
    def get_method(
        cls,
        class_bases: tuple[type],
        class_dict: dict[str, Any],
    ) -> Callable:
        """Search the method for which the arguments shall be updated.

        Args:
            class_bases: The base classes.
            class_dict: The class namespace.

        Returns:
            The method.

        Raises:
            RuntimeError: If the method cannot be found
                or if the method cannot be updated.
        """
        method = class_dict.get(cls.method_name)
        if method is None:
            for base in class_bases:
                method = getattr(base, cls.method_name)
                if method is not None:
                    break
            else:
                msg = f"The method named {cls.method_name} cannot be found."
                raise RuntimeError(msg)

        if hasattr(method, "__func_impl__"):
            # The method has already been processed, pick its original signature.
            return method.__func_impl__

        return method

    @staticmethod
    def create_signature(
        model: ModelType,
        method: Callable,
    ) -> Signature:
        """Update the signature of a method from a model.

        Args:
            model: The pydantic model.
            method: The method to be processed.

        Returns:
            The updated signature.
        """
        param_without_defaults = [Parameter("self", kind=Parameter.POSITIONAL_ONLY)]
        param_with_defaults = []

        for name, field in model.model_fields.items():
            if field.is_required():
                param_without_defaults += [
                    Parameter(
                        name,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=field.annotation,
                    )
                ]
            else:
                param_with_defaults += [
                    Parameter(
                        name,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        default=field.default,
                        annotation=field.annotation,
                    )
                ]

        return Signature(
            param_without_defaults + param_with_defaults,
            return_annotation=signature(method).return_annotation,
        )

    @classmethod
    def create_docstring(
        cls,
        model: ModelType,
        method: Callable,
    ) -> str:
        """Update the docstring of a method from a model.

        Args:
            model: The pydantic model.
            method: The method to be processed.

        Returns:
            The updated docstring.
        """
        args_section = {}
        for name, field in model.model_fields.items():
            description = field.description
            if not description:
                description = cls.__MISSING_DESCRIPTION
            # This is the format expected by GoogleDocstringInheritor.
            args_section[name] = f": {description}"

        docstring_sections = GoogleDocstringInheritor._DOCSTRING_PARSER.parse(
            method.__doc__
        )
        docstring_sections["Args"] = args_section

        return GoogleDocstringInheritor._DOCSTRING_RENDERER.render(docstring_sections)


def update_signature(
    model_attr_name: str,
    method_name: str,
) -> type:
    """Create a metaclass to update the signature of a method from a pydantic model.

    The class that will derive from the created metaclass shall have an attribute
    that holds the Pydantic model.

    Args:
        model_attr_name: The name of the attribute that holds the model.
        method_name: The name of the method.

    Returns:
        The metaclass that can update the method signature.
    """
    # Create a unique copy of the metaclass to avoid sharing its state.
    metaclass = type(
        SignatureUpdater.__name__,
        SignatureUpdater.__bases__,
        dict(SignatureUpdater.__dict__),
    )
    metaclass.model_attr_name = model_attr_name
    metaclass.method_name = method_name
    return metaclass
