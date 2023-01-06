"""Example Google style docstrings.

This module demonstrates the documentation as specified by the Google style:
https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
Docstrings may extend over multiple lines.
Sections are created with a section header and
a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples`` sections.
    Sections support any reStructuredText formatting,
    including literal blocks::

        $ python example_google.py

Section breaks are created by resuming un-indented text.
Section breaks are also implicitly created anytime a new section starts.
"""
from __future__ import annotations

from typing import ClassVar
from typing import Generator

from docstring_inheritance import GoogleDocstringInheritanceMeta
from gemseo.utils.python_compatibility import Final

MODULE_LEVEL_VARIABLE: Final[int] = 98765
"""Module level constant variable documented inline.

The docstring may span multiple lines.
"""


def example_function(
    arg1: int,
    arg2: str | None = None,
    *args: int,
    **kwargs: str,
) -> bool:
    """This is an example of a module level function.

    Function arguments are documented in the ``Args`` section.
    The types of the arguments and the return type
    are described as comments in the signature,
    according to PEP 484 (https://www.python.org/dev/peps/pep-0484/#
    suggested-syntax-for-python-2-7-and-straddling-code).

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``,
    and typed according to their values.

    The format for an argument is::

        name: The description, starts with a capital letter and ends with a dot.
            The description may span multiple lines.
            Following lines should be indented.

            Multiple paragraphs are supported in parameter descriptions.

    Args:
        arg1: The first parameter.
        arg2: The second parameter.
            Second line of description should be indented.
        *args: A variable length argument list of int values.
        **kwargs: An arbitrary keyword arguments with str values.

    Returns:
        True if successful, False otherwise.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'arg1': arg1,
                'arg2': arg2
            }

    Raises:
        AnError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `arg2` is equal to `arg1`.
    """
    if arg1 == arg2:
        raise ValueError("arg1 may not be equal to arg2")
    return True


def example_generator(
    n: int,
) -> Generator[int]:
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n: The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        The next number in the range of 0 to `n` - 1.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]
    """
    yield from range(n)


class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    The public attributes of the class are documented here in an ``Attributes``.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Instances attributes are defined in the class body with no default values,
    see below.
    """

    attr1: int
    """The description of ``attr1``."""

    attr2: int
    """The description of ``attr2``."""

    class_attr1 = 0
    """A class attribute that can be re-assigned and turned into an instance attribute by
    an instance."""

    class_attr2: ClassVar[int] = 0
    """A class attribute that shall not be re-assigned by an instance."""

    CLASS_CONSTANT_ATTR: Final[int] = 0
    """A class attribute that shall not be re-assigned or overridden by a subclass."""

    OTHER_CLASS_CONSTANT_ATTR: int = 0
    """A class attribute that shall not be re-assigned but can be overridden by a
    subclass."""

    def __init__(
        self,
        arg1: int,
        arg2: int,
        arg3: float,
    ) -> None:
        """
        Args:
            arg1: Description of `arg1`.
            arg2: Description of `arg2`.
                Multiple lines are supported.
            arg3: Description of `arg3`.

        Note:
            Do not include the `self` parameter in the ``Args`` section.
            For the __init__ method only:
            add to the return type annotation
            another comment to discard the legit style checks
            for missing docstring parts, the comment shall be

        """  # noqa: D205, D212, D415
        self.attr1 = arg1
        self.attr2 = arg2
        self._attr3 = arg3

    @property
    def readonly_property(self) -> str:
        """Property is documented in the getter method."""
        return "readonly_property"

    @property
    def readwrite_property(self) -> list[str]:
        """Property with both getter and setter is only documented in the getter method.

        If the setter method contains notable behavior, it should be mentioned here.
        """
        return ["readwrite_property"]

    @readwrite_property.setter
    def readwrite_property(
        self,
        value: int,
    ) -> None:
        self.attr1 = value

    def example_method(
        self,
        arg1: int,
        arg2: int,
    ) -> bool:
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            arg1: The first parameter.
            arg2: The second parameter.

        Returns:
            True if successful, False otherwise.
        """
        return True


class BaseClass(metaclass=GoogleDocstringInheritanceMeta):
    """A base class such that derived classes inherit docstrings.

    If a base class declares the metaclass ``GoogleDocstringInheritanceMeta``,
    then a derived class could inherit the docstrings of its methods.
    A method of a derived class will inherit a docstring
    if it has no docstring defined.

    Only the base class is decorated, not the derived classes.
    """

    def method1(self) -> None:
        """Docstring of method1."""
        raise NotImplementedError

    def method2(self) -> None:
        """Docstring of method2."""


class DerivedClass(BaseClass):
    """An example of a class that inherits docstrings from its base class."""

    def method1(self) -> None:  # noqa: D102
        # This method will inherit its docstring from the same method of its base class,
        # because it has no docstring.
        # WARNING: you have to suffix the signature with # noqa: D102 to prevent this
        # legit style violation
        pass

    def method2(self) -> None:
        """This method will NOT inherit its docstring from its base class.

        Because it has a docstring.
        """
