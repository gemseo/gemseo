- Support for Python 3.10; the minimum supported version is now Python 3.11.
- The `typing-extensions` dependency, whose features are now provided by the standard library.
- The `strenum` dependency; the enumerations now derive from `enum.StrEnum`.
  The member names and values did not change.
  `enum.StrEnum` is not a subclass of `strenum.StrEnum`, so code that is type-annotated with `strenum.StrEnum` must be changed to `enum.StrEnum`.
