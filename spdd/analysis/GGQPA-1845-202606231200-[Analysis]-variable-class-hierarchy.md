<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: Variable Class Hierarchy

> **GitLab issue [#1845](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1845)**
> — "Variable class hierarchy in the DesignSpace".
> This is the **v1 foundation refactor** that the discrete/categorical work
> (issue [#1791](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1791)) and the
> catalog work depend on. It is **behavior-preserving**:
> no user-visible functional change, no serialization change, no public-API
> change. It only turns the single `Variable` model into a polymorphic
> hierarchy so kind-specific behavior lives on the type.

## Original Business Requirement

> Each design-variable kind should be its own object (`ContinuousVariable`,
> `IntegerVariable`, and later `DiscreteVariable` / `CategoricalVariable` /
> `CatalogVariable`) rather than a single `Variable` carrying a `type` enum
> and an ever-growing set of `if variable.type == …` branches. Kind-specific
> behavior (bound validation, normalization policy, integer masking,
> membership, default-value initialization, dtype casting) should be defined
> **on the variable type itself**, so new kinds extend the hierarchy instead
> of editing every consumer.

This need surfaced while scoping ticket 1791 (discrete/enumerated variables):
adding a third and fourth kind to the `type`-switch pattern would multiply the
branch sites. The 1791 analysis explicitly tracked this as a follow-up
("extract behavior into per-kind strategies"). This story promotes that
follow-up to a **prerequisite**, done first as a clean refactor.

## Domain Concept Identification

### Existing Concepts (from codebase)

The design-space code lives in `src/gemseo/space/`: the façade and its collaborators in
`space/design/`, and the `Variable` model one level up in `space/_variable.py`, where it
is shared with `ParameterSpace` and re-exported by `gemseo/enum/__init__.py`.

- **`Variable`** (`src/gemseo/space/_variable.py:100`): a Pydantic `BaseModel`
  declared **`frozen=True`** (immutable — *not* `validate_assignment`).
  Fields `size`, `type: DataType`, `lower_bound`, `upper_bound` (`:115-125`).
  A single `@model_validator(mode="after") __validate_variable` (`:127-142`) drives
  everything: for each bound it calls `__convert_bound` (`:144`) then `__check_bound`
  (`:176`), and finally enforces `upper >= lower` (`:138`).
    - `__convert_bound` picks the dtype via `TYPE_MAP[self.type]` (`:161`), **freezes**
      the resulting array with `setflags(write=False)` (`:171`) so an in-place mutation
      cannot bypass the version bump, and writes it back through
      `self.__dict__[bound_name] =` (`:174`) to bypass the frozen model.
    - `__check_bound` rejects finite non-integer bounds when `type == INTEGER` (`:216`).
- **Copy and pickle hooks on `Variable`**: `__copy__` (`:229`) and `__deepcopy__`
  (`:236`) both return `self`, because a variable is immutable and its bound arrays are
  read-only; `model_copy` (`:239`) returns `self` when there is no update and otherwise
  rebuilds through `model_validate` so the update is converted, checked and frozen;
  `__setstate__` (`:261`) re-freezes the bound arrays after unpickling.
  `__eq__` (`:269`) compares `isinstance(other, self.__class__)` + size + type + bounds.
- **`DataType`** (`space/_variable.py:84`): `StrEnum{FLOAT, INTEGER}`, plus module-level
  `TYPE_MAP` → numpy dtype (`int64` / `float64`, `:94`). Re-exported publicly as
  `gemseo.enum.DesignVariableType` (`enum/__init__.py:116`, lazy map at `:275`).
- **`format_components`** (`space/_variable.py:67`): renders the offending components
  of a bound or value in an error message; it goes through `pretty_str`, so error
  snapshots depend on it.
- **`Variables`** (`space/design/_variables.py:48`): a `MutableMapping[str, Variable]`
  storing `__name_to_variable`, subclass-agnostic (only reads `.size`, `.type`, bounds).
  Owns the `type`-branching behaviors: `__compute_normalization_mask` (`:276`, branch
  `:290`), `get_integer_mask` (`:251`, branch `:263`), `has_integer_variable` property
  (`:269`, branch `:272`), `enable_integer_variables_normalization` setter (`:145`,
  branch `:151`). Constructs a `Variable` directly in `filter_components`
  (def `:219`, builds at `:231`). Carries the monotonic `version` / `bump_version`
  (`:131` / `:158`) bumped on every mutation.
- **`UnknownVariableError`** (`space/design/_variables.py:41`): a `KeyError` subclass
  whose `__str__` returns the raw message; `__delitem__` and `rename` report an unknown
  name by routing through `self[name]`.
- **`RegistryDerivedData`** (`space/design/_registry_derived_data.py:31`) +
  **`StalenessGuard`** (`space/design/_staleness_guard.py:28`): the caching abstraction.
  Collaborators derive from `RegistryDerivedData`, register one or more guards via
  `_register_guard(rebuild, name="")` (`:57`), and a guard's `rebuild` fires only when
  `_get_version_key()` (`:78`, default `Variables.version`) changed. This is where any
  per-variable polymorphic call belongs (rebuild path, not the hot path).
- **`Normalizer`** (`space/design/_normalizer.py:48`, `RegistryDerivedData`):
  `_rebuild` (`:86`), `normalize` (`:104`), `denormalize` (`:168`). It does not own the
  integer mask — it delegates rounding to an `IntegerRounder`.
- **`IntegerRounder`** (`space/design/_integer_rounder.py:32`, `RegistryDerivedData`):
  owns the integer-component mask, rebuilt from `Variables.get_integer_mask()`
  (`_rebuild`, `:51`); `has_integer` (`:59`), `round` (`:64`).
- **`Bounds`** (`space/design/_bounds.py:44`, `RegistryDerivedData`): concatenates the
  per-variable bounds into `full_lower_bound` / `full_upper_bound`, built lazily in
  `_rebuild` (`:128`). Subclass-agnostic (reads `.lower_bound` / `.upper_bound`) and
  hands out **read-only views** of the frozen bound arrays.
- **Helpers** (`space/design/_codec.py`): `split_full_value` (`:36`) /
  `concatenate_values` (`:50`). Shared constants live in `space/design/_constants.py`,
  where `BOUND_ATOL` is `100.0 * EPSILON` (`:23`).
- **Membership checks** (`space/design/_checking.py` — free functions taking
  `variables: Variables` / `bounds: Bounds`, **no `Membership` class**):
  `check_addable_value` (`:101`, integer branch `:164`), `check_out_array` (`:179`),
  `check_membership` (`:206`), `check` (`:257`), `_check_membership_array` (`:276`),
  `_check_membership_dict` (`:316`, per-component integer branch `:348-351`).
- **`Value`** (`space/design/_value.py:51`, `RegistryDerivedData`): default
  initialization (midpoint / finite bound / zero) in `initialize_missing` (`:331`,
  rule at `:348-355`, dtype branch at `:359-366`), `set` (`:222`, integer cast
  `:276-278`), `__compute_normalization_values` (`:512`, integer cast `:531`). It is the
  only **multi-guard** collaborator (three guards at `:107-109`) and overrides
  `_get_version_key` (`:125`) with a compound `(version, mutation_count)` key.
- **I/O** (`space/design/_io.py`): serializes `size`, `lower_bound`, `upper_bound`,
  `type` per component; reconstructs via
  `DesignSpace.add_variable(name, size, var_type, lb, ub, value)` on read
  (`to_hdf` `:81`, `from_hdf` `:138`, `to_csv` `:208`, `from_csv` `:232`; type branch
  `:195`). No `model_dump`; per-field access.
- **`View`** (`space/design/_view.py`): tabular export; `get_pretty_table` (`:39`)
  branches on `type` at `:78`.
- **`DesignSpace.add_variable`** (`space/design/__init__.py:310`): the public factory;
  constructs `Variable(...)` directly (`:342`) then `self._variables[name] = variable`
  (`:348`). The class also exposes `DesignVariableType = DataType` (`:123`) and
  `VARIABLE_TYPES_TO_DTYPES = TYPE_MAP` (`:126`).
- **`ParameterSpace`** (`space/parameter.py`): calls
  `add_variable(..., self.DesignVariableType.FLOAT, ...)` (`:333-336`).
- **Neighbours**: `space/factory.py` (`DesignSpaceFactory`) and `space/util.py` sit
  beside the design-space package.

### New Concepts Required

- **Abstract `BaseVariable`** — Pydantic base model holding the shared field
  `size` and declaring the **polymorphic interface** as `@abstractmethod`s.
  Being abstract (`inspect.isabstract` is True) it is never instantiated and is
  skipped by the factory's class discovery.
  It stays `frozen=True`, and the bound-conversion contract must be preserved in
  whichever subclass hook performs the conversion: dtype selection, `setflags(write=False)`
  freezing, and the `self.__dict__[...] =` write-back that bypasses the frozen model.
  The name is `BaseVariable`, not `Variable`, to match GEMSEO's `Base*` convention
  (`BaseFactory`, `BaseMDA`, `BaseDiscipline`).
- **Concrete subclasses (this phase)**, in a dedicated package
  **`src/gemseo/space/_variable/`** — the current `space/_variable.py` module is turned
  into a package (base + subclasses + factory), so the factory scan scope is tight.
  The package sits **outside** `space/design/`, at the import path the module already
  occupies, so `ParameterSpace` and `gemseo.enum` keep importing it from the same place.
    - **`ContinuousVariable`** — `type` pinned to `FLOAT`; `lower_bound` /
      `upper_bound`; normalized when both bounds finite; `float64` dtype.
    - **`IntegerVariable`** — `type` pinned to `INTEGER`; integer-bound
      validation; integer component mask; rounding; `int64` dtype;
      not normalized unless `enable_integer_variables_normalization`.
- **`VariableFactory(BaseFactory[BaseVariable])`** — the **GEMSEO factory** that
  builds variable instances, **not** a `BaseVariable.create` classmethod
  (`_CLASS = BaseVariable`, `_PACKAGE_NAMES = ("gemseo.space._variable",)`;
  singleton via the inherited `BaseABCMultiton`). `create(class_name, *args, **kwargs)`
  resolves the class by `__name__` via `get_class` and calls the Pydantic constructor.
  A helper `create(data_type, *args, **kwargs)` resolves a `DataType` to its
  subclass by reading each discovered subclass's pinned `type` default
  (`model_fields["type"].default`) — so adding a kind inside this package is just
  adding a subclass. A kind cannot be plugged in from outside, though: `DataType` is a
  closed `StrEnum`, so a new kind also requires a new `DataType` member and a new
  `TYPE_MAP` entry. Retiring `TYPE_MAP` in favour of `BaseVariable.component_type`,
  which would open the discriminator, is follow-up work for #1791. This is the single
  internal construction entry.
- **Package import contract** — `gemseo/enum/__init__.py:116` imports
  `DataType` from `gemseo.space._variable`, and the lazy-import map at `:275` binds
  `"DesignVariableType"` to the string `"gemseo.space._variable:DataType"`. Turning the
  module into a package therefore requires the package `__init__.py` to re-export
  `DataType`, plus `TYPE_MAP`, `BoundType` / `BoundArray` and `format_components`, which
  `space/design/*` import from the same module.
  `tests/test_enums.py::test_all_exports_are_enums` guards the lazy string, since it
  `getattr`s every name in `enum.__all__`.
- **Polymorphic methods on `BaseVariable`** (replacing scattered `type ==`
  branches), each mapping to a current site:

    | Method (on `BaseVariable`) | Replaces | Current site |
    |---|---|---|
    | `compute_normalization_mask(enable_integer_normalization)` | norm-policy branch | `space/design/_variables.py:276` (`__compute_normalization_mask`, branch `:290`), toggle setter `:145` (branch `:151`) |
    | `check_finite_bound_components(bound, prefix)` (in validator) | integer-bound check | `space/_variable.py:216` (inside `__check_bound` `:176`, driven by `__validate_variable` `:127-142`) |
    | `find_components_outside_domain(x_real)` | integer membership branch | `space/design/_checking.py:348-351` (`_check_membership_dict`), `:164` (`check_addable_value`) |
    | `compute_default_component(lb_i, ub_i)` | midpoint/bound/zero rule | `space/design/_value.py:348-355` (`initialize_missing`) |
    | `component_type` / `cast(value)` | `TYPE_MAP[type]` casting | `space/_variable.py:94`/`:161`, `space/design/_value.py:276-278`/`:359-366`/`:531`, `space/design/_view.py:78`, `space/design/__init__.py:126` |

### Conceptual Relationships

- **`Variables` holds a heterogeneous collection** of `BaseVariable` subclasses
  with no change to storage/iteration — it already treats elements
  polymorphically through `.size` / `.type` / bounds.
- **`Variables` / `Normalizer` / `IntegerRounder` / `Value` and the
  `_checking` functions call polymorphic methods** instead of switching on `type`.
  Aggregate arrays (integer mask, normalization mask) are still **built once inside
  the `RegistryDerivedData` guard `rebuild` callbacks, keyed on
  `Variables.version`** (and `Value`'s compound `(version, mutation_count)` key) — the
  per-variable method is called during cache rebuild, never in the hot transform path;
  vectorized arithmetic is preserved.
- **I/O reconstructs through the factory**: `from_hdf` / `from_csv` keep calling
  `add_variable(name, size, type, …)`, which routes to `VariableFactory`.
  Serialized bytes are unchanged (`type` already stored per component).
- **`DataType` remains the discriminator**: each subclass pins its `type`, so
  `variable.type` keeps working for any external reader, for `gemseo.enum.DesignVariableType`
  and for serialization.

### Key Business Rules

- **Behavior-preserving**: identical functional behavior, identical
  serialization layout, identical public API. Existing `tests/space/**` tests must
  pass **unchanged**.
- **`add_variable` signature unchanged**: still
  `add_variable(name, size, type_=FLOAT, lower_bound, upper_bound, value)`;
  it returns/stores the correct subclass internally.
- **`variable.type` still readable** and equal to the same `DataType` member as
  before, for every consumer and for HDF/CSV.
- **Bounds stay frozen**: whatever subclass hook converts a bound must keep the
  `setflags(write=False)` freeze, and `__setstate__` must keep re-freezing after
  unpickling.
- **Equality is data-based across the hierarchy**: two variables are equal iff
  same `type` + same `size` + same bounds — not merely same Python class. A
  `ContinuousVariable` and an `IntegerVariable` are unequal because their
  `type` differs, not because of `isinstance`.
- **The base class is abstract**: callers use `add_variable` /
  `VariableFactory`, never `BaseVariable(...)` directly.

## Use Cases

### UC-1: Construct a variable through the public API

`add_variable("n", type_=DataType.INTEGER, lower_bound=1, upper_bound=10)`
yields an `IntegerVariable`; `add_variable("x", lower_bound=0.0,
upper_bound=1.0)` yields a `ContinuousVariable`. The `Variables` registry stores the
heterogeneous instances; every downstream collaborator works unchanged.

### UC-2: Round-trip persistence is identical

A design space written to HDF/CSV before the refactor reloads to an **equal**
design space after it (same `type` strings, same bounds, same values). The
factory reconstructs the right subclass from the stored `type`.

### UC-3: A new kind extends the hierarchy, not the consumers

Adding `DiscreteVariable` (ticket 1791, v2) means adding one subclass that
implements the polymorphic interface — `Normalizer`, `Value` and the
`_checking` functions need **no new `if` branch**. Two things must still be
declared by hand: a new `DataType` member with its `TYPE_MAP` entry, and, for
a kind whose components are integral (a discrete kind over an integer grid,
say), its place in the integer mask. That mask is the one branch the hierarchy
does not carry: `Variables.get_integer_mask` and `Variables.has_integer_variable`
test `isinstance(variable, IntegerVariable)` (`space/design/_variables.py:260`
and `:268`), and feed `IntegerRounder._rebuild`
(`space/design/_integer_rounder.py:51`). A kind whose components are integral
without deriving from `IntegerVariable` would silently drop out of the rounding
mask.

## Strategic Approach

### Solution Direction

- Introduce an **abstract `BaseVariable`** + concrete `ContinuousVariable`
  and `IntegerVariable`, in the package `src/gemseo/space/_variable/`. Move the
  `type`-specific logic out of the single `__validate_variable` model validator and
  out of the collaborators **onto the subclasses** as the polymorphic methods
  tabulated above.
- Keep `type: DataType` as a **per-subclass-pinned discriminator** (a `Literal`
  default on each subclass). This preserves `variable.type`, the HDF/CSV
  layout, `gemseo.enum.DesignVariableType`, and any external `match variable.type`
  (no new enum members in this phase).
- Add a `VariableFactory(BaseFactory[BaseVariable])` (GEMSEO factory pattern); route
  the **two** direct construction sites through its singleton:
  `DesignSpace.add_variable` (`space/design/__init__.py:342`) and
  `Variables.filter_components` (`space/design/_variables.py:231`, must rebuild the
  entry with the **same** subclass as the source, via
  `factory.create(type(source).__name__, …)`).
- Replace each `variable.type == DataType.X` branch in `space/design/_variables.py`,
  `_normalizer.py`/`_integer_rounder.py`, `_checking.py`, `_value.py` and
  `_view.py` with a polymorphic call. Keep the guard-`rebuild`-cached-on-`version`
  pattern intact.
- Relax `__eq__` from `isinstance(other, self.__class__)` to
  `isinstance(other, BaseVariable)` **and** compare `type` + `size` + bounds.
- Preserve the copy/pickle hooks: `__copy__` / `__deepcopy__` return `self`,
  `model_copy` rebuilds through `model_validate` and must return the **same subclass**,
  `__setstate__` re-freezes the bound arrays.

### Key Design Decisions

- **Rich behavioral subclasses vs. separate `Domain` strategy objects**: a
  composition design (`Variable` + a `Domain` strategy) keeps one model shape
  but adds an indirection and a second object per variable. The user wants
  **variable objects**. → **Decided — rich subclasses**: behavior lives on
  `ContinuousVariable` / `IntegerVariable` (and later kinds).
- **Keep `DataType` as discriminator vs. drop it for `isinstance`**: dropping
  it would break serialization, `gemseo.enum.DesignVariableType` and external
  readers. → **Keep `DataType`**; each subclass pins its member; `isinstance` is used
  only internally where a method is the cleaner expression.
- **Base named `BaseVariable`**: consistent with GEMSEO's `Base*` convention and with
  the fact that the base is abstract and never instantiated. The rename ripples through
  `space/design/_checking.py`, `_io.py`, `_value.py`, `_variables.py`, `_view.py`,
  `space/design/__init__.py`, `gemseo/enum/__init__.py` and `tests/space/test_variable.py`.
- **Hierarchy lives in `space/_variable/`, not `space/design/_variable/`**: `Variable`
  is already shared between `DesignSpace` and `ParameterSpace` and re-exported by
  `gemseo.enum`; keeping it at the same import path avoids touching either consumer
  and keeps the deprecation aliases valid.
- **GEMSEO `BaseFactory` over a `create` classmethod** (user requirement):
  construction goes through `VariableFactory`, consistent with every other GEMSEO
  subsystem (Cache/MDA/DesignSpace factories). It keys on the class `__name__`, is a
  singleton via `BaseABCMultiton`, auto-discovers subclasses in `gemseo.space._variable`,
  and resolves a data type to its class without a hardcoded branch — none of which
  a classmethod gives. The abstract base is not instantiated; the factory is the sole
  internal constructor, so the two construction sites stay the only places that know
  how a variable is built.
- **Equality is data-based**: relaxed `__eq__` so the hierarchy does not change
  equivalence semantics for any existing test (guarded by comparing `type`).
- **Behavior-preserving scope**: no new `DataType` member, no IO format change,
  no new public method. The only observable change is the **Python class** of a
  variable instance (internal).

### Alternatives Considered

- **Status quo (single `Variable` + `type` switch)**: rejected — the user
  wants variable objects, and each new kind multiplies branch sites.
- **`Domain` strategy composition**: rejected for this story — extra
  indirection without the "variable is an object" ergonomics the user asked
  for. (Could still be used *inside* a subclass later if a kind needs pluggable
  domains.)
- **Putting the hierarchy under `space/design/_variable/`**: rejected — it would
  force `ParameterSpace` and `gemseo.enum` to import a design-space-private package,
  and would break the `gemseo.space._variable:DataType` lazy-import contract.
- **`Variable.create` classmethod factory**: rejected (user requirement) — a
  classmethod is inconsistent with GEMSEO's factory convention, gives no plugin
  discovery, and re-implements name→class dispatch that `BaseFactory` already
  provides. Use `VariableFactory` instead.
- **Pydantic discriminated-union `TypeAdapter` for parsing**: not required —
  I/O reconstructs through `add_variable` → `VariableFactory`, not by parsing a
  tagged union. Noted as an option if a direct `model_validate` path is ever needed.
- **Doing the hierarchy and the enumerated kinds in one MR**: rejected —
  bundling a refactor with a feature is exactly the scope problem this split
  solves. Hierarchy first, enumerated kinds (1791 v2) second.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Where do `ContinuousVariable` / `IntegerVariable` live?** **Decided** —
  dedicated package `src/gemseo/space/_variable/` (the current `space/_variable.py`
  module becomes a package holding base + subclasses + factory), so
  `VariableFactory._PACKAGE_NAMES = ("gemseo.space._variable",)` scans a tight scope
  and the existing import path is preserved. The singular `_variable` name does not
  clash with the registry module `space/design/_variables.py`: they sit in different
  packages.
- **Is the base exported publicly?** **Decided — no.** `BaseVariable` stays private in
  `space/_variable/`, re-exported only through the package `__init__` for internal
  typing; construction goes through `add_variable` / `VariableFactory`. This keeps the
  story behavior-preserving; a public export can follow with 1791 if users need to
  subclass.
- **Interaction with issue [#1812](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1812)**:
  `gemseo/enum/__init__.py:113` carries a TODO to rename `DataType` to
  `DesignVariableType` at the definition site (it clashes with `SobieskiBase.DataType`).
  Decide explicitly whether 1845 absorbs that rename or stays out of it — doing both at
  once would enlarge an otherwise behavior-preserving MR. **Recommendation: stay out.**

### Edge Cases

- **`filter_components`** (`space/design/_variables.py:219`) mutates the registry
  **in place**; it must rebuild the entry with the **same subclass** as its source
  (build site `:231`).
- **`ParameterSpace.add_variable(FLOAT)`** must transparently yield a
  `ContinuousVariable` — no `ParameterSpace` change (`space/parameter.py:333-336`).
- **Mixed continuous + integer space**: aggregate integer/normalization masks
  must be byte-identical to today (concatenation order preserved).
- **`__eq__` across subclasses**: `Continuous([0,1]) != Integer([0,1])` because
  `type` differs; a future `DiscreteVariable` with the same `type` as another
  kind cannot occur (each kind has a distinct `type`). The comparison is driven by
  `type(self).model_fields`, so a field added by a new kind takes part in it
  without touching `BaseVariable.__eq__`.
- **`model_copy` must not lose the subclass**: it rebuilds via
  `self.model_validate({**self.__dict__, **update})`, which returns the class it is
  called on — verify it stays so once `type` is a pinned `Literal` (an update that
  contradicts the pinned `type` must fail loudly rather than silently downcast).
- **Direct `Variable(...)` in existing tests**: any test constructing
  `Variable(type=…)` directly must move to `VariableFactory` / `add_variable`.
  Grep `tests/space/` during implementation, in particular
  `tests/space/test_variable.py` and `tests/space/design/test_variables.py`.
- **Snapshot churn**: exception messages emitted from the relocated validators
  must keep their current text, or `__snapshots__/*.ambr` regenerate (without
  `-n`, per the testing conventions). Those messages are rendered by
  `format_components` (`space/_variable.py:67`) through `pretty_str`, so its wording
  drives the baselines: `tests/space/__snapshots__/test_variable.ambr`,
  `tests/space/__snapshots__/test_design_space.ambr` and
  `tests/space/design/__snapshots__/test_checking.ambr`.
- **Unknown-name errors** flow through `UnknownVariableError`
  (`space/design/_variables.py:41`); no new error path introduced by the hierarchy may
  regress that message.

### Technical Risks

- **Pydantic abstract base + factory discovery + inherited validators**: the
  base must be `inspect.isabstract`-true (≥1 `@abstractmethod`) so
  `BaseFactory` skips it — verify Pydantic's metaclass composes with `ABCMeta`
  (a model that mixes `BaseModel` + `ABC` and declares an abstractmethod). Ensure
  the `@model_validator(mode="after")` on the base runs for subclasses, and that
  `frozen=True` and field defaults are inherited. The frozen model writes converted
  bounds via `self.__dict__[...] =` and freezes them with `setflags(write=False)`;
  keep both working through the subclass hook. Risk of validators firing in the wrong
  order; cover with construction tests per subclass.
- **Factory keys on `__name__`**: subclass names must be unique and stable
  (they are the factory keys); `create(class_name, **fields)` raises a clear
  error on an unknown name. `BaseFactory.create` wraps `TypeError`, but Pydantic
  raises `ValidationError`, which passes through unchanged.
- **Module-to-package conversion breaks imports if incomplete**: `space/design/*`
  import `DataType`, `TYPE_MAP`, `BoundType`, `BoundArray`, `Variable` and
  `format_components` from `gemseo.space._variable`, and `gemseo/enum/__init__.py:275`
  resolves the string `"gemseo.space._variable:DataType"` lazily. Every one of these
  must remain importable from the package root, and `tests/test_enums.py` must stay green.
- **Performance regression** (largely mitigated by the design): the
  per-variable polymorphic call must stay in the `RegistryDerivedData` guard
  **`rebuild`** path only; the per-`normalize_vect` hot path keeps operating on the
  cached aggregate masks. The `RegistryDerivedData`/`StalenessGuard` seam already
  isolates rebuilds to version changes, so the task is to keep the polymorphic calls
  inside those callbacks. Benchmark a high-dimensional space to confirm no per-call
  Python dispatch crept in.
- **Copy, pickling and deepcopy**: `Variable` ships `__copy__` / `__deepcopy__` /
  `model_copy` / `__setstate__` (`space/_variable.py:229-266`). Subclasses must inherit
  all four unchanged; in particular `__setstate__` must keep re-freezing the bound
  arrays, and a round-trip through pickle must return the same subclass with read-only
  bounds. Cover it with a per-subclass copy/pickle test.
- **Serialization back-compat**: old HDF/CSV (float/integer only) must reload
  identically; `VariableFactory` must later resolve `DISCRETE` / `CATEGORICAL`
  (added by 1791 v2) — automatic, since those subclasses self-register on import
  — without changing this phase's format.
- **Pickle of an earlier release**: such a pickle refers to the single `Variable`
  class by name, so that name must keep resolving; the legacy class rebuilds the
  variable through the factory on `__setstate__`, and `DesignSpace.__setstate__`
  replays the flat attribute layout through the components. One deliberate
  exception to a faithful restore: a stored value that falls outside the domain of
  the kind of its variable — an earlier release stored it untouched — is rejected
  on the way in, since restoring it would build a design space that its own
  membership check rejects.
- **Equality contract change** could surprise code relying on exact-class
  identity; low risk (no such site found) but document in the changelog.

### Acceptance Criteria Coverage

| AC# | Description | Addressable? | Notes |
|-----|-------------|--------------|-------|
| 1 | `BaseVariable` is an abstract base; `ContinuousVariable` / `IntegerVariable` are concrete subclasses | Yes | Base not directly instantiable (`@abstractmethod`s). |
| 2 | `add_variable(type_=FLOAT)` returns a `ContinuousVariable`; `type_=INTEGER` returns an `IntegerVariable` | Yes | Via `VariableFactory.create`. |
| 3 | `variable.type` still equals the original `DataType` member for every kind | Yes | Discriminator preserved. |
| 4 | All `type ==` branches in `space/design/_variables.py` / `_normalizer.py` + `_integer_rounder.py` / `_checking.py` / `_value.py` / `_view.py` are replaced by polymorphic calls | Yes | Table above lists each site. |
| 5 | HDF/CSV round-trip yields an **equal** design space (byte-identical layout) | Yes | Reconstruct via `VariableFactory`. |
| 6 | `__eq__` is data-based (`type` + size + bounds), not exact-class | Yes | Relax `isinstance`. |
| 7 | `filter_components` rebuilds the entry with the same subclass as its source | Yes | Route through `VariableFactory`; in-place mutation preserved. |
| 7b | `VariableFactory` is a `BaseFactory[BaseVariable]` singleton; discovers subclasses in `gemseo.space._variable`; no `create` classmethod exists | Yes | Mirror `CacheFactory`; `reset_factory` fixture clears the cache. |
| 7c | `gemseo.space._variable` still exports `DataType`, `TYPE_MAP`, `BoundType`, `BoundArray` and `format_components` after becoming a package | Yes | `gemseo/enum/__init__.py:275` resolves `"gemseo.space._variable:DataType"` lazily; `tests/test_enums.py` guards it. |
| 7d | Bound arrays stay read-only, including after `copy`, `deepcopy`, `model_copy` and unpickling | Yes | Inherit `__copy__` / `__deepcopy__` / `model_copy` / `__setstate__`. |
| 8 | `ParameterSpace` works unchanged (continuous yields `ContinuousVariable`) | Yes | No `ParameterSpace` edit. |
| 9 | **All existing `tests/space/**` tests pass unchanged** | Yes | Behavior-preserving; this is the headline AC. |
| 10 | Changelog fragment documents the (internal) refactor + equality-contract note | Yes | Towncrier: `changed`/`refactor`. |
