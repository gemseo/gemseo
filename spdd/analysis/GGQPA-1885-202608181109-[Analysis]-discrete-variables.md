<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: Discrete Variables in the DesignSpace

> **GitLab issue [#1885](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1885)**
> — "Discrete variables in the Design Space".
>
> This story **builds on** the variable class hierarchy of issue
> [#1845](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1845)
> ([MR !2629](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2629)) and merges
> **after** it. Everything below assumes the post-1845 code: `BaseVariable` /
> `ContinuousVariable` / `IntegerVariable` / `VariableFactory` in
> `src/gemseo/space/variable/`.
>
> Scope is deliberately **the data model only**. A discrete variable becomes
> declarable, checkable, persistable and viewable. Making it *solvable* — encoding or
> relaxing it so an optimizer or a DOE can drive it — is a separate story, as are
> unordered categorical variables and catalog-backed variables. See
> [Out of Scope](#out-of-scope).

## Original Business Requirement

> ## Objective
>
> The current DesignSpace allows to handle integer variables.
> Now, we want to add another type of variable: DiscreteVariable.
>
> It can contain a list of values (integers or continuous).
>
> For instance:
>
> $x \in \{1, 4, 6, 9\}$
>
> or
>
> $x \in \{0.4, 0.47\}$

### Clarifications obtained from the requester

The requirement above is a short feature statement; the following points were settled
with the requester before this analysis and are treated as requirements:

- A discrete variable is **scalar**: its size is always 1. A vector of discrete
  quantities is declared as several discrete variables.
- The potential values are **always numeric** (integer or real) and are **sorted** at
  construction. Unordered user input is sorted, not rejected.
- Declaration goes through a **new** `DesignSpace.add(name, variable, value=None)` that
  accepts a `BaseVariable` instance. `add_variable` is left untouched in this story; its
  deprecation in favour of `add` is a follow-up.
- Values are stored as NumPy arrays, per the GEMSEO-wide convention that a value is
  never a scalar.
- The tabular view gains a `potential_values` column, elided when the set is long.

## Domain Concept Identification

The design-space code lives in `src/gemseo/space/`: the façade and its collaborators in
`space/design/`, the variable hierarchy one level up in `space/_variable/`, where it is
shared with `ParameterSpace` and re-exported by `gemseo/enum/__init__.py`.

### Existing Concepts (from codebase)

- **`BaseVariable`** (`space/_variable/_base.py:100`): abstract Pydantic model,
  `frozen=True`, fields `size`, `type`, `lower_bound`, `upper_bound`, plus the
  `component_type` `ClassVar` (`:122`) holding the NumPy type of the components. A
  single `@model_validator(mode="after")` (`:137`) converts each bound
  (`__convert_bound`, `:154`) then checks it (`__check_bound`, `:186`), and finally
  enforces `upper >= lower`. `__convert_bound` picks the dtype from
  `self.component_type` (`:171`), **freezes** the array with `setflags(write=False)`,
  and writes it back via `self.__dict__[...] =` to bypass the frozen model. The
  polymorphic interface a new kind must satisfy is **one** `@abstractmethod`,
  `compute_normalization_mask` (`:280`), plus five overridable hooks that already carry
  a permissive default: `check_finite_bound_components` (`:293`, a no-op — *any finite
  component is a valid bound unless a subclass restricts the domain*),
  `find_components_outside_domain` (`:310`, returns `set()`), `compute_default_value`
  (`:262`), its per-component helper `compute_default_component_value` (`:240`, a
  `@staticmethod`) and `cast` (`:228`). `__eq__` (`:363`) compares **every field
  declared by either kind** — it walks the union of both `model_fields` — so a field
  added by a subclass takes part in the comparison with no edit.
- **`DataType`** (`space/_variable/_base.py:93`): `StrEnum{FLOAT, INTEGER}`. The
  module-level `TYPE_MAP` sits one level up, in `space/_variable/__init__.py:32`, and is
  **derived rather than written by hand**: a comprehension over the hard-coded tuple
  `(ContinuousVariable, IntegerVariable)` reading each class's `component_type`. Adding
  a kind therefore means editing that tuple and `__all__` in the same file. `DataType`
  is publicly re-exported as `gemseo.enum.DesignVariableType` (`enum/__init__.py:116`,
  lazy map at `:275`).
- **`ContinuousVariable`** (`space/_variable/_continuous.py:38`) and
  **`IntegerVariable`** (`space/_variable/_integer.py:74`): the two concrete kinds. Each
  pins its `type` as a `Literal` default and its `component_type` as a `ClassVar`
  (`float64` at `_continuous.py:41`, `int64` at `_integer.py:77`); the pinned `type`
  default is what makes the kind discoverable (see the factory below).
  `IntegerVariable.find_components_outside_domain` (`:106`) returns the indices of
  non-integer components; `compute_normalization_mask` (`:83`) returns all-`False`
  unless `enable_integer_normalization`.
- **`VariableFactory`** (`space/_variable/_factory.py`): `BaseFactory[BaseVariable]`
  scanning `gemseo.space.variable`. `create(data_type, *args, **kwargs)` (`:78`)
  resolves a `DataType` to its class through the cached `_data_type_to_class_name` map
  (`:46`), built by reading each discovered class's `model_fields["type"].default` and
  raising if two classes pin the same type. A new kind self-registers by existing as a
  module in that package — no factory edit. It **cannot** be plugged in from outside,
  though: `DataType` is a closed `StrEnum`, so a new kind also needs a new `DataType`
  member and a new `TYPE_MAP` entry. Retiring `TYPE_MAP` in favour of
  `BaseVariable.component_type`, which would open the discriminator, is follow-up work
  for #1791. `create_from_settings` (`:70`) raises `NotImplementedError`: there is no
  settings-based construction path.
- **`Variables`** (`space/design/_variables.py`): ordered, versioned
  `MutableMapping[str, BaseVariable]`; every mutation bumps `version`. Only
  `__compute_normalization_mask` (`:273`) delegates to a polymorphic hook;
  `get_integer_mask` (`:251`) and `has_integer_variable` (`:266`) test
  `isinstance(variable, IntegerVariable)`, so a kind that is not an `IntegerVariable` is
  excluded from both masks with **no hook to implement**. `filter_components` (`:216`)
  rebuilds an entry with `variable.model_copy(update=...)` (`:229`), which preserves the
  kind and every field of that kind, but re-validates through
  `model_validate({**__dict__, **update})` (`_base.py:333`) with both bounds passed
  explicitly.
- **`_variable/_legacy.py`**: a `Variable` shim (`:29`) kept so that a design space
  pickled before the 1845 split still loads; its `__setstate__` (`:56`) rebuilds through
  `VARIABLE_FACTORY.create` with `size` and the two bounds, and warns. Old pickles are
  never discrete, so this story adds nothing here — but the pickle path is no longer
  untouched territory.
- **`Bounds`** (`space/design/_bounds.py`): concatenates the per-variable bounds into
  `full_lower_bound` / `full_upper_bound` (`_rebuild`, `:128`) and hands out read-only
  views. `set_lower_bound` (`:104`) / `set_upper_bound` (`:116`) do not mutate: they
  rebuild the variable with `variable.model_copy(update={...})`.
- **`Normalizer`** / **`IntegerRounder`** (`space/design/_normalizer.py`,
  `_integer_rounder.py`): both `RegistryDerivedData`, caching aggregate masks keyed on
  `Variables.version`. Normalization is a linear `[lb, ub] → [0, 1]` map applied only
  where the normalization mask is `True`; rounding is applied only where the integer
  mask is `True`.
- **Membership checks** (`space/design/_checking.py`, free functions):
  `check_addable_value` (`:67`) validates a value before it is stored;
  `check_membership` (`:172`) dispatches to `_check_membership_dict` (`:335`, the
  per-variable path) or `_check_membership_array` (`:242`, the full-vector path);
  `check_domain` (`:311`) and `_check_index_in_domain` (`:282`) are two further sites
  that call `find_components_outside_domain` and phrase the failure themselves.
- **`Value`** (`space/design/_value.py`): owns the current values. `set` (`:219`),
  `set_variable` (`:277`), `initialize_missing` (`:336`, delegating to the
  whole-variable `variable.compute_default_value()` at `:349`, which casts to
  `component_type` itself), `to_complex` (`:324`), `check_value` (`:351`).
- **I/O** (`space/design/_io.py`): `to_hdf` (`:81`) writes one group per variable with
  the datasets named in `space/design/_constants.py` — `_SIZE_GROUP`, `_LB_GROUP`,
  `_UB_GROUP`, `_VAR_TYPE_GROUP`, `_VALUE_GROUP`. `from_hdf` (`:136`), `_to_dataframe`
  (`:168`), `to_csv` (`:202`) and `from_csv` (`:226`) all speak the five fields of
  `_TABLE_NAMES` (`_constants.py:49`). Both readers end by calling
  `add_variable(name, size, var_type, l_b, u_b, value)`.
- **`View`** (`space/design/_view.py`): `get_pretty_table` (`:38`) renders one row per
  scalar component over `_TABLE_NAMES`, reading `lower_bound[i]`, `upper_bound[i]`,
  `type` and the current value.
- **`DesignSpace`** (`space/design/__init__.py`): the façade. `add_variable` (`:310`) is
  today the single public construction entry, routing through `VARIABLE_FACTORY.create`
  (`:342`). `extend` (`:1390`) and `_add_variable_from` (`:1445`, reached from
  `add_variables_from` `:1435`) rebuild variables by destructuring them into
  `(size, type, lower_bound, upper_bound, value)`. `filter_dimensions` (`:277`),
  `set_lower_bound` (`:1095`), `to_complex` (`:1224`), `rename_variable` (`:1409`).
  The class also exposes `DesignVariableType = DataType` (`:123`) and
  `VARIABLE_TYPES_TO_DTYPES = TYPE_MAP` (`:126`).
- **`TYPE_MAP` consumers outside the space package**:
  `core/problem/database.py:1102` and `doe/core/base_doe_library.py:209` both index
  `VARIABLE_TYPES_TO_DTYPES` by a variable type.
- **Documentation** (`docs/user_guide/concepts/design_space.md:32`): currently reads
  *"a type, either `"float"` (continuous, default) or `"integer"` (discrete)"* — it uses
  "discrete" as a **synonym** for "integer". This story reclaims the word, so the
  sentence becomes wrong and must be corrected in the same MR.

### New Concepts Required

- **Potential values** — the finite, explicit, sorted set of numeric values a discrete
  variable may take. It is the **sole** definition of the variable's domain. This is the
  new domain vocabulary the requirement introduces, distinct from the
  `(lower_bound, upper_bound)` interval that defines a continuous or integer domain.
- **`DiscreteVariable`** (`space/_variable/_discrete.py`) — a `BaseVariable` subclass
  pinning `type` to a new `DataType.DISCRETE` member, pinning its `component_type`
  `ClassVar` to `float64`, and carrying the potential values as a frozen 1-D numeric
  array. It is **scalar by construction**: `size` is pinned to 1. Its bounds are
  **derived**, not supplied: the set is sorted, so the first and last elements are the
  tightest true bounds of the domain. It implements the polymorphic
  interface with a set-membership domain instead of an interval domain.
- **`DataType.DISCRETE`** — a third enum member, serialized as `"discrete"`. It is the
  discriminator that lets `VariableFactory`, HDF, CSV and every external reader of
  `variable.type` recognise the kind, exactly as `FLOAT` and `INTEGER` do today.
- **A per-kind membership diagnostic** — the 1845 hierarchy moved membership
  *detection* onto the variable but left the *wording* in the caller: `_checking.py`
  hard-codes "neither None nor integer while variable is of type integer" (`:136-138`)
  and "The variable X is of type integer; got …" (`:305`). A discrete variable needs to
  say that a value is not among its potential values. This is the **only** extension to
  the 1845 interface that this story forces: the message becomes a polymorphic
  responsibility of the variable.
- **A candidate-set accessor on the façade** — a space reloaded from HDF or CSV must be
  able to report the potential values of a variable without reaching into the private
  `_variables` registry.
- **`DesignSpace.add(name, variable, value=None)`** — a public method taking a
  `BaseVariable` instance. It is the only way to declare a discrete variable, because
  the potential values are a kind-specific field that `add_variable`'s
  `(size, type_, lower_bound, upper_bound, value)` signature cannot express. It also
  becomes the construction primitive that `add_variable` and the I/O readers delegate
  to.

### Conceptual Relationships

- **The potential values are owned by the variable**, like bounds. `Variables` stores a
  `DiscreteVariable` exactly as it stores any other kind; the registry, its ordering,
  its versioning and its index ranges are unaffected.
- **The set is the sole source of truth for membership**; the derived bounds are a
  *consequence* of the set, never an independent constraint. This ordering matters: a
  value inside the derived interval is still invalid unless it is one of the potential
  values.
- **`Bounds` and every bounds consumer keep working unchanged** precisely because the
  bounds are derived rather than absent. `full_lower_bound` and `full_upper_bound` stay
  finite and meaningful, the tabular view keeps rendering both columns, and nothing in
  the aggregate-array machinery learns about discrete variables.
- **`Normalizer` and `IntegerRounder` need no new branch**: a discrete variable reports
  an all-`False` normalization policy through `compute_normalization_mask`, and it falls
  out of the integer mask for free because `Variables.get_integer_mask` selects on
  `isinstance(variable, IntegerVariable)`. Both collaborators skip its component through
  the mask arithmetic they already perform.
- **`Value` derives the default from the set**, not from the bounds: the midpoint rule
  that `compute_default_component_value` implements for an interval domain has no meaning for
  a set, and would generally land between two potential values. The override happens at
  the whole-variable level, `compute_default_value`, which is what
  `Value.initialize_missing` calls.
- **`add` becomes the funnel**: `add_variable`, `extend`, `_add_variable_from`,
  `from_hdf` and `from_csv` all converge on it, so any future kind is persisted and
  copied without those five sites learning its fields.
- **`ParameterSpace`** keeps its deterministic-versus-random split untouched. Random
  variables remain continuous; a discrete random variable would need a discrete
  distribution and is a separate story.

### Key Business Rules

- **Membership**: a value is admissible **iff** it is one of the potential values.
  Comparison is strict equality — a user-supplied set must round-trip exactly, and no
  tolerance contract is introduced.
- **Scalarity**: a discrete variable has `size == 1`. Any other size is rejected at
  construction.
- **Numeric-only**: the potential values are integers or reals. Non-numeric entries are
  rejected at construction. The design vector must stay numeric; an object dtype would
  silently degrade every vectorized operation downstream.
- **Sorted and de-duplicated at construction**: the set is sorted ascending, and
  duplicates are rejected rather than silently collapsed — a duplicate is a user
  mistake, not an expressible intent.
- **Non-empty**: an empty set has no admissible value and is rejected.
- **Immutable**: the set is frozen at construction, like `size` and `type`. Changing it
  means removing the variable and adding a new one. This keeps the cache-invalidation
  story identical to today's.
- **Derived bounds are read-only**: `lower_bound` is the first potential value and
  `upper_bound` the last. Supplying either explicitly is an error, and
  `set_lower_bound` / `set_upper_bound` on a discrete variable fail with a message
  explaining that the domain is the set.
- **Default current value**: the **first** potential value. Deterministic and trivially
  testable, and it is by construction admissible.
- **Every value is an array**: a scalar accepted for convenience at the call site is
  promoted to a shape-`(1,)` array before storage, per the GEMSEO-wide convention.
- **Round-trip fidelity**: a space serialized to HDF or CSV and reloaded must be
  **equal** to the original, potential values included. A lossy round-trip is a defect,
  not a limitation.
- **Additive public surface**: every existing `add_variable` call, every existing
  serialized file and every existing behavior of continuous and integer variables is
  unchanged.

## Use Cases

### UC-1: Declare a design space mixing all three kinds

```python
ds = DesignSpace()
ds.add("x", ContinuousVariable(lower_bound=0.0, upper_bound=1.0), 0.5)
ds.add("n", IntegerVariable(lower_bound=1, upper_bound=10), 4)
ds.add("t", DiscreteVariable(potential_values=[0.72, 0.45, 0.55]), 0.55)
ds.add("n_plies", DiscreteVariable(potential_values=[2, 4, 6, 8]))
```

`t` stores its set sorted as `[0.45, 0.55, 0.72]`, reports `lower_bound == 0.45` and
`upper_bound == 0.72`, and accepts `0.55` as its current value. `n_plies` gets no
explicit value, so `initialize_missing_current_values` gives it `2` — the first
potential value, not the midpoint `5` of its derived interval. Assigning `0.6` to `t`
fails: it lies inside `[0.45, 0.72]` but is not one of the potential values.

### UC-2: Persistence round-trips the set

`ds.to_hdf(path)` writes the potential values as one dataset in each discrete variable's
group; `DesignSpace.from_hdf(path)` reconstructs a `DiscreteVariable` and reloads to an
**equal** space. `ds.to_csv(path)` writes the set as one space-free cell so the
whitespace-delimited format survives, and `from_csv` splits it back. A file written
before this story has no such column or dataset, and reloads exactly as it does today.

### UC-3: Reading a space that contains discrete variables

Printing the space shows the potential values, elided when the set is long enough to
break the table layout:

```text
+---------+-------------+-------+-------------+----------+---------------------------------+
| name    | lower_bound | value | upper_bound | type     | potential_values                |
+---------+-------------+-------+-------------+----------+---------------------------------+
| x       | 0.0         | 0.5   | 1.0         | float    |                                 |
| t       | 0.45        | 0.55  | 0.72        | discrete | [0.45, 0.55, 0.72]              |
| n_plies | 2           | 2     | 98          | discrete | [2, 4, ..., 96, 98] (50 values) |
+---------+-------------+-------+-------------+----------+---------------------------------+
```

A space with no discrete variable renders exactly as it does today, five columns wide.

## Strategic Approach

### Solution Direction

- **Add one subclass, plus two registration lines.** `DiscreteVariable` implements the
  1845 polymorphic interface, so `Variables`, `Normalizer`, `IntegerRounder` and the
  aggregate-mask machinery need **no new branch** — `Variables` because its two integer
  queries select on `isinstance(variable, IntegerVariable)`, the others because they
  read cached masks. The unavoidable non-polymorphic edits are the new `DataType`
  member and the `(ContinuousVariable, IntegerVariable)` tuple that builds `TYPE_MAP`
  in `space/_variable/__init__.py:32`. This is the payoff the 1845 refactor
  was done for, and the measure of whether this story is designed correctly: every
  `if variable.type == …` that would be needed is a design smell to remove.
- **Derive the bounds from the sorted set.** This is the pivotal decision. The naive
  reading of "the lower and upper bounds have no meaning" is to report `±inf`, but that
  would push a fake unboundedness into `Bounds`, the view, the normalization policy and
  every consumer of `full_lower_bound`. Because the set is sorted and numeric, its
  extremes *are* the true bounds of the domain, so deriving them keeps every existing
  bounds consumer correct and honest while the set does the real work in the membership
  check.
- **Enforce the domain in the membership layer, not the bounds layer.** The bound
  comparison stays as a necessary-but-insufficient check; the set-membership check
  runs alongside it through `find_components_outside_domain`, which the 1845 hierarchy
  already routes per variable.
- **Introduce `add` as the construction funnel.** A kind whose domain is a set cannot be
  described by `add_variable`'s bound-shaped signature. Rather than widening that
  signature, take the variable object itself — which the 1845 factory already builds —
  and let `add_variable` and both I/O readers delegate to `add`.
- **Fix the two places where the 1845 refactor stopped short**: the membership error
  wording (still hard-coded to "integer" in the caller) and the full-vector membership
  path (still bounds-only). Both are load-bearing for this story.
- **Keep the view honest but bounded.** The tabular view is a human artefact, not a
  serialization format: it elides a long set and appends the count. The full set is
  always available from the variable and from the accessor on the façade, and HDF and
  CSV never elide.

### Key Design Decisions

- **Derived bounds versus `±inf`**: reporting `±inf` is the literal reading of "bounds
  have no meaning", but it makes `Bounds._rebuild` mix infinities into arrays that today
  only see them for genuinely unbounded variables, and it makes the tabular view
  actively misleading. Deriving `values[0]` / `values[-1]` costs nothing (the set is
  already sorted) and keeps every consumer correct. → **Decided — derived bounds**,
  with explicit bounds rejected at construction and the two setters failing with a
  message naming the set as the domain.
- **Sort versus reject unordered input**: rejecting would surface a user typo, but the
  order of a set carries no information — `{4, 2}` and `{2, 4}` are the same domain — so
  rejecting punishes the user for a non-mistake. Duplicates are different: they *are* a
  mistake, since no intent is expressible by repeating a value. → **Decided — sort
  silently, reject duplicates.**
- **`size` pinned to 1 versus a vector of sets**: a per-component set is strictly more
  general, but it multiplies the validation, serialization and view surface, and the
  requirement's examples are all scalar. A vector need is expressible today as several
  variables. → **Decided — `size` pinned to 1** (per the requester). This also makes
  `filter_components` and `filter_dimensions` trivial for the kind.
- **`add(name, variable, value)` versus a `name` field on `BaseVariable`**: putting the
  name on the variable would let `add(variable, value)` read more naturally, but
  `Variables` is keyed by name, the model is frozen, and `rename` would have to rebuild
  the variable; `__eq__`, the factory call sites and both I/O readers would all have to
  learn about the field. → **Decided — the name stays outside the variable**, so the
  registry, renaming, equality and the factory are untouched.
- **`add` now, `add_variable` deprecated later**: `add_variable` has 81 call sites in
  `src/`, 588 in `tests/` and 133 in `docs/`. Deprecating it here would bury the
  discrete feature under mechanical migration churn and make the MR unreviewable. →
  **Decided — introduce `add`, leave `add_variable` untouched and undeprecated**;
  `add_variable` delegates to `add` internally. Deprecation and caller migration is a
  follow-up issue.
- **Dtype of a discrete variable**: `component_type` is a `ClassVar`, so it is fixed
  per kind, not per instance, and `TYPE_MAP` is derived from it — one dtype per
  `DataType` member, read by two consumers outside the space package
  (`core/problem/database.py:1102`, `doe/core/base_doe_library.py:209`). Inferring
  `int64` when every potential value happens to be integral would need a per-instance
  dtype, which the `ClassVar` shape forbids. → **Decided — `component_type = float64`
  for `DISCRETE`, unconditionally.** The cost is visible and small: `potential_values=[2, 4, 6, 8]`
  yields a current value rendered as `2.0`. The alternative is recorded under
  Ambiguities.
- **CSV serialization of the set**: `to_csv` defaults to a **space** delimiter, so a
  `[2, 4, 6, 8]` cell would corrupt the file. Refusing CSV export for discrete spaces
  would be safe but removes working functionality; a lossy CSV would be a silent defect.
  → **Decided — one space-free cell**, `2|4|6|8`, in a `potential_values` column, with
  `to_csv` raising if the caller's `delimiter` is the separator itself. `from_csv` reads
  the column from the string pass of `genfromtxt` and splits it.
- **View column always present versus conditional**: adding `potential_values` to
  `_TABLE_NAMES` unconditionally gives a dead column on the overwhelming majority of
  spaces, which are continuous-only. → **Decided — conditional**: the column is appended
  only when the space holds at least one discrete variable, so no existing output
  changes.
- **Elision rule**: the view must stay one line per component. → **Decided —** full list
  up to six values; beyond that, two head and two tail values around `...`, followed by
  the count, since the count is the only thing an elided list hides.
- **Membership message as a polymorphic hook versus a `type` branch in `_checking.py`**:
  a branch would work and is two lines, but it reintroduces exactly the pattern 1845
  removed, and the next kind would add a third arm. → **Decided — a formatting hook on
  the variable**, called by the `_checking.py` sites that phrase a domain failure:
  `check_addable_value` (`:136-138`), `_check_index_in_domain` (`:305`),
  `check_domain` (`:330`) and `_check_membership_dict` (`:367`).

### Alternatives Considered

- **Reuse `INTEGER` with an index convention**: encode the set as integer indices
  `0…n−1` and let the user maintain the index-to-value map. Rejected — it pushes the
  domain out of the design space and back onto the user, and the values the user reads
  back are not the values they declared.
- **An optional `potential_values` field on `BaseVariable`**: one model shape, no new
  subclass. Rejected — it makes a field whose meaning depends on `type`, forces every
  consumer to check both, and contradicts the 1845 decision that a kind is a class.
- **`values` as the field name**: shorter at the constructor call. Rejected in favour of
  `potential_values` so that one name spans the field, the HDF dataset, the CSV header
  and the view column; recorded under Ambiguities for confirmation.
- **Index-based normalization** (map the set to `[0, n−1]` and normalize that):
  rejected for this story — it is an *encoding*, which belongs to the solver-facing
  story, not to the data model. Here a discrete component is simply not normalized,
  which matches the default treatment of integer variables.
- **A driver guard rejecting discrete design spaces**: extending
  `BaseDriverLibrary._check_integer_handling` to raise, with a bypass flag, would turn
  "not yet solvable" into an explicit error. Rejected for this story (requester's call)
  to keep the MR inside `src/gemseo/space/`; the consequence is recorded as a known
  limitation below.
- **Delivering discrete and categorical together**: rejected — the two kinds share only
  the "finite set" idea. Categorical variables need label handling, forbid ordering, and
  cannot derive bounds at all, so bundling them doubles the review surface and forces
  premature abstraction over the two.

### Out of Scope

Explicitly not in this story, each tracked separately:

- **Solving.** No encoder, no relaxation, no working design space, no
  `handle_discrete_variables` capability flag, no `round_ints` rework.
- **A driver guard.** See the known limitation in Technical Risks.
- **Categorical (unordered) variables** and **catalog-backed variables**.
- **Deprecating `add_variable`** and migrating its callers.
- **Discrete random variables** in `ParameterSpace`.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Field name**: `potential_values` is used throughout this analysis, for the model
  field, the HDF dataset, the CSV header and the view column — one name end to end. The
  shorter `values` is the alternative and reads better at the constructor call
  (`DiscreteVariable(values=[2, 4, 6, 8])`), at the cost of a name that does not match
  the serialized artefacts. **Confirm with the requester before implementation.**
- **Dtype**: decided as `float64` unconditionally, so
  `potential_values=[2, 4, 6, 8]` gives a current value of `2.0`. The alternative —
  infer `int64` when every value is integral — reads better for ply counts and similar
  quantities, but `TYPE_MAP` admits one dtype per `DataType` member and two consumers
  outside the space package read it, so the variable and the map would disagree.
  Resolving that properly means making dtype a per-instance concern across those
  consumers, which is a wider change than this story. **Flagged, not silently assumed.**
- **`DataType.INTEGER` keeps its meaning**: a bounded integer range with rounding. It is
  **not** deprecated and **not** redefined, even though the documentation currently
  presents "discrete" as its synonym. The doc wording is what changes.
- **Comparison semantics for real-valued sets**: decided as strict equality. A value
  that has drifted by one ULP through arithmetic is rejected. This is deliberate — a
  tolerance contract would need its own semantics for overlapping neighbourhoods — but
  it means the eventual solver-facing story must snap values to the set before assigning
  them, rather than relying on near-equality here.
- **Interaction with `enable_integer_variables_normalization`**: no analogous toggle is
  introduced for discrete variables. A discrete component is never normalized in this
  story.

### Edge Cases

- **Empty set** (`potential_values=[]`): rejected at construction.
- **Singleton set** (`potential_values=[42]`): a legitimate fixed-value variable. Derived
  bounds are equal, which the normalizer's existing `lb == ub` guard already handles, and
  the normalization policy is `False` anyway.
- **Duplicates** (`[2, 4, 4]`): rejected, not de-duplicated.
- **Unsorted input** (`[6, 2, 4]`): sorted to `[2, 4, 6]`. The derived bounds and the
  default current value are computed **after** sorting.
- **Non-numeric entries** (`["a", "b"]`, `None`, `nan`): rejected at construction with a
  message naming the offending components, consistent with how `__check_bound` reports
  bad bound components through `format_components`.
- **Explicit bounds** (`DiscreteVariable(potential_values=[2, 4], lower_bound=0)`):
  rejected. This is also what makes `set_lower_bound` / `set_upper_bound` fail, since
  `Bounds.set_lower_bound` (`_bounds.py:104`) rebuilds through
  `model_copy(update={"lower_bound": …})` and the validator refuses.
- **`size != 1`**: rejected at construction.
- **A scalar current value**: promoted to shape `(1,)`. `add` must apply the same
  `atleast_1d` → `check_addable_value` → cast → store sequence that `add_variable`
  performs today (`space/design/__init__.py:349-368`), including its rollback: a value
  that fails validation must leave the variable removed rather than half-registered.
- **`filter_components` on a discrete variable** (`_variables.py:216`): the rebuild at
  `:229` uses `variable.model_copy(update=...)`, which now preserves the kind and the
  set — but `model_copy` re-validates through `model_validate({**__dict__, **update})`
  with `lower_bound` and `upper_bound` in the update, which is exactly what the
  explicit-bounds rejection refuses. With `size == 1` the only valid component list is
  `[0]`, so the operation is an identity — return the same frozen instance rather than
  rebuilding it. `DesignSpace.filter_dimensions` (`:277`) inherits this.
- **`extend` and `add_variables_from`** (`__init__.py:1390`, `:1445`): both destructure
  the variable into `(size, type, lower_bound, upper_bound, value)`, which silently drops
  the set. Routing both through `add` fixes it and — because variables are frozen and
  `__copy__` / `__deepcopy__` return `self` — lets the same instance be shared rather
  than rebuilt.
- **`rename_variable`**: safe by construction, the set travels on the variable object.
- **`to_complex`** (`_value.py:324`): a complex potential value is meaningless; discrete
  variables are skipped, mirroring how the method already leaves valueless variables
  alone.
- **HDF `append=True`**: writing a discrete variable into an existing file must add its
  dataset under that variable's group without disturbing siblings — the existing
  `require_dataset` pattern in `to_hdf` covers this, but the new dataset must follow it.
- **Legacy files**: an HDF file or CSV written before this story has no set dataset or
  column. Absent means "not discrete", and such files must keep loading byte-for-byte
  identically.
- **CSV delimiter collision**: `to_csv(delimiter="|")` on a space holding a discrete
  variable must raise rather than emit a file it cannot read back.
- **Mixed dtypes in one space**: a discrete (`float64`) variable beside an integer one
  goes through the existing `Value.common_dtype` promotion, which already handles
  float/int mixtures — no new path, but worth a targeted test.
- **`ParameterSpace` holding both a discrete deterministic variable and random
  variables**: equality between two spaces must compare the sets, and
  `BaseVariable.__eq__` (`_base.py:363`) already does — it walks the union of
  `type(self).model_fields` and `type(other).model_fields`, treats a field declared by
  only one kind as an inequality, and compares arrays elementwise with a shape guard.
  So `potential_values` participates for free and **no `__eq__` edit is needed**; the
  round-trip assertions are load-bearing rather than vacuous. Still assert the failure
  mode directly (two sets sharing extremes must compare unequal), since it is what the
  file-format ACs rest on.

### Technical Risks

- **Silently wrong results when solving.** With no driver guard, running any optimizer or
  DOE on a space containing a discrete variable treats it as a bounded continuous
  variable and can return values that are not among the potential values, with no
  warning. This is a **deliberate, accepted limitation** of this story and must be stated
  in the changelog fragment and the user documentation, not left implicit. It is the
  single most likely source of user confusion between this MR and the solver-facing one.
- **Equality is already covered — do not re-derive it.** `BaseVariable.__eq__`
  (`_base.py:363`) compares every field declared by either kind, so a subclass field is
  included with no edit. The residual risk is the opposite one: assuming it is missing
  and adding a redundant `__eq__` override on `DiscreteVariable`, which would then have
  to be kept in step with the base implementation. Assert the behavior, add no override.
- **Membership message coupling.** Four call sites in `_checking.py` currently phrase a
  membership failure as an integer problem (`:136-138`, `:305`, `:330`, `:367`). Moving
  the wording onto the variable changes the text emitted for `IntegerVariable` unless
  the hook reproduces it exactly; `tests/space/__snapshots__/test_design_space.ambr`,
  `tests/space/__snapshots__/test_variable.ambr` and
  `tests/space/design/__snapshots__/test_checking.ambr` are the baselines that will
  detect a drift. Regenerate with `pytest --snapshot-update` and **without** `-n`, per
  the testing conventions, and review the diff rather than accepting it.
- **The full-vector membership hole.** `_check_membership_array` (`_checking.py:242`)
  checks only against `full_lower_bound` / `full_upper_bound`. With derived bounds this
  path accepts any value inside the interval, so `check_membership(array)` would pass a
  non-candidate while `check_membership(dict)` rejects it — two public entry points
  disagreeing about validity. Either route the array path through
  `_check_membership_dict` when the space holds a discrete variable, or add the
  per-component set check there. Note the cost: the array path is the fast, fully
  vectorized one, so a naive fix pushes a Python loop into it.
- **Normalization hot path.** The 1845 design keeps polymorphic calls inside
  `RegistryDerivedData` rebuild callbacks, with the hot `normalize_vect` path operating
  on cached aggregate masks. The new kind must respect that: `compute_normalization_mask`
  is called on rebuild only, and the integer mask is a rebuild-time `isinstance` sweep in
  `Variables.get_integer_mask`. Verify no per-call dispatch creeps in, on a
  high-dimensional space.
- **`TYPE_MAP` completeness.** The map is derived in `space/_variable/__init__.py:32`
  from a hard-coded class tuple, so a new kind that is not added to that tuple is simply
  absent from it — silently, at import time. `core/problem/database.py:1102` and
  `doe/core/base_doe_library.py:209` index the map by variable type and will `KeyError`
  the first time a discrete variable reaches them unless the new member is registered.
- **Public enum surface.** `gemseo.enum.DesignVariableType` resolves lazily to
  `gemseo.space.variable:DataType` (`enum/__init__.py:275`), so the new member is public
  immediately. `tests/test_enums.py::test_all_exports_are_enums` guards the lazy string.
  Any third-party plugin that matches exhaustively on `DataType` with no default branch
  breaks — additive at the type level, not silent for exhaustive consumers. Changelog
  note required.
- **HDF and CSV forward compatibility.** Reading old files stays safe (absent field means
  not discrete), but a file written with a discrete variable cannot be read by an older
  GEMSEO. Document in the changelog. CSV is the more fragile of the two, since its
  header is positional and its delimiter is caller-supplied.
- **Pydantic mechanics.** The new subclass must pin `type` and `size` as `Literal`
  defaults and `component_type` as a `ClassVar`, validate and freeze the set the way
  `__convert_bound` (`_base.py:154`) treats bounds — `setflags(write=False)` plus the
  `self.__dict__[...] =` write-back that bypasses the frozen model — and must derive the
  bounds inside the same `mode="after"` validator, in the right order relative to the
  inherited bound checks. `__setstate__` (`:355`) must re-freeze the set after
  unpickling, as it already does for bounds, and `model_copy` (`:333`) must return a
  `DiscreteVariable`. Note that `__copy__` (`:323`) and `__deepcopy__` (`:330`) return
  `self`, and `model_copy` short-circuits to `self` on an empty update. Cover
  construction, copy, deepcopy and pickle per kind.
- **Factory discovery.** `VariableFactory` resolves a kind through
  `_data_type_to_class_name` (`_factory.py:46`), which reads
  `model_fields["type"].default` and **raises if two classes pin the same data type**,
  so the new module in `space/_variable/` self-registers. The map is cached and cleared
  by `update()` (`:117`), so the `reset_factory` fixture must be used in tests that
  assert on discovery, and the package `__init__.py` must re-export the new class — in
  `__all__`, and in the `TYPE_MAP` tuple — for internal typing.
- **Documentation drift.** `docs/user_guide/concepts/design_space.md:32` currently equates
  "discrete" with "integer". Leaving it would ship documentation that contradicts the
  feature in the same release.

### Acceptance Criteria Coverage

Issue 1885 states a feature, not enumerated ACs. The ACs below are derived for
implementation and cover only this story's scope.

| AC# | Description | Addressable? | Gaps/Notes |
|-----|-------------|--------------|------------|
| 1 | `DesignSpace.add(name, variable, value=None)` accepts any `BaseVariable`; `add_variable` behavior is unchanged and delegates to it | Yes | Includes `add_variable`'s existing rollback-on-invalid-value semantics. |
| 2 | `DiscreteVariable(potential_values=[…])` pins `type` to `DataType.DISCRETE` and `size` to 1; other sizes rejected | Yes | Kind self-registers with `VariableFactory` via its pinned `type` default. |
| 3 | Potential values are numeric, sorted ascending at construction, non-empty, duplicate-free, and stored as a frozen array | Yes | Non-numeric, empty and duplicate inputs each get a distinct error message. |
| 4 | `lower_bound` / `upper_bound` are derived as the first and last potential values; explicit bounds rejected; `set_lower_bound` / `set_upper_bound` fail with a domain-naming message | Yes | The setter failure comes from the validator via `Bounds.set_lower_bound`'s `model_copy`. |
| 5 | Membership accepts a value iff it is a potential value, on **both** `check_membership` paths (mapping and full array) | Yes | The array path (`_checking.py:242`) is bounds-only today; closing it is required, and the vectorization cost must be watched. |
| 6 | The membership failure message names the potential values, and the `IntegerVariable` message is textually unchanged | Yes | New polymorphic hook, called from four `_checking.py` sites; three snapshot files are the baseline. |
| 7 | A discrete component is not normalized and not rounded | Yes | All-`False` norm policy; the integer mask excludes the kind via `isinstance`, with no hook to write. Assert `normalize_vect` and `round_vect` are the identity on it. |
| 8 | `initialize_missing_current_values` sets the first potential value, not the midpoint of the derived bounds | Yes | Overrides `compute_default_value`, which `Value.initialize_missing` calls. |
| 9 | A scalar current value is stored as a shape-`(1,)` array | Yes | GEMSEO-wide convention; assert the shape, not just the value. |
| 10 | HDF round-trip yields an **equal** space, potential values preserved, including with `append=True`; legacy files still load | Yes | New dataset per variable group; absent means not discrete. |
| 11 | CSV round-trip yields an **equal** space via the `potential_values` column serialized as `2\|4\|6\|8`; `to_csv` raises when `delimiter` is the separator; legacy files still load | Yes | Read the column from the string pass of `genfromtxt`. |
| 12 | Two discrete variables sharing derived bounds but not their sets compare unequal | Yes | Already satisfied by `BaseVariable.__eq__`, which walks both kinds' `model_fields`. Assert it; add no override. |
| 13 | The tabular view shows `potential_values` only for spaces holding a discrete variable, eliding long sets as `[2, 4, ..., 96, 98] (50 values)` | Yes | Continuous-only spaces render byte-identically to today. |
| 14 | `extend`, `add_variables_from` and `rename_variable` preserve the potential values | Yes | Requires routing the first two through `add`. |
| 15 | `filter_components` / `filter_dimensions` on a discrete variable is an identity | Yes | Only `[0]` is valid at `size == 1`; must not go through `model_copy`, whose re-validation passes explicit bounds. |
| 16 | The potential values are readable from the façade without touching `_variables` | Yes | Needed by any space reloaded from a file. |
| 17 | `ParameterSpace` works unchanged, including a discrete deterministic variable beside random variables | Yes | No `ParameterSpace` edit expected; verify equality semantics. |
| 18 | Copy, deepcopy and pickle round-trips return a `DiscreteVariable` with a read-only set | Yes | Mirrors the 1845 per-kind copy/pickle coverage. The `_variable/_legacy.py` shim is untouched: a pre-1845 pickle is never discrete. |
| 19 | `docs/user_guide/concepts/design_space.md` no longer presents `"integer"` as meaning "discrete", and documents the new type | Yes | Code and docs agree within the same MR. |
| 20 | All existing `tests/space/**` tests pass unchanged | Yes | Additive story; snapshot updates are the only sanctioned churn, and each must be reviewed. |
| 21 | Changelog fragment (`added`) documents the new variable type, the new `add` method, the file-format change, the widened `DataType`, and **explicitly** the limitation that discrete variables are not yet solvable | Yes | The limitation is the highest-value line in the fragment. |
