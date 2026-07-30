<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: DesignSpace Composition Façade Refactor

## Original Business Requirement

The `DesignSpace` class is too big and has too many responsibilities. The purpose of
this user story is to refactor it following the **composition façade approach**
defined in section 2.1 of GitLab work item
[gemseo/dev/gemseo#1801, note 3336536205](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1801#note_3336536205).

The referenced "Proposal B: Compose Focused Parts (Façade Approach)" describes:

> **Overview** — Reorganize the monolithic `DesignSpace` class by decomposing it into
> focused, single-responsibility components while maintaining a public façade that
> preserves the existing API.
>
> **Proposed Structure**
>
> `text ``
> algos/design_space/
> ├── **init**.py # DesignSpace facade
> ├── variables.py # VariableSet (versioned)
> ├── bounds.py # BoundsAccessor
> ├── state.py # CurrentValue
> ├── normalizer.py # Normalizer (policy + transforms, cached)
> ├── membership.py # check_membership
> ├── codec.py # convert_array_to_dict / convert_dict_to_array
> ├── io.py # HDF + CSV free functions
> └── view.py # pretty_table, repr
>
> `text ``
>
> **Key Components**
>
> - **VariableSet**: variable container with explicit versioning to track mutations
>   and invalidate dependent caches.
> - **BoundsAccessor**: lower/upper bound retrieval and manipulation, active bounds
>   calculation and projection logic.
> - **CurrentValue**: getting/setting current state, initialization, complex-number
>   conversion.
> - **Normalizer**: normalization policies, forward/inverse transforms, gradient
>   scaling, rounding — with explicit cache invalidation tied to `VariableSet.version`.
> - **Codec**: bidirectional conversion between array and dictionary representations,
>   plus common dtype tracking.
> - **Supporting modules**: `membership.py` validates membership; `io.py` and
>   `view.py` handle serialization and rendering as separate concerns.
>
> **Benefits** — Eliminates silent staleness bugs by making cache invalidation
> explicit and observable. Changes to variable definitions (via
> `VariableSet.version`) automatically invalidate dependent caches in `Normalizer`
> and `BoundsAccessor`. Each component becomes independently testable, and new
> behaviors (log-scale transforms, mixed-integer support) can be introduced by
> swapping implementations. The `DesignSpace` façade preserves the public API while
> delegating to collaborators, so existing code requires no changes.
>
> **Trade-offs** — Medium-effort refactoring (~7 new files), introduces an
> intermediate composition layer. Long-term ergonomic gains and foundational
> improvements to correctness.

## Domain Concept Identification

### Existing Concepts (from codebase)

- **DesignSpace** (`src/gemseo/algos/design_space.py`, ~2551 LOC, single class):
  current monolith. Holds variables, bounds, current value, normalization policy
  and cache, membership/projection logic, codec (array↔dict), HDF/CSV I/O, pretty
  table rendering, filtering, renaming, extension. Today owns roughly 80 public
  and private methods spanning all of the responsibilities listed above.
- **Variable** (`src/gemseo/algos/_variable.py`): Pydantic `BaseModel` describing
  a single variable (size, `DataType`, lower/upper bound, with validation). Stays
  unchanged; the new `VariableSet` is a versioned container of these.
- **DataType** (`_variable.py`): `StrEnum` of FLOAT / INTEGER. Drives integer
  rounding in the normalizer and dtype handling in the codec.
- **ParameterSpace** (`src/gemseo/algos/parameter_space.py`): subclass adding
  uncertain variables, distributions, copulas. Must keep working through the
  façade without internal-attribute breakage.
- **Problem-specific subclasses**: `AerostructureDesignSpace`, `SellarDesignSpace`,
  `SobieskiDesignSpace`, `ScalableDesignSpace` — all inherit from `DesignSpace`
  and only call public construction methods (`add_variable`, `set_current_value`).
- **DesignSpaceFactory** (`src/gemseo/algos/design_space_factory.py`): plugin
  factory using `DesignSpace` as `_CLASS`. Plugin discovery contract must hold.
- **design_space_utils.get_value_and_bounds**: free helper already calling the
  public API; reference example of the desired façade-only access pattern.

### New Concepts Required

- **VariableSet** — ordered, versioned collection of `Variable`s with the
  name→indices map, total dimension, integer-component mask, normalization
  policy per variable. Single source of truth for "what variables exist."
  Emits/exposes a monotonically increasing `version` on every mutation.
- **BoundsAccessor** — read/write of per-variable and aggregate lower/upper
  bounds (`get_lower_bound[s]`, `get_upper_bound[s]`, `set_lower_bound`,
  `set_upper_bound`, `get_active_bounds`, `project_into_bounds`). Caches array
  views keyed by `VariableSet.version`.
- **CurrentValue** — current-state container: `set_current_value`,
  `set_current_variable`, `get_current_value`, `has_current_value`,
  `initialize_missing_current_values`, `to_complex`, complex→real conversion.
  Owns the in-memory `__current_value` dict and the cached concatenated array.
- **Normalizer** — normalization policy plus transforms: `normalize_vect`,
  `unnormalize_vect`, `transform_vect`, `untransform_vect`, `normalize_grad`,
  `unnormalize_grad`, `round_vect`, and `enable_integer_variables_normalization`.
  Holds `_norm_factor`, `_norm_factor_inv`, `__norm_inds`, `__integer_components`,
  `__no_integer`, `__common_dtype`, with cache invalidation observable via
  `VariableSet.version`.
- **Membership** — `check_membership`, `check`, internal x_vect / dict variants.
  Pure validators that read from `VariableSet` + `CurrentValue`.
- **Codec** — `convert_array_to_dict`, `convert_dict_to_array`,
  `names_to_indices`, `__get_common_dtype`. Stateless apart from the indices
  view derived from `VariableSet`.
- **io module** — free functions: `to_hdf` / `from_hdf` / `to_csv` / `from_csv`
  / `to_file` / `from_file` (today methods/classmethods on `DesignSpace`).
- **view module** — `get_pretty_table`, `_repr_html_`, `__repr__`, `__str__`,
  `_get_string_representation`.
- **DesignSpace façade** — orchestrates the seven collaborators, exposes the
  same public surface used today by 60+ importers in the repository.

### Conceptual Relationships

- `DesignSpace` (façade) **owns** one `VariableSet`, one `BoundsAccessor`, one
  `CurrentValue`, one `Normalizer`, one `Codec`. Membership/io/view are mostly
  free functions taking these components as arguments.
- `VariableSet.version` is the **invalidation signal**: `BoundsAccessor`,
  `Normalizer`, `Codec` cache derived data keyed by this version and recompute
  on miss.
- `CurrentValue` depends on `VariableSet` (names/sizes/types) and on
  `Normalizer` (to produce normalized variants).
- `Normalizer` depends on `VariableSet` (policy, integer mask) and
  `BoundsAccessor` (lower/upper arrays).
- `io` / `view` depend on all of the above but only via their public methods,
  no internal state.
- Subclasses (`ParameterSpace`, problem-specific ones) sit above the façade,
  must not see the decomposition.

### Key Business Rules

- **API preservation**: the public surface of `DesignSpace` documented for users
  must remain backward-compatible. Imports `from gemseo.space.design
  import DesignSpace` must keep working unchanged.
- **Cache coherence**: every mutation of variables, bounds, or normalization
  policy must invalidate downstream caches before the next read. Today this is
  done by manually flipping `__norm_data_is_computed = False` in scattered
  call sites; the new design must enforce it through `VariableSet.version`.
- **Integer-variable handling**: bounds with integer type are checked for
  integer or infinite components; rounding is applied after unnormalization;
  normalization can be opted in via `enable_integer_variables_normalization`.
- **lb == ub guard**: when `_norm_factor == 0`, the inverse uses a safe
  divisor (`where(==0, 1, factor)`) — must be preserved.
- **Complex-number support**: `to_complex` converts current values to
  `complex128`; transforms must be tolerant of complex dtypes (used by adjoint
  / complex-step gradients).
- **HDF & CSV format compatibility**: existing files written by `to_hdf` /
  `to_csv` must remain readable by the refactored `from_hdf` / `from_csv` and
  vice-versa.
- **Subclass extension points**: `ParameterSpace` overrides `__init__`,
  pretty-table rendering, and adds methods that read deterministic-variable
  state — the façade must expose enough hooks for it to keep working without
  reaching into private attributes.

## Strategic Approach

### Solution Direction

- Introduce a new package `src/gemseo/algos/design_space/` whose `__init__.py`
  re-exports the `DesignSpace` façade class. Keep
  `from gemseo.space.design import DesignSpace` valid (Python treats the
  package as the module). The single-file `design_space.py` is removed in the
  same change.
- Extract responsibilities into the seven proposed modules. Each module owns a
  small class (or set of free functions) with a narrow contract.
- Drive cache invalidation by a monotonically increasing `VariableSet.version`
  bumped on add/remove/filter/rename/dimension-filter/extend and on bound
  mutations that affect the normalization factor (or expose a separate
  `BoundsAccessor.version` if bounds should not invalidate everything).
- `DesignSpace` becomes a thin coordinator: each public method delegates to one
  of the collaborators. State previously held on `DesignSpace` (e.g.
  `__lower_bounds_array`, `__norm_inds`, `__current_value`) moves into the
  owning collaborator.
- Keep `design_space_utils.get_value_and_bounds` and `DesignSpaceFactory`
  unchanged — they only consume the public API.

### Key Design Decisions

- **Package vs. module for `design_space`**: trade-off between minimal
  blast-radius (keep a single file, add a `_design_space/` package alongside)
  and the cleaner layout proposed in §2.1. → **Adopt the package layout** as
  specified, since the façade lives in `__init__.py` and existing imports keep
  working. Risk: any code that internally referenced
  `gemseo.space.design` as a *module attribute* (rare) breaks.
- **Version source**: option A — single `VariableSet.version`; option B — a
  version per concern (variables vs. bounds vs. current value). → **Start
  with single version on `VariableSet`** (matches §2.1) and add finer-grained
  versions only if profiling shows redundant recomputation. Simpler invariant,
  matches today's coarse `__norm_data_is_computed` flag.
- **Collaborator lifetime**: option A — collaborators created in
  `DesignSpace.__init__` and reused; option B — recreated lazily on every
  call. → **Create once, hold references**, so caches survive across calls.
- **IO and view as functions vs. methods**: §2.1 prescribes free functions. →
  **Free functions in `io.py` / `view.py`**, but keep thin façade methods
  (`to_hdf`, `to_csv`, `get_pretty_table`, `_repr_html_`, `__repr__`,
  `__str__`) that delegate to them. Preserves public API and discoverability.
- **`ParameterSpace` adaptation**: option A — keep `ParameterSpace` as a
  subclass of the façade; option B — turn it into another façade composing
  the same collaborators plus a `DistributionSet`. → **Keep subclass relation
  for this story**, scope a follow-up issue for option B. The current US is
  about `DesignSpace`; subclass changes should be the minimum required to
  unblock the refactor.
- **Naming of private accessors moved on collaborators**: option A — preserve
  current name-mangled `__x` attributes by re-exposing them; option B — drop
  them and rely on the façade. → **Drop private attribute access at the
  façade level**; audit downstream code and treat any `_DesignSpace__*` or
  protected attribute reach-in as a bug to fix in the same change.
- **Migration strategy**: option A — big-bang single MR; option B — strangler
  pattern (introduce collaborators behind feature flag, migrate methods one
  by one). → **Strangler-style sequence inside one branch**: land
  `VariableSet` first (with version + indices), then `Codec`, then
  `BoundsAccessor`, then `Normalizer`, then `CurrentValue`, then split
  `io`/`view`/`membership`. Each step keeps tests green. The user's branch is
  `design-space-refacto` — long-lived, one MR is acceptable as long as commits
  are scoped.

### Alternatives Considered

- **Proposal A "extract mixins"** (implicit alternative): split responsibilities
  via multiple inheritance. Rejected — multiplies the MRO without giving
  independent testability or explicit cache invalidation.
- **Inlining everything into `Variable`**: rejected — `Variable` is a Pydantic
  model used elsewhere; loading it with cross-cutting concerns would couple
  validation to caching.
- **Per-variable lazy `Normalizer`**: rejected — current normalization is
  vectorized across the full design vector; per-variable caching would lose
  the vectorized fast path.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Scope of "preserve the public API"**: the requirement implies no callsite
  changes outside `algos/design_space*`. Clarify whether protected attributes
  starting with a single underscore (`_variables`, `_current_value`,
  `_lower_bounds`, `_upper_bounds`, `_norm_factor`, `_norm_factor_inv`,
  `_add_variable_from`, `_check_variable_name`, `_check_value`,
  `_check_current_value`, `_check_current_names`, `_add_norm_policy`,
  `_get_string_representation`) are part of the "API" or fair game to move.
  `ParameterSpace` reaches into several of these — its behavior is the de
  facto contract.
- **HDF group-name constants**: `DESIGN_SPACE_GROUP`, `NAME_GROUP`, etc. are
  `ClassVar` on `DesignSpace` today and externally observable. Clarify
  whether they remain on the façade or move to `io.py` (recommend keeping
  them on the façade as re-exports).
- **`DesignVariableType` alias**: `DesignSpace.DesignVariableType = DataType`
  is a documented entry point — clarify whether it stays on the façade
  (recommended) or only on `_variable`.
- **`VARIABLE_TYPES_TO_DTYPES` ClassVar**: same question.
- **Behavior of `__eq__`, `__contains__`, `__iter__`, `__len__`**: equality
  semantics today compare variable dicts and current values; clarify whether
  `ParameterSpace`-specific state (distributions) participates in equality
  (it does today via the inherited method reading only `_variables` and
  current values).
- **Bound mutation invalidation**: does `set_lower_bound` / `set_upper_bound`
  bump `VariableSet.version`, or does `BoundsAccessor` have its own version?
  §2.1 only specifies the variables version.

### Edge Cases

- **Empty design space**: every collaborator must accept zero variables
  without raising — used in tests and by formulations during initialization.
- **lb == ub**: avoid divide-by-zero in `Normalizer`, currently handled by
  `where(==0, 1, factor)`. Must be preserved in the extracted code.
- **All-integer design space with `enable_integer_variables_normalization`
  off**: `normalize_vect` must short-circuit; today guarded by `__norm_inds`.
- **Sparse arrays in `normalize_vect` / `unnormalize_vect`**: special-cased
  via `isinstance(out, sparse_classes)`. Must remain.
- **`out=` aliasing**: many transform methods accept an `out` array that
  may alias `x_vect`. The extracted `Normalizer` must keep the exact aliasing
  semantics, including the `out *= 0; out = x_vect` quirk on line 1414.
- **Complex current values + integer variables**: `__get_common_dtype`
  upgrades dtype across variables; the codec must preserve this when arrays
  and dicts round-trip.
- **`filter_dimensions` reshaping current value**: cache invalidation must
  survive a partial-dimension keep.
- **Deepcopy in `filter(copy=True)`**: collaborators must be deep-copyable
  (no unpicklable handles, no `weakref` to outer façade).
- **CSV round-trip with missing fields**: today `from_csv` infers columns;
  the free function in `io.py` must keep the same lenient parsing.
- **HDF append mode**: `to_hdf(append=True)` writes into an existing file;
  the free function must keep this signature.
- **Subclass `_repr_html_` override** in `ParameterSpace`: the view module
  must accept a delegated render path that subclasses can extend.

### Technical Risks

- **Plugin discovery**: `DesignSpaceFactory._CLASS = DesignSpace` must keep
  resolving. Risk: if `from gemseo.space.design import DesignSpace` is
  re-routed through a package import that triggers side effects, plugins may
  fail to load. Mitigation: keep `__init__.py` import-light; no top-level
  factory instantiation.
- **Pickle / serialization**: `DesignSpace` instances are pickled by some
  scenarios (caching, MDA history). Moving attributes onto collaborators
  changes the pickle layout. Mitigation: implement `__getstate__` /
  `__setstate__` on the façade for an upgrade path, or accept a breaking
  change and gate it via the changelog (`changelog/` towncrier fragment).
- **Performance regression**: today many caches are inlined; adding
  collaborator dispatches risks per-call overhead in hot paths
  (`normalize_vect`, `convert_array_to_dict`). Mitigation: keep the
  collaborators referenced via plain attributes (no `__getattr__`
  delegation), inline hot dispatches if profiling shows >5% regression on
  scenario suites.
- **Backward-compatible private state**: any third-party plugin reaching
  into `_DesignSpace__lower_bounds_array` etc. will break. Mitigation:
  search across the codebase confirmed only internal usage, but document
  in the changelog.
- **`ParameterSpace` over-reach**: it currently overrides private/protected
  internals (e.g. pretty table rendering uses superclass attributes). Each
  reach-through must be re-routed to the façade's new public API or to a
  new protected hook on the façade.
- **Test surface**: tests in `tests/algos/` directly instantiate
  `DesignSpace` and inspect attributes — extraction must keep tests green or
  update them as part of the same MR (no separate stabilization step).
- **Snapshot tests**: project convention prefers `assert_raises_snapshot`
  (`__snapshots__/*.ambr`). Changes to exception messages from the
  refactor must be tracked via `pytest --snapshot-update` (without `-n`).

### Acceptance Criteria Coverage

The original requirement is a refactoring user story without enumerated ACs.
The implicit ACs are derived below and assessed.

| AC# | Description | Addressable? | Gaps/Notes |
|-----|-------------|--------------|------------|
| 1 | New package `algos/design_space/` exists with the 8 modules from §2.1 | Yes | Decide whether to keep `algos/design_space.py` as a compat shim (probably no, since the package shadows it). |
| 2 | `DesignSpace` public API unchanged (signatures + behavior) | Yes | Need an explicit list of what counts as "public" — see ambiguity on protected attrs. |
| 3 | All existing tests pass without modification, OR test updates are limited to test-only files | Yes | Some tests read private state; those tests need updates and that should be called out. |
| 4 | `VariableSet.version` increments on every mutation, observable in tests | Yes | Need test cases per mutation method. |
| 5 | Caches (`Normalizer`, `BoundsAccessor`) invalidate on version change | Yes | Verify via a unit test that bumps version manually and asserts recomputation. |
| 6 | `ParameterSpace` and problem-specific subclasses continue to work | Yes | May require small adjustments in `ParameterSpace`'s pretty-table override. |
| 7 | HDF and CSV files written by the old code remain readable | Yes | Add round-trip regression tests using a checked-in fixture file. |
| 8 | Plugin discovery (`DesignSpaceFactory`) keeps working | Yes | Smoke test loading at least one problem subclass via the factory. |
| 9 | No performance regression beyond a documented budget on scenario benchmarks | Partial | The project does not have a standard performance gate — propose ad-hoc timing of `normalize_vect` / `unnormalize_vect` on a 10k-variable space as a sanity check. |
| 10 | Changelog fragment in `changelog/` documents the internal refactor | Yes | Towncrier convention; classify as `refactor` (or whichever type matches the project's towncrier config). |
