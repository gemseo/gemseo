<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: Split Discipline Local Data into Input and Output Stores

## Original Business Requirement

## Split discipline local data into input and output stores

## Context

`gemseo.core.discipline.io.IO` currently stores both inputs and outputs of the
last discipline execution in a single `DisciplineData` dict exposed as
`IO.data` (mirrored on `BaseDiscipline.local_data`). Inputs and outputs share
the same namespace; only the grammars distinguish which keys are which. This
mixing forces filtering by grammar every time callers need a side
(`get_input_data`, `get_output_data`, `_store_cache`, auto-coupling deepcopy),
and makes it easy to silently overwrite an input with an output of the same
name (auto-coupled variables).

## Goal

Split `IO.data` into two separate stores — one for input data, one for output
data — so the input/output side of every value is explicit and there is no
ambiguity at any execution stage.

## Drivers

1. **Clarity / API** (primary): make explicit which keys are inputs vs.
   outputs at any stage of the execution lifecycle (init, run, finalize,
   cache load).
2. **Performance** (secondary, follow-up): once the data is split, remove
   per-call filtering by grammar and reduce dict copies during execution and
   MDA coupling.

## Backward compatibility

- Keep `BaseDiscipline.local_data` and `IO.data` as read-only / merged
  views with a `DeprecationWarning`, returning the union of the new input
  and output stores. Existing user code reading `discipline.local_data[name]`
  must keep working.
- Internal call sites in `gemseo` are migrated to the new attributes.

## Out of scope

- Performance optimizations (left for a follow-up once the split lands).
- Changes to grammar API, cache API, or `_run()` contract.
- Removal of `local_data`/`io.data` (only deprecation in this iteration).

## Domain Concept Identification

### Existing Concepts (from codebase)

- **IO** (`src/gemseo/core/discipline/io.py`): owns `input_grammar`,
  `output_grammar`, the merged `data` (`DisciplineData`), `data_processor`,
  residual mappings, and linear-relationship metadata. Provides
  `prepare_input_data`, `get_input_data`, `get_output_data`,
  `update_output_data`, `initialize`, `finalize`. Today, `data`/`_data` is the
  single source of truth for both sides.
- **DisciplineData** (`src/gemseo/core/discipline/discipline_data.py`):
  thin `dict` subclass with picklable `Path` handling. Used as the storage
  type for the merged data — and will be reused for the split stores.
- **BaseDiscipline** (`src/gemseo/core/discipline/base_discipline.py`):
  exposes `local_data` as a property delegating to `self.io._data`, drives
  the execution lifecycle (`execute`, `_execute`, cache load/store), and
  performs auto-coupling deepcopy by intersecting input and output grammars.
- **BaseGrammar** (`src/gemseo/core/grammars/`): provides the membership
  oracle (`name in grammar`) used to filter the merged dict into "the input
  side" or "the output side". After the split, grammars stop being a
  filter and become only a validation/defaults source.
- **Auto-coupled variables**: names belonging to *both* input and output
  grammars (e.g., MDA fixed-point iterates). Today they cohabit a single key
  in `io.data`; the most recent write wins, which is precisely the ambiguity
  the split must remove.
- **DataProcessor** (`data_processor.py`): pre/post-processes input/output
  data around `_run()`. Documentation contract already states it must not
  use `io.data`/`local_data` directly — the split makes that contract
  enforceable in practice.
- **Cache layer** (`gemseo.caches.*`, consumed by `_store_cache`,
  `_set_data_from_cache`): writes outputs from the merged dict via
  `output_data = self.io._data.copy(); del keys not in output_grammar`. After
  the split, cache I/O can read the output store directly.
- **MDA / chains** (`src/gemseo/mda/*`, `src/gemseo/core/chains/*`): heavily
  read and mutate `io.data` to propagate couplings between disciplines
  (`base.py`, `base_solver.py`, `gauss_seidel.py`, `jacobi.py`,
  `quasi_newton.py`, `newton_raphson.py`, `chain.py`, `sequential.py`,
  `additive_chain.py`, `parallel_chain.py`, `warm_started_chain.py`). MDAs
  treat the merged dict as the working state across the fixed-point loop and
  need a clear answer to "where does an auto-coupled key live now".

### New Concepts Required

- **Input data store** (`IO.input_data` / `IO._input_data`): a
  `DisciplineData` holding only items whose name belongs to the input
  grammar. Written by `initialize()` (from validated input data) and by
  cache hits restoring the cached inputs. Read by `_execute`/`_run`,
  validators, `prepare_input_data` callers, MDAs propagating couplings, and
  Jacobian assembly.
- **Output data store** (`IO.output_data` / `IO._output_data`): a
  `DisciplineData` holding only items whose name belongs to the output
  grammar. Written by `update_output_data()` and by cache hits restoring
  the cached outputs. Read by `finalize()`, `_store_cache`, MDA coupling
  propagation, and downstream disciplines.
- **Merged read-only view** for `local_data`/`io.data` backwards
  compatibility: returns a `DisciplineData` reflecting the union of the two
  stores at read time, raising `DeprecationWarning` on access; writes to it
  must route into the two stores (or be deprecated outright if not used
  externally).
- **Auto-coupled storage policy**: an explicit rule about where an
  auto-coupled name lives during execution — almost certainly *both* stores,
  with `update_output_data` writing to `output_data` only and the next
  execution / MDA iteration copying it back to `input_data`. This rule must
  be documented because it changes the observable layout for auto-coupled
  variables.

### Key Business Rules

- **Side-purity**: a name belonging to the input grammar but not the output
  grammar MUST NOT appear in the output store, and vice versa. The split
  exists precisely to make this invariant structural.
- **Auto-coupling visibility**: a name in *both* grammars MUST be readable
  from both stores after a full execution (so MDAs and chains continue to
  observe the latest value regardless of which side they query).
- **Deprecation visibility**: every public read or write through
  `local_data` / `io.data` MUST emit a `DeprecationWarning` exactly once
  per call site / per process (per project convention — verify against
  existing deprecations in the repo).
- **Backward-compatible reads return the union**: `local_data[name]` must
  resolve regardless of whether `name` is an input or output, matching
  today's behavior.
- **Cache semantics unchanged**: cache hits restore inputs into the input
  store and outputs into the output store; the discipline's external
  observable behavior after `execute()` is unchanged.
- **`_run()` contract unchanged**: subclasses keep receiving the same
  `input_data` mapping and keep returning output data via
  `update_output_data`. The split is internal.

## Strategic Approach

### Solution Direction

Refactor `IO` so the storage is two `DisciplineData` instances —
`_input_data` and `_output_data` — exposed as `input_data` and `output_data`
properties on `IO` (and forwarded as `BaseDiscipline.input_data` /
`BaseDiscipline.output_data`). Migrate every internal write to target the
correct store: `initialize()` populates `_input_data`; `update_output_data()`
populates `_output_data`; `_set_data_from_cache` splits cache entries into
both stores. Migrate every internal read to query the correct store directly
instead of filtering the merged dict by grammar.

`local_data` and `io.data` become thin compatibility shims that emit a
`DeprecationWarning` and return a union view, preserving existing user code
during the transition. After the rename, `get_input_data` /
`get_output_data` collapse to a near-trivial copy of the respective store
(namespace stripping still applies).

Data flow stays the same end-to-end:
`execute(input) → initialize → _run/_execute → update_output_data → finalize
→ cache store`. What changes is *which attribute* each step touches; the
sequence and the public methods are preserved.

### Key Design Decisions

- **Two stores vs. tagged single store**: separate `_input_data` /
  `_output_data` vs. keeping one `DisciplineData` with per-key side tags.
  → **Two stores**. Trade-off: tagged single store wins on backward layout
  but loses the type-level guarantee that callers can't see the wrong side;
  the requirement's primary driver (clarity) demands the structural split.
- **Auto-coupled storage**: store auto-coupled names in only one store (and
  pay an extra lookup on the other side) vs. duplicate the reference in
  both. → **Duplicate the reference** (shallow, same `ndarray` object).
  Trade-off: a tiny memory bookkeeping cost and a "two writes on update"
  rule for auto-coupled names, in exchange for O(1) reads on both sides and
  preservation of the current MDA semantics where reading "the input" or
  "the output" of an auto-coupled name yields the latest written value.
  This decision must be validated against MDA fixed-point semantics during
  REASONS Canvas — see Risk § "Auto-coupling semantics".
- **`local_data` deprecation form**: read-only union view + warning vs.
  full read-write proxy vs. immediate removal. → **Read-only union view
  with `DeprecationWarning`**, plus a writable setter that emits a warning
  and routes writes by grammar membership (matching today's implicit
  routing). Trade-off: a writable shim costs implementation effort but
  matches today's `self.local_data = data` and `io.data = ...` usages
  inside the codebase (e.g., `mda/chain.py:318`, `mda/sequential.py:81`,
  `parallel_execution/discipline_execution.py:96`) — those internal
  call sites get migrated, but external user code retains the option.
- **`update_output_data` contract**: keep current semantics (silently
  drop items not in output grammar) but write only to `_output_data`,
  with the auto-coupled mirror handled by the *caller*
  (`base_discipline._execute`) rather than `IO`. → **Keep silent drop;
  centralize auto-coupled mirroring in `IO`** so `_execute` stays
  unchanged. Trade-off: a small amount of coupling logic moves into
  `IO`, but `_execute` and `_run` subclasses remain untouched, which is
  far cheaper than changing the discipline contract.
- **Scope of internal migration**: migrate all gemseo-internal call sites
  in one PR vs. land the split with shims and migrate sites incrementally.
  → **Migrate internal sites in the same change** (≈ 40+ source files,
  ≈ 43+ test files). Trade-off: bigger diff, but otherwise the
  deprecation warning would fire from gemseo's own code, polluting test
  output and forcing whitelists. A single PR keeps the warning surface
  clean — only third-party callers trigger it.

### Alternatives Considered

- **Tag-based single store** (each key annotated as input/output/both):
  rejected — does not eliminate the "input vs. output ambiguity at any
  stage" the requirement targets, and complicates `DisciplineData`'s dict
  contract (pickling, copying).
- **Move `data` out of `IO` entirely into `BaseDiscipline`**: rejected —
  `IO` is already the right home (it owns the grammars that define each
  side); moving storage breaks the existing `io.update_output_data` API
  and increases the migration cost without addressing the requirement.
- **Hard-break `local_data` immediately**: rejected — user code reading
  `discipline.local_data[...]` is pervasive in tutorials, examples, and
  third-party disciplines; the requirement explicitly asks for an alias
  kept with deprecation. Removal is a later step.
- **Lazy union view computed only on `local_data` access** (no separate
  dict object): accepted as the implementation strategy for the
  compatibility shim — avoids a third copy of the data and keeps the
  warning narrow.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Auto-coupling storage**: the requirement does not say whether an
  auto-coupled variable should live in the input store, the output store,
  or both. Today it lives in one merged dict and the last write wins. The
  chosen direction (mirror into both) needs explicit confirmation during
  REASONS Canvas — it is the single most behaviorally observable choice in
  this refactor.
- **`local_data` setter behavior**: callers today do
  `self.io.data = cache_entry.inputs` (`base_discipline.py:251`) and
  `self.io.data = mda.execute(...)` (`mda/sequential.py:81`,
  `mda/chain.py:318`). Should the deprecated setter accept the union and
  route by grammar, or should it require callers to pick a side? Internal
  call sites will be migrated; the question is only about external users.
- **DeprecationWarning frequency**: emit on every access (noisy, but
  precise), once per process (clean, but misses some call sites in tests),
  or once per call site via `stacklevel`/cache (most useful, more code).
  Project convention should govern — check what other gemseo deprecations
  do.
- **Removal timeline**: when does the `local_data` / `io.data` alias get
  removed? The requirement says "deprecation in this iteration" but does
  not name a version. Need a changelog/towncrier statement that pins the
  removal window.
- **Public new attribute names**: `input_data` / `output_data` collide
  with the existing `BaseDiscipline.default_input_data` and
  `default_output_data` only by family resemblance, not by name. Confirm
  the names are acceptable, or pick alternatives (`local_input_data` /
  `local_output_data`) that mirror `local_data`.

### Edge Cases

- **`io.update_output_data` with auto-coupled name**: must update *both*
  stores, or downstream code reading `input_data[name]` will see stale
  data. Today this works because there's one dict.
- **Cache hit (`_set_data_from_cache`)**: currently sets `io.data =
  cache_entry.inputs` then `io._data.update(cache_entry.outputs)`. After
  the split, cache outputs that are also inputs (auto-coupled) must land
  in both stores.
- **`DataProcessor` interaction**: doc claims `_run` "does not use the
  `io.data` or `local_data` attributes" when a `data_processor` is set
  (`io.py:63`). Today this is a soft contract. After the split, the
  attributes still exist (deprecated). Need to update the doc; behavior is
  unchanged.
- **Residual / state-variable handling**: `residual_to_state_variable`
  maps output names to input names. Code paths that compute residuals
  (`mda/base_solver.py:323`, `mda/quasi_newton.py:217`) write into
  `io.data` for keys that include `NORMALIZED_RESIDUAL_NORM` — which is
  *not* in either grammar today. Need to decide where non-grammar keys
  live (third store? output store with relaxed rule? or migrate
  residual norms to a dedicated MDA attribute).
- **Items that fall outside both grammars**: e.g., the residual norm
  above, and anything users sneak into `local_data`. The current dict is
  permissive. The split must either keep an "extras" bucket or reject
  unknown keys. The chosen policy decides whether the change is purely
  internal or observable.
- **Namespaced names**: `get_input_data(with_namespaces=False)` strips
  the namespace prefix. The split stores must remain *namespaced* so the
  grammar membership check stays simple; stripping happens only at read
  time, as today.
- **Pickling**: `DisciplineData.__getstate__` and `__setstate__` already
  handle paths. Two stores means two pickled dicts on `IO` — fine, but
  any external code that pickles a discipline expecting `local_data` in
  the state will need the compatibility shim to be re-hydratable. Likely
  out of scope but must be checked.
- **`io.data.copy()` patterns**: many MDA/chain call sites snapshot the
  merged dict before iteration (`mda/jacobi.py:147`,
  `mda/gauss_seidel.py:169`, `mda/quasi_newton.py:202`,
  `mda/newton_raphson.py:165`). After the split, the snapshot must
  capture both sides — these are exactly the call sites where
  auto-coupled semantics show up.
- **`__create_input_data_for_cache` auto-coupling deepcopy**: intersects
  `input_grammar` and `output_grammar` to deepcopy values shared by both
  (`base_discipline.py:234-241`). The logic remains correct after the
  split (the intersection is the same set) but the implementation should
  read from `_input_data` rather than the merged dict.

### Technical Risks

- **Migration churn**: ≈ 46 source files and ≈ 43 test files reference
  `io.data` / `io._data` / `local_data`. Splitting and updating these in
  one change is mechanically large; a missed migration leaves a path
  emitting `DeprecationWarning` inside gemseo's own code. → Mitigation:
  grep-based audit + a CI rule that turns the new warning into an error
  *only inside `src/gemseo/`* (tests still allowed during transition).
- **MDA fixed-point regression**: MDAs iterate by mutating `io.data` and
  relying on auto-coupled keys being instantly visible on the input side.
  If the auto-coupled mirroring rule is wrong by one iteration, fixed-point
  loops will diverge or converge to wrong values. → Mitigation: keep the
  MDA test suite (Sellar, Sobieski, springs) as the regression line; do
  *not* land the change without those tests passing unchanged.
- **Parallel execution serialization**: `parallel_execution/*.py` sends
  `disc.io.data` across worker boundaries and reassigns it on return
  (`discipline_execution.py:96`, `discipline_linearization.py:137,155`).
  The wire format effectively encodes the merged dict. After the split
  it must encode both stores (or be migrated to encode the union and
  re-route on reception). → Mitigation: explicit cross-process protocol
  decision recorded in the REASONS Canvas.
- **Jacobian assembly indexing**: `jacobian_assembly.py:366,911` reads
  `discipline.io.data[name]` for both input and output names (e.g.,
  coupling outputs and design variables). After the split these reads
  must route to the right store by grammar lookup. Wrong routing yields
  silent KeyErrors at linearization time. → Mitigation: centralize the
  lookup in a helper on `IO` (`get(name)` returning from the appropriate
  store) and replace all `io.data[name]` reads with it.
- **Performance regression of the compatibility shim**: each call to
  `local_data` allocates a union dict and emits a warning. Hot MDA loops
  that today touch `io.data` thousands of times would degrade if anything
  inside `gemseo` still reads through the alias. → Mitigation: full
  internal migration (no alias reads inside `src/gemseo/`); accept that
  external users opting into the shim pay the cost until they migrate.
- **Pickling format change**: serialized disciplines from older versions
  will not load against the new `IO` layout unless `__setstate__` is
  taught to accept the legacy `_data` key and split it. → Mitigation:
  add a one-shot upgrade path in `IO.__setstate__` (or
  `BaseDiscipline.__setstate__`) and a test pinning a legacy pickle.
- **Towncrier fragment + upgrading doc**: `docs/software/upgrading.md`
  already documents `local_data`-related upgrades. A new entry is
  required, and the deprecation must appear in `changelog/fragments/`.
  Easy to forget. → Mitigation: include in the REASONS Canvas
  Operations sequence.

### Acceptance Criteria Coverage

The provided requirement has no numbered ACs. The implicit acceptance
criteria derived from the requirement text:

| AC# | Description | Addressable? | Gaps/Notes |
|-----|-------------|--------------|------------|
| 1 | `IO` exposes two separate stores for input and output data | Yes | Names (`input_data` / `output_data` vs. `local_input_data` / `local_output_data`) need confirmation in REASONS Canvas |
| 2 | At any stage, the input/output side of every value is unambiguous | Partial | Holds for grammar-typed keys; ambiguous for non-grammar keys like `NORMALIZED_RESIDUAL_NORM` until storage policy decided |
| 3 | `BaseDiscipline.local_data` keeps working with `DeprecationWarning` | Yes | Setter behavior (writable shim vs. read-only) needs confirmation |
| 4 | `IO.data` keeps working with `DeprecationWarning` | Yes | Same setter question as AC#3 |
| 5 | Internal gemseo call sites migrated to the new attributes | Yes | ≈ 46 source files; migration list comes from grep audit |
| 6 | No behavioral change to `execute()` / `_run()` / cache / MDA outputs | Yes | Validated by existing test suite (Sellar, Sobieski, MDA convergence tests) |
| 7 | Public API for `get_input_data` / `get_output_data` unchanged | Yes | Internals become trivial; external behavior identical |
| 8 | Performance work is *not* in this iteration | Yes | Tracked as follow-up; do not optimize filtering in this PR |
| 9 | Changelog fragment + upgrading doc updated | Yes | Add towncrier fragment under `changelog/fragments/` |
