<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: BiLevel Sub-Scenario History Dataset

## Original Business Requirement

Currently, a BiLevel scenario in GEMSEO offers users the possibility of keeping the databases of the sub-scenarios either in memory, in HDF5 files, or both. A user that wants to post-process the results then relies on the fact that the order of each database corresponds to the iterations of the system-level scenario and is able to link the system-level design vector to the optimal solution of each of the sub-scenario runs.
However, in some contexts it is interesting to have the entire history of the sub-scenarios linked to the starting point set by the system level in one go.
For instance, a dataset that looks as follows:

```text
GROUP     | designs | objectives | upper_level_designs | executions
VARIABLE  | x_1     | y_11       | x_shared            | execution  sub_iteration
1         | 0.25    | 0.98       | 0.05 ...            | 0          1
2         | 0.27    | 0.99       | 0.05 ...            | 0          2
3         | 0.25    | 0.98       | 0.06 ...            | 1          1
```

Where `x_1` is the design variable of the sub-scenario, `y_11` is the objective of the sub-scenario, `x_shared` is the design point coming from the outer scenario. The `executions` column tracks the execution from the outer scenario and the sub-iterations for the current sub-scenario.
GEMSEO does not offer a simple way to achieve this result. The objective of this feature is to define the way to implement this feature so that users can get all the relevant information in one place, without having to assemble the data themselves.

## Domain Concept Identification

### Existing Concepts (from codebase)

- `BiLevel` formulation (`src/gemseo/formulation/bilevel.py`) — orchestrates, at each system-level iteration, MDA1 → sub-scenarios → MDA2; owns the `_scenario_adapters` wrapping each sub-scenario.
- `MDOScenarioAdapter` / `EvaluationScenarioAdapter` (`src/gemseo/scenario/adapter/{mdo,evaluation}.py`) — wraps a sub-scenario as a discipline; owns the `keep_databases`, `save_databases`, `database_file_prefix` and `naming` settings that control history retention.
- `adapter.databases: list[Database]` — one `Database` snapshot appended per system-level iteration when `BiLevel_Settings.keep_opt_history=True`; already relied upon by `BiLevelScenarioResult` to fetch the sub-problem's database at the system optimum's iteration index (`i_opt`). This list is index-aligned with the outer scenario's own iteration history — the exact linkage the requirement wants to exploit for every iteration, not only the optimum.
- HDF5 export — when `BiLevel_Settings.save_opt_history=True`, each adapter execution's database is also written to `{database_file_prefix}_{generated_name}.h5`, `generated_name` produced by `NameGenerator` in either `NUMBERED` or `UUID` mode.
- `Database.to_dataset()` / `OptimizationProblem.to_dataset()` (`src/gemseo/core/problem/database.py`, `src/gemseo/optimization/problem.py`) — build a single `OptimizationDataset` (or `IODataset`) from one `Database`, exposing exactly the `designs` / `objectives` / `equality_constraints` / `inequality_constraints` / `observables` groups shown in the requirement's example table. This is the per-iteration building block the requirement's dataset must be assembled from.
- `BiLevelScenarioResult` (`src/gemseo/scenario/scenario_result/bilevel_scenario_result.py`) — deliberately reads a single entry, `adapter.databases[i_opt]`, to reconstruct the sub-optimization result at the system-level optimum only. Today's API intentionally picks one iteration, not the full history in one go.
- `Dataset.add_group()` (`src/gemseo/dataset/dataset.py`) — generic mechanism to append an arbitrarily named group of columns to an existing `Dataset`, provided the row count matches. This is the extension point through which new groups (`upper_level_designs`, `executions`) can be attached to a per-iteration dataset.

#### New Concepts Required

- **Execution / sub-iteration index** — a pair of counters that, for every row of the combined dataset, identify which system-level iteration produced it and which local iteration within that sub-scenario run it is. Neither `Database` nor `Dataset` currently models this.
- **Combined sub-scenario history dataset** — the row-wise concatenation of the per-iteration `OptimizationDataset` built from every entry of `adapter.databases` (or every saved HDF5 file), decorated with two new groups: the execution/sub-iteration group, and an `upper_level_designs` group carrying the outer system-level design vector for that iteration, broadcast across every row of that iteration's block.
- **Dataset row-wise concatenation** — no utility in the current source merges several `Dataset`/`OptimizationDataset` instances of identical column structure into one (only stale compiled bytecode references `concat`/`vstack`-like helpers; nothing exists in `src/gemseo/dataset` or `src/gemseo/utils` today). This must be built before it can be decorated.
- **Disk-based reconstruction** — when `save_opt_history` is used, the same combined dataset must be re-buildable from the saved `.h5` files, read back in the correct system-level iteration order via `OptimizationProblem.from_hdf`.

#### Key Business Rules

- **Row ordering**: rows must preserve, within a block, the sub-scenario's own iteration order, and, across blocks, the outer scenario's execution order — governs both the in-memory (`adapter.databases` list order) and on-disk (file ordering) paths.
- **Row-to-outer-iteration linkage**: every row of a sub-scenario's combined dataset must carry the exact outer design vector (`x_shared`) that was current for that system-level iteration, sourced from the main problem's database/x-history at the same index as the corresponding entry of `adapter.databases`.
- **Per sub-scenario, not global**: each sub-scenario adapter of a BiLevel formulation gets its own combined dataset (different sub-scenarios may have different design/objective variables) — never one dataset mixing several sub-scenarios' variables.
- **History must exist**: the feature is only meaningful when `keep_opt_history=True` (in-memory) and/or `save_opt_history=True` (on-disk) is enabled; with neither, there is no history to assemble.

## Strategic Approach

### Solution Direction

For each sub-scenario adapter, iterate `i, sub_database in enumerate(adapter.databases)` (or, for the on-disk case, the ordered list of saved `.h5` files loaded via `OptimizationProblem.from_hdf`); build each iteration's dataset by reusing the existing `Database.to_dataset()` / `OptimizationProblem.to_dataset()`; attach the execution index `i` and a per-row sub-iteration counter, and the outer design vector at index `i` (read from the system-level problem's `x`-history); then concatenate all per-iteration datasets into a single `Dataset` returned to the user.

Data flow: BiLevel execution → `adapter.databases` (and/or saved `.h5` files) accumulate one entry per system-level iteration → new assembly step (concatenate + decorate) → single `Dataset` with groups `designs` / `objectives` / … / `upper_level_designs` / `executions` → handed to the user for post-processing.

This reuses existing building blocks for the per-iteration dataset construction (`to_dataset`, groups, dtypes) and confines the new work to the concatenation-and-decoration layer, rather than reimplementing dataset assembly.

#### Key Design Decisions

- **Where to expose the new API**: on `EvaluationScenarioAdapter`/`MDOScenarioAdapter` (closest to where `databases` already lives, usable outside BiLevel) vs. on `BiLevelScenarioResult` (matches where users already look for BiLevel post-processing) vs. a standalone dataset-building utility (keeps adapter/result classes free of dataset-construction responsibility). → Recommend exposing the capability from the adapter, since `databases`/`save_databases`/naming already live there and the mechanism is formulation-agnostic, with an optional convenience accessor on `BiLevelScenarioResult` for discoverability.
- **Order guarantee for the on-disk case**: `NameGenerator.Naming.UUID` (required for multiprocess-safe parallel sub-scenarios) does not preserve execution order in the file name. → Recommend requiring the in-memory `databases` list order when available, and for disk-only reconstruction, either requiring `NUMBERED` naming for this feature or introducing an explicit order marker in each saved file, rather than inferring order from UUID file names.
- **Scope of reuse**: the "one database per outer iteration" pattern is specific to how BiLevel's adapters populate `databases`, but the underlying "concatenate N per-iteration `OptimizationDataset`s with execution/sub-iteration bookkeeping" mechanism is not BiLevel-specific. → Recommend building the concatenation/decoration logic as a reusable, formulation-agnostic building block, with BiLevel-specific code supplying only its own linkage rule (the outer `x_shared` value per iteration).

#### Alternatives Considered

- Leave assembly to the user, outside GEMSEO — rejected, this is exactly the friction the requirement describes.
- Track "outer iteration" metadata natively inside `Database` at `store()` time instead of post-hoc dataset assembly — rejected: `Database` is used far beyond BiLevel, so embedding BiLevel-specific linkage there has a much larger blast radius for no benefit over decorating the dataset after the fact.

## Risk & Gap Analysis

### Requirement Ambiguities

- Indexing convention for the `executions` group (`execution`, `sub_iteration`) — 0- vs. 1-indexed, and whether it must match the outer scenario's own iteration numbering, which may not equal raw call count if the outer driver restarts or revisits points.
- Whether `disciplines_as_sub_scenario` entries (`BiLevel_Settings.disciplines_as_sub_scenario`) — which are not `MDOScenarioAdapter` instances and have no `databases` history at all — are in scope, since the requirement only discusses "sub-scenarios".
- Whether the combined dataset is meant to be available only post-hoc (after the whole BiLevel run, e.g. from a scenario result) or queryable incrementally during the run.

#### Edge Cases

- `keep_opt_history=False` with `save_opt_history=True` (or vice versa) — the assembly must support both the in-memory and file-based sources, and fail informatively when neither is enabled.
- `parallel_scenarios=True` — per `BiLevel_Settings.keep_opt_history`'s own documentation, databases are not propagated back to the main process under multiprocessing, so `keep_opt_history` is expected `False` there; only the on-disk path remains available, which is exactly the path where order is hardest to reconstruct (UUID naming).
- An outer iteration with no corresponding sub-scenario re-evaluation (e.g. a warm-started or reused point via `reset_x0_before_opt`/`set_x0_before_opt`) — must not silently misalign the `upper_level_designs` broadcast against the sub-scenario's row blocks.
- Vector-valued shared design variables in `upper_level_designs` — must reuse the same per-component column expansion that `to_dataset` already applies to `designs`/`objectives`, not a collapsed single column.

#### Technical Risks

- No dataset row-wise concatenation utility currently exists in the live source; it must be built, including its interaction with `Dataset`'s pandas `MultiIndex`-based column model.
- Order reconstruction from disk-only history is unreliable under `NameGenerator.Naming.UUID`; the implementation must mandate `NUMBERED` naming for this feature, add an explicit order marker, or clearly document the limitation.
- Memory cost: concatenating the full iteration history across all system-level iterations, for every sub-scenario, compounds the "very memory consuming" warning already documented for `keep_opt_history` — this feature turns that documented edge case into the common case for anyone using it.

#### Acceptance Criteria Coverage

No formal Acceptance Criteria were provided — the business input is a narrative description plus one illustrative example table. Capturing explicit ACs (e.g. row-count and column-shape expectations, in-memory vs. on-disk parity) is itself a gap to close before REASONS Canvas.

| AC# | Description | Addressable? | Gaps/Notes |
|-----|-------------|--------------|------------|
| 1 | Combined dataset exposes `designs`/`objectives`/`upper_level_designs`/`executions` groups, one row per sub-iteration, linked to the outer design point | Yes | Achievable by decorating per-iteration `to_dataset()` output; needs the new concatenation utility |
| 2 | Works for the in-memory (`keep_opt_history`) case | Yes | Straightforward from `adapter.databases` |
| 3 | Works for the on-disk (`save_opt_history`/HDF5) case | Partial | Order reconstruction unreliable with `UUID` naming; needs an explicit ordering guarantee |
| 4 | No manual assembly required from the user | Yes | A single new method/accessor covers it once the placement decision (adapter vs. scenario result) is made |
