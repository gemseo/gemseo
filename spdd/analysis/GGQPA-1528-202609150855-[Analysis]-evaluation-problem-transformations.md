<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: Separate function preprocessing from `EvaluationProblem`

> **GitLab issue [#1528](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1528)**
> — "Keep the initial optimization problem".
>
> The requirement below was written by the assignee as a reframing of that issue.
> A detailed, tactical design proposal was already posted on the issue on 2026-09-11
> (two-layer split: a recording `DatabaseFunction` owned by the problem and an adaptation
> `PreprocessedFunction` built by the driver into a derived problem). This analysis stays
> at the strategic level, grounds the requirement in the current code, and records where
> the requirement **extends** the posted proposal: a transformation that the *user* can
> apply, organised as a *series*, and open to the *discrete and categorical* variable
> features of issue [#1885](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1885).
>
> The current branch `1528` is at the tip of `develop`; no implementation exists yet.
>
> **Update (2026-09-15).** A review note on MR
> [!2759](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2759#note_3836074409)
> compared this analysis with issue
> [#1886](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1886) — "Transformations
> around the `DesignSpace`" — whose analysis, which **specifies** a
> `BaseSpaceTransformation` / `TransformationChain` contract, already exists on the branch
> `design-space-transformations` (MR !2719). Seven decisions taken there bind the
> design-space part of this requirement; they are recorded in
> [Relationship with #1886](#relationship-with-1886-transformations-around-the-designspace)
> and they amend the design-space-mapping concept, three business rules, six design
> decisions, one ambiguity and two acceptance criteria below.
>
> **Update (2026-09-16).** The branch was rebased onto `develop` after the `random_space`
> merge (commit `32d104b`). Three facts this analysis rests on changed:
> `EvaluationProblem` is now `Generic[_SpaceT]` and owns an `input_space` that is only
> *sometimes* a `DesignSpace` — `ReliabilityProblem` is an in-tree
> `EvaluationProblem[RandomSpace]` that calls `preprocess_functions`;
> `transform_vect` / `untransform_vect` became an **abstract pair on
> `BaseVariableSpace`**, implemented as normalization by `DesignSpace` and as the
> iso-probabilistic transform by `RandomSpace`; and `ParameterSpace` was removed in favour
> of `RandomSpace`. The affected bullets are updated in place, and one business rule, one
> edge case and one technical risk are added.
>
> **Update (2026-09-16, second pass).** #1886's "three spaces" framing was re-examined
> against the code. The separation it defends is real but is a separation of *maps*, not
> of spaces: the typed domain is a subset of the original space in the same
> coordinates. Decision 1, the design-space-mapping concept, the corresponding business
> rule, one ambiguity and AC#6 now read **two spaces and a projection**. The same pass
> found that MR !2719 carries no transformation source — the contract is specified in its
> analysis only.
>
> **Update (2026-09-16, new requirement).** The assignee added one requirement: **the
> working problem gets its own database, in its own coordinates, when a debug mode is
> enabled; the mode is disabled by default.** It settles the open question on the derived
> problem's database — the answer is *both*, a shared one and an opt-in second one — and
> adds a concept, a business rule, a design decision, an edge case, a risk and AC#9 below.
>
> **Update (2026-09-16, blocking points resolved).** The four points listed as blocking
> the canvas were taken: the transformation entry point is **public** (B1); **#1528
> implements** `BaseSpaceTransformation` / `TransformationChain` to #1886's specification
> (B2); `preprocess_functions`, `reset(preprocessing=…)` and the setter guards are
> **removed outright in 7.0.0** (B3); and the vectorized-plus-normalized DOE combination is
> **refused by a guard** rather than defined (B4). The same pass corrected a wrong claim
> below: four sibling plugins do call the APIs being removed.
>
> **Update (2026-09-16, vocabulary and surface settled).** The four points that become
> names or signatures were taken: the layers are **`DatabaseFunction`** (recording) and
> **`TransformedInputFunction`** (adaptation), with "transformation" as the user-facing word;
> **`project` takes a original point** and the middle space is called **original**;
> the debug database is an **independent path** that records whenever its flag is on,
> including under `use_database=False`; and **`to_dataset` / `to_hdf` stay on the
> problem**. Export had been listed as an ambiguity but never triaged — that omission is
> fixed below.
>
> **Update (2026-09-16, review round on !2759).** Three points raised by the reviewer are
> addressed: the approximated Jacobian must be recorded while its **perturbation
> evaluations must not**, which today holds only through an undocumented detail; the
> position of derivative approximation in the step order was never pinned; and the
> tolerance-based termination criteria must be evaluated against the **original** problem,
> because the user's tolerances are expressed in the user's scales. Two business rules,
> the step-order concept, the approximation decision, one risk and AC#10 below carry the
> answers.
>
> **Update (2026-09-16, adaptation wrapper renamed).** The adaptation wrapper is
> `TransformedInputFunction`, not `PreprocessedFunction`, superseding the naming decision
> recorded above: once the class holds only the transformation chain and the NaN checks,
> "preprocessed" describes what it no longer does, and **input** says which part is
> transformed — the function value is untouched, so a bare `TransformedFunction` would
> collide with the semantic transformations the business rules exclude. The contract maps are likewise
> `transform_value` / `untransform_value` / `transform_jacobian` / `untransform_jacobian`,
> which the appendix records as a sixth amendment to #1886's text.
>
> **Update (2026-09-16, unblocked).** #1886's author agreed offline to B2 and to the seven
> amendments, so #1528 owns `BaseSpaceTransformation` / `TransformationChain`. The two
> prerequisites have landed with their regression tests. Implementation is unblocked; the
> four plugin MRs remain open but gate only the 7.0.0 bump, not the work.

## Original Business Requirement

> EvaluationProblem is a set of ArrayFunction objects, called observables, to be evaluated on a DesignSpace. OptimizationProblem is a subclass of EvaluationProblem where the functions are either observables, objectives or constraints. EvaluationProblem can preprocess the functions, evaluate the functions or the preprocessed functions, export the evaluations to a dataset of HDF5 file if the Database has been attached to the functions during the preprocessing stage. IMO EvaluationProblem does too much things. Furthermore, I don't think it is EvaluationProblem's responsibility to preprocess functions. I have also in mind (just a proposal!) to let the user or the driver take an original EvaluationProblem and transform it into a new problem where the functions would wrap the original ones with a series of transformation, e.g., normalization and rounding as already proposed now, but also transformations specific to the future features of GEMSEO related to discrete and categorical variables.

### Context from the GitLab issue

The issue description (by the requester) asks that *all* modifications of an optimization
problem — sign change, normalization, relaxation, aggregation — be applied to **another**
problem, so that the problem given by the user can be retrieved at any time. The
motivating use case is a warm start: define a mixed problem, relax it, solve the relaxed
problem, inject the solution, then retrieve and solve the original mixed problem. The
requester accepts a higher RAM footprint. The posted proposal narrows the first
deliverable to normalization, rounding and database attachment, and explains why sign
change, aggregation and relaxation are of a different nature (they change *what is
recorded*, not *how a point is spelled*).

## Domain Concept Identification

The problem code lives in `src/gemseo/core/problem/` (`EvaluationProblem`, `Database`,
`EvaluationCounter`) and `src/gemseo/optimization/problem.py` (`OptimizationProblem`).
The function wrappers live in `src/gemseo/core/function/`. The driver orchestration lives
in `src/gemseo/core/algorithm/base_driver_library.py`.

### Existing Concepts (from codebase)

- **`EvaluationProblem`** (`core/problem/evaluation.py`, 1074 lines): since the
  `random_space` merge it is `Generic[_SpaceT]` with `_SpaceT` bound to
  `BaseVariableSpace`, so it owns an `input_space` that is a `DesignSpace` for an
  optimization or a DOE and a `RandomSpace` for a reliability study; a private
  `_design_space` property narrows it back with an `isinstance` and returns `None`
  otherwise. It also owns two `Observables` collections (plain and "new iter"), a
  `Database`, an `EvaluationCounter`, differentiation settings and a `stop_if_nan`
  flag. It currently carries **four responsibilities**: (1) *composition* — adding and
  naming functions; (2) *evaluation* — `get_functions` / `evaluate_functions` /
  `_preprocess_inputs`, with the `no_db_no_norm` flag to reach the raw functions;
  (3) *preprocessing* — `preprocess_functions` / `_preprocess_function`, a five-branch
  builder that wraps every function **in place** and sets `_functions_are_preprocessed`,
  with `reset(preprocessing=…)` as the undo and two setter guards that refuse changes once
  wrapped; it is now preceded by a sixth, space-kind branch — when `_design_space` is
  `None`, `round_ints` is forced off and a normalized input raises `ValueError` — and
  `_preprocess_inputs` carries the symmetric pair of `ValueError`s;
  (4) *export* — `to_dataset` and `to_hdf`, which are thin delegations to `Database.to_dataset` /
  `Database.to_hdf`. Responsibility (3) is the bulk of the class-specific logic
  (roughly 200 lines including its undo and guards); responsibility (4) is about ten lines
  of forwarding.
- **`OptimizationProblem`** (`optimization/problem.py`): adds an objective and a
  `Constraints` collection, hooks into the base class through `_sequence_of_functions`,
  `_function_names` and `_get_output_functions`, and overrides `reset` **only** to undo
  the preprocessing of the objective and constraints. `is_linear` inspects
  `function.original`, so it depends on the wrapper being exactly one level deep.
- **`ArrayFunction`** (`core/function/array_function.py`): the function abstraction. It
  has a plain `original` attribute (base case `self`, written only by the preprocessing
  step), an `expects_normalized_inputs` flag and an `f_type` (objective, observable,
  equality or inequality constraint).
- **`PreprocessedFunction`** (`core/function/preprocessed_function.py`): the single
  wrapper produced by preprocessing. It mixes four concerns: coordinate adaptation
  (denormalize, round integers, normalize the gradient, densify sparse Jacobians),
  recording (database lookup and store, call counting, the `pre_compute_at_new_point`
  hook that drives the iteration counter and progress bar), algorithm control (the NaN
  checks raising `TerminationCriterion` subclasses) and derivative approximation
  (replacing the Jacobian by a finite-difference or complex-step approximator). Its
  dispatch selects one of four code paths from `database`, `with_normalized_inputs` and
  `vectorize`; the "no database" path performs no NaN check at all. It also carries
  `enable_statistics`, a **`ClassVar` mutated from outside the class**: the validator
  `GlobalConfiguration.__validate_enable_function_statistics` assigns
  `PreprocessedFunction.enable_statistics = v` as a side effect
  (`util/global_configuration.py:160-162`), the class reads it at four sites, and
  `gemseo._log_settings` prints a user-visible section literally headed
  `PreprocessedFunction` from it (`__init__.py:1586-1590`). Splitting or renaming this
  class therefore reaches the global configuration and the settings log, not only the
  function layer.
- **`Functions` / `Observables` / `Constraints`** (`core/function/collection/`,
  `optimization/core/constraints.py`): mutable sequences with `get_originals`, `reset`
  (rebuilds from the originals) and `format`. `Constraints.format` **already wraps
  functions at add time** (offset by the constraint value, negation for positive
  constraints), and `Constraints.aggregate` replaces constraints in place. These are
  transformations of a different kind: they define what the problem *is*, and the
  database records their result.
- **`Database`** (`core/problem/database.py`): a mapping keyed on hashed design vectors
  in real coordinates, holding function values and gradients, with store and new-iteration
  listeners, HDF and dataset export. It is shared by `OptimizationHistory`, the KKT
  checker, the progress bar and the parallel-DOE result collection, which all read
  `problem.database` during a run. Its constructor is already
  `Database(name="", input_space=None)` with `input_space: BaseVariableSpace | None`, and
  each instance owns its listeners, its `HDFDatabase` node and its own
  `dataset.misc["input_space"]`, so a second database over a *different* space costs no
  new machinery.
- **`GlobalConfiguration`** (`util/global_configuration.py`, set through
  `gemseo.configure`): a pydantic model of `enable_*` booleans — `enable_discipline_cache`,
  `enable_function_statistics`, `enable_progress_bar`, … — plus a `fast` aggregate whose
  validator forces the expensive ones off. It is the existing home for a diagnostic that
  must be off unless asked for.
- **`DesignSpace`** (`space/design.py`): already owns every coordinate
  transformation used by preprocessing — `normalize_vect`, `denormalize_vect`,
  `normalize_grad`, `denormalize_grad`, `round_vect` — and implements the generic
  `transform_vect` / `untransform_vect` pair as normalization. Its
  `Normalizer` and `IntegerRounder` collaborators are mask-based, so they act only on the
  components concerned.
- **`BaseVariableSpace` / `RandomSpace`** (`space/base.py`, `space/random.py`): the
  `transform_vect` / `untransform_vect` pair is **abstract on the base space** — map to
  and from the unit hypercube — with two implementations: normalization for `DesignSpace`
  and the iso-probabilistic (Rosenblatt) transform for `RandomSpace`. `BaseDOELibrary`
  already samples through this pair rather than through `denormalize_vect`. The generic
  coordinate hook this analysis wanted to lean on therefore already exists *and* already
  has a non-design implementation; what is missing is a **series** and the gradient half
  (`transform_jacobian` / `untransform_jacobian`) — exactly #1886's contract.
- **`BaseDriverLibrary.execute`**: the orchestrator. It checks the problem, calls
  `preprocess_functions` with the driver settings (`normalize_design_space`,
  `use_database`, `round_ints`, `eval_obs_jac`, `store_jacobian`, `vectorize`, sparse
  support), installs the `pre_compute_at_new_point` hook and the new-iteration listener,
  runs, builds the result and tears down the hook and listeners — but **never undoes the
  preprocessing**. Drivers then evaluate `self._problem.objective` with *normalized*
  vectors (NLopt, SciPy global, MNBI, and external plugins).
- **Sub-problem builders**: the augmented Lagrangian, multi-start and the slack-variable
  reformulation each construct a new `OptimizationProblem` from `function.original`. The
  augmented Lagrangian preprocesses its sub-problem itself, then the sub-algorithm's
  `execute` hits the idempotence guard and its own settings are silently ignored.
  `scaling_threshold` in the optimization library replaces the objective and constraints
  with scaled copies in place, with no undo.
- **Workarounds caused by in-place mutation**: `EvaluationScenario` resets the problem
  before every DOE run (seven-line comment explaining that a previous optimizer left the
  functions expecting normalized inputs); `MDOScenarioAdapter` and `LagrangeMultipliers`
  pass `preprocessing=False` to `reset`; `Animation` swaps `problem.database` in and out
  for reading.
- **Non-driver preprocessing clients**: the reliability algorithms preprocess with
  `is_function_input_normalized=False` and then read the wrapped observables to hand
  them to OpenTURNS. Their problem is a `ReliabilityProblem(EvaluationProblem[RandomSpace])`,
  so this is the in-tree proof that preprocessing must run on a problem whose input space
  is **not** a `DesignSpace`, with the coordinate half of the series empty.
- **External consumers** (sibling plugins under `GEMSEO/`): `gemseo-bilevel-outer-approximation`
  reads `problem.objective.original` twice (coefficients, discipline adapter);
  `gemseo-mlearning` reads `acquisition_criterion.original.func`. Five call sites across
  four plugins also use the APIs this story removes: `reset(preprocessing=False)` in
  **production** code — `gemseo-umdo` twice (`_functions/base_statistic_function.py`,
  `_functions/statistic_function_for_control_variate.py`) and `gemseo-mlearning` once
  (`active_learning/active_learning_algo.py`) — and `preprocess_functions()` in **tests**
  only, in `gemseo-calibration` and `gemseo-benchmark`. All four are first-party, so the
  updates can land in lockstep, but they are a coordination obligation, not a no-op.
- **Discrete variables (#1885)**: the data model only. Its analysis and prompt state
  explicitly that *solving* — encoder, relaxation, working design space, `round_ints`
  rework, a `handle_discrete_variables` driver flag — is out of scope and tracked
  separately. Those deferred items are the "future features" this requirement wants the
  transformation mechanism to accommodate.
- **Design-space transformations (#1886)**: a sibling story, analysed and prototyped on
  the branch `design-space-transformations` (MR !2719), that already specifies a
  `BaseSpaceTransformation` contract — `transform_space`, `transform_value` /
  `untransform_value`, `transform_jacobian` / `untransform_jacobian`, `project`,
  `create_constraints`, and the `is_affine` / `changes_dimension` /
  `requires_finite_bounds` flags — plus a `TransformationChain` and implementations for
  normalization, integer relaxation and the Rosenblatt transform of the random space
  (`ParameterSpace` on the !2719 branch, `RandomSpace` on `develop` since the
  `random_space` merge — !2719 needs the same rename when it rebases).
  It covers exactly the design-space half of the "series of transformations" this
  requirement asks for, and it is deliberately independent of the problem.
- **Test footprint**: 13 test files call `preprocess_functions`; 5 reference
  `PreprocessedFunction`; the user guide page `docs/user_guide/concepts/problems/evaluation.md`
  and `functions.md` cross-reference both, and the docs build is strict.

### New Concepts Required

- **Recording attachment** — the part of today's preprocessing that the problem keeps:
  binding its functions to its `Database` and `EvaluationCounter`, in the problem's own
  (real) coordinates, together with derivative approximation, whose results must reach the
  database. It is the only wrapping the problem performs on itself, and it must be
  idempotent with respect to repeated executions with different settings.
- **Problem transformation** — an operation that takes an `EvaluationProblem` and returns a
  **new** problem whose functions wrap the original ones. The original problem is never
  mutated. The derived problem is disposable: a driver builds one for the duration of a
  run; a user builds one to solve a variant (the warm-start use case).
- **Transformation step** — one composable unit of adaptation applied to every function of
  a problem: today normalization and integer rounding (plus Jacobian densification for
  drivers without sparse support); later relaxation of discrete variables to a continuous
  domain, encoding of categorical variables into numeric coordinates, and their inverse
  mappings. Steps are ordered: denormalize before round, decode before evaluate,
  normalize the gradient after the Jacobian. Derivative approximation has a place in that
  order too, and it is not a step of the series: the approximator perturbs in **working**
  coordinates, each perturbed evaluation reaching the raw function through the backward
  map, and the working Jacobian it returns is mapped back with `untransform_jacobian`
  before being recorded. Written out, one evaluation is: working point →
  `untransform_value` →
  original point → evaluate and record; and one approximated Jacobian is: perturb in
  working coordinates → evaluate each perturbation **without recording** →
  `untransform_jacobian` → record the original gradient under the original key, while the
  algorithm receives the working Jacobian.
- **Transformation series** — the ordered composition of steps applied by one
  transformation. Its shape is what makes the mechanism extensible: a new variable kind
  contributes a new step rather than a new branch in a builder.
- **Design-space mapping** — a transformation may change the *space* the algorithm sees,
  not only the spelling of a point: relaxation widens an integer or discrete domain to a
  continuous one; encoding changes the dimension. The derived problem then has its own
  design space, and a two-way mapping to the original one is needed so that the
  database, the results and the current value are expressed in the user's coordinates.
  Normalization and rounding are the degenerate case where the mapping is a bijection on
  the same variables and the design space can be shared. Per #1886 this mapping needs
  **two** maps that must never be merged: `untransform` (working → original), run at every
  evaluation, and `project`, run once on the result. #1886 counts the target of `project`
  — the typed domain — as a third space; this analysis counts it as a **subset of
  the original space in the same coordinates**, so the model is two spaces and a
  projection. See
  [Relationship with #1886](#relationship-with-1886-transformations-around-the-designspace).
- **Working-coordinate database** — a *second* `Database`, owned by the derived problem
  and keyed on the points the algorithm actually produced (normalized, relaxed, encoded),
  over the working space rather than the user's. It is populated only when a debug mode is
  enabled, and it is **additive**: the shared database in the user's coordinates keeps
  being written exactly as before. It is the only place where the working-space history is
  observable, because #1886's BR-2 applies the backward at write time, so the shared
  database never sees a working vector. It is written by the adaptation layer, the only one
  holding a working point, and never by the recording layer, which would perturb the call
  counts and the listener order it is meant to observe. For a dimension-changing step it is
  also the only
  store whose keys have the working dimension at all.
- **Original-problem invariant** — at any time, the problem handed to a driver is
  retrievable with its raw functions, its database and its design space intact. This is
  the requester's core ask in #1528.

### Key Business Rules

- **The user's problem is never mutated by a driver** — governs `EvaluationProblem`,
  `OptimizationProblem`, `BaseDriverLibrary`; it replaces today's "preprocess in place,
  reset afterwards" contract and makes the `EvaluationScenario` DOE reset and the
  `preprocessing=False` flags unnecessary.
- **Recording is the problem's; adaptation is the algorithm's** — governs the split
  between recording attachment and problem transformation, carried by `DatabaseFunction`
  and `TransformedInputFunction` respectively. Recording covers database
  keys, call counts, the new-iteration hook and gradient approximation — which places the
  `enable_statistics` `ClassVar`, and the configuration validator that writes it, on
  `DatabaseFunction`, since what it gates is counting. The `_log_settings` section headed
  `PreprocessedFunction` follows the flag and becomes `DatabaseFunction`; that user-visible
  change happens whatever the adaptation wrapper ends up being called. Adaptation
  covers coordinates and the NaN checks, which exist to *stop an algorithm* and which a
  DOE explicitly disables. It does **not** cover the tolerance-based criteria: those are
  governed by the rule below and belong to the original problem.
- **The database is recorded in the original problem's coordinates** — governs
  `Database` and every transformation. Keys are real (denormalized, rounded) design
  vectors of the user's design space; stored gradients are in real coordinates. A
  transformation that changes the design space must map back before recording.
- **One shared `Database`, plus an optional debug database** — governs the derived
  problem. Listeners, the history, the KKT checker and the progress bar read
  `problem.database` during the run; a copy or `None` would silently lose results, so the
  original problem's database stays the single object every run-time reader sees. On top
  of that, and only when the debug mode is enabled, the derived problem owns a **second**
  database in working coordinates. The second one is never read by the result, the
  history, the KKT checker or the progress bar; enabling the debug mode must leave every
  number the user gets unchanged. `use_database` scopes to the **original** problem's
  database alone: it turns that store and lookup on and off and governs nothing else —
  not the working-coordinate store, not call counting, not derivative approximation. The
  two flags are orthogonal and all four combinations are valid.
- **`.original` reaches the user's function through any depth of wrapping** — governs
  `ArrayFunction` and its fourteen internal readers plus three plugin readers. Today's
  one-level assumption breaks as soon as two wrappers stack.
- **`.original` means "before preprocessing", not "before any algebra"** — constraint
  formatting (offset, negation) and aggregation are part of the problem's definition and
  must remain visible through `get_originals`.
- **Coordinate transformations compose; semantic transformations do not belong in the
  series** — sign change, aggregation, scaling and penalty change what the database
  records and how feasibility is judged; they remain problem-definition operations
  (some already return a new problem, e.g. the slack reformulation).
- **Numbers are preserved** — finite-difference gradients keep being computed in
  normalized coordinates when the algorithm normalizes (the absolute step must not
  silently scale by the variable range); each evaluation is counted once; NaN handling
  keeps stopping optimizers and not DOEs.
- **An approximated Jacobian is recorded; its perturbations are not** — governs derivative
  approximation and the recording layer. Today this holds only because the approximator is
  built on `self._compute_output`, the **non-recording** callable
  (`core/function/preprocessed_function.py:181`), while the function itself evaluates
  through the dispatched `_compute_output_db*` variants (lines 150-161). Nothing names the
  invariant, and a split that puts approximation above recording would store every
  perturbed point in the database, inflating the history and the call counts and polluting
  every post-processing. It must become an explicit contract with a regression test
  asserting the database grows by one entry per approximated Jacobian, not by one per
  perturbation.
- **Tolerance-based termination criteria are evaluated against the original problem** —
  governs `_pre_run`, the tolerance testers and the `_KKTChecker` listener. The user
  supplies `ftol_*`, `xtol_*` and the KKT tolerances in the scales of their own functions
  and variables, so the comparison must happen there. Today this holds because all three
  read the database, whose keys are real coordinates
  (`DesignToleranceTester._check` through `get_last_n_x_vect`, `_KKTChecker` through
  `database.get_function_value`). After the split the numbers stay real through the shared
  database, but the **problem object** passed to `_pre_run` decides which
  `constraints.is_point_feasible` and which `standardized_objective_name` are consulted,
  so the criteria must be registered on the original problem even though the algorithm
  iterates on the derived one.
- **Re-execution with different settings takes effect** — a second driver run on the
  same problem (bi-level, adapters executing hundreds of times, augmented Lagrangian
  sub-problems) must honour the second driver's settings instead of the first's.
- **The typed domain is reached once, at the end** — governs every step that
  relaxes a domain. `untransform` (working → original) runs at every evaluation and must
  be a true inverse; `project` (original → the typed subset, in the same coordinates)
  runs once, on `x_opt`. This is not hypothetical: today `_preprocess_function` composes
  `denormalize_vect → round_vect → func` on **every** call
  (`core/problem/evaluation.py:862-879`), and `round_vect` appears nowhere else in `src/`,
  so the projection is currently folded into the backward map. That fusion is **correct
  while nothing relaxes**: `round_ints` exists so the discipline receives `3` rather than
  `3.4`, and unfolding it in this delivery would change what every integer-handling
  discipline sees. It becomes wrong the moment a step relaxes a domain, because rounding at
  every evaluation turns the function into a step function and zeroes the gradient, which
  is exactly what relaxing an integer exists to avoid. So the rule is per step: a
  **relaxing** step must not round in its backward and defers to `project`; a
  non-relaxing `round_ints` step rounds in its backward, and its `project` is then
  idempotent.
- **The user's `DesignSpace` object is written to once, at the end** — governs the
  derived problem and the driver teardown. During the run the untransformed point lives
  in a plain vector; it is never written into the typed space object, which would reject
  a real value for an `IntegerVariable`. The space's setter is called after `project`.
- **Finite-bound requirements are checked on the transformed space** — governs the
  series. A step that widens a bound (an unbounded integer relaxed) can break a
  normalization placed after it, so `requires_finite_bounds` is evaluated on the space
  each step actually receives, not on the user's space.
- **A step applies on a space *capability*, never on a space class** — governs the series
  and the driver. `preprocess_functions` decides today with
  `isinstance(input_space, DesignSpace)` and two hard-coded `ValueError`s; the series must
  ask instead for what the step needs (bounded, normalizable, has integer variables, has a
  joint distribution), so that `RandomSpace` — and any future space — contributes or omits
  a step instead of adding a branch.

## Strategic Approach

### Solution Direction

- **Split preprocessing into two layers with different owners.** The problem keeps a
  minimal *recording attachment* of its own functions to its database and counter. The
  *adaptation* (normalization, rounding, densification, NaN control) becomes a
  *transformation* that produces a derived problem. The driver is the first client of the
  transformation and discards the derived problem after the run; the original is left
  untouched and is what `execute` returns results on.
- **Make the transformation a first-class, composable object rather than a method with
  seven booleans.** Today's five-branch builder encodes the cross product of
  `normalize × round_ints × linear` by hand. A series of steps, each knowing how to wrap a
  function's value and Jacobian (and, when needed, how to map a design vector both ways),
  turns that into ordered composition, and gives the discrete and categorical work of
  #1885 a slot to plug into without touching the problem class.
- **Expose the transformation to users, with the driver as its main consumer.** The
  requirement and the issue both want the user to be able to derive a problem (the
  warm-start use case relaxes, solves, then returns to the original). The posted proposal
  kept the builder driver-private; this analysis recommends a public entry point built on
  the same primitive, so that the user and the driver go through one code path.
- **Design the derived problem around a shared database and a possibly distinct design
  space.** Coordinate steps share the design space; relaxation and encoding steps produce
  a new one plus a mapping. The abstraction must allow the mapping from the start, even if
  the first delivery implements only the shared-space case.
- **Give the derived problem a place to record what the algorithm saw, off by default.**
  The derived problem already has a working space; letting it own a database over that
  space turns "what did the optimizer actually evaluate" from a debugging expedition into
  a `to_dataset` call. Because it doubles the store cost, it is gated by a global debug
  flag that is disabled by default, and it changes nothing when disabled.
- **Leave export where it is.** `to_dataset` and `to_hdf` are ten lines of delegation to
  `Database`; they do not contribute to the "does too much" problem and moving them would
  churn user code for no structural gain.
- **Leverage existing conventions**: `BaseVariableSpace` already declares the generic
  `transform_vect` / `untransform_vect` hook, filled with normalization by `DesignSpace`
  and with the iso-probabilistic transform by `RandomSpace`, and the `DesignSpace` already
  owns the coordinate maps; the `Functions` collections
  already know how to rebuild from originals; the slack reformulation already shows the
  "return a new problem" shape; capability flags on the space (`has_integer_variables`)
  already gate rounding.

### Key Design Decisions

- **Ownership of the transformation entry point** — driver-private helper (posted
  proposal) versus public API usable by users (this requirement). A private helper is a
  smaller surface and easier to change; a public one satisfies the warm-start use case and
  the requirement's "let the user or the driver" phrasing, at the cost of committing to a
  contract in a major release. → **Decided (B1): public**, with the driver calling the same
  primitive, and the contract kept narrow (a problem in, a problem out, a series of steps).
- **Granularity of a step** — whole-problem transformation versus per-function wrapper.
  Per-function wrappers alone cannot express a design-space change; whole-problem
  transformations alone hide the composition. → **Recommend a problem-level
  transformation that composes function-level steps**, with an optional design-space
  mapping carried by the step that needs it.
- **Shared versus mapped design space** — sharing keeps every current-value, bounds and
  history consumer trivially correct; mapping is required for relaxation and encoding. →
  **Recommend allowing a mapping in the abstraction, implementing sharing first**; the
  requirement's discrete and categorical steps are the ones that will need the mapping,
  and they do not exist yet (#1885 defers them). The mapping abstraction should be
  #1886's `BaseSpaceTransformation`, not a second one.
- **Where derivative approximation lives** — with the algorithm (it depends on the
  algorithm's normalization) or with the recording layer (its output must be stored). The
  approximator replaces the Jacobian; outside the recording layer the approximated
  gradient never reaches the database and the KKT criterion, Lagrange multipliers and
  gradient export silently break. → **Recording layer**, parameterized by whether to
  perturb in normalized coordinates so that finite-difference numbers are preserved.
  Three things this decision must spell out, because each is load-bearing and none is
  visible in today's code:
  **(i)** the *result* is recorded and the *perturbations* are not — the approximator is
  wired to the non-recording callable, and that must be stated rather than inherited;
  **(ii)** the approximator perturbs in working coordinates while the database receives
  the original gradient, so `untransform_jacobian` sits between the two;
  **(iii)** approximation is therefore **not** a step of the transformation series — it
  wraps the series, which is why it can sit in the recording layer while the coordinate
  steps sit in the adaptation one.
- **Where NaN control lives** — with the recording layer (as today, when a database is
  used) or with the adaptation layer. The checks raise termination criteria caught by the
  driver, and a DOE turns them off. → **Adaptation layer**; this also decouples NaN
  checking from `use_database`, which today silently disables it.
- **Where the debug switch lives** — a field on `GlobalConfiguration`
  (`gemseo.configure(...)`), a driver setting, or an argument of the transformation entry
  point. A driver setting only covers the driver path, and the derived problem is built
  inside `BaseDriverLibrary.execute` where the user holds no handle on it. →
  **`GlobalConfiguration`**, mirroring `enable_function_statistics` exactly: default
  `False`, added to the list `__validate_fast` forces off when `fast=True`, and to the
  **second** list of the `fast=False` branch, the one forced back to `False` — putting it
  in the first list would switch debugging *on* for anyone who writes
  `configure(fast=False)`. The transformation entry point may still take an explicit
  override for the user-built path.
- **Idempotence strategy** — a boolean guard that skips the second call (today) versus
  rebuilding from the originals on each call. The guard drops the second caller's settings
  (visible in the augmented-Lagrangian path). → **Rebuild from originals**, carrying call
  counts across.
- **Migration policy** — remove `preprocess_functions`, `reset(preprocessing=…)` and the
  setter guards outright in 7.0.0, or deprecate with shims. Internal callers are few and
  the user guide already warns that they are for algorithms only, but four sibling plugins
  do use them (see External consumers). → **Decided (B3): outright removal in 7.0.0**,
  with a changelog entry, the `bump-version` mapping updated, and **four coordinated
  plugin MRs** — `gemseo-umdo`, `gemseo-mlearning`, `gemseo-calibration`,
  `gemseo-benchmark` — landing with the bump. Until those land, `gemseo-umdo` and
  `gemseo-mlearning` break at runtime, so they are part of the delivery, not follow-up.
- **Scope of the first delivery** — coordinate steps only (normalization, rounding,
  densification) plus the recording split, or also the semantic operations the issue lists
  (sign change, aggregation, relaxation). The semantic ones change the recorded history
  and the feasibility semantics. → **Coordinate steps and the recording split now**; make
  the "sibling problem sharing a design space" primitive public so the semantic ones and
  the #1885 solving story can build on it later.
- **Own space-transformation contract versus #1886's** — defining the step contract
  inside #1528 keeps the two stories independent; reusing `BaseSpaceTransformation` /
  `TransformationChain` avoids two competing designs for one idea. #1528 needs *some* step
  contract to express a series at all, so the real question is whose shape it carries, not
  whether one is written. → **Decided (B2): #1528 implements
  `BaseSpaceTransformation` / `TransformationChain` to #1886's specification**, together
  with the normalization step it needs anyway; #1886 then rebases onto it and contributes
  `IntegerRelaxation` and the encoders. #1528 keeps the recording/adaptation split and the
  derived problem as its own contribution. This enlarges #1528's scope by the contract and
  requires #1886's author to accept an interface merged by another story.
- **`LinearFunction` fast path** — accept one denormalization per linear evaluation (the
  position taken in AC#5 below) versus keeping the fold into the coefficients. → **Keep
  the fast path**, guarded by #1886's `is_affine` flag on the step and generalized from
  `LinearFunction.normalize` to `LinearFunction.fold(transformation)`. Without it
  `ScipyLinprog` and `ScipyMILP` lose an order of magnitude; this supersedes the
  performance loss AC#5 accepted.
- **Gradient signature of a step** — `transform_jacobian(jac)` while every step is affine,
  versus `transform_jacobian(jac, x=None)` from the start. → **Take the point from the
  start.** The Rosenblatt transform already used by `RandomSpace`, and any smooth
  relaxation, need it; adding it afterwards changes every implementation and every call
  site, and the preprocessing code already has the point in hand where it is needed.
- **Dimension-changing steps** — left open by this analysis (see Ambiguities). → **Close
  it the way #1886 does**: the contract allows a dimension change, a guard rejects it
  when the series is built, so the user gets an explicit error at `execute()` instead of
  a crash mid-run, and the follow-up story lifts a guard instead of changing a contract.
- **Constraints created by a transformation** — no coordinate step needs one, but a
  relaxation or an encoding may (for example `Σt = 1`). → **Specify the hook now**:
  `create_constraints`, called **once per problem** and never per function, with the
  produced constraints tagged so they are hidden from the result shown to the user.
- **What a step may know about the problem** — a step that attaches the database, counts
  evaluations or checks for NaN itself forces every future step to repeat that wiring. →
  **Keep the space-level step independent of the problem**, as #1886 does; all wiring
  stays on the #1528 side (recording attachment and the problem transformation). This is
  the same separation of recording from adaptation that this analysis is built on.

### Alternatives Considered

- **Keep in-place preprocessing and harden the undo** — extend `reset` and have `execute`
  restore the functions in its teardown. Rejected: it keeps the mutation contract that
  produces the workarounds, cannot undo `scaling_threshold`, and does not give users a
  derived problem.
- **Deep-copy the whole problem before preprocessing** — snapshot and restore. Rejected:
  the copy severs the shared `Database` identity that ten run-time readers rely on,
  duplicates listeners, and doubles memory for every execution rather than only when a
  user asks for a variant.
- **Push all transformations into `DesignSpace`** — make the space do normalization and
  rounding transparently. Rejected: the functions must still be bound to the database and
  the counter, and the drivers still need a function object that speaks their coordinates.
- **Have drivers store into the database themselves** — leave functions raw and let each
  algorithm record. Rejected: it moves the recording responsibility to every driver and
  plugin and loses the evaluation cache that keys on the point.

## Relationship with #1886 (Transformations around the `DesignSpace`)

Issue [#1886](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1886) — "Transformations
around the `DesignSpace`" — attacks the *space* half of what this requirement calls a
series of transformations: normalization, relaxation, encoding, and the backward maps that
return the solution in the user's coordinates. Its analysis already exists on the branch
`design-space-transformations` (MR !2719) and **specifies** a `BaseSpaceTransformation` /
`TransformationChain` contract. The contract is specified there, not implemented: that
branch carries no transformation source, and its `src/` changes are the #1845
variable-hierarchy work, still on the pre-rebase `space/design/` paths.

The two stories overlap on purpose. #1528 changes the API more deeply and goes first, so it
should **reuse** #1886's decisions rather than solve the same problems again — or, worse,
break something #1886 already fixed. The seven points below are the ones that bind.

| # | Decision taken in #1886 | What #1528 must do | Cost of ignoring it |
|---|---|---|---|
| 1 | **Two coordinate systems plus a projection** — recorded in #1886 as "three spaces, not two", amended here: *working*, what the algorithm sees, continuous and normalized (`0.24`); *original*, the real value at the scale the user declared (`3.4`, because the integer was relaxed), which is what the discipline is evaluated at and what the database keys on; and the *typed domain*, the values the user actually asked for (the integers), which is a **subset of the original space in the same coordinates**, not a third space. Two maps, never merged: `untransform` (working → original, every evaluation, bijective) and `project` (onto the typed domain, once, on `x_opt`). | Model the original coordinates explicitly and keep the two maps separate; never write the untransformed point into the typed `DesignSpace` object during the run — only into a plain vector — and call the space's setter once, after `project`. Do **not** build a third space object. | Rounding at every evaluation makes the function a step function and the gradient identically zero, which defeats the relaxation; never projecting returns a non-integer `x_opt` for an `IntegerVariable`. |
| 2 | **`is_affine` on the step**, guarding the `LinearFunction` fold. | Keep the fold — generalize `LinearFunction.normalize` into `LinearFunction.fold(transformation)` — instead of accepting one denormalization per linear evaluation. | `ScipyLinprog` and `ScipyMILP` lose an order of magnitude. |
| 3 | **`transform_jacobian(jac, x=None)`**, the point taken from the start. | Fix the gradient signature now, even though the first steps are all affine and ignore `x`. | Rosenblatt (already used by `RandomSpace`) and any smooth relaxation force a second refactor of every implementation and every call site. |
| 4 | **Dimension change: allowed by the contract, refused by a guard** when the series is built. | Close this analysis's open question the same way. | The follow-up story has to change a contract instead of lifting a guard, and a user gets a mid-run crash instead of a clear error at `execute()`. |
| 5 | **`create_constraints`, called once per problem**, never per function, with the produced constraints tagged so they are hidden from the result shown to the user. | Specify the hook, even though no coordinate step exercises it. | A relaxation or an encoder needing `Σt = 1` has nowhere to put it: such a constraint belongs to no function. |
| 6 | **`requires_finite_bounds` is checked on the transformed space.** | Check it after each step, not once on the original space. | A step that widens a bound (an unbounded integer relaxed) silently breaks a normalization placed after it. |
| 7 | **The space-level transformation knows nothing about the problem.** | Keep all wiring — database attachment, evaluation counting, NaN checks — on the #1528 side (recording attachment and `EvaluationProblem.transform`). | Every new step (relaxation, future encoders) repeats the wiring, and #1528 loses the separation between recording and adaptation that is its whole point. |

Four further consequences for this analysis:

- **Two spaces, not three.** The separation #1886's BR-1 defends is real — the two maps
  have different schedules and different mathematical natures — but it is a separation of
  **maps**, not of spaces. Working and original are two coordinate systems: different
  units, different bounds, a change of variables between them. The typed domain is
  the *same* coordinates as the original space with the integrality or discreteness
  constraint reinstated, and membership in it is already expressible with
  `check_membership` and the variable types. Counting it as a third space invites two
  concrete mistakes: building a third `DesignSpace` object that has to be kept in sync —
  which the business rules above forbid — and placing `project` in the chain beside
  `transform` / `untransform`, when its schedule (once, on the result) is entirely
  different. #1886's own document already shows the confusion: its signature block
  comments `project(x)  # working -> admissible domain`, while its sequence diagram
  applies `project` *after* the backward map (`3.4` → `3`), that is, from the original
  coordinates. **Settled: the diagram is right and the signature comment is the error —
  `project` takes a original point**, and the middle space is called *original* in both
  stories. Vocabulary for both: **two spaces and a projection onto the typed
  domain**, with `project()` defaulting to the identity on the base contract.
- **The projection is degenerate for every step that exists today.** `project` is the
  identity for normalization and for the iso-probabilistic transform of a `RandomSpace`,
  where nothing is relaxed; it is non-trivial only for integer relaxation, which is not
  implemented and which #1885 defers. No current algorithm needs it either: `ScipyMILP`
  passes `integrality=problem.input_space.get_integer_mask()` straight to SciPy
  (`optimization/scipy_milp/scipy_milp.py:149`) instead of relaxing. Keep the hook —
  retrofitting it later would touch every implementation — but size it as one method with
  a default, not as a third first-class space in the vocabulary.
- **Naming of the middle space.** #1886 calls it the "user" space. The name is misleading:
  it sounds like the `DesignSpace` object the user created, but that object — with its
  type constraint, e.g. `IntegerVariable` — is what carries the *typed* domain, and
  it would reject `3.4`. The middle space is a plain vector of reals with no type
  constraint, needed because the discipline must sometimes be evaluated at a real value.
  "Relaxed" is no better: it fits `IntegerRelaxation` only, not a plain normalization run
  nor the Rosenblatt transform, where the split still holds and nothing is relaxed.
  "Physical" was carried for a while and then dropped: not every space a user passes holds
  physical quantities — a `RandomSpace` carries random variables, and a design space may
  hold abstract coefficients — so the word claims an engineering meaning the concept does
  not have. **The middle space is the *original* space**, which is what the rest of this
  document already calls the things the user handed over: the original problem, the
  original-problem invariant, `ArrayFunction.original`. "Real" was rejected outright, since
  `to_complex()` makes "real" mean non-complex here. Whatever the label, the rule matters
  more than the name (see the business rules above).
- **Ownership of the contract.** #1528 should mention #1886 explicitly and either reuse
  `BaseSpaceTransformation` / `TransformationChain` for its design-space part, or state
  clearly why it does something different. Anything else leaves two competing designs for
  the same idea.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Public or driver-only transformation API**: the requirement says "the user or the
  driver"; the posted proposal decided driver-only. **Resolved (B1): public.**
- **What "a series of transformations" is composed of**: function-level steps, or
  problem-level steps that may also change the design space? The discrete and categorical
  steps mentioned require a design-space change; nothing else in the requirement does.
  **Settled: leave room, do not deliver them** — the contract accepts a dimension change
  and a guard refuses it for now, so the follow-up story lifts a guard instead of changing
  a contract.
- **Semantics of the derived problem's database**: shared object versus its own. The new
  requirement answers it — the shared object for everything the user reads, *plus* a
  second database in working coordinates when the debug mode is enabled. Three follow-up
  questions remain, below.
- **Debug database versus `use_database=False`**: when a driver disables the database, the
  wrapper loses its store path entirely. **Resolved: the debug store is an independent
  path and records whenever its flag is on**, including under `use_database=False`, which
  is precisely when a user hunting a bug wants it. The two settings do not overlap at all:
  `use_database` names the **original** problem's database and scopes to it alone. The
  debug store therefore cannot be hung off the existing database branch, and lives on the
  adaptation layer, the only one holding a working point.
- **Debug database under parallel execution**: wrappers are pickled to workers and results
  are collected centrally. Either the working-coordinate entries are collected like the
  shared ones, or the mode is documented as single-process only. Confirm which.
- **Name of the debug flag**: `enable_working_database` is the spelling that matches
  `enable_discipline_cache` / `enable_function_statistics`. Confirm.
- **Export responsibility**: the requirement lists export among the things the problem
  does. **Resolved: `to_dataset` and `to_hdf` stay on the problem.** They are ten lines
  delegating to `Database`, they do not contribute to the "does too much" problem, and
  moving them would be a second breaking change in 7.0.0 touching far more user code than
  the algorithm-facing APIs already being removed.
- **Migration policy**: outright removal in 7.0.0 versus deprecation shims. **Resolved
  (B3): outright removal**, with the four plugin MRs treated as part of the delivery.
- **Naming**: whether the adaptation wrapper keeps the name `PreprocessedFunction` and
  whether "transformation" or "preprocessing" is the user-facing vocabulary. **Resolved:
  `DatabaseFunction` for recording and `TransformedInputFunction` for adaptation**, with
  "transformation" as the user-facing word for the series and the entry point. The posted
  proposal's `PreprocessedFunction` is dropped: once the class holds only the chain and
  the NaN checks, "preprocessed" names what it no longer does, and "preprocessing"
  disappears from the vocabulary entirely. The name says **input** because the class does
  not transform the function: `f(x)` returns the same number in either coordinate system,
  and only the argument, and by chain rule the Jacobian, are mapped. A bare
  `TransformedFunction` would collide with the operations that genuinely do transform a
  function — `Constraints.format`'s offset and negation, `Constraints.aggregate`, and
  `scaling_threshold` — which the rule above deliberately keeps out of the series.
  The `_log_settings` heading moves to `DatabaseFunction` regardless, because it follows
  the statistics flag rather than the adaptation wrapper.
- **Name of the middle space, and whether the typed domain is a space at all**:
  #1886 calls the middle one the "user" space, which names the wrong thing — the object
  the user created is what carries the *typed* domain. "Relaxed" covers only the
  integer-relaxation case; "original coordinates" covers every transformation kind. This
  analysis further recommends dropping "three spaces" for "two spaces and a projection".
  **Resolved: the middle space is called *original*, `project` takes a original point,
  and the third domain is called *typed*** — #1886's sequence diagram is right and its
  signature comment is the error. "Admissible" is dropped because in optimization it means
  *feasible*, that is, satisfying the constraints; this document already uses "feasible"
  in that sense, including `constraints.is_point_feasible`. The third domain is about what
  the **variables** accept, not what the constraints allow: an `IntegerVariable` rejects
  `3.4` whether or not any constraint is satisfied. Both stories use this vocabulary.
- **Discrete and categorical steps**: their actual semantics (relaxation, rounding to the
  nearest potential value, one-hot or ordinal encoding) are not specified anywhere yet;
  #1885 defers them. This analysis treats them as future steps that must be *pluggable*,
  not as deliverables.

### Edge Cases

- **Nested drivers**: bi-level formulations and `MDOScenarioAdapter` execute the same
  problem hundreds of times, sometimes from inside another driver's evaluation. The
  iteration hook is detected by inspecting the functions; two layers of wrapping must not
  make the inner driver clobber the outer driver's callback.
- **Second execution with different settings**: DOE after optimizer (normalization
  off after on), augmented-Lagrangian sub-algorithms, multi-start. Today's guard silently
  keeps the first settings.
- **Functions added between two executions**: must be wrapped by the second execution.
- **Vectorized DOE with normalization and a database**: today's dispatch gives the
  normalized path precedence and stores one entry keyed on the whole sample matrix; the
  block-diagonal Jacobian is normalized only on its first block. The combination is not on
  any default path — `BaseDOESettings` overrides `normalize_design_space` to `False` and
  `vectorize` defaults to `False` — so it is latent, not live. → **Decided (B4): refuse
  it**, with a third `NotImplementedError` in `BaseDOESettings.__check` beside the existing
  `n_processes > 1` and `preprocessors` guards. The vectorized path is then only ever
  reached without normalization, and AC#8's "history preserved" becomes well defined.
- **Parallel DOE**: wrappers are pickled to workers; bound methods and closures in the
  series must survive pickling; call counters do not propagate under `spawn`.
- **Integer variables with complex step**: rounding zeroes the imaginary perturbation
  today; the refactor may move or fix this and must not hide it.
- **Linear problems**: the `LinearFunction.normalize` fast path returns a new function
  with rescaled coefficients rather than a composition; LP and MILP libraries read
  `.original.coefficients`. A composition-based series cannot use that fast path without
  bypassing recording. #1886's answer is the `is_affine` flag on the step plus a
  generalized `LinearFunction.fold(transformation)`, which keeps the fold for any affine
  series.
- **New-iteration observables**: evaluated from inside a database store notification in
  real coordinates; they need recording but no adaptation and must not raise termination
  criteria from a listener.
- **A problem that is attached but never handed to a driver**: no adaptation layer, so no
  NaN check. Acceptable by the rule "no algorithm, nothing to stop", but it changes
  behaviour for scripts that call `evaluate_functions` by hand after preprocessing.
- **Problems reloaded from HDF**: constructed without any wrapping; must behave like a
  fresh problem.
- **Debug databases under nested drivers**: bi-level formulations and `MDOScenarioAdapter`
  build a derived problem per execution, hundreds of times. With the debug mode on, each
  one would own a database that is never released, turning a diagnostic into an
  out-of-memory error. The name of each must also identify its driver, or the HDF export
  of the outer run is unreadable.
- **A problem whose input space is not a `DesignSpace`**: `ReliabilityProblem` preprocesses
  a `RandomSpace` problem. The derived problem must be buildable with an empty coordinate
  series, and the two `ValueError`s guarding normalization and rounding on such a space
  must still fire on the same trigger.
- **`stop_if_nan` set by a DOE on the problem it received**: must reach the functions
  that actually perform the check, whichever layer that is.

### Technical Risks

- **Silent wrong coordinates.** Drivers evaluate through `self._problem` with normalized
  vectors. If the driver keeps a reference to the original problem while running, a
  normalized vector fed to a real-coordinate function stays within bounds and converges
  to the wrong point with no error. Mitigation: the driver's working reference during the
  run is the derived problem; the original is held separately for results and teardown.
- **Criteria registered on the derived problem.** `_pre_run` builds the tolerance testers
  and the `_KKTChecker` listener from the `problem` it is handed, and the driver's working
  reference during the run is the derived one. The values would stay real through the
  shared database, so nothing would fail loudly; only the functions consulted for
  feasibility and for the objective name would be the wrapped ones, and a user's `xtol`
  would silently start meaning something else the day a step changes the design space.
  Mitigation: pass the original problem to `_pre_run` explicitly, and assert in a test
  that the tolerances behave identically with and without normalization.
- **Finite-difference numbers drift.** Moving the approximator into real coordinates
  naively scales the absolute step by the variable range. Mitigation: perturb in
  normalized coordinates when the algorithm normalizes; expect re-pinning of snapshots on
  wide-range problems within FD noise.
- **Call-count semantics change.** Counting once at the surviving layer makes counts
  cumulative across executions. Mitigation: state it in the changelog; adjust the
  termination-criteria tests that assert per-execution counts.
- **Database key rounding.** Keys become the rounded real point on every path; the
  normalized path already rounds, the raw path may shift on integer spaces; parallel DOE
  pre-seeds keys at the unrounded sample. Mitigation: assert that untransformed samples are
  integral on integer components.
- **Non-design input spaces.** `EvaluationProblem` is generic over `BaseVariableSpace`,
  and `ReliabilityProblem` reaches `preprocess_functions` with a `RandomSpace`. Today two
  `isinstance` narrowings turn that into explicit `ValueError`s; a series that silently
  built an empty chain instead would turn a clear error into a wrong run on a space that
  cannot be normalized. Mitigation: pin both messages in the regression set before the
  split, and route the decision through a capability check rather than the class.
- **Depth of `.original`.** Fourteen internal readers and three plugin readers assume one
  level. Mitigation: make it transitive as a prerequisite, without forwarding it through
  algebraic operations (offset, negation), which would return the un-offset constraint.
- **Design-space-changing steps have no support in `Database`.** Keys are full vectors
  over the problem's design space; a relaxed or encoded space needs a mapping before every
  store and after every read. This is the main cost of the discrete and categorical
  extension and is not covered by the coordinate-only delivery. #1886 halves it by
  applying the backward at **write** time — the key is the untransformed (original) point
  — so the history is already in the user's coordinates and no decoding pass over the
  database is needed at the end of the run.
- **A debug mode that changes the result.** The whole value of the working-coordinate
  database is that it observes the run without taking part in it. If the debug store is
  placed inside the recording layer it will shift call counts, listener order or the
  `pre_compute_at_new_point` hook, and the bug being hunted moves when the mode is turned
  on. Mitigation: make "every output identical with the flag on and off" an explicit
  regression test on a gradient-based run, a DOE and a bi-level case, not a review
  comment.
- **Scope of code touched.** About seventy test occurrences across fourteen files, two
  user-guide pages under a strict docs build, the global configuration plumbing — both
  the new debug flag and the existing `enable_function_statistics`, whose validator writes
  `PreprocessedFunction.enable_statistics` directly and must be repointed in the same
  commit as the split, alongside the `_log_settings` section that reads it — and the
  `bump-version` mapping.
  Mitigation: land the two prerequisites (transitive `.original`, and the
  `NotImplementedError` guard refusing vectorization with normalization) as separate
  commits with regression tests before the structural change.
- **External plugins.** Only `.original` is read from plugins; drivers in plugins that
  evaluate `self._problem` keep working if the driver's working reference is the derived
  problem. Any plugin that subclasses `PreprocessedFunction` or calls
  `preprocess_functions` would break; none was found among the sibling repositories.

### Acceptance Criteria Coverage

The requirement states no numbered acceptance criteria. The following are derived from
its sentences and from the issue description; each is marked as inferred.

| AC# | Description (inferred) | Addressable? | Gaps/Notes |
|-----|------------------------|--------------|------------|
| 1 | `EvaluationProblem` no longer preprocesses its functions; that responsibility leaves the class. | Yes | Recording attachment stays on the problem by necessity (database, counter, gradient approximation must be recorded). State this as a responsibility split, not a full removal. |
| 2 | The problem given by the user is retrievable, unmutated, at any time, including after a driver run (issue #1528). | Yes | Requires the driver to build and discard a derived problem and to return results on the original. The `scaling_threshold` leak is fixed as a side effect. |
| 3 | A user **or** a driver can take an original problem and obtain a new problem whose functions wrap the originals. | Yes | Both paths covered: B1 settled the user-facing entry point as public, diverging from the posted proposal, with the driver calling the same primitive. |
| 4 | The wrapping is expressed as an ordered series of transformations. | Yes | Coordinate steps compose naturally; semantic operations (sign, aggregation, scaling, penalty) are excluded from the series by design. |
| 5 | Normalization and integer rounding are available as transformations, equivalent to today's behaviour. | Yes | The `LinearFunction` fast path is **kept**, not traded away: #1886's `is_affine` flag lets an affine series be folded into the coefficients (`LinearFunction.fold`), which `ScipyLinprog` and `ScipyMILP` depend on. This replaces the denormalization-per-linear-evaluation cost accepted earlier. FD gradients are preserved within FD noise, not bitwise. |
| 6 | The mechanism is extensible to discrete and categorical variable transformations. | Partial | Pluggable step slot is addressable now; the design-space mapping those steps need is not in the coordinate-only delivery, and their semantics are not yet specified (#1885 defers solving). #1886 supplies the mapping contract — two spaces plus a projection, `untransform` versus `project`, `create_constraints` — that the coordinate-only delivery must leave room for. |
| 7 | Evaluation of the original or the transformed functions, and export to dataset or HDF, keep working. | Yes | Export unchanged; `get_functions(no_db_no_norm=True)` becomes simply "the raw functions". |
| 8 | Recorded history semantics (keys, gradients, dataset and HDF content) are preserved. | Yes | Database shared by the derived problem; keys rounded on every path; densification kept before store; NaN outputs now stored before the stop is raised. |
| 9 | With a debug mode enabled, the working problem owns a database in its own coordinates; disabled by default. | Yes | Additive to the shared database, never a replacement. Disabled is the no-op case and must cost nothing: same numbers, same call counts, same memory as today. Enabled, the working history is exportable through the usual `to_dataset` / `to_hdf`, over the working space, which stay on the problem. It records whenever its flag is on, including under `use_database=False`, so it is an independent path rather than a branch of the shared store. Open: behaviour under parallel execution, and the flag's name. |
| 10 | User-supplied tolerances and the approximated-gradient history keep their meaning. | Yes | `ftol_*`, `xtol_*` and the KKT tolerances are compared in the user's scales, on the original problem. An approximated Jacobian adds one gradient entry to the database; its perturbation evaluations add none. Both hold today and must be asserted rather than assumed. |

## Blocking Before Canvas

The open points above are not equally urgent. This section triages them by one test: can
the REASONS Canvas be written without the answer, or would the canvas have to invent it?
The four blocking ones were taken on 2026-09-16 and are recorded below with their answer.

### Blocking — resolved

| # | Question | Decision | What it commits the canvas to |
|---|----------|----------|-------------------------------|
| B1 | Public transformation API, or driver-only helper? | **Public.** | A documented 7.0.0 contract: a problem in, a problem out, a series of steps, with the driver calling the same primitive. The canvas carries a user-facing surface, not just an internal refactor. |
| B2 | Who lands `BaseSpaceTransformation` / `TransformationChain`? | **#1528 implements it to #1886's specification.** Agreed with #1886's author on 2026-09-16, offline, together with the seven amendments in the appendix. | The contract and the normalization step enter #1528's scope; #1886 rebases onto them and adds `IntegerRelaxation` and the encoders. Requires #1886's author to accept an interface merged by another story. |
| B3 | Outright removal in 7.0.0, or deprecation shims? | **Outright removal** of `preprocess_functions`, `reset(preprocessing=…)` and the setter guards. | A changelog entry, the `bump-version` mapping, and four coordinated plugin MRs (`gemseo-umdo`, `gemseo-mlearning`, `gemseo-calibration`, `gemseo-benchmark`) landing with the bump — part of the delivery, since two of them break at runtime otherwise. |
| B4 | Vectorized DOE with normalization: reference behaviour, or bug? | **Refuse the combination.** | A third `NotImplementedError` in `BaseDOESettings.__check`, beside the `n_processes > 1` and `preprocessors` guards, landed as its own commit before the refactor. The vectorized path is then only reached without normalization, so AC#8's "history preserved" is well defined. |

Evidence behind B3 and B4, since both reversed an earlier reading of the code: four
sibling plugins do call the APIs being removed (see External consumers), and the
vectorized-plus-normalized combination is latent rather than live because
`BaseDOESettings` overrides `normalize_design_space` to `False` while `vectorize` also
defaults to `False`.

### Names and signatures — resolved

These never blocked the canvas, but it would have committed to an answer whether or not
one was chosen, because each becomes a name or a signature. They were taken on 2026-09-16.
The fourth row was listed as an ambiguity but omitted from the first triage; that is
corrected here.

| Question | Decision | Consequence |
|----------|----------|-------------|
| Layer names and user-facing word | **`DatabaseFunction`** for recording, **`TransformedInputFunction`** for adaptation; "transformation" is the user-facing word. | "Preprocessing" leaves the vocabulary altogether, and **input** distinguishes the coordinate wrapper from the semantic function transformations (offset, negation, aggregation, scaling) that the business rules keep out of the series. The rename costs a `bump-version` `classes:` entry and a changelog line; no sibling plugin references the class, so nothing downstream breaks. |
| `project`'s domain, and the names of the domains | **`project` takes a original point**; the middle space is called **original** and the third domain **typed**, not "admissible". | #1886's sequence diagram is right and its signature comment (`working -> admissible`) is the error; both stories adopt the correction. "Admissible" means *feasible* in optimization — satisfying the constraints, as in `constraints.is_point_feasible` — whereas this domain is about what the variables accept: an `IntegerVariable` rejects `3.4` whether or not the constraints hold. |
| Debug database under `use_database=False` | **Independent path**: it records whenever its flag is on. | Debugging a run that deliberately has no database is exactly when the working history is wanted. The debug store cannot reuse the existing database branch, which costs a little more code. |
| Export responsibility | **`to_dataset` and `to_hdf` stay on the problem.** | Ten lines delegating to `Database`; moving them would be a second breaking change in 7.0.0, touching far more user code than the algorithm-facing APIs already removed. AC#7 is unaffected. |

One consequence is worth stating separately, because it is user-visible and independent of
the naming choice: `enable_statistics` gates call counting, counting is a recording
concern, so the `ClassVar` and the `_log_settings` section headed `PreprocessedFunction`
both move to `DatabaseFunction`.

### Safe to defer

- **Debug database under parallel execution** — document the first delivery as
  single-process and lift it later.
- **Name of the debug flag** — `enable_working_database` unless someone prefers otherwise.
- **Design-space-changing steps in the first delivery** — already settled: the contract
  allows a dimension change, a guard refuses it for now.
- **Discrete and categorical step semantics** — out of scope, deferred by #1885; they only
  need a slot to plug into.

### Remaining before the canvas

Nothing. The three items listed here are closed:

1. **Agree B2 and the contract with #1886's author** — **done**, agreed offline on
   2026-09-16, covering both the ownership and the seven amendments the appendix makes to
   their text: three forced by the `random_space` rebase, four vocabulary. The interface
   is drafted in
   [Appendix: The Space-Transformation Contract](#appendix-the-space-transformation-contract).
2. **Land the two prerequisites** — **done**, each as its own commit with a regression
   test: the transitive `.original`, and the `NotImplementedError` refusing vectorization
   combined with normalization.
3. **Open the four plugin MRs** that B3 implies — **still open**: `gemseo-umdo` and
   `gemseo-mlearning` break at runtime on the 7.0.0 bump without them, `gemseo-calibration`
   and `gemseo-benchmark` only in their tests. They are part of the delivery, not
   follow-up, but they do not gate the implementation.

The four deferred points can be revisited during implementation.

## Appendix: The Space-Transformation Contract

Specified by #1886's analysis, merged by #1528 per decision B2. This appendix turns that
prose into the interface the canvas needs, and records where #1528 amends it. The package
is `src/gemseo/space/transformation/`.

### What each story delivers

| Class | Role | Lands in |
|-------|------|----------|
| `BaseSpaceTransformation` | The contract: one space-to-space map plus its vector, tangent and projection maps. | **#1528** |
| `TransformationChain` | Ordered composition, original → working; reversed for the backward; chain rule on tangents. | **#1528** |
| `NormalizationTransformation` | Wraps `Normalizer` and `IntegerRounder` (`space/_design/normalizer.py`, `space/_design/integer_rounder.py`). Carries today's behaviour. | **#1528** |
| `IdentityTransformation` | The neutral element; dropped at chain construction. | **#1528** |
| `IntegerRelaxation` | `IntegerVariable` → `ContinuousVariable` with the same bounds. The new capability. | #1886 |
| `RosenblattTransformation` | Thin wrapper over `RandomSpace.transform_vect` / `untransform_vect`, which already implement the iso-probabilistic map. | #1886 |
| `SpaceTransformationFactory` | Discovery, so plugins can contribute transformations. | #1886 |

Issue #1528 needs the contract, the chain and normalization to express "a series of
transformations" at all; the rest are #1886's capabilities and have no caller here.

### The interface

```python
class BaseSpaceTransformation(ABC):
    """One space-to-space map, with its vector, tangent and projection maps."""

    is_affine: ClassVar[bool] = False
    """Whether the map is affine, which lets `LinearFunction.fold` keep the fast path."""

    changes_dimension: ClassVar[bool] = False
    """Whether the working space has a different dimension from the original one."""

    requires_finite_bounds: ClassVar[bool] = False
    """Whether the map needs finite bounds, checked on the space this step receives."""

    @abstractmethod
    def transform_space(self, space: BaseVariableSpace) -> BaseVariableSpace:
        """Build the working space from the original one."""

    @abstractmethod
    def transform_value(
        self, x: NumberArray, out: NumberArray | None = None
    ) -> NumberArray:
        """Map a original point to the working space."""

    @abstractmethod
    def untransform_value(
        self, x: NumberArray, no_check: bool = False, out: NumberArray | None = None
    ) -> NumberArray:
        """Map a working point back to the original space.

        Bijective, and run at every evaluation.
        """

    @abstractmethod
    def transform_jacobian(
        self, jac: NumberArray, x: NumberArray | None = None
    ) -> NumberArray:
        """Map a original tangent to the working space, at the original point `x`."""

    @abstractmethod
    def untransform_jacobian(
        self, jac: NumberArray, x: NumberArray | None = None
    ) -> NumberArray:
        """Map a working tangent back to the original space, at the point `x`."""

    def project(self, x: NumberArray) -> NumberArray:
        """Project a original point onto the typed domain. Runs once, on `x_opt`.

        Defaults to the identity: only a relaxing step has anything to do here.
        """
        return x

    def create_constraints(
        self, working_space: BaseVariableSpace
    ) -> tuple[ArrayFunction, ...]:
        """Constraints the transformation imposes, called once per problem.

        Defaults to `()`. The produced constraints are tagged so they are hidden from
        the result shown to the user.
        """
        return ()
```

### What #1528 amends in #1886's text

| # | #1886 says | #1528 amends to | Why |
|---|------------|-----------------|-----|
| 1 | `transform_space(space: DesignSpace) -> DesignSpace` | `BaseVariableSpace -> BaseVariableSpace` | Written before the `random_space` merge. `EvaluationProblem` is now generic over `BaseVariableSpace`, and `RosenblattTransformation` acts on a `RandomSpace`, which is not a `DesignSpace`. |
| 2 | `project(x)  # working -> admissible domain` | `project(x)  # original -> typed` | #1886's own sequence diagram applies `project` after `untransform_vect` (`3.4` → `3`). The signature comment contradicts it; the diagram is right. |
| 3 | `transform_vect` maps "user → working" | "original → working" | "User" names the wrong thing: the object the user created carries the *typed* domain and would reject `3.4`. |
| 4 | `RosenblattTransformation` extracts the joint CDF from `parameter.py` | Wraps `RandomSpace.transform_vect` / `untransform_vect` | `ParameterSpace` was removed; the iso-probabilistic pair is already implemented on `RandomSpace` (`space/random.py`). |
| 5 | `NormalizationTransformation` wraps `Normalizer` / `IntegerRounder` | Same, at `space/_design/` | The package was flattened by the `random_space` merge. |
| 6 | `transform_vect` / `untransform_vect` / `transform_grad` / `untransform_grad` on the contract | `transform_value` / `untransform_value` / `transform_jacobian` / `untransform_jacobian` | `value` says what the argument is and matches the `input_value` / `current_value` vocabulary already used across the problem and space APIs, where `vect` only says it is an array; `jacobian` is accurate where `grad` is not, since the maps carry a Jacobian matrix, not a gradient. The rename applies to the **contract only** — the `transform_vect` / `untransform_vect` pair on `BaseVariableSpace`, `DesignSpace` and `RandomSpace` is shipped API and keeps its name, as does the `normalize_vect` / `normalize_grad` family. |
| 7 | The third domain is called **admissible** | **typed** | In optimization "admissible" means *feasible*, that is, satisfying the constraints — a sense both documents already use, including `constraints.is_point_feasible`. This domain is about what the **variables** accept: an `IntegerVariable` rejects `3.4` whether or not any constraint holds. |

Amendments 2, 3, 6 and 7 are vocabulary corrections that both stories adopt; 1, 4 and 5 are
consequences of the rebase and are not optional. Amendment 6 leaves the codebase with two
conventions side by side — `_vect` / `_grad` on the spaces, `_value` / `_jacobian` on the
transformations — which is the price of not breaking shipped space API in this story.

### Two constraints the contract must respect

- **Flatten the chain into the evaluation sequence.** `PreprocessedFunction` evaluates by
  looping over a tuple of callables (`core/function/preprocessed_function.py:246-258`).
  The chain's steps are spliced into that tuple rather than hidden behind one method that
  loops internally, so the number of Python frames per evaluation is unchanged.
- **`is_affine` guards the linear fast path.** `LinearFunction.normalize`
  (`core/function/linear_function.py:341`) folds the affine map into the coefficients, so
  linear problems never evaluate point by point. It is generalized to
  `LinearFunction.fold(transformation)` and guarded by `all(step.is_affine ...)` over the
  chain. Losing it costs `ScipyLinprog` and `ScipyMILP` an order of magnitude.

### One point where #1528 diverges from #1886 by design

Issue #1886 weighed a database in working coordinates and decided against it, because
every reader — post-processings, `OptimizationHistory.optimum`, `from_hdf`, `gemseo-benchmark` —
assumes original coordinates. #1528 keeps that decision for the *shared* database and adds
the working-coordinate store as a **second, opt-in** one (see the debug-mode requirement).
The two are compatible: #1886's objection is about the database everyone reads, and the
debug store is read by no one unless asked for.

### Relationship with the machine-learning transformers

`gemseo.machine_learning.transformer` already contains an abstraction with the same shape,
and a reviewer will notice. It is **not** reused, deliberately.

| `machine_learning.transformer` | This contract |
|---|---|
| `BaseTransformer` | `BaseSpaceTransformation` |
| `Pipeline`, itself a transformer, composing a sequence | `TransformationChain`, itself a transformation |
| `MinMaxScaler`, affine, `offset + coefficient·z` | `NormalizationTransformation` |
| `BaseDimensionReduction`, with `n_components` | a step declaring `changes_dimension` |
| `TransformerFactory` | `SpaceTransformationFactory` |
| `transform` / `inverse_transform` | `transform_value` / `untransform_value` |
| `compute_jacobian` / `compute_jacobian_inverse` | `transform_jacobian` / `untransform_jacobian` |

Five reasons the contract stands on its own:

- **Dependency direction.** `core/`, `space/` and `optimization/` import nothing from
  `machine_learning`; the dependency runs the other way. Deriving a problem-layer class
  from an ML base class inverts it.
- **Fitted from data versus derived from a space.** `BaseTransformer` has `fit(data)` and
  `is_fitted`. A `MinMaxScaler` fitted on samples uses the *sample* extrema; normalization
  uses the *declared bounds*. Same arithmetic, different meaning, and nothing in a space is
  fitted.
- **The Jacobian methods are not the same method.** `compute_jacobian(data)` returns the
  transformer's own Jacobian at a point; `transform_jacobian(jac, x)` applies the chain
  rule to a **given** function Jacobian. One can be built from the other; they do not unify.
- **Three members have no counterpart**: `transform_space`, `project` and
  `create_constraints`. An ML transformer knows nothing of bounds or variable types.
- **Shape and cost.** ML transformers work on 2D sample arrays and read their parameters
  through a dict; these maps work per point with an `out=` buffer, and the chain is
  flattened into the evaluation sequence to hold the per-evaluation frame count.

`crossed` is not an analogue of `is_affine` either: it records that `fit()` needs two data
arrays, a fitting concern rather than a property of the map.

Two consequences worth stating:

- **Reuse at the implementation level is welcome.** A future dimension-reducing step on the
  design variables can wrap `machine_learning`'s `PCA` inside its implementation. What is
  refused is inheriting the *contract*, not calling the code.
- **A naming divergence is accepted.** ML says `inverse_transform`, spaces say
  `untransform_*`. The space side has precedent — `untransform_vect` is shipped API — so
  the divergence stands, but as a known choice rather than an oversight.

After this story lands, "transformer" and "transformation" name three unrelated things in
GEMSEO: machine-learning data transformers, the MDA sequence transformers (Aitken, secant,
relaxation) under `mda/sequence_transformer/`, and these space transformations. Docstrings
should say which one they mean.
