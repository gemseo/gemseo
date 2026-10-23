<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: adoption of the `BaseVariableSpace` API

> **Historical record** — This analysis was written while `ParameterSpace` still
> existed and every uncertain space was converted into one at the entry points of
> the execution machinery. Tier A and Tier B were carried out afterwards, and
> `ParameterSpace` was then removed altogether (`464d2cdf32`), so the body below
> describes a state of the code that no longer exists: `ParameterSpace`,
> `convert_to_parameter_space` and the conversion sites inventoried here are all
> gone, and the space of an `EvaluationProblem` is now `input_space`. It is kept
> as written, being the audit trail behind those changes; the only section
> describing the current code is **Implementation status**, at the end.

## Original Business Requirement

The `random_space` branch introduced `BaseVariableSpace` and `RandomSpace` in the
new `gemseo.space` package. A `RandomSpace` is nevertheless converted into a
`ParameterSpace` at every entry point of the execution machinery. The comment at
`src/gemseo/scenario/evaluation.py:185-186` states the reason:

> The evaluation machinery consumes the DesignSpace API;
> bridge the uncertain space at entry, once and for all.

The question this analysis answers: **how much of that machinery genuinely needs
`DesignSpace`, and what would it cost to widen it to `BaseVariableSpace`?**

This is an audit: it proposes no code change. It classifies the API, locates the
consumers and sizes the work. Its main result is negative and repeated across
every layer examined: neither the **current value**, nor geometric
**normalization**, nor **membership checking**, nor the **database**, nor the
**formulations** used for sampling, nor the **drivers**, nor the **disciplines**,
nor the **sensitivity analyses**, nor the **PCE/FCE regressors**, nor
**post-processing** requires a `DesignSpace` on that path. Each is already
inactive, already generic, or structurally unreachable there, so the remaining
coupling is a handful of ungated call sites and of call sites still reading the
`ParameterSpace` shortcuts instead of the `variables` view — not a missing
abstraction.

## Domain Concept Identification

### Existing concepts (from the codebase)

| Class | Location | Base |
|---|---|---|
| `BaseVariableSpace` | `src/gemseo/space/base.py:61` | `Generic[_VariablesT, _VariablesViewT]`, ABC |
| `DesignSpace` | `src/gemseo/space/design/__init__.py:73` | `VariableAccessorsMixin`, `BaseVariableSpace` |
| `RandomSpace` | `src/gemseo/space/random/__init__.py:54` | `BaseVariableSpace` |
| `ParameterSpace` | `src/gemseo/space/parameter.py:62` | `DesignSpace`, **owning** a `RandomSpace` (`:98`) |

Supporting: `Variables` (`space/_core/variables.py:40`) and its read-only live view
`VariablesView` (`space/variables_view.py:30`), reachable from any space through
the `variables` property (`space/base.py:105`).

### Conceptual relationship that frames everything else

`ParameterSpace` is not a competitor of `RandomSpace`; it is the **composition**
"`RandomSpace` + deterministic facilities". Each random variable is registered in
the owned uncertain space *and* mirrored as a deterministic variable whose bounds
are the mathematical support of the distribution and whose current value is its
mean (`space/parameter.py:244-251`).

Consequently the bridge is not an accident to be removed wholesale. It is the
mechanism that supplies bounds and a current value to a space that has neither.
The goal is to stop paying for it where it buys nothing — not to abolish it.

The unifying abstraction already in place is the **unit-hypercube mapping**:
`transform_vect` / `untransform_vect` are abstract on `BaseVariableSpace`
(`space/base.py:359,376`), implemented geometrically by `DesignSpace`
(`design/__init__.py:637,654`) and iso-probabilistically (Rosenblatt) by
`RandomSpace` (`random/__init__.py:174,181`). `ParameterSpace` overrides them to
mix both (`parameter.py:649,654`).

## API classification

The three buckets below are the core result of the audit.

### Already generic — on `BaseVariableSpace` today

`dimension`, `variables`, `get_variables_indexes`, `get_indexed_variable_names`,
`convert_array_to_dict`, `convert_dict_to_array`, `filter`, `filter_dimensions`,
`remove_variable`, `rename_variable`, `transform_vect` / `untransform_vect`,
`get_pretty_table`, `__contains__` / `__iter__` / `__len__`.

### Generic in nature, `DesignSpace`-only today

**(a) Name / size / type metadata.** Reachable only through
`VariableAccessorsMixin` (`space/_core/accessors.py:32`), which is mixed into
`DesignSpace` alone.

The mixin is *not* the migration target. Its own docstring describes it as a set
of conveniences over the `variables` view, and it may be removed: everything it
exposes is a one-liner on the view, which every space already has. Call sites
should be rewritten against the view rather than the mixin being lifted into the
base class.

| mixin accessor | view-based equivalent |
|---|---|
| `space.variable_names` | `list(space.variables)` |
| `space.variable_sizes` | `{n: v.size for n, v in space.variables.items()}` |
| `space.variable_types` | same, with `v.type` |
| `space.name_to_indices` | `space.variables.name_to_indices` |
| `space.get_size(n)` | `space.variables[n].size` |
| `space.get_type(n)` | `str(space.variables[n].type)` |
| `DesignSpace.VARIABLE_TYPES_TO_DTYPES` | `TYPE_MAP` from `space/variable.py` |
| `space.has_integer_variables` | no view equivalent; a single call site, resolved on its own below |

`uncertainty/sensitivity/form.py:155` already uses `list(uncertain_space.variables)`
and needs no conversion at all. It is the pattern the other sites should follow.

**(b) Current value.** `has_current_value`, `get_current_value` (non-normalized
paths), `set_current_value`, `set_current_variable`, `to_complex`. Not
intrinsically bound-dependent, but the implementation is: `Value`
(`space/design/_value.py:51,87-101`) is constructed from `Bounds` and
`Normalizer`. Lifting the family would require splitting that coupling —
**which the sampling path does not require**; see the section below.

**(c) `round_vect`.** Depends on the variable *type*, not on bounds.

### Genuinely bounds-dependent — must stay on `DesignSpace`

`normalize_vect`, `denormalize_vect`, `normalize_grad`, `denormalize_grad`,
`get_lower_bound(s)` / `get_upper_bound(s)`, `set_lower_bound` /
`set_upper_bound`, `project_into_bounds`, `get_active_bounds`,
`check_membership`, `check`, `name_to_normalization_mask`,
`enable_integer_variables_normalization`, `initialize_missing_current_values`,
`extend`, `add_variables_from`, `to_hdf` / `from_hdf` / `to_file` / `from_file` /
`to_csv` / `from_csv`, `add_variable(lower_bound=, upper_bound=)`, and
`get_current_value(normalize=True)`.

## Where the conversions happen

Five sites in `src/gemseo`, plus two defensive re-conversions in the ML layer.

| # | Site | What forces it |
|---|---|---|
| 1 | `space/util.py:85` — `convert_to_parameter_space` | the canonical bridge; its docstring names the machinery |
| 2 | `scenario/evaluation.py:184-187` | open-coded, does not call the helper; `EvaluationProblem` declares `design_space: DesignSpace` |
| 3 | `uncertainty/reliability/problem.py:48-50` | open-coded; also narrows the attribute to `ParameterSpace` (`:38`) because the OpenTURNS algorithms read `.distribution` |
| 4 | `uncertainty/sensitivity/core/base.py:1003` | needed only for `.variable_names` at `:1010` |
| 5 | `uncertainty/sensitivity/sobol.py:243` and `is_form_sobol.py:173` | `.dimension`, plus persistence of the space in `dataset.misc["uncertain_space"]` (`sobol.py:291`) so the control-variate path can re-sample it (`:369`) |
| — | `machine_learning/regression/core/base_fce.py:154`, `model/pce.py:211` | `dataset.misc["input_space"]` may hold either kind; `pce` genuinely needs `.uncertain_variables` and `.distributions` |
| — | `problem/uncertainty/ishigami/ishigami_problem.py:39` | `OptimizationProblem` needs bounds and a current value |

Redundancy worth fixing regardless of the rest:

- `SobolAnalysis` converts at `sobol.py:243` and again in
  `super().compute_samples` (`core/base.py:1003`).
- `base_fce.py:151-154` recomputes the conversion on **every** property access,
  rebuilding the joint distributions each time.

No redundancy, contrary to what the shape of the code suggests: a
`ReliabilityScenario` built from a `RandomSpace` converts **once**, not twice.
`EvaluationScenario` converts first and then instantiates the problem with the
result (`scenario/evaluation.py:210`), so the guard in `ReliabilityProblem` sees
a `ParameterSpace` and does nothing. The same reasoning applies to
`ISFORMSobolAnalysis`: each entry point guards on the type, so only the outermost
one converts.

Deliberate rejections, not gaps: `MDOScenario` refuses a `RandomSpace` outright
(`scenario/mdo.py:117-119`) — optimization needs bounds and a current value.

## The already-generalized consumer: the model to follow

`BaseDOELibrary.sample_space` (`doe/core/base_doe_library.py:444-492`) is typed
`BaseVariableSpace | int` and gates the bound-aware parts behind
`isinstance(space, DesignSpace)` (`:471`), sampling through
`space.untransform_vect` (`:486`).

Every DOE algorithm needs only `.dimension` — plus `.variables` for
`diagonal_doe`, and `convert_dict_to_array` / `transform_vect` for `custom_doe`.
Yet all of them narrow `_generate_unit_samples` back to `DesignSpace`, against a
base declaration that already says `BaseVariableSpace` (`:219`). The exception is
`doe/diagonal_doe/diagonal_doe.py:58`, which is typed correctly and proves the
others can be weakened.

The public wrapper `compute_doe` (`src/gemseo/__init__.py:1506`) re-narrows to
`DesignSpace | int` although the callee already accepts a `RandomSpace`: a random
space works there today but is not advertised.

## Sampling does not need a current value

This is the main negative result of the audit: **the current value is an
optimization concept and the sampling path can do without it entirely.** It does
not have to be promoted to `BaseVariableSpace`, and `Value` does not have to be
split from `Bounds` / `Normalizer`.

Two facts make it work already:

- `Value.get(names=None, as_dict=True, normalize=False)` returns a partial —
  possibly empty — dictionary instead of raising when no variable has a value
  (`space/design/_value.py:430-442`);
- `set_current_value` treats an empty mapping as "clear every current value"
  rather than as an error (`design/__init__.py:717-718`).

So a `DesignSpace` carrying no value already flows through `EvaluationProblem`
and a DOE today; that is what the branch test *"a plain design space can be used
to compute the samples of a sensitivity analysis"* exercises.

Only four sites touch the current value on the sampling path, and none of them
needs a value to exist:

| Site | Today | Behavior without a current value |
|---|---|---|
| `core/problem/evaluation.py:170-172` — snapshot for `reset` | `get_current_value(as_dict=True)` | returns `{}` |
| `core/problem/evaluation.py:906` — `reset(design_space=True)` | `set_current_value(self.__initial_current_x)` | clearing no-op; guard on the snapshot being non-empty |
| `core/problem/evaluation.py:571-572` — fallback when `input_value is None` | reads the current value | unreachable from a driver, which always supplies the samples; keep it as an explicit error path |
| `core/algorithm/base_driver_library.py:237` — `_post_run` | `set_current_value(result)`, which fires after a DOE too | writing the "best sample" into an uncertain space is meaningless; gate it |

`formulation/core/base.py:568-573` already self-gates on `has_current_value`.
`idf.py:168` and `to_complex` (`scenario/evaluation.py:294`) sit on the
optimization and complex-step differentiation paths, out of scope.

### Minimal base surface this implies

Instead of the whole current-value family:

- `BaseVariableSpace.has_current_value -> bool`, defaulting to `False`, so the
  four sites above and `formulation/core/base.py:570` can gate uniformly;
- `BaseVariableSpace.check()` meaning "the space is not empty", with
  `DesignSpace.check` adding the current-value name consistency. The existing
  helper already splits along exactly that line
  (`space/design/_checking.py:228-246`).

No `Value` split, no distribution mean, no bounds. This removes the largest item
from Tier B and turns the open decision recorded there into a decided *no*.

### Behavior difference to accept

With `has_current_value` false for a `RandomSpace`,
`_set_default_input_values_from_design_space` becomes a no-op, whereas the
`ParameterSpace` bridge currently seeds the default input values of the top-level
disciplines with the distribution means (`space/parameter.py:249`). The
discipline defaults and the MDA initial couplings must cover that gap; it needs a
dedicated test before the gate is adopted.

## Sampling does not need normalization either

Same conclusion, same shape of evidence: geometric normalization is an
optimization concern, and it is **already switched off on the sampling path**.

- `BaseDOESettings.normalize_design_space` defaults to `False`
  (`doe/core/base_doe_settings.py:86-89`), overriding the base driver default
  (`core/algorithm/base_driver_settings.py:74-77`).
- The DOE worker evaluates with `preprocess_design_vector=False` and
  `design_vector_is_normalized=False` (`doe/core/base_doe_library.py:381-386`),
  so `_preprocess_inputs` (`core/problem/evaluation.py:554-589`) — the whole
  normalize / denormalize / `check_membership` block — is never reached from a
  DOE. It is only on the public `evaluate_functions(preprocess_design_vector=True)`
  path.
- `preprocess_functions` already disables `round_ints` when no variable is an
  integer (`core/problem/evaluation.py:660-665`), so for a float-only space
  `_preprocess_function` takes its final `else` branch (`:764-767`) and stitches
  **no** design-space method into the function or Jacobian sequences.
- The sampling itself goes through `untransform_vect`
  (`doe/core/base_doe_library.py:200`), which is the generic base contract.

What still touches normalization on that path:

| Site | Why it fires | Resolution |
|---|---|---|
| `core/function/preprocessed_function.py:158-161` | binds `denormalize_vect`, `normalize_grad` and `denormalize_grad` **unconditionally** at construction, although only `_compute_output_db_norm` (`:399`) and `_compute_jacobian_db_norm` (`:423-437`) ever read them | bind them only when `with_normalized_inputs` |
| `doe/core/base_doe_library.py:161-165` (`_pre_run`) | calls `__enable_integer_variables_normalization` and `__check_unnormalization_capability` unconditionally | apply the gate that `sample_space` already uses at `:471`; the asymmetry between the two entry points is the actual defect |
| `doe/core/base_doe_library.py:202-212` | `variable_types` plus the class attribute `DesignSpace.VARIABLE_TYPES_TO_DTYPES` | the `variables` view and `TYPE_MAP`; the block is skipped anyway when all variables share one type |
| `core/algorithm/base_driver_library.py:331` — `_check_integer_handling` | `has_integer_variables` | its only call site in the codebase; see the driver section |
| `core/problem/evaluation.py:572-587` | reachable only through the public `evaluate_functions` | gate it, or express it through `transform_vect` / `untransform_vect` |

### If normalization is explicitly requested

Setting `normalize_design_space=True` on a DOE still has a generic meaning for
the *value* path: `DesignSpace.transform_vect` / `untransform_vect` **are**
`normalize_vect` / `denormalize_vect` (`design/__init__.py:637,654`), and
`RandomSpace` provides the iso-probabilistic counterpart. Only the *gradient*
scaling has no generic equivalent — the Jacobian of the Rosenblatt transform is
not constant, unlike the affine scaling `normalize_grad` assumes — and gradients
are not part of sampling.

Gating `__check_unnormalization_capability` off for non-design spaces loses
nothing: an unbounded support, a Gaussian for instance, is precisely what the
iso-probabilistic mapping is able to handle, whereas an unbounded *design* space
cannot be denormalized.

## Nor does it need membership checking

`check_membership` has only four call sites outside the space package, and none
of them is on the sampling path:

- `core/problem/evaluation.py:581` sits inside `_preprocess_inputs`, unreachable
  from a driver as shown above, and is further gated by the class flag
  `check_bounds` (`:82`, default `True` through `util/constant.py:45` and
  overridable via `util/global_configuration.py:63`);
- `optimization/core/constraints.py:324` and
  `optimization/lagrange_multipliers.py:229` are optimization;
- `discipline/surrogate.py:167` checks the surrogate's *validity domain*, a
  `DesignSpace` synthesized from the training data
  (`machine_learning/core/model/base_supervised.py:130-135`), which has nothing
  to do with the input space of the problem.

The calls internal to `DesignSpace` are all reachable only through the current
value, so they disappear with it:

| Internal site | Reached by | Behavior without a current value |
|---|---|---|
| `design/__init__.py:733,738,743` in `set_current_value` | `_post_run` (`core/algorithm/base_driver_library.py:237`) | already covered by the current-value gate |
| `design/__init__.py:382` `__check_current_names`, via `check()` (`:291`) | `problem.check()` (`core/algorithm/base_driver_library.py:343`) | guarded by `has_current_value`, so a no-op |
| `design/__init__.py:355` in `get_active_bounds` | optimization | out of scope |

Worth recording on its own: **nothing validates that the samples belong to the
space.** `CustomDOE` maps its file or array rows through `transform_vect`
(`doe/custom_doe/custom_doe.py:146`), which performs no validation, and `_pre_run`
then calls `untransform_vect(..., no_check=True)`
(`doe/core/base_doe_library.py:200`), which skips even the unit-hypercube check.
Out-of-bounds custom samples therefore pass silently today. That is a
pre-existing gap, independent of this migration, but it should not be mistaken
for a guarantee that the move would break.

Should a generic membership check ever be wanted, it is definable:
`RandomVariable.lower_bound` and `upper_bound` derive from the mathematical
support of the distribution (`space/random/variable.py:139-147`), so membership
in a `RandomSpace` means membership in the support. Nothing on the sampling path
requires it today.

## Nor does the database need a design space

`Database` is typed on `DesignSpace` throughout (`core/problem/database.py:183,
189, 203, 212`), but what it actually reads from the space to record a sampling
run is base API:

| Site | Member | Verdict |
|---|---|---|
| `database.py:918,922` | `.dimension` | generic |
| `database.py:1127` | `for name in input_space` | generic |
| `database.py:1099-1103` | `.variables` → `.size` | generic, through the view |
| `database.py:1105` | `input_space.VARIABLE_TYPES_TO_DTYPES[variable.type]` | `TYPE_MAP` from `space/variable.py` |
| `database.py:1169` | `deepcopy` into `dataset.misc["input_space"]` | generic |

Three `DesignSpace`-specific behaviors remain, none of them on the sampling path:

- the **lazy `add_variable(DEFAULT_INPUT_NAME, size=...)`** (`:215-217`) — the one
  genuine signature clash, since `RandomSpace.add_variable` takes distribution
  settings — is guarded by `if self and not self.__input_space`: entries exist but
  the space is empty. It never fires when a real space is passed in; it exists
  only to describe the inputs of a bare `Database()`;
- **`DesignSpace.from_file`** (`:841`) is on the reload path;
- **`input_space.to_hdf`** (`core/problem/_hdf_database.py:562-564`) fires only on
  HDF export or backup, which `EvaluationScenario` leaves off by default
  (`scenario/evaluation.py:201`). It is already lossy: `ParameterSpace.to_hdf`
  drops the probabilistic information (`space/parameter.py:71-86`), so backing up
  a sampled uncertain space already writes a plain deterministic one.

The only genuine coupling is therefore `DesignSpace()` as the **empty default
placeholder** (`:203`) — and that placeholder is deterministic by construction: a
synthetic variable of known size under `DEFAULT_INPUT_NAME`, with no distribution
knowable. Keeping `DesignSpace()` as the default is the right call; what changes
is the annotation, widened to `BaseVariableSpace`, and a guard so that the lazy
branch only ever mutates the internally created default.

`database.py:1169` deserves separate attention for a different reason: it is the
channel through which the space type leaks into the ML and post-processing
layers, which is why `pce.py` and `base_fce.py` re-convert defensively. Widening
the annotation makes that leak explicit rather than creating it.

## Nor do the formulations, for the two that matter

Every scenario builds a formulation, so the formulation layer is on the sampling
path by construction. Exactly one of its space accesses is unconditional:
`BaseFormulation.__init__` (`formulation/core/base.py:121`) reads
`problem.design_space.variable_sizes`. The replacement is already written inline
a few lines below, at `base.py:484-486`.

Everything else is per-formulation:

| Formulation | Space usage | Verdict |
|---|---|---|
| **MDF** — the `EvaluationScenario` default | `_remove_couplings_from_ds`: `in` + `remove_variable` (`mdf.py:104-107`), then `_remove_unused_variables`: `variable_names` + `remove_variable` (`base.py:508-520`) | base API plus the accessor |
| **DisciplinaryOpt** | `set(...).intersection(design_space)` + `filter` (`disciplinary_opt.py:60-62`) | base API |
| **IDF** | `get_upper_bound` / `get_lower_bound` (`idf.py:221-223`) through `_get_normalization_factor`, reached from `core/function/consistency_constraint.py:69` behind `normalize_constraints` (default `True`); `get_current_value` / `set_current_variable` (`:168,175`) behind `start_at_equilibrium` (default `False`) | genuinely bounds-dependent, but setting-gated |
| **BiLevel** | `variable_names` at `:203,228,230,236,411`, `remove_variable` at `:436`, and `get_current_value` on the **sub-scenario** spaces (`:601`) | optimization by construction: the sub-scenarios are `MDOScenario`s, which reject a `RandomSpace` (`scenario/mdo.py:117-119`) |

`_set_default_input_values_from_design_space` (`base.py:568-573`) is the
current-value item already covered, self-gated on `has_current_value`.
`get_optim_variable_names`, `get_x_names_of_disc` and `_get_mask_from_datanames`
(`base.py:340,540,554`) are `variable_names` reads and nothing more.

Removing a coupling variable from an uncertain space, which is what MDF does, is
meaningful: `remove_variable` is base API and a coupling is not an uncertain
input. So MDF and DisciplinaryOpt — the two formulations that matter for
sampling — become pure base API as soon as the accessor reads are rewritten. IDF
scales its consistency constraints by the coupling bound ranges and is
bounds-dependent by design; BiLevel is optimization by construction. Neither is a
sampling formulation.

## The sensitivity analyses are the closest layer, not the hardest

Three of the five conversion sites live here, which makes the sensitivity package
look like the main obstacle. It is the opposite: these analyses consume almost
nothing but the base API.

They all sample through a DOE library driven by an `EvaluationScenario`
(`uncertainty/sensitivity/core/base.py:1006,1017`), and the DOE algorithms
generate in the unit hypercube from **`dimension` alone** — including the Sobol'
one, whose generator is
`doe_algo.generate_samples(design_space.dimension, settings)`
(`doe/openturns/openturns.py:235-241`), the `DesignSpace` annotation being
typing-only. The space's `untransform_vect` then applies the mapping, which for a
`RandomSpace` is precisely the iso-probabilistic transform. The sensitivity
analyses are therefore the most natural consumers of the unit-hypercube contract
already declared on `BaseVariableSpace`.

| Analysis | Space API used | Needs a `ParameterSpace`? |
|---|---|---|
| `CorrelationAnalysis` (`correlation.py:129`, `OT_MONTE_CARLO`) | base only, through `BaseSensitivityAnalysis.compute_samples` | no — the conversion at `core/base.py:1003` serves only `.variable_names` (`:1010`) |
| `HSICAnalysis` (`hsic.py:210`) | same | no |
| `MorrisAnalysis` (`morris.py:186`) | `.dimension` | no |
| `SobolAnalysis` | `.dimension` (`:256`), persistence in `dataset.misc` (`:291`), re-sampling on the control-variate path (`:367-369`) | no — `RandomSpace.compute_samples` exists (`space/random/__init__.py:140`); only the `-> ParameterSpace` annotation of `__read_uncertain_space` (`:297-309`) and its legacy `"parameter_space"` key need revisiting |
| `FORMAnalysis` (`form.py:155`) | `list(space.variables)`, already conversion-free | the conversion happens downstream, in `ReliabilityProblem` |
| `ISFORMSobolAnalysis` | `.variables`, `.dimension`, `.distribution.distribution` (`:275`) | see below |

### `.distribution` is reachable through the view, which is the documented API

`ParameterSpace.distribution` (`space/parameter.py:140-143`) is a pure delegation:
it returns `self.__uncertain_space.variables.distribution`. A `RandomSpace`
exposes the very same object as `space.variables.distribution`
(`space/random/variables_view.py:57`).

So `is_form_sobol.py:275`, `uncertainty/reliability/openturns/base.py:70` and
`uncertainty/reliability/openturns/form.py:82` require a `ParameterSpace` only
because they read the shortcut rather than the view.

The fix is *not* to add `distribution` and `distributions` to `RandomSpace`. The
upgrade guide states the opposite as a deliberate decision
(`docs/software/upgrading.md:42-62`): a `RandomSpace` has no bounds setters, no
current value, no normalization and no integer management, and `variables` is its
only accessor, not duplicated by the shortcuts of a `ParameterSpace`. It even
publishes the mapping to apply:

| shortcut | documented replacement |
|---|---|
| `ParameterSpace.distribution` | `space.variables.distribution` |
| `ParameterSpace.distributions[name]` | `space.variables[name].distribution` |
| `ParameterSpace.get_range(name)` / `get_support(name)` | `space.variables[name].distribution.range` / `.support` |
| `ParameterSpace.uncertain_variables` | `list(space.variables)` |

The consumers are what must change — the same conclusion already reached for the
accessor mixin, and for the same reason.

**Caveat, and it is a real constraint.** That mapping holds only when the space
is a `RandomSpace`. A `ParameterSpace` is a `DesignSpace`, so its `variables` view
is a plain `VariablesView` (`space/base.py:87`), not the `RandomVariablesView`
that carries `distribution` (`space/random/variables_view.py:57`). There is
therefore **no uniform, view-based way to read the joint distribution of a space
declared as `RandomSpace | ParameterSpace`**: the view works for the first, the
shortcut for the second.

Every consumer that accepts both kinds — `is_form_sobol.py:275`,
`uncertainty/reliability/openturns/base.py:70`,
`uncertainty/reliability/openturns/form.py:82`, `pce.py:233` and `fce.py:207` —
must therefore keep converting until this is resolved. Resolving it means
choosing between a helper in `space/util.py` and giving `ParameterSpace` a
`RandomVariablesView`-like access; both are design decisions beyond an audit.

That places the reliability layer closer than the tiering suggests: the
OpenTURNS FORM/SORM algorithms build their own `RandomVector` and
`PythonFunction` and drive the search themselves, so what they need from the
GEMSEO space is the joint distribution and the dimension. `ReliabilityProblem`
remains an `EvaluationProblem` and inherits the same gates as any other, but it
has no bounds requirement of its own.

## Tests: the thinnest layer, with the gap exactly on the anchor

Nine test files touch `RandomSpace`:

| File | What it pins |
|---|---|
| `tests/space/test_base.py`, `tests/space/test_random_space.py`, `tests/space/test_random_space_factory.py`, `tests/space/random/*` | the space itself |
| `tests/uncertainty/reliability/test_problem.py:53` | `ReliabilityProblem` accepts one |
| `tests/uncertainty/sensitivity/test_is_form_sobol.py:122-128` | `compute_samples` accepts one |
| `tests/machine_learning/regression/test_pce.py:307-312`, `tests/machine_learning/regression/test_fce.py:179-189` | `dataset.misc["input_space"]` may hold one |
| `tests/test_gemseo.py:758` | `create_uncertain_space()` returns one |
| `tests/problem/uncertainty/ishigami/`, `tests/problem/uncertainty/wing_weight/` | the benchmark spaces |

**Nothing under `tests/scenario/` or `tests/doe/` passes a `RandomSpace`.** The
bridge at `scenario/evaluation.py:184-187`, the most-cited line of this analysis,
is covered only transitively by the sensitivity and reliability tests that happen
to route through a scenario. It should be tested directly *before* being removed,
not after.

Three further consequences for Tier B.

**Docstrings that encode the conversion.** `tests/machine_learning/regression/test_pce.py:309`
states that "A RandomSpace must be converted on the fly into an equivalent
ParameterSpace". After the Tier A rewrite against the `variables` view the test
still passes, but its stated rationale is false — the kind of stale comment that
misleads the next reader.

**`__str__` in the driver logs.** `core/algorithm/base_driver_library.py:269,382`
logs `str(problem.design_space)`. A sampled uncertain space renders today as a
`ParameterSpace`; afterwards it renders through the `__str__` of
`BaseVariableSpace`, under the title derived from the class name ("Random
space"), and the driver introduces it with the same class-derived wording
("over the random space:"). Only
`tests/space/__snapshots__/test_random_space.ambr` and
`tests/post/_engine/__snapshots__/test_hessian.ambr` currently contain space
tables, so the blast radius is small, but any regeneration must run **without
`-n`**: parallel workers race on the `.ambr` files.

**The positive control exists in the other direction.**
`tests/uncertainty/sensitivity/test_correlation.py:132` pins that a plain design
space — bounds, no distributions — can be sampled. Tier B needs its mirror: a
`RandomSpace` sampled through an `EvaluationScenario` with no conversion, and an
assertion that `scenario.design_space` *is* the space that was passed.

## Examples and documentation: the target state is already written down

The branch has already documented the design this audit is measuring against, and
that changes what remains to be done.

### The upgrade guide states the rule

`docs/software/upgrading.md:42-62` is explicit: a `RandomSpace` deliberately has
no bounds setters, no current value, no normalization and no integer management,
and `variables` is its **only** accessor, not duplicated by the shortcuts of a
`ParameterSpace`. The page publishes the replacement mapping for both the
probabilistic reads and the metadata accessors.

Two consequences for this analysis:

- every recommendation above that rewrites a call site against the `variables`
  view is simply the application of a published rule, not a new proposal;
- conversely, any recommendation to *add* a shortcut to `RandomSpace` would
  contradict it. This applies to `distribution` and `distributions` as much as to
  the accessor mixin.

### The guide also already describes the behavior Tier B would change

`docs/software/upgrading.md:102-106` currently reads that `EvaluationScenario`,
`ReliabilityProblem`, `ReliabilityScenario`, `sample_disciplines()`,
`PCERegressor` and `FCERegressor` *"accept a `RandomSpace` and convert it
internally into an equivalent `ParameterSpace`"*, while
`BaseDOELibrary.sample_space` samples any `BaseVariableSpace` directly.

Deleting the bridge makes that sentence false. Since the entry is in the same
unreleased cycle, it must be **amended in place** rather than followed by a
correcting entry. The same holds for the changelog fragment: the net effect
relative to the last release is "a `RandomSpace` is sampled directly", not "the
conversion was added and then removed".

The user-visible part is worth stating plainly, because it is a type change on a
public attribute: after Tier B, `scenario.design_space` returns the `RandomSpace`
that was passed, not a `ParameterSpace` built from it. The current docstring
promising the opposite (`scenario/evaluation.py:166-172`) goes with it.

### Examples

The examples are already `RandomSpace`-first and need little:

| File | Current state |
|---|---|
| `docs/examples/howtos/uncertainty/distribution/plot_howto_define_uncertain_space.py` | teaches `create_uncertain_space()`, `add_variable(name, *settings)` and `compute_samples()`; unaffected |
| `.../plot_howto_define_parameter_space.py:42-46` | a note redirecting a purely uncertain space to `RandomSpace`; unaffected |
| `.../plot_howto_propagate_uncertainty_through_discipline.py:36-40` | same note, next to a `sample_disciplines()` call over a `ParameterSpace`; the natural place to show sampling a `RandomSpace` directly once the bridge is gone |
| `docs/examples/howtos/machine_learning/regression/plot_pce_regression.py` | builds a `ParameterSpace`; stays valid, since `PCERegressor` accepts both |

Prose pages to re-check after Tier B:
`docs/user_guide/concepts/uncertainty/uncertainty_propagation.md:33`,
`.../sensitivity_analysis.md:300`, `.../uncertainty_characterization.md:175`,
`docs/user_guide/use_cases/uncertainty_quantification.md` and
`docs/user_guide/concepts/benchmarking/uncertainty_problems.md`. They describe
which space types are accepted, which is exactly what changes.

## The benchmark problems are fixtures, and one of them exposes a defect

The use cases under `problem/` produce spaces rather than consuming them, so the
migration does not act on them. They are what it should be tested against.

**Deterministic MDO benchmarks.** `SobieskiDesignSpace`
(`problem/mdo/sobieski/standalone/design_space.py:27`), `SellarDesignSpace`
(`problem/mdo/sellar/sellar_design_space.py:42`) and `AerostructureDesignSpace`
(`problem/mdo/aerostructure/aerostructure_design_space.py:28`) are plain
`DesignSpace` subclasses carrying bounds and current values. Nothing to migrate:
they are design spaces. Their role is regression — every gate introduced in
Tier B must leave them behaving exactly as today, which the branch's own test
*"a plain design space can be used to compute the samples of a sensitivity
analysis"* already probes from the opposite direction.

**Uncertain benchmarks.** `IshigamiSpace` (`problem/uncertainty/ishigami/ishigami_space.py:26`)
and `WingWeightUncertainSpace` (`problem/uncertainty/wing_weight/uncertain_space.py:24`)
already subclass `RandomSpace`. They are the natural end-to-end fixtures: sample
them without any conversion.

**`IshigamiProblem`.** The wrap at
`problem/uncertainty/ishigami/ishigami_problem.py:38-39`,
`ParameterSpace(uncertain_space=IshigamiSpace(...))`, exists because the class is
an `OptimizationProblem`. It **stays** after the migration, since
`OptimizationProblem` re-narrows to `DesignSpace`. It is a correct use of the
composition, not a workaround to be removed.

**`ScalableDesignSpace`.** This one turns the `isinstance` finding from
theoretical into demonstrated. It subclasses `ParameterSpace`
(`problem/mdo/scalable/parametric/scalable_design_space.py:54`) but its
`add_uncertain_variables` flag defaults to `False` (`:69`), so by default it holds
**only deterministic variables**, added with explicit bounds and default values
(`:96-103`).

`doe/core/base_doe_library.py:429` early-returns on
`isinstance(design_space, ParameterSpace)` to mean "the mapping from the unit
hypercube is iso-probabilistic, so the unboundedness check does not apply". For
this space that is false: the check is skipped for a fully geometric space. It
does not fail today, because the bounds are 0 and 1 and the check would pass
anyway, but the predicate is testing the wrong property. With
`add_uncertain_variables=True` (`:105-110`) the space becomes genuinely mixed,
which is the case the early return was written for.

## The scenario layer: Tier B deletes the bridge rather than gating it

This is the anchor of the whole question, and what it does with the space is
thin.

| Site in `scenario/evaluation.py` | Today | After Tier B |
|---|---|---|
| `:158` | `design_space: DesignSpace \| RandomSpace` | `BaseVariableSpace` |
| `:184-187` | the `ParameterSpace` bridge | **deleted**; the `RandomSpace` reaches `EvaluationProblem` unchanged |
| `:166-172` | docstring promising the conversion | drops |
| `:210` | `self._evaluation_problem_class(design_space)` | unchanged |
| `:240-242` | the `design_space` property, returning a `DesignSpace` | returns a `BaseVariableSpace`; its *name* becomes wrong for an uncertain space |
| `:293` | `to_complex()` | reached only through `set_differentiation_method(COMPLEX_STEP)` (`:292`); gate it, or leave it to `DesignSpace` |
| `:440` | `reset(..., design_space=False)` | already avoids the current value, and only on the DOE branch (`:436`) |

`MDOScenario` gains importance rather than losing it. Its explicit rejection of a
random space (`scenario/mdo.py:117-121`) is today a courtesy check, since the base
class converts anyway; once the base stops converting, it becomes the sole guard
keeping an uncertain space out of optimization. It also overrides
`_evaluation_problem_class` (`:96`) and reads `variable_names` (`:266`), one view
rewrite.

`ReliabilityScenario` (`uncertainty/reliability/scenario.py:40-64`) accepts a
`RandomSpace | ParameterSpace` and forwards it unchanged. Its problem converts
too (`uncertainty/reliability/problem.py:48-50`), but only when reached directly:
through the scenario, the space it receives has already been converted, so the
conversion happens once. `ReliabilityProblem` cannot drop it, because it narrows
its attribute to `ParameterSpace` (`:38`) precisely to read `distribution`, which
a `ParameterSpace` exposes and a `RandomSpace` does not — see the caveat below.

`scenario/scenario_result/bilevel_scenario_result.py:51` reads `variable_names`
on a sub-scenario space, which belongs to an `MDOScenario`: optimization.

The layer therefore contributes one deletion, one annotation, two view rewrites
and one gate. It also makes the naming question unavoidable: a property called
`design_space` on a pure evaluation object was tolerable while every uncertain
space was silently converted into one, and stops being so once it is not.

## The optimization problem is out of scope, and the inheritance already says so

`EvaluationScenario._evaluation_problem_class` is `EvaluationProblem`
(`scenario/evaluation.py:132`); only `MDOScenario` overrides it with
`OptimizationProblem` (`scenario/mdo.py:96`). The sampling path never constructs
one.

What `OptimizationProblem` reads from the space is mostly base API already:

| Member | Site | Verdict |
|---|---|---|
| `dimension` | `optimization/problem.py:409,412,438` | generic |
| `get_variables_indexes` | `:435` | generic |
| `get_indexed_variable_names` | `:446` | generic |
| `deepcopy(design_space)` | `:392` | generic |
| iteration, for `pretty_str` | `:625` | generic |
| `add_variable(size=, value=0, upper_bound=0)` for the slack variables | `:400-405` | **bounds**, in the inequality-constraint reformulation |
| `has_current_value` and `get_current_value(normalize=True)`, to infer an output dimension | `:949-954` | **bounds** |
| `database.input_space` fed back into `OptimizationProblem(...)` | `:714-715` | HDF reload; always a plain `DesignSpace`, since `Database.from_hdf` reads one (`core/problem/database.py:841`) |
| the space forwarded to `Constraints`, `EvaluationProblem` and `OptimizationHistory` | `:143,157,162,171` | type binding |

Only three of those are genuinely design-space, and each is an optimization
feature. So `OptimizationProblem` needs no gate and no widening: it keeps
`design_space: DesignSpace` and simply **re-narrows** the attribute inherited
from a widened `EvaluationProblem`. That is exactly what `ReliabilityProblem`
already does in the other direction
(`uncertainty/reliability/problem.py:38`, `design_space: ParameterSpace`).

This is the structural payoff of the audit: the widening belongs entirely to
`EvaluationProblem`, and every specialization re-narrows to what it needs.

It also confirms a standing finding — the space is handed to `OptimizationHistory`
at `:171`, and `optimization/history.py:65,90,101` never dereferences a member of
it.

## The driver layer already splits along the right boundary

The driver hierarchy is the most generic layer of all, and its `DesignSpace`
coupling falls exactly on the DOE / optimization subclass boundary.

| Class | Space usage | Verdict |
|---|---|---|
| `BaseAlgorithmLibrary` | none | — |
| `BaseDriverLibrary` | `result.design_space = problem.design_space` (`:234`), `.dimension` (`:263,375`), `str(problem.design_space)` (`:269,382`) | generic: `__str__` is defined on the base (`space/base.py:401`) as an alias of `__repr__` and is not overridden by `RandomSpace`, the title of the table being derived from the class name |
| `BaseDriverLibrary` | `set_current_value(result)` (`:237`), `_check_integer_handling` (`:331`), `normalize_design_space` forwarded to `preprocess_functions` (`:345`), `problem.check()` (`:343`) | the four gates already listed; nothing new |
| `BaseDOELibrary` | `_pre_run:161-165`, `:200-212`, `sample_space:444-492`, the throwaway `DesignSpace()` at `:510-511`, integer normalization at `:519-547` | already covered above |
| `BaseOptimizationLibrary` | `initialize_missing_current_values()` (`:209`), `to_complex()` (`:211`) | bounds and current value, correctly placed on the optimization subclass |
| DOE algorithms | `.dimension` only, plus `.variables` for `diagonal_doe` and `convert_dict_to_array` / `transform_vect` for `custom_doe` | annotations narrowed to `DesignSpace` everywhere except `doe/diagonal_doe/diagonal_doe.py:58` |
| Optimization algorithms | the `get_value_and_bounds` funnel (`scipy_local:145`, `scipy_global:131,150`, `nlopt:393`, `scipy_linprog:113`, `scipy_milp:104`) and `project_into_bounds` (`scipy_linprog:154`) | genuinely bounds; Tier C |

The shared ancestor has only three `DesignSpace`-specific touches, all of them
already on the gate list. Everything bounds-dependent lives in
`BaseOptimizationLibrary` and below.

### This settles `has_integer_variables`

The one metadata accessor with no `variables`-view equivalent has exactly **one**
call site in the whole codebase, `core/algorithm/base_driver_library.py:291`, and
its definition is `any(variable.type == DataType.INTEGER for variable in
self.values())` on `DesignVariables` (`space/design/_variables.py:125-127`).

No base API is needed for it. Either inline that comprehension over the view at
the single site, or gate `_check_integer_handling` on the space being a
`DesignSpace` — integer variables are a design-space concept, and the check is
meaningless for a random space whose variables are all floats
(`space/random/variable.py:135-137`).

## The discipline layer is space-free by construction

Disciplines exchange input and output data dictionaries; the space belongs to the
problem and the scenario and never enters the discipline. `core/discipline/`
touches no space at all, and the whole of `discipline/` contains a single space
access:

- `SurrogateDiscipline.__check_validity_domain` (`discipline/surrogate.py:159-167`)
  calls `self.regressor.validity_domain.check_membership(...)`. That domain is
  the training-data hypercube (`machine_learning/core/model/base_supervised.py:130-135`),
  classified above as a design space by nature. It is not the input space of any
  problem.

The one heavy consumer is a discipline only in the structural sense:
`MDOScenarioAdapter` (`scenario/adapter/mdo_scenario_adapter.py:77`), which wraps
a scenario as a `ProcessDiscipline`. It reads bounds at `:229,233` — behind the
`set_bounds_before_opt` flag — and at `:424-425`, and the current value at
`:247,291,335,417,445,476,659`. It is typed `scenario: MDOScenario` (`:100,130`),
and `MDOScenario` refuses a `RandomSpace` (`scenario/mdo.py:117-119`). Same
verdict as BiLevel, which is the formulation that uses these adapters:
optimization by construction. `MDOObjectiveScenarioAdapter`
(`scenario/adapter/mdo_objective_scenario_adapter.py:46`) derives from it.

Nothing to migrate, nothing to gate.

## Machine learning splits in two, and only one half is in scope

### (a) PCE and FCE — consumers of the uncertain input space

Everything these regressors read from the space is delegation:

| Need | Site | `RandomSpace` equivalent |
|---|---|---|
| `variable_names` | `regression/model/pce.py:213` | `list(space.variables)` |
| `variable_sizes` | `regression/model/pce.py:247` | comprehension over the view |
| `distributions` | `regression/model/pce.py:233` | `space.variables[name].distribution`, per the documented mapping |
| `distribution` | `regression/model/fce.py:207` | `space.variables.distribution` |
| `convert_array_to_dict` | `regression/model/fce.py:106` | already base API |
| `uncertain_variables` | `regression/model/pce.py:216` | every variable of a `RandomSpace` is uncertain, so `list(space.variables)` |

`uncertain_variables` is the only entry that is not pure delegation, and it only
*discriminates* inside a mixed `ParameterSpace` — where the space already is one
and `convert_to_parameter_space` returns it untouched (`space/util.py:104-106`).

Independently of this migration, `BaseFCERegressor._input_space`
(`regression/core/base_fce.py:151-154`) is a plain property, so every read
rebuilds a `ParameterSpace`, deep-replaying all variables and rebuilding their
joint distributions (`space/parameter.py:112-122`). One of its two readers is
`_compute_sobol_indices` (`regression/model/fce.py:106`).

### (b) Validity, optimization and calibration domains — out of scope

These are `DesignSpace` by nature and are not input spaces at all:

- `machine_learning/core/model/base_supervised.py:130-135` — `validity_domain`,
  the hypercube derived from the bounds of the training data;
- `regression/model/ot_gpr_settings.py:100` with
  `regression/model/ot_gpr.py:151-153` — the hyperparameter search domain;
- `machine_learning/calibration.py:192,217`,
  `machine_learning/selection.py:148`,
  `regression/model/moe.py:283,309,335` — spaces a DOE or an optimizer runs over.

A search domain or a validity domain is a design space; nothing to migrate and
nothing to gate.

The root cause on side (a) is the one already identified for post-processing:
`dataset.misc["input_space"]` (`core/problem/database.py:1169`) carries a space of
unknown type, so the consumer converts defensively.

## Post-processing is structurally out of the sampling path

The bounds-heavy `post` package looks like the largest remaining obstacle, but no
sampling entry point can reach it:

- `post_process` is defined **only on `MDOScenario`** (`scenario/mdo.py:309`);
  `EvaluationScenario` has no such method;
- `BasePost.__init__` accepts an `OptimizationProblem` or an
  `OptimizationDataset` and raises for anything else
  (`post/core/base_post.py:98-120`). An `EvaluationProblem` is neither, and
  `Database.to_dataset` produces a plain `Dataset` by default
  (`core/problem/database.py:1067`);
- the family that *is* meant for sampling output, `post/dataset` — the
  `DatasetPlot` plots — never touches a space at all.

The residual risk is the dataset channel rather than the scenario: a
post-processor can be constructed directly from a dataset, and
`dataset.misc["input_space"]` (`core/problem/database.py:1169`) carries whatever
space the problem held. Today a sampled uncertain space arrives there as a
`ParameterSpace` and satisfies every bounds call; after the migration it would
arrive as a `RandomSpace`, and six modules would fail:

| Module | Member | Line |
|---|---|---|
| `post/opt_history_view.py` | `normalize_vect`, `get_lower_bounds`, `get_upper_bounds` | `:106,237-238,374` |
| `post/robustness.py` | `get_upper_bounds() - get_lower_bounds()` | `:60` |
| `post/quad_approx.py` | `get_lower_bounds`, `get_upper_bounds` | `:201-203` |
| `post/parallel_coordinates.py` | `normalize_vect` | `:81` |
| `post/gradient_sensitivity.py` | `denormalize_vect` | `:145` |
| `post/_engine/hessian.py` | `normalize_vect`, `normalize_grad` | `:297-298` |

The rest of the package is already space-generic:
`scatter_plot_matrix.py:63,102`, `pareto_front.py:75,202`,
`variable_influence.py:186`, `core/base_post.py:276`, `hessian_history.py:86`
(`get_variables_indexes`, base API), `quad_approx.py:112,145` (`dimension`) and
`parallel_coordinates.py:77` (`variables.items()`).

This is the same channel that makes `pce.py:211` and `base_fce.py:154` re-convert
defensively, so the consumer-side conversion pattern is already established and
working. Whether the bounds-based post-processors should convert likewise, or
simply fail loudly on a space that has no bounds, is a Tier C question — they are
optimization tools by construction.

## Remaining hard blockers for a generic `EvaluationProblem`

| Blocker | Site | Nature |
|---|---|---|
| HDF round-trip of the input space | `core/problem/database.py:841`, `core/problem/_hdf_database.py:562-564` | I/O; opt-in, and already lossy for uncertain spaces |

## Strategic Approach

### Distance, in tiers

**Tier A — cheap, no semantic change.** Rewrite the metadata call sites against
the `variables` view (table above); widen the `_generate_unit_samples` overrides
and `compute_doe` to `BaseVariableSpace`; drop the conversions in
`sensitivity/core/base.py:1003`, `sobol.py:243` and `is_form_sobol.py:173`; route
`scenario/evaluation.py` and `reliability/problem.py` through
`convert_to_parameter_space` so the double conversion disappears. This removes
most of the conversion sites without touching any semantics.

Note for `sobol.py`: what is persisted in `dataset.misc["uncertain_space"]` must
stay re-samplable. A `RandomSpace` has `compute_samples`
(`random/__init__.py:140`), so storing it unconverted is viable, but the legacy
`"parameter_space"` key and the `-> ParameterSpace` return type of
`__read_uncertain_space` (`sobol.py:297-309`) must be revisited together.

Tier A stops short of the distribution reads. Rewriting them against the
`variables` view is the documented direction
(`docs/software/upgrading.md:42-62`), and adding `distribution` /
`distributions` to `RandomSpace` would contradict it — but neither is applicable
to a consumer typed `RandomSpace | ParameterSpace`, for the reason given in the
caveat above. `is_form_sobol.py:173`, `pce.py:211` and `base_fce.py:154`
therefore keep converting, and the choice of a uniform accessor is left open.

**Tier B — the real work.** Type `EvaluationProblem` and `Database` on
`BaseVariableSpace` so a `RandomSpace` flows end-to-end for *sampling*, letting
`OptimizationProblem` re-narrow the attribute to `DesignSpace` the way
`ReliabilityProblem` narrows it to `ParameterSpace`:

- keep the current value a `DesignSpace` concept and add only
  `has_current_value` (default `False`) and a non-empty `check()` to the base
  class, gating the four call sites listed above;
- keep geometric normalization a `DesignSpace` concept too, and close the five
  ungated sites listed above — chiefly the unconditional binding in
  `PreprocessedFunction.__init__` and the `_pre_run` / `sample_space` asymmetry
  in `BaseDOELibrary`;
- widen the `Database` annotations to `BaseVariableSpace`, keep `DesignSpace()`
  as the empty placeholder, guard its lazy `add_variable` branch, and import
  `TYPE_MAP` directly instead of going through `DesignSpace`;
- rewrite the single unconditional formulation access,
  `BaseFormulation.__init__` (`formulation/core/base.py:121`), leaving IDF and
  BiLevel typed on `DesignSpace`;
- resolve `has_integer_variables` at its single call site
  (`core/algorithm/base_driver_library.py:291`), without adding it to the base
  class;
- delete the bridge at `scenario/evaluation.py:184-187` — the point of the whole
  exercise — and decide what the `design_space` parameter and property should be
  called on a scenario that only evaluates;
- amend `docs/software/upgrading.md:102-106` in place, since it currently
  documents the conversion being removed, and write the changelog fragment
  against the last release rather than against this branch's own history.

**Tier C — not a target.** Optimization, gradient normalization, the bounds-based
post-processors and HDF persistence of distributions. `ParameterSpace` already is
the composition that supplies those facilities, `MDOScenario` already refuses a
random space on purpose, and `BasePost` is reachable only from an
`OptimizationProblem` or an `OptimizationDataset`. The one thing Tier C must
decide is what a bounds-based post-processor should do when
`dataset.misc["input_space"]` holds a space without bounds: convert, as the ML
layer already does, or fail loudly.

### Headline answer

Sampling-side GEMSEO is **one cheap tier away** — Tier A is type-hint widening
plus a handful of view-based rewrites. Evaluation-side (Tier B) is markedly
smaller than it first looks: none of the current value, normalization, membership
checking, the database, the sampling formulations, the sensitivity analyses, the
PCE/FCE regressors or post-processing requires a `DesignSpace` on that path, so
what remains is a set
of gates at known call sites, a few annotation widenings and the opt-in HDF
plumbing. No new abstraction and no split of `Value`, `Bounds` or `Normalizer` is
required. Optimization-side is not a target at all.

## Risk & Gap Analysis

### Technical risks

- **Gradients under an iso-probabilistic mapping.** `PreprocessedFunction`
  (`core/function/preprocessed_function.py:158-161`) binds `denormalize_vect`,
  `normalize_grad` and `denormalize_grad` as fixed scalings. The Jacobian of the
  Rosenblatt transform is not constant, so the differentiated path cannot simply
  be re-expressed through `transform_vect`. Tier B must keep it gated, not
  generalized.
- **Pickle / HDF back-compatibility.** `DesignSpace.__setstate__`
  (`design/__init__.py:113`) and `ParameterSpace.__setstate__` (`parameter.py:151`)
  already carry pre-refactor compatibility code; moving state between classes
  compounds it.

### Smaller findings

- `optimization/history.py:65,90,101` stores a `DesignSpace` and never
  dereferences a member — dead binding.
- `doe/core/base_doe_library.py:429` uses `isinstance(design_space, ParameterSpace)`
  as a proxy for "the mapping is iso-probabilistic". Unreliable in both
  directions, and demonstrably so: `ScalableDesignSpace`
  (`problem/mdo/scalable/parametric/scalable_design_space.py:54`) subclasses
  `ParameterSpace` yet holds only deterministic variables by default, so the
  unboundedness check is skipped for a fully geometric space; conversely a bare
  `RandomSpace` would fall through to `name_to_normalization_mask` and fail, and
  only never reaches that code because of the conversion upstream.
- `doe/core/base_doe_library.py:420` — `__check_unnormalization_capability` has an
  untyped `design_space` parameter.
- `doe/core/base_doe_library.py:510-511` synthesizes a throwaway `DesignSpace`
  just to obtain an identity unit-hypercube mapping.
- `util/hdf5.py:88` guards `isinstance(value, Mapping) and not isinstance(value, DesignSpace)`.
  `BaseVariableSpace` is not a `Mapping` (it exposes only `__iter__`, `__len__`
  and `__contains__`), so the guard looks vestigial from the era when
  `DesignSpace` was a `MutableMapping`; if it ever fires again it would not cover
  `RandomSpace`.
- Naming drift: `scenario/evaluation.py:158,240` names its parameter and property
  `design_space` on a pure evaluation object, and
  `uncertainty/reliability/scenario.py:55` does the same for an uncertain space.
  Tolerable while every uncertain space was converted on entry; a defect once it
  is not.
- `formulation/core/base.py:568-573` reads `has_current_value` and
  `get_current_value(as_dict=True)` on every scenario, including evaluation ones —
  a second consumer of the current-value decision in Tier B.

## Test plan implied by the analysis

This document proposes no code change and therefore needs no test run and no
changelog fragment of its own. Every `file:line` it cites was verified against
the branch at the time of writing; note that the tree was restructured here
(`gemseo/algos/*` → `gemseo/core/problem/*`, `gemseo/optimization/*`,
`gemseo/space/*`), so line numbers taken from earlier branches do not apply. The
two central claims stay re-checkable by inspection:

```text
grep -n "isinstance(space, DesignSpace)" src/gemseo/doe/core/base_doe_library.py
grep -rn "convert_to_parameter_space\|ParameterSpace(uncertain_space=" src/gemseo
```

Should Tier A or Tier B be undertaken, the test work it implies is:

1. add a direct test of `EvaluationScenario` with a `RandomSpace` **before**
   touching `scenario/evaluation.py:184-187`, which has no direct coverage today;
2. mirror `tests/uncertainty/sensitivity/test_correlation.py:132` with a
   `RandomSpace` sampled end to end, asserting that `scenario.design_space` is
   the space that was passed rather than a conversion of it;
3. keep the deterministic benchmarks — `SobieskiDesignSpace`,
   `SellarDesignSpace`, `AerostructureDesignSpace` — as unchanged regression
   fixtures across every gate that is added;
4. cover `ScalableDesignSpace` with `add_uncertain_variables` both `False` and
   `True`, the case that defeats the `isinstance(..., ParameterSpace)` predicate
   at `doe/core/base_doe_library.py:429`;
5. update the docstring of `tests/machine_learning/regression/test_pce.py:309`,
   which asserts in prose a conversion that will no longer happen;
6. regenerate any affected `.ambr` snapshot **without `-n`**, since parallel
   workers race on those files.

## Implementation status

Tier A and Tier B were carried out after this analysis was written. What the
implementation confirmed, and where it departed from the plan:

- **Tier A** dropped the conversions of `BaseSensitivityAnalysis.compute_samples`
  and `SobolAnalysis.compute_samples`, widened the DOE algorithms and
  `compute_doe` to `BaseVariableSpace`, and routed the two open-coded bridges
  through `convert_to_parameter_space`.
- **Tier A did not** rewrite the distribution reads against the `variables` view,
  for the reason given in the caveat above: the view of a `ParameterSpace` has no
  `distribution`. `ISFORMSobolAnalysis`, `PCERegressor` and `FCERegressor` keep
  converting.
- **Tier B** added `has_current_value` and `check` to `BaseVariableSpace`, gated
  the current-value, normalization and integer-rounding paths, widened
  `EvaluationProblem`, `Database` and `EvaluationScenario`, and deleted the
  bridge at `scenario/evaluation.py`.
- **One blocker materialized that the analysis had classified as opt-in**: the
  HDF export of the input space (`core/problem/_hdf_database.py`) is reached by
  the backup path of a sensitivity analysis, not only by an explicit `to_hdf`.
  It now logs a warning and omits the space when that space is not a
  `DesignSpace`.
- The gate on the initial current value uses `isinstance(..., DesignSpace)`
  rather than `has_current_value`: the latter is all-or-nothing, and a design
  space with a *partial* current value must still have it snapshotted and
  restored by `reset`.
- **The naming drift was settled** once the bridge was gone: the space of an
  `EvaluationProblem` and of an `EvaluationScenario` is now `input_space`, the
  name `Database` already used, and `OptimizationProblem` and `MDOScenario` keep
  a narrowed `design_space` property. `BaseFormulation.design_space` was renamed
  `input_space` as well (`formulation/core/base.py:200`), and so was the space of
  a reliability analysis, named `uncertain_space` there.

### After the removal of `ParameterSpace`

The analysis assumed `ParameterSpace` would stay, as the composition supplying
bounds and a current value to an uncertain space; Tier C rested on that. It was
removed instead, which changes three of the conclusions above:

- **The behavior difference was not accepted.** The analysis proposed to let
  `_set_default_input_values_from_space` become a no-op for a space without a
  current value, and to cover the gap with the discipline defaults. That loses
  two properties of the bridge: a default input value had to be set discipline by
  discipline, and it no longer followed a change of the probability
  distributions. `BaseVariableSpace.reference_value` restores both — the
  current value of a `DesignSpace`, the means of the distributions of a
  `RandomSpace` — and the formulation reads it instead of gating on
  `has_current_value`. The means are read from the distributions on each call, so
  they now follow their settings, which the frozen current value of a
  `ParameterSpace` did not.
- **The distribution reads were rewritten after all.** The caveat that blocked
  them — the view of a `ParameterSpace` has no `distribution` — died with the
  class. `ISFORMSobolAnalysis`, `PCERegressor` and `FCERegressor` now require a
  `RandomSpace` and read its `variables` registry.
- **The old names are mapped rather than left to fail.** `bump-version.yml`
  points `gemseo.algos.parameter_space` at `gemseo.space.random`, renames
  `ParameterSpace` to `RandomSpace`, and records the members that did not survive.

Of the smaller findings listed above, the dead `DesignSpace` binding of
`optimization/history.py` and the vestigial guard of `util/hdf5.py` were removed,
and `has_integer_variables` was resolved by exposing `has_integer_variable` on
the `variables` view, which a consumer typed `BaseVariableSpace` can reach.
`sample_unit_hypercube` still builds a throwaway `DesignSpace`: `diagonal_doe`
reads the individual variables of the space, so `_generate_unit_samples` cannot
be narrowed to a dimension.
