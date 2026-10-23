---
description: "Migrate your code from GEMSEO 6 to GEMSEO 7: run bump-gemseo, migrate what it leaves, check the silent changes."
search:
  boost: 2
---

<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# From GEMSEO 6 to GEMSEO 7 { #upgrading-gemseo }

GEMSEO 7 changes the API in depth: the packages are reorganized by domain, objects are configured
with Pydantic settings models instead of keyword arguments, and random variables have their own
space, `RandomSpace` (see [What's new in GEMSEO 7](../../whats_new_7.md)). Most of these changes are
mechanical: a module moved, a class or an argument was renamed, keyword arguments became a settings
model. A tool written by the GEMSEO developers, bump-gemseo, makes these changes in your code for you.

This guide takes you through the migration in four steps, then documents, domain by domain, the
changes that bump-gemseo cannot make on its own, each with the code before and after. A coding agent
can do most of the work that bump-gemseo leaves to you, see
[Let a coding agent do the manual work](#coding-agent).

!!! info "How much bump-gemseo does on its own"

    The changes of GEMSEO 7 were turned into 2109 small programs, one or a few per change, each
    written for GEMSEO 6.3 and executed before and after migration. bump-gemseo alone migrates
    1807 of them correctly (85.7 %). The 302 others are what this guide is about:
    169 are explained with an example, the rest are names removed without replacement,
    listed in the sections [*Removed names*](#sections-removed-names) of the domain pages.

## The compatibility layer of GEMSEO 7 { #compatibility-layer }

With GEMSEO 7, your code can still use most of the GEMSEO 6.3 names: the old import paths of the
modules, and the old names of the classes, functions, attributes and enumeration members. When your
code runs, GEMSEO 7 migrates each old name to its new one and emits a `DeprecationWarning` giving the
new name to write in your code:

```python
from gemseo.algos.design_space import DesignSpace
# DeprecationWarning: The module 'gemseo.algos.design_space' is deprecated; use 'gemseo.space.design' instead.
# DeprecationWarning: The attribute 'DesignSpace' of the module 'gemseo.algos.design_space' is deprecated; use 'gemseo.space.DesignSpace' instead.
```

The old names work until GEMSEO 8, which will remove them: this period is the *deprecation cycle*.
During it, an old name that you have not migrated yet does not break your code, it only emits a
warning.

The compatibility layer migrates names only, not the way your code calls GEMSEO: a renamed or removed
argument still raises an error. So it does not replace bump-gemseo, which migrates both in your code.

### What it does

- **Moved and renamed modules**: a GEMSEO 6.3 import path imports the GEMSEO 7 module, e.g.
  `gemseo.algos.design_space` imports `gemseo.space.design`.
- **Renamed classes, functions and constants** can still be imported under their old names from
  their GEMSEO 6.3 import paths, e.g. `SciPyMILP_Settings` from `gemseo.algos.opt.scipy_milp` gives
  `gemseo.optimization.MILP_Settings`.
- **Renamed class attributes, methods and enumeration members**: `Class.OLD` and `instance.OLD`
  are migrated to the new attribute, e.g. `cache.last_entry` gives `cache.last_item` and
  `DatasetClassName.Dataset` gives `DatasetClassName.DATASET`. If one of your subclasses still
  defines or overrides an old name, its definition is moved to the new name.
- **The package `gemseo.settings`, split into several packages**: importing it warns and lists the
  packages that now hold its settings.
- **Names moved to a plugin**, e.g. the import path `gemseo.disciplines.wrappers.xls_discipline`,
  now `gemseo_excel.xls_discipline`: when the plugin is not installed, the error names the plugin to
  install.
- **Removed names**: importing a name removed without replacement raises an error that says so.
  For the few names that cannot be migrated automatically, the error says how to migrate, e.g.
  `use logging.getLogger("gemseo.space.random") instead` for `gemseo.algos.parameter_space.LOGGER`.

### What it does not do

GEMSEO 6.3 code that relies on any of the following fails with GEMSEO 7, even with the layer:

- **renamed or removed arguments**, e.g. `MDAGaussSeidel(disciplines, tolerance=1e-10)` raises
  `TypeError: MDAGaussSeidel.__init__() got an unexpected keyword argument 'tolerance'`;
- **keyword arguments that became a settings model**, e.g.
  `scenario.execute(algo_name="SLSQP", max_iter=30)`;
- **methods removed in favour of another API**, e.g. `design_space.get_lower_bound("x")`;
- **names given as strings**, e.g. the algorithm name `"Scipy_MILP"`, now `"MILP"`;
- **changes of behaviour**: when GEMSEO 7 keeps a name but changes what it does, the compatibility
  layer does not restore the GEMSEO 6.3 behaviour (see [Step 4](#step-4-check-the-silent-changes)).

### Who does what

- **The compatibility layer** keeps the old names working until GEMSEO 8.
- **bump-gemseo** renames them in your code, which removes the warnings. It also rewrites the calls
  that the layer cannot keep working: arguments, settings models, removed methods with a
  replacement ([Step 1](#step-1-run-bump-gemseo)).
- **You** handle the rest:
    - the `# TODO(bump-gemseo)` comments
      ([Step 2](#step-2-migrate-the-code-flagged-by-the-todo-comments));
    - the errors that come with no comment ([Step 3](#step-3-switch-to-gemseo-7-and-run-your-code));
    - the silent changes ([Step 4](#step-4-check-the-silent-changes));
    - the old names that bump-gemseo did not see. They still work with a warning; migrate them before
      GEMSEO 8 ([Step 3](#step-3-switch-to-gemseo-7-and-run-your-code)).
- **A coding agent** can do most of your part, see
  [Let a coding agent do the manual work](#coding-agent).

### Limits

- **Patching an old import path does not reach GEMSEO.**
  `mock.patch("gemseo.utils.string_tools.MultiLineString")` replaces the name under its old import
  path only: GEMSEO imports it from its new import path, `gemseo.util.string`, and keeps using the
  original. Patch the new import path. bump-gemseo migrates the `mock.patch` targets written as
  strings.
- **Assigning an old class attribute after the class is defined** does not change the new attribute,
  e.g. `Class.OLD = value` or `monkeypatch.setattr(Class, "OLD", ...)`: GEMSEO keeps reading the new
  attribute, unchanged. Assign the new name.

## Before you start

1. **Upgrade to GEMSEO 6.3 first.** bump-gemseo migrates code written for GEMSEO 6.3 only. If your
   code uses an older version, follow the [upgrading guides](../index.md) from your version up to
   [From GEMSEO 5 to GEMSEO 6](../5_to_6.md), then the changelog up to 6.3, and make sure your code runs with
   GEMSEO 6.3 without deprecation warnings.
2. **Make sure your tests pass with GEMSEO 6.3, then commit.** bump-gemseo rewrites your files in
   place and has no dry-run mode: a clean working tree lets you review its changes with `git diff`
   and undo them.
3. **Keep the results of your studies.** Some changes are *silent*: the code runs with GEMSEO 7
   without any error, but gives other results. Tests comparing values, or the outputs of a reference
   run saved with GEMSEO 6.3, are the only way to see them.

## Step 1: run bump-gemseo

Install bump-gemseo in the environment where GEMSEO 6.3 is installed, then run it on the directories
of your project at once:

```shell
bump-gemseo <path/to/directory> ...
```

e.g. `bump-gemseo src tests docs` for a package in `src`, its tests and its documentation.

[Running bump-gemseo](bump_gemseo.md) shows how to install it, what it changes, how to read its report
and what it cannot do, on a complete example.

## Step 2: migrate the code flagged by the TODO comments

Where bump-gemseo cannot make a change, it leaves a comment above the line,

```python
# TODO(bump-gemseo): <what to do>  # noqa: E501
```

and lists these comments in its report. Search for the text of a comment in this guide (with the
search box, or in the sections [*Changes flagged by bump-gemseo*](#sections-flagged) of the domain
pages): the entry explains the change and shows the code to write. A comment about a removed name that
no entry explains refers to the section [*Removed names*](#sections-removed-names) of the domain page;
other comments with no example are listed in the sections
[*Other comments of bump-gemseo*](#sections-other-comments).

## Step 3: switch to GEMSEO 7 and run your code

Install GEMSEO 7 (see [Installation](../../installation.md)) and run your tests and scripts. An error
that no comment announced is in a section
[*Changes that fail without a comment*](#sections-fail-without-comment), which quotes the error
message: search for it.

Old names that bump-gemseo did not rename, such as names on an object whose type it could not infer,
or names built at run time, still work through the [compatibility layer](#compatibility-layer), with a
`DeprecationWarning`, and will fail with GEMSEO 8. Run your tests once with
`python -W error::DeprecationWarning` to find them, then migrate them.

## Step 4: check the silent changes

These changes do not raise any error and bump-gemseo writes no comment about them. Check the ones that
concern your code, and compare your results with those of GEMSEO 6.3:

- [Core](core.md#silent-changes):
    - An `HDF5Cache` file of GEMSEO 6.3 is rewritten when GEMSEO 7 opens it
    - Parallel executions with `n_processes=1` run in the calling process
    - `split_namespace` splits on the last separator only
- [Disciplines](discipline.md#silent-changes):
    - `execute` returns the output data only
- [Spaces](space.md#silent-changes):
    - `get_current_value` returns copies
    - `project_into_bounds` may return a float array
    - The bounds of a random variable no longer bound the finite differences
    - Random variables are of type `VariableType.RANDOM`
    - The design space copies and casts the values it stores
- [Problems and functions](problem.md#silent-changes):
    - A `Database` over a `RandomSpace` is written to HDF without its space
    - A `NaN` output is recorded in the database before the run stops
- [Optimization](optimization.md#silent-changes):
    - The augmented Lagrangian leaves `normalize_design_space` to its sub-algorithm
    - `MultiStart` skips the starting points whose optimization fails
- [DOE](doe.md#silent-changes):
    - The unit hypercube variable is no longer named `"x"`
- [Datasets](dataset.md#silent-changes):
    - Dataset variable and group names keep insertion order
    - `IODataset` input and output names keep insertion order
    - The names of an `OptimizationDataset` keep insertion order
- [Machine learning](machine_learning.md#silent-changes):
    - RBF models invert `epsilon` and add a polynomial term
- [Utilities](util.md#silent-changes):
    - `epsilon` is a builtin `float` and no longer upcasts `float32` arrays
    - `get_sub_disciplines` replaces any object with a `disciplines` attribute by its disciplines

## Let a coding agent do the manual work { #coding-agent }

The changes that bump-gemseo leaves to you, steps
[2](#step-2-migrate-the-code-flagged-by-the-todo-comments) and
[3](#step-3-switch-to-gemseo-7-and-run-your-code), are a good job for a coding agent. The
GEMSEO developers tested coding agents on programs written for GEMSEO 6.3 that cover the breaking
changes of GEMSEO 7, and compared the values that the migrated programs compute with those of
GEMSEO 6.3. Claude Opus 5.5, Claude Sonnet 5.5 and the open-weight models GLM 5.3 and Qwen 3.8 27B
all migrated these programs successfully: each reproduced more than 99 % of the checked behaviours of
GEMSEO 6.3. An open-weight model such as Qwen 3.8 27B can run on your own infrastructure, which helps
when your code must not leave it.

In these tests, the agent had nothing that you do not have: bump-gemseo, the GEMSEO 7 changelog and
a GEMSEO 6.3 environment to run the original programs. To migrate your code with an agent:

1. Run bump-gemseo yourself ([Step 1](#step-1-run-bump-gemseo)) and commit its result, so that you
   review the changes of the agent on their own.
2. Give the agent GEMSEO 7, the GEMSEO 6.3 environment where your tests pass (see
   [Before you start](#before-you-start)) and your tests: it can then run the original code and the
   migrated code, and compare their results.
3. Point it to this guide and to the [changelog](../../../changelog/index.md). Ask it to migrate the code
   flagged by the `# TODO(bump-gemseo)` comments, make your tests pass with GEMSEO 7, and leave no
   `DeprecationWarning` (see [the compatibility layer](#compatibility-layer)).
4. Review its changes with `git diff`, as you would review those of a colleague.

No model handles reliably the silent changes of [Step 4](#step-4-check-the-silent-changes), which
raise no error and get no comment, or the few changes that need a design decision, such as the
methods that a custom formulation overrides. Check these yourself.

## The guide, domain by domain

The domains are those of the [GEMSEO issue tracker](https://gitlab.com/gemseo/dev/gemseo/-/issues).
Each domain page groups the changes that bump-gemseo leaves to you in these sections:

- *Silent changes*{ #sections-silent-changes }: the code runs after bump-gemseo but gives other
  results. [Step 4](#step-4-check-the-silent-changes) lists them, page by page.
- *Changes that fail without a comment*{ #sections-fail-without-comment }: the code fails with
  GEMSEO 7 and bump-gemseo wrote no comment. In
  [Core](core.md#changes-that-fail-without-a-comment),
  [Disciplines](discipline.md#changes-that-fail-without-a-comment),
  [Spaces](space.md#changes-that-fail-without-a-comment),
  [Problems and functions](problem.md#changes-that-fail-without-a-comment),
  [Optimization](optimization.md#changes-that-fail-without-a-comment),
  [DOE](doe.md#changes-that-fail-without-a-comment),
  [Formulations](formulation.md#changes-that-fail-without-a-comment),
  [MDA](mda.md#changes-that-fail-without-a-comment),
  [Scenarios](scenario.md#changes-that-fail-without-a-comment),
  [Post-processing](post.md#changes-that-fail-without-a-comment),
  [Machine learning](machine_learning.md#changes-that-fail-without-a-comment),
  [Uncertainty](uncertainty.md#changes-that-fail-without-a-comment) and
  [Utilities](util.md#changes-that-fail-without-a-comment).
- *Changes flagged by bump-gemseo*{ #sections-flagged }: bump-gemseo wrote a `# TODO(bump-gemseo)`
  comment. In
  [Core](core.md#changes-flagged-by-bump-gemseo),
  [Disciplines](discipline.md#changes-flagged-by-bump-gemseo),
  [Spaces](space.md#changes-flagged-by-bump-gemseo),
  [Problems and functions](problem.md#changes-flagged-by-bump-gemseo),
  [Optimization](optimization.md#changes-flagged-by-bump-gemseo),
  [DOE](doe.md#changes-flagged-by-bump-gemseo),
  [Formulations](formulation.md#changes-flagged-by-bump-gemseo),
  [MDA](mda.md#changes-flagged-by-bump-gemseo),
  [Scenarios](scenario.md#changes-flagged-by-bump-gemseo),
  [Datasets](dataset.md#changes-flagged-by-bump-gemseo),
  [Post-processing](post.md#changes-flagged-by-bump-gemseo),
  [Machine learning](machine_learning.md#changes-flagged-by-bump-gemseo),
  [Uncertainty](uncertainty.md#changes-flagged-by-bump-gemseo) and
  [Utilities](util.md#changes-flagged-by-bump-gemseo).
- *Other comments of bump-gemseo*{ #sections-other-comments }: the comments with no example. In
  [Optimization](optimization.md#other-comments-of-bump-gemseo),
  [Formulations](formulation.md#other-comments-of-bump-gemseo),
  [Scenarios](scenario.md#other-comments-of-bump-gemseo) and
  [Utilities](util.md#other-comments-of-bump-gemseo).
- *Deprecated names*{ #sections-deprecated-names }: the code works, with a deprecation warning:
  migrate what it names. In [Core](core.md#deprecated-names).
- *Removed names*{ #sections-removed-names }: names removed without replacement. In
  [Core](core.md#removed-names),
  [Disciplines](discipline.md#removed-names),
  [Spaces](space.md#removed-names),
  [Problems and functions](problem.md#removed-names),
  [Optimization](optimization.md#removed-names),
  [Formulations](formulation.md#removed-names),
  [MDA](mda.md#removed-names),
  [Scenarios](scenario.md#removed-names),
  [Post-processing](post.md#removed-names),
  [Machine learning](machine_learning.md#removed-names),
  [Uncertainty](uncertainty.md#removed-names) and
  [Utilities](util.md#removed-names).

The domains:

- [Core](core.md): grammars, caches, namespaces, derivatives, execution and monitoring.
- [Disciplines](discipline.md): disciplines, chains and wrappers.
- [Spaces](space.md): design spaces and random spaces.
- [Problems and functions](problem.md): evaluation and optimization problems, functions, databases,
  benchmark problems.
- [Optimization](optimization.md): optimization libraries and their settings.
- [DOE](doe.md): design of experiments libraries and their settings.
- [Formulations](formulation.md): MDO formulations and custom formulations.
- [MDA](mda.md): multidisciplinary analyses and coupled derivatives.
- [Scenarios](scenario.md): scenarios, scenario adapters and their results.
- [Datasets](dataset.md): datasets and their metadata.
- [Post-processing](post.md): post-processors and dataset plots.
- [Machine learning](machine_learning.md): regressors, classifiers, clustering and their quality.
- [Uncertainty](uncertainty.md): distributions, statistics and sensitivity analysis.
- [Utilities](util.md): configuration, logging, testing helpers and other utilities.

## How to read an entry

Each entry has a title saying what changed and an explanation of the change and of how to migrate.
Then come the comment bump-gemseo writes, if any, the error you get if you leave the code as
bump-gemseo wrote it, and the issue and the merge request of the change. Two tabs show the code for
GEMSEO 6.3 and for GEMSEO 7. Both were executed, with GEMSEO 6.3.3 and with GEMSEO 7.0.0rc0, and
print the same values, unless the entry says that GEMSEO 7 cannot give the GEMSEO 6.3 value. The code
for GEMSEO 7 also contains the changes that bump-gemseo makes on its own, such as the new import paths,
so that you can run it as it is.

## See also

- [What's new in GEMSEO 7](../../whats_new_7.md): the new features.
- [Changelog](../../../changelog/index.md): every change of the release.
- [From GEMSEO 5 to GEMSEO 6](../5_to_6.md): the breaking changes of the previous major version; the
  [upgrading overview](../index.md) lists the guides of the older versions.
