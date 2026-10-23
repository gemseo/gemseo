---
description: "Migrate your code from GEMSEO 6 to GEMSEO 7: run bump-gemseo, resolve what it leaves, check the silent changes."
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
    listed at the end of each domain page.

## The compatibility layer of GEMSEO 7 { #compatibility-layer }

GEMSEO 7 keeps most of the GEMSEO 6.3 *names* working for one deprecation cycle. When you import
`gemseo`, it installs a compatibility layer. The layer maps an old name to its new one and emits a
`DeprecationWarning` that gives the replacement:

```python
from gemseo.algos.design_space import DesignSpace
# DeprecationWarning: The module 'gemseo.algos.design_space' is deprecated; use 'gemseo.space.design' instead.
# DeprecationWarning: The attribute 'DesignSpace' of the module 'gemseo.algos.design_space' is deprecated; use 'gemseo.space.DesignSpace' instead.
```

The layer reads the same rename rules as bump-gemseo. It is a safety net, not a replacement for
bump-gemseo: it resolves names, not calls.

### What it does

- **Moved and renamed modules**: importing a GEMSEO 6.3 module path imports its GEMSEO 7 module,
  e.g. `gemseo.algos.design_space` gives `gemseo.space.design`.
- **Renamed classes, functions and constants** still resolve under their old names in their old
  modules, e.g. `SciPyMILP_Settings` of `gemseo.algos.opt.scipy_milp` gives
  `gemseo.optimization.MILP_Settings`.
- **Renamed class attributes, methods and enumeration members**: `Class.OLD` and `instance.OLD`
  both read the new attribute, e.g. `cache.last_entry` gives `cache.last_item` and
  `DatasetClassName.Dataset` gives `DatasetClassName.DATASET`. If one of your subclasses still
  defines or overrides an old name, its definition is moved to the new name.
- **The dissolved package `gemseo.settings`**: importing it warns and lists the packages that now
  hold its settings.
- **Names moved to a plugin**, e.g. `gemseo.disciplines.wrappers.xls_discipline`, now
  `gemseo_excel.xls_discipline`: when the plugin is not installed, the error names it.
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
- **changes of behaviour**: a name that GEMSEO 7 keeps but whose behaviour changed is not aliased
  (see [Step 4](#step-4-check-the-silent-changes)).

### Who does what

- **The compatibility layer** keeps the old names working until you rename them.
- **bump-gemseo** renames them in your code, which removes the warnings. It also rewrites the calls
  that the layer cannot keep working: arguments, settings models, removed methods with a
  replacement ([Step 1](#step-1-run-bump-gemseo)).
- **You** handle the rest:
    - the `# TODO(bump-gemseo)` comments ([Step 2](#step-2-resolve-the-todo-comments));
    - the errors that come with no comment ([Step 3](#step-3-switch-to-gemseo-7-and-run-your-code));
    - the silent changes ([Step 4](#step-4-check-the-silent-changes));
    - the old names that bump-gemseo did not see. They still work with a warning; rename them before
      the layer drops them ([Step 3](#step-3-switch-to-gemseo-7-and-run-your-code)).
- **A coding agent** can do most of your part, see
  [Let a coding agent do the manual work](#coding-agent).

### Limits

- **Patching an old path does not reach GEMSEO.** An old module path gives a stand-in that owns its
  own namespace. `mock.patch("gemseo.utils.string_tools.MultiLineString")` replaces the name on the
  stand-in, but GEMSEO code that imports the new path does not see the replacement: patch the new
  path. bump-gemseo migrates the `mock.patch` targets written as strings.
- **Assigning an old class attribute after the class is defined** shadows the alias instead of going
  through it, e.g. `Class.OLD = value` or `monkeypatch.setattr(Class, "OLD", ...)`. Assign the new
  name.

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

Install bump-gemseo in the environment where GEMSEO 6.3 is installed, then run it on your package,
its tests and its documentation at once:

```shell
bump-gemseo src tests docs
```

[Running bump-gemseo](bump_gemseo.md) shows how to install it, what it changes, how to read its report
and what it cannot do, on a complete example.

## Step 2: resolve the TODO comments

Where bump-gemseo cannot make a change, it leaves a comment above the line,

```python
# TODO(bump-gemseo): <what to do>  # noqa: E501
```

and lists these comments in its report. Search for the text of a comment in this guide (with the
search box, or in the sections *Changes flagged by bump-gemseo* of the domain pages): the entry
explains the change and shows the code to write. A comment about a removed name that no entry explains
refers to the section *Removed names* of the domain page; other comments with no example are listed in
the sections *Other comments of bump-gemseo*.

## Step 3: switch to GEMSEO 7 and run your code

Install GEMSEO 7 (see [Installation](../../installation.md)) and run your tests and scripts. An error
that no comment announced is in a section *Changes that fail without a comment*, which quotes the
error message: search for it.

Old names that bump-gemseo did not rename, such as names on an object whose type it could not infer,
or names built at run time, still work through the [compatibility layer](#compatibility-layer), with a
`DeprecationWarning`, and will fail once the layer drops them. Run your tests once with
`python -W error::DeprecationWarning` to find them, then rename them.

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

The changes that bump-gemseo leaves to you, steps 2 and 3, are a good job for a coding agent. The
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
3. Point it to this guide and to the [changelog](../../../changelog/index.md). Ask it to resolve the
   `# TODO(bump-gemseo)` comments, make your tests pass with GEMSEO 7, and leave no
   `DeprecationWarning` (see [the compatibility layer](#compatibility-layer)).
4. Review its changes with `git diff`, as you would review those of a colleague.

No model handles reliably the silent changes of [Step 4](#step-4-check-the-silent-changes), which
raise no error and get no comment, or the few changes that need a design decision, such as the
methods that a custom formulation overrides. Check these yourself.

## The guide, domain by domain

The domains are those of the [GEMSEO issue tracker](https://gitlab.com/gemseo/dev/gemseo/-/issues).
For each domain, the number of changes that bump-gemseo leaves to you:

- *Silent*: the code runs after bump-gemseo but gives other results.
- *Fail without comment*: the code fails with GEMSEO 7 and bump-gemseo wrote no comment.
- *Flagged*: bump-gemseo wrote a `# TODO(bump-gemseo)` comment.
- *Deprecated*: the code works, with a deprecation warning: rename what it names.
- *Removed names*: names removed without replacement.

| Domain | Scope | Silent | Fail without comment | Flagged | Deprecated | Removed names |
|---|---|---|---|---|---|---|
| [Core](core.md) | grammars, caches, namespaces, derivatives, execution and monitoring | 3 | 8 | 2 | 1 | 23 |
| [Disciplines](discipline.md) | disciplines, chains and wrappers | 1 | 8 | 2 | 0 | 3 |
| [Spaces](space.md) | design spaces and random spaces | 5 | 13 | 8 | 0 | 31 |
| [Problems and functions](problem.md) | evaluation and optimization problems, functions, databases, benchmark problems | 2 | 12 | 7 | 0 | 6 |
| [Optimization](optimization.md) | optimization libraries and their settings | 2 | 2 | 10 | 0 | 4 |
| [DOE](doe.md) | design of experiments libraries and their settings | 1 | 2 | 3 | 0 | 0 |
| [Formulations](formulation.md) | MDO formulations and custom formulations | 0 | 4 | 7 | 0 | 1 |
| [MDA](mda.md) | multidisciplinary analyses and coupled derivatives | 0 | 5 | 7 | 0 | 3 |
| [Scenarios](scenario.md) | scenarios, scenario adapters and their results | 0 | 5 | 3 | 0 | 1 |
| [Datasets](dataset.md) | datasets and their metadata | 3 | 0 | 2 | 0 | 0 |
| [Post-processing](post.md) | post-processors and dataset plots | 0 | 5 | 2 | 0 | 2 |
| [Machine learning](machine_learning.md) | regressors, classifiers, clustering and their quality | 1 | 8 | 6 | 0 | 7 |
| [Uncertainty](uncertainty.md) | distributions, statistics and sensitivity analysis | 0 | 7 | 6 | 0 | 32 |
| [Utilities](util.md) | configuration, logging, testing helpers and other utilities | 2 | 4 | 3 | 0 | 21 |

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
