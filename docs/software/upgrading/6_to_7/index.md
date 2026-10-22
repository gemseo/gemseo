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
changes that bump-gemseo cannot make on its own, each with the code before and after.

!!! info "How much bump-gemseo does on its own"

    The changes of GEMSEO 7 were turned into 2081 small programs, one or a few per change, each
    written for GEMSEO 6.3 and executed before and after migration. bump-gemseo alone migrates
    1804 of them correctly (86.7 %). The 277 others are what this guide is about:
    144 are explained with an example, the rest are names removed without replacement,
    listed at the end of each domain page.

## Before you start

1. **Upgrade to GEMSEO 6.3 first.** bump-gemseo migrates code written for GEMSEO 6.3 only. If your
   code uses an older version, follow the page [From GEMSEO 5 to GEMSEO 6](../5_to_6.md) and those of the previous versions up to 6.3,
   and make sure it runs with GEMSEO 6.3 without deprecation warnings.
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
explains the change and shows the code to write. A comment saying that a name *was removed* refers to
the section *Removed names* of the domain page.

## Step 3: switch to GEMSEO 7 and run your code

Install GEMSEO 7 (see [Installation](../../installation.md)) and run your tests and scripts. An error
that no comment announced is in a section *Changes that fail without a comment*, which quotes the
error message: search for it.

GEMSEO 7 still accepts most GEMSEO 6.3 names for one deprecation cycle, with a `DeprecationWarning`.
bump-gemseo renames them, but a name it did not see (in a string, or built at run time) still works
for now and will fail in a later version. Run your tests once with `python -W error::DeprecationWarning`
to find them.

## Step 4: check the silent changes

These changes do not raise any error and bump-gemseo writes no comment about them. Check the ones that
concern your code, and compare your results with those of GEMSEO 6.3:

- [Core](core.md#silent-changes):
    - `split_namespace` splits on the last separator only
- [Disciplines](discipline.md#silent-changes):
    - `execute` returns the output data only
- [Spaces](space.md#silent-changes):
    - `get_current_value` returns copies
    - `project_into_bounds` may return a float array
    - The design space copies and casts the values it stores
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

## The guide, domain by domain

The domains are those of the [GEMSEO issue tracker](https://gitlab.com/gemseo/dev/gemseo/-/issues).
For each domain, the number of changes that bump-gemseo leaves to you:

- *Silent*: the code runs after bump-gemseo but gives other results.
- *Fail without comment*: the code fails with GEMSEO 7 and bump-gemseo wrote no comment.
- *Flagged*: bump-gemseo wrote a `# TODO(bump-gemseo)` comment.
- *Deprecated*: the code works, with a deprecation warning.
- *Removed names*: names removed without replacement.

| Domain | Scope | Silent | Fail without comment | Flagged | Deprecated | Removed names |
|---|---|---|---|---|---|---|
| [Core](core.md) | grammars, caches, namespaces, derivatives, execution and monitoring | 1 | 8 | 1 | 1 | 23 |
| [Disciplines](discipline.md) | disciplines, chains and wrappers | 1 | 7 | 2 | 0 | 3 |
| [Spaces](space.md) | design spaces and random spaces | 3 | 11 | 8 | 0 | 31 |
| [Problems and functions](problem.md) | evaluation and optimization problems, functions, databases, benchmark problems | 0 | 8 | 7 | 0 | 6 |
| [Optimization](optimization.md) | optimization libraries and their settings | 0 | 2 | 10 | 0 | 4 |
| [DOE](doe.md) | design of experiments libraries and their settings | 1 | 1 | 3 | 1 | 0 |
| [Formulations](formulation.md) | MDO formulations and custom formulations | 0 | 1 | 6 | 0 | 1 |
| [MDA](mda.md) | multidisciplinary analyses and coupled derivatives | 0 | 4 | 5 | 0 | 3 |
| [Scenarios](scenario.md) | scenarios, scenario adapters and their results | 0 | 4 | 3 | 1 | 1 |
| [Datasets](dataset.md) | datasets and their metadata | 3 | 0 | 2 | 0 | 0 |
| [Post-processing](post.md) | post-processors and dataset plots | 0 | 4 | 2 | 1 | 1 |
| [Machine learning](machine_learning.md) | regressors, classifiers, clustering and their quality | 1 | 6 | 6 | 0 | 7 |
| [Uncertainty](uncertainty.md) | distributions, statistics and sensitivity analysis | 0 | 7 | 6 | 0 | 32 |
| [Utilities](util.md) | configuration, logging, testing helpers and other utilities | 1 | 2 | 3 | 0 | 21 |

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
- [From GEMSEO 5 to GEMSEO 6](../5_to_6.md): the breaking changes of the previous major version; the pages of the
  older versions follow it.
