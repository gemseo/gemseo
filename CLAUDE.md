<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# CLAUDE.md

Guidance for Claude Code when working in this repository.

The team-wide conventions (Python tooling, code style, docstrings, changelog policy,
the gemseo-provided pytest fixtures and the `assert_exception` helper) come from the
team-common `common.md`, which is loaded automatically at session start. This file only
records what is specific to this repository; do not duplicate `common.md` here.

## Project Overview

GEMSEO (Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization)
is a Python library for Multidisciplinary Design Optimization (MDO). It orchestrates
disciplines (computational modules), MDO formulations, multidisciplinary analysis (MDA)
solvers, optimization algorithms and surrogate models.

Python `>=3.11, <3.15`.

## Commands

Recipes live in `.justfile` (run `just --list`). They all go through `uv`; the
interpreter is the one of `.python-version` (3.11), overridable per machine with
`UV_PYTHON` (the `.justfile` loads a git-ignored `.env`, a local preference, not a
project setting).

| Recipe | What it does |
|---|---|
| `just test [args]` | `uv run --extra all pytest` — the `--extra all` matters, a plain `uv run pytest` misses optional backends |
| `just coverage [args]` | Tests plus XML/HTML coverage reports |
| `just coverage-diff upstream/develop` | Coverage then `diff-cover` on the changed lines, `--fail-under=100` |
| `just test-min-deps [args]` | Tests on Python 3.11 with `--resolution lowest-direct` |
| `just check` | Installs the git hooks then runs `prek` on all files |
| `just check-typing` | mypy, strict, on the opt-in file list of `.mypy.ini` |
| `just doc` / `just doc-fast` | Serve the docs with mkdocs; `doc-fast` uses `mkdocs-fast.yml` and skips the API pages |
| `just install` / `just fresh` | `uv sync --extra all`; `fresh` wipes `.venv` and the caches first |
| `just dist` / `just publish` | Build and check the wheel, then upload |

Run a single test or a selection with the usual pytest arguments:
`just test tests/path/test_x.py::test_fn`, `just test tests/ -k keyword`,
`just test tests/ -m doc_examples`.

Prefer `-n 6` for any run over more than a handful of tests; the suite is long.

## Testing

Markers: `doc_examples`, `integration`, `post`, `slow`, `skip_under_windows`. The
`doc_examples` tests are skipped unless `-m` is passed, by a
`pytest_collection_modifyitems` hook in `tests/conftest.py`; run them with
`-m doc_examples`.

`tests/conftest.py` re-exports the shared fixtures from
`gemseo.util.testing.pytest_conftest` and adds this repository's own:
`sellar_disciplines` (a built Sellar problem), `two_virtual_disciplines`,
`sellar_with_2d_array`.

## Architecture

### Public API

The user-facing subpackages re-export their public names from their `__init__.py` —
classes, the matching `_Settings` models and the factory singleton. **Prefer these
aliases** in examples, tests and docs — `from gemseo.mda import MDAGaussSeidel`,
`from gemseo.formulation import MDF_Settings` — over an import from the defining
module (`gemseo.mda.gauss_seidel`) or a `create_*` helper. `core/` and `util/` are
internal and export nothing this way.

The aliases are lazy: an `__init__.py` maps exported name to
`"module[:attribute.chain]"` in a `_name_to_location` mapping and calls
`install_lazy_reexport(globals(), _name_to_location)`, which installs `__all__`,
`__dir__` and a resolving `__getattr__`, so importing the subpackage does not import
its submodules. The same imports are repeated in a `if TYPE_CHECKING:` block for mypy
and IDEs. A new public class must be added to **both**, in alphabetical order.

`src/gemseo/__init__.py` still exposes ~40 top-level functions (`create_scenario()`,
`create_discipline()`, `create_mda()`, `execute_algo()`, `compute_doe()`, ...); they
remain part of the stable user-facing API, they are just no longer the preferred way to
build objects.

### Core Abstractions

**Discipline** (`core/discipline/`): the fundamental building block. A `Discipline`
wraps a computational function, declares its I/O through grammars, manages caching,
computes or approximates Jacobians, and tracks execution statistics.
`BaseDiscipline` → `Discipline` is the main chain.

**Grammar** (`core/grammar/`): JSON-schema-based I/O specification attached to every
discipline. `JSONGrammar` is the default, `SimpleGrammar` the lighter one. Grammars
validate data and make coupling detection automatic.

**Scenario** (`scenario/`): top-level orchestrator combining disciplines, an MDO
formulation and an algorithm. `MDOScenario` and `DOEScenario` are the concrete classes.

**Formulations** (`formulation/`): MDF, IDF, BiLevel, DisciplinaryOpt. A formulation
assembles disciplines and MDAs into an optimization problem.

**MDA** (`mda/`): coupling solvers — Gauss-Seidel, Jacobi, Newton-Raphson,
quasi-Newton variants — all deriving from `BaseMDA`.

**Algorithms**: one top-level package per family — `optimization/`, `doe/`, `linear/`,
`ode/`. Each keeps its base classes and factory in a `core/` subpackage, one subpackage
per backend (`scipy_local/`, `openturns/`, `nlopt/`, ...), and the per-algorithm
Pydantic settings. Design and parameter spaces live in `space/`.

**Machine learning** (`machine_learning/`): regression, classification and clustering
models, all deriving from `BaseMLModel`.

### Package Map

| Package | Purpose |
|---|---|
| `core/` | Base machinery: discipline, grammar, cache, function, problem, factory, data converters, parallel execution |
| `discipline/` | Built-in discipline types and wrappers (Excel, executables, job schedulers) |
| `optimization/`, `doe/`, `ode/`, `linear/` | Algorithm families: base classes and factory in `core/`, one subpackage per backend |
| `space/` | Design space, parameter space, variables |
| `formulation/` | MDO formulations |
| `mda/` | MDA solvers |
| `scenario/` | Scenario orchestration and scenario adapters |
| `problem/` | Benchmark problems: Sellar, Sobieski, Rosenbrock, Power2, ... |
| `machine_learning/` | Surrogate models, quality measures, transformers |
| `post/` | Post-processing and visualization |
| `uncertainty/` | Distributions, sensitivity analysis, statistics, reliability |
| `dataset/` | Dataset containers |
| `enum/` | Shared enumerations |
| `util/` | Derivative approximation, testing helpers, XDSM, multiton, logging |
| `_deprecation/` | Import redirection and the `bump-version.yml` rename tables |

### Factories

Every subsystem has a factory (`GrammarFactory`, `DisciplineFactory`, `MDAFactory`,
`PostFactory`, `RegressorFactory`, ...) deriving from `BaseFactory[T]`. Factories are
singletons per subclass, through the `BaseABCMultiton` metaclass. They discover
implementations by scanning their package, the directories of the `GEMSEO_PATH`
environment variable, and the `gemseo_plugins` entry points of installed packages.

Each factory module exposes a module-level singleton annotated `Final`, named in lower
snake case: `grammar_factory`, `discipline_factory`, `mda_factory`, `post_factory`,
`regressor_factory`, `dataset_factory`, ... **Import and use that global** instead of
calling `MDAFactory()` yourself: the call returns the same singleton, so instantiating
explicitly is only noisier. Use the `reset_factory` fixture in tests that touch them.

### Pydantic Settings

Algorithm, MDA, formulation and post-processing options are Pydantic models named
`<ClassName>_Settings` (or `Base*Settings`), with `extra="forbid"`. They sit next to the
class they configure — a sibling `<name>_settings.py` (e.g. `formulation/mdf_settings.py`)
or a `settings/` subpackage of the backend (e.g. `doe/pydoe/settings/`). There is no
central settings package.

### Deprecation and Renames

`src/gemseo/_deprecation/` keeps a rename cycle working for users:

- `bump-version.yml` is the single rename map. Its `modules:`, `attributes:`,
  `dissolved:` and `manual:` sections are read at runtime; `classes:` is read too, but
  only its scalar entries (renamed class attributes and methods) — nested
  method-parameter blocks and `null` targets are for the external codemod only.
- `aliases.py` computes the old→new tables from that file at import time.
- `install()` registers a `MetaPathFinder` that redirects old imports, adds a module
  `__getattr__` when only a class was renamed, and sets data descriptors for renamed
  class attributes, each emitting a `DeprecationWarning`.
- Names under `manual:` cannot be migrated automatically and raise an `ImportError`
  explaining the migration.

**Any user-visible rename must be registered in `bump-version.yml`.** Entries stay there
until their scheduled removal, which is what makes the aliases accumulate across
releases.

### Changelog Fragments

Fragments go in `changelog/fragments/` as `<issue>.<type>.md`, e.g. `1269.fixed.md`;
types are `added`, `changed`, `deprecated`, `fixed`, `removed`. Preview with
`towncrier build --draft`.

A fragment named only for its type (`added.md`, `fixed.md`, ...) is **silently dropped**:
towncrier reads the segment before the first dot as the issue reference and the one
after it as the type, so `added.md` is read as issue `added` of type `md`, which is not
a configured type. With no issue number, prefix the name with `+` — the first segment is
then taken as "no issue" and the type is read correctly, e.g. `+boxplot-options.changed.md`.

### Docs

mkdocs, sources in `docs/`, API pages generated by `docs/gen_ref_nav.py`. `just doc-fast`
skips the API generation and is the one to use while editing prose. Docstring
inheritance is enabled through `DOCSTRING_INHERITANCE_ENABLE`. Bibliography entries
live in `docs/references.bib`.

Never edit `docs/software/upgrading.md`.

### Type Checking

`.mypy.ini` runs in strict mode but only over the explicit `files =` list — most of the
package is not type checked yet. Adding a module to that list is a deliberate move, not
a side effect of another change.
