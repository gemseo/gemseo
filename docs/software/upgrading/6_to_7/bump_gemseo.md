---
description: "Install and run bump-gemseo, the tool that migrates your code from GEMSEO 6.3 to GEMSEO 7, read its report and know its limits."
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

# Running bump-gemseo

[bump-gemseo](https://gitlab.com/gemseo/dev/bump-gemseo) is a command-line tool, written by the
GEMSEO developers, that rewrites your Python code from GEMSEO 6.3 to GEMSEO 7. It applies the
migration rules of GEMSEO, a list of every module, class, function, attribute and argument that
GEMSEO 7 renamed, moved, converted or removed, and it analyses the names and the types of your code to
apply them only where they belong. It does about nine tenths of the work. This page shows how to run
it and what to do with what it leaves.

## What it changes

- **Imports and names**: modules, classes, functions, attributes, methods, arguments and enumeration
  members get their GEMSEO 7 names, e.g. `from gemseo.algos.design_space import DesignSpace` becomes
  `from gemseo.space import DesignSpace`.
- **Settings**: the keyword arguments configuring an object become a settings model, e.g.
  `scenario.execute(algo_name="SLSQP", max_iter=30)` becomes
  `scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=30))`, with the import of
  `SLSQP_Settings`.
- **Removed members** with a direct replacement are rewritten, e.g. `design_space.get_lower_bound("x")`
  becomes `design_space.variables["x"].lower_bound`.
- **Special cases**: the calls to `configure(...)` and `configure_logger(...)` become fields of
  `gemseo.configuration`; a discipline using JSON grammar files keeps them
  (`default_grammar_type = Discipline.GrammarType.JSON`); the tests comparing images move to the
  `snapshot_matplotlib` fixture; a distribution of one variable gets its new base class.
- **High-level functions keep their keyword arguments**: `create_scenario`, `execute_post`,
  `sample_disciplines` and the `create_*_model` functions still accept the settings as keyword arguments
  in GEMSEO 7, so bump-gemseo leaves these calls as they are, but for the renamed settings.
- **Around the code**: the targets of `mock.patch` given as strings, the cross-references of the
  documentation (`[DesignSpace][gemseo.space.DesignSpace]`) and the lists of classes of the ruff
  configuration.
- **What it cannot change** is marked with a `# TODO(bump-gemseo)` comment, see below.

## Install it

Install bump-gemseo **in the environment of the code to migrate, where GEMSEO 6.3 is installed**:

```shell
pip install bump-gemseo@git+https://gitlab.com/gemseo/dev/bump-gemseo.git@develop
```

bump-gemseo reads the GEMSEO 6.3 installed next to it, and the dependencies of your project, to know
what each name of your code refers to: they must be importable in the same environment. Do not install it in an isolated environment such as the one
`pipx` creates, and do not run it on code written for an older version of GEMSEO.

## A first run

Take a small study written for GEMSEO 6.3, in `my_study/optimize.py`:

```python
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel

first = AnalyticDiscipline({"y_1": "x + 0.2 * y_2"}, name="first")
second = AnalyticDiscipline({"y_2": "0.5 * y_1 + 1"}, name="second")
mda = MDAGaussSeidel([first, second], tolerance=1e-10, linear_solver_tolerance=1e-12)
print(mda.execute()["y_1"])

objective = AnalyticDiscipline({"f": "(x - 1)**2 + y_1"}, name="objective")
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=0.0)

scenario = create_scenario(
    [first, second, objective], "f", design_space, formulation_name="MDF"
)
scenario.execute(algo_name="SLSQP", max_iter=30)
print(scenario.optimization_result.x_opt, design_space.get_lower_bound("x"))
```

With GEMSEO 6.3, it prints `[0.22222222]`, the result of the MDA, then `[0.44444444] [-2.]`, the optimum and the lower bound. Commit it, then run bump-gemseo from the directory
containing `my_study`:

```shell
bump-gemseo my_study
```

bump-gemseo prints its report:

```text
Migrating 1 file
my_study/optimize.py  migrated
    10: cannot transform: linear_solver_tolerance is passed without linear_solver, whose default names no Settings class, and linear_solver_tolerance has a rule in a Settings class: migrate it by hand
    11: use the rtol field of the linear_solver_settings instead
1 file migrated, 2 TODOs to review
```

and `git diff` shows what it changed:

```diff
 from gemseo import create_scenario
-from gemseo.algos.design_space import DesignSpace
-from gemseo.disciplines.analytic import AnalyticDiscipline
-from gemseo.mda.gauss_seidel import MDAGaussSeidel
+from gemseo.space import DesignSpace
+from gemseo.discipline import AnalyticDiscipline
+from gemseo.mda import MDAGaussSeidel
+from gemseo.optimization import SLSQP_Settings
+from gemseo.mda import MDAGaussSeidel_Settings

 first = AnalyticDiscipline({"y_1": "x + 0.2 * y_2"}, name="first")
 second = AnalyticDiscipline({"y_2": "0.5 * y_1 + 1"}, name="second")
-mda = MDAGaussSeidel([first, second], tolerance=1e-10, linear_solver_tolerance=1e-12)
+# TODO(bump-gemseo): cannot transform: linear_solver_tolerance is passed without linear_solver, whose default names no Settings class, and linear_solver_tolerance has a rule in a Settings class: migrate it by hand  # noqa: E501
+# TODO(bump-gemseo): use the rtol field of the linear_solver_settings instead  # noqa: E501
+mda = MDAGaussSeidel([first, second], settings=MDAGaussSeidel_Settings(tolerance=1e-10, linear_solver_tolerance=1e-12))
 print(mda.execute()["y_1"])
 ...
-scenario.execute(algo_name="SLSQP", max_iter=30)
-print(scenario.optimization_result.x_opt, design_space.get_lower_bound("x"))
+scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=30))
+print(scenario.optimization_result.x_opt, design_space.variables["x"].lower_bound)
```

Every import, the settings of the MDA and of the optimizer and the lower bound are migrated. One
change is left to you: GEMSEO 7 has no `linear_solver_tolerance` setting any more, the tolerance of the
linear solver being a field of its own settings model. Run with GEMSEO 7 as it is, the MDA line raises
`ValidationError: ... linear_solver_tolerance Extra inputs are not permitted`. The entry of the
[MDA page](mda.md) about this comment explains the change; here, the default linear solver of GEMSEO 6.3
was LGMRES, so the line becomes:

```python
from gemseo.linear import LGMRES_Settings

mda = MDAGaussSeidel(
    [first, second],
    settings=MDAGaussSeidel_Settings(
        tolerance=1e-10, linear_solver_settings=LGMRES_Settings(rtol=1e-12)
    ),
)
```

Remove the two comments: the study runs with GEMSEO 7 and prints the same values as with GEMSEO 6.3.

## Read the report

The report lists each file, sorted by path, with its status:

- `migrated`: bump-gemseo changed the file;
- `unchanged`: there was nothing to change;
- `failed`: the file could not be read, e.g. because of a syntax error; it is left unchanged, and the
  error is given at the end of the report.

Under a file come the comments bump-gemseo added, with their line in the migrated file. The last line
counts the files and the comments to review.

## The TODO comments

A comment is written on the line above the code it is about:

```python
# TODO(bump-gemseo): <message>  # noqa: E501
```

The `# noqa: E501` pragma tells ruff not to report the length of the line. To list the comments left
in your code at any time:

```shell
git grep -n "TODO(bump-gemseo)"
```

Each comment says what to do. The domain pages of this guide explain the change behind each message,
in their sections *Changes flagged by bump-gemseo*, and list the names removed without replacement.
Once a change is made, remove its comment. Run bump-gemseo once, on the code written for GEMSEO 6.3:
running it again on migrated code is not supported, as it may change some calls again.

## Which files it reads

`bump-gemseo <path> ...` takes files and directories, searched recursively; the hidden directories,
such as `.venv` or `.git`, are skipped. It runs one worker process per CPU; set their number with
`-j`, e.g. `bump-gemseo -j 4 src`.

- **Python files** are migrated.
- **Markdown files**: the cross-references of the documentation are migrated, but not the Python code
  blocks, since a documentation may execute them: the names they use that GEMSEO 7 renamed are listed
  in the report, for you to migrate by hand.
- **ruff configuration files** (`.ruff.toml`, `ruff.toml`, `pyproject.toml`): the class names of the
  options `runtime-evaluated-base-classes` and `runtime-evaluated-decorators` are migrated.

Migrate a package together with its tests and its documentation, in one run: bump-gemseo then knows
the classes of your package, e.g. the settings classes of a plugin, and the modules that the tests
patch by name. Run it from the root of the project: `bump-gemseo src tests docs`.

## After the run

1. Review the changes with `git diff`.
2. Format the code: bump-gemseo adds the new imports after the existing ones and writes long lines,
   so run your formatter, e.g. `ruff check --fix` and `ruff format`.
3. Migrate the code flagged by the TODO comments (step 2 of the [overview](index.md)).
4. Install GEMSEO 7, run your tests, and check the silent changes (steps 3 and 4).

## What it cannot do

bump-gemseo works on the text of your code and on the types it can infer. Some changes are out
of its reach:

- **Silent changes of behaviour.** When GEMSEO 7 keeps a name but changes what it does, there is
  nothing to rename and bump-gemseo writes no comment: the domain pages list these changes, in their
  sections *Silent changes*.
- **Removed features** have no replacement: bump-gemseo marks each use, and you decide what to do.
- **Design changes**, such as the methods a custom formulation or a custom driver library overrides,
  need your judgment: bump-gemseo marks them with a comment explaining the new design.
- **Types it cannot infer.** An object whose class is not known from the code, e.g. the result of a
  method called on a class rather than on an instance, `BaseFactory.create(DisciplineFactory(), ...)`,
  keeps the old names of its attributes. A keyword argument of a base class is not renamed in a call to a subclass
  that redefines `__init__`, e.g. `algo=` in `MSEMeasure(algo=...)`.
- **Names in strings** other than the targets of `mock.patch`, and names built at run time.

If bump-gemseo changes something wrongly, report it on its
[issue tracker](https://gitlab.com/gemseo/dev/bump-gemseo/-/issues).
