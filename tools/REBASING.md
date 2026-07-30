<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Rebasing a branch created before the subpackage reorganization

This branch renamed and moved most packages: `gemseo.algos` was split into `gemseo.doe`,
`gemseo.optimization`, `gemseo.ode`, `gemseo.linear`, `gemseo.space` and
`gemseo.core.algorithm`, the plural package names were singularized, `gemseo.utils`
became `gemseo.util`, and so on. A branch based on `develop` therefore uses names that no
longer exist here, and its changes do not apply as they are.

`tools/bump-version-from-develop.yml` is the rename table to fix them, from the `develop`
names to the ones of this branch. Read it and apply it to the code of the branch being
rebased; its header states how, in short:

- `modules:` maps a module or package path to its new path — replace the longest matching
  prefix of a dotted name with the new one;
- `names:` maps a class, function or variable, given with the module defining it, to its
  new module and name — import it from the new module and rename it at its use sites.

Both sides of every entry are complete dotted paths, so the entries are independent of
one another and of their order. Apply a single entry per name: a value is already final.

Beyond the names, no public class attribute, method or signature was renamed between
`develop` and this branch. Private internals did move though, so a branch that touches
them, a factory internal or a private helper, has to reconcile its design with the one of
this branch and not merely its names.

## Procedure

Rebase first, rewrite afterwards. The merge base is the old `develop`, so rewriting the
names of the branch before rebasing makes both sides differ from that base and
manufactures conflicts instead of sparing them.

1. Squash the branch, or be ready to resolve the same conflicts once per commit.
2. `git rebase upstream/develop`. Rename detection carries the modified files to their new
   paths (`src/gemseo/algos/base_algo_factory.py` becomes
   `src/gemseo/core/algorithm/base_algorithm_factory.py`, `tests/algos/...` becomes
   `tests/core/algorithm/...`) and merges the changes there, so the conflicts are mostly
   import blocks. The files added by the branch are renamed by nothing: move them to the
   directory they now belong to by hand.
3. Resolve the conflicts. Mind that `--ours` is the side of this branch and `--theirs` the
   side of the branch being rebased, which reads backwards. For every hunk, ask first
   whether this branch already does the same thing by another route: what a branch did on
   `develop` to circumvent a problem has often been done here too, differently, and is
   then to be dropped rather than ported.
4. Apply the table to the old names that are left, and regenerate what is generated: the
   `src/gemseo/factory-files/` listings are regenerated with
   `uv run gemseo-generate-factory-files src/gemseo`, not rewritten, since the classes
   they list have changed as well.

Dotted paths are not only in `import` statements. Rewrite them also in the entry points of
`pyproject.toml`, in the mkdocs cross-references of the docstrings and of the documentation
pages (`[Name][dotted.path]`), in the string literals of the tests (a `sys.modules` key, a
module name given to `monkeypatch`) and in the test fixture packages under `tests/*/data/`.

Conversely, leave the old names alone where they are recorded on purpose:
`tests/test_deprecated_imports.py`, `docs/software/upgrading.md`, `changelog/fragments/`
and `src/gemseo/_deprecation/bump-version.yml`.

## Checking the result

- Grep the old top-level names, `gemseo.algos`, `gemseo.caches`, `gemseo.disciplines`,
  `gemseo.formulations`, `gemseo.problems`, `gemseo.settings` and `gemseo.utils`, over
  `src`, `tests` and `docs`, excluding the files listed above. This catches what no import
  error ever will: a docstring, a comment, a string literal.
- Compare the rebased commit with the original one, `git range-diff <old-base>..<old-tip>
  upstream/develop..<new-tip>`. Every line of it shall be either an entry of the table or a
  decision taken while resolving a conflict. This is also how a file that merged cleanly
  but still needs adapting is found: a test of the branch may keep testing an
  implementation that the resolution has replaced with the one of this branch.
- Do not expect a green test suite: a few hundred image comparison tests fail in a typical
  development environment. Run the suite on `upstream/develop` too and compare the sets of
  failing tests, not their number.

## Which table to use

| File | Old names | Purpose |
|---|---|---|
| `tools/bump-version-from-develop.yml` | `develop` | rebasing a branch onto this one |
| `src/gemseo/_deprecation/bump-version.yml` | last gemseo 6 release | migrating user code to gemseo 7 |

The shipped one drives the external codemod and is read at runtime by
`gemseo._deprecation` to redirect the imports of the gemseo 6 names to their new
location. It does not fit a rebase: its old names are the gemseo 6 ones, whereas a branch
based on `develop` already uses the gemseo 7 names for the most part.

## Maintenance

`tools/bump-version-from-develop.yml` is the composition of those two tables: the
`develop` → this branch renamings are what the shipped table adds to the one of
`develop`. Every old name in it exists on `develop` and every new name exists here,
checked against both source trees, and every module of `develop` and every `gemseo`
import of its `src` and `tests` resolves to a module of this branch. Extend it whenever a
package or a module-level name is renamed here.
