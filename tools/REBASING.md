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

Beyond the names, nothing needs adapting: no class attribute, no method and no signature
was renamed between `develop` and this branch.

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
