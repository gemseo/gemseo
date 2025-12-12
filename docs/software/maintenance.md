<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

 <!--
 _pypi: https://pypi.org
 _anaconda: https://anaconda.org
 _conda-forge: https://conda-forge.org
 _pre-commit: https://pre-commit.com
 _uv: https://docs.astral.sh/uv/
 -->

# Maintainers information

## Packages upgrading

This section contains information about how and when to upgrade the
packages used by GEMSEO and by the development environments.

### Dependencies  { #maintenance-dependencies }

GEMSEO is a library and not a self-contained application, it can be
installed in environments with varying and unknown constraints on the
versions of its dependencies. Thus, the versions of its dependencies
cannot be pinned, but a range of compatible versions shall be defined.

All the dependencies of GEMSEO are defined in `pyproject.toml`, this
files does not tell where the packages will be pulled from.

Getting GEMSEO to work with a set of packages versions common to several
platforms and python versions is tricky and challenging. This kind of
work is mostly done by trials and errors.

In addition to the dependencies of GEMSEO, `pyproject.toml` also defines
extra dependencies used for developing, testing, or building the
documentation.

### Dependencies for development

The dependencies used for development shall be fully controlled so all
developers are provided with reproducible and working environments. The
dependencies shall be updated to benefit from packages improvements,
security and bug fixes.

Development dependencies are defined in `pyproject.toml` under
[dependency-groups (PEP 735)](https://peps.python.org/pep-0735/):

- `dev`: testing dependencies (pytest, coverage, etc.)
- `doc`: documentation dependencies (mkdocs, etc.)
- `check`: linting dependencies (pre-commit, ruff)
- `check-types`: type checking dependencies (mypy, stubs)

All dependencies are stored in `uv.lock`, a
[universal lockfile](https://docs.astral.sh/uv/concepts/resolution/)
which is cross-platform and cross-Python-version,
so it can be generated on any platform.

!!! warning
      All environments and tools shall be checked whenever dependencies have been changed.

Whenever a dependency defined in `pyproject.toml` is changed or added,
update the lockfile:

``` shell
tox -e update-deps
```

## Testing pypi packages

Run

``` shell
tox -e pypi-pyX
```

for all the supported Python versions `X`, e.g. `tox -e pypi-py310`.

## Updating the changelog

To avoid rebase and merge conflicts, the changelog is not directly
updated in a branch but updated once a release is ready from changelog
fragments. Changelog fragment is a file that contains the part of the
changelog of a branch, named with `<issue number>.<change kind>.md` and
stored under `changelog/fragments`. The update is done with
[towncrier](https://github.com/twisted/towncrier):

``` shell
towncrier build --version <version number>
```

## Publishing process

The publishing of the distribution archives of a package at the version
X.Y.Z (where Z may contain a rcW suffix) is done automatically by the CI
on the following conditions:

- a CI variable with a PyPI token has be set,
- a branch named release-X.Y.Z is merged to the master branch.

A tag named X.Y.Z is also automatically created on the master branch.

## Making a new release

1. Create a release branch named release-X.Y.Z.
2. For plugins only:
    1. Update the required gemseo version in `pyproject.toml`.
3. Update the changelog.
4. Push the branch.
5. Create a MR to master.
6. Make sure the full test suite passes.
7. Merge master to develop so the last tag is a parent commit for
    defining the dev versions.
8. Push develop.
9. For GEMSEO only:
    1. Update the recipe for conda-forge once the update bot sends the
        PR.
    2. Test the conda-forge packages.
    3. Create the anaconda stand alone distribution.

## Mirroring to github

To mirror a project from gitlab to github:

- Clone the repository on github,
- Enable push mirroring on the gitlab repository setting page.
