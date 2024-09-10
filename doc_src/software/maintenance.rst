..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _pypi: https://pypi.org
.. _anaconda: https://anaconda.org
.. _conda-forge: https://conda-forge.org
.. _pre-commit: https://pre-commit.com
.. _uv: https://docs.astral.sh/uv/

Maintainers information
=======================

Packages upgrading
------------------

This section contains information about how and when to upgrade
the packages used by |g| and by the development environments.

Dependencies
~~~~~~~~~~~~

|g| is a library
and not a self contained application,
it can be installed in environments
with varying and unknown constraints
on the versions of its dependencies.
Thus the versions of its dependencies cannot be pinned,
but a range of compatible versions shall be defined.

All the dependencies of |g| are defined in :file:`pyproject.toml`,
this files does not tell where the packages will be pulled from.

Getting |g| to work with
a set of packages versions common to several platforms
and python versions is tricky and challenging.
This kind of work is mostly done by trials and errors.

In addition to the dependencies of |g|,
:file:`pyproject.toml` also defines extra dependencies
used for developing, testing, or building the documentation.

Dependencies for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependencies used for development shall be fully controlled
so all developers are provided
with reproducible and working environments.
The dependencies shall be updated
to benefit from packages improvements,
security and bug fixes.

The dependencies update is done with `uv`_
and eventually from input requirements files.
These input requirements files may contain
the minimum pinning requirements
and are intended to be modified by maintainers.
The `uv`_ package provides :command:`uv pip compile`
which may process an input requirements file
to produce a fully pinned requirements file.
The actual call to `uv`_ is done via ``tox`` (see below).

.. warning::

   All environments and tools shall be checked
   whenever dependencies have been changed.

Documentation files like :file:`CREDITS.rst`
and :file:`dependencies.rst` shall also be updated accordingly.

Whenever a dependency defined in :file:`pyproject.toml` is changed,
update the requirements for the testing and ``doc`` environments of ``tox``:

.. _update-deps:

.. code-block:: shell

    tox -e update-deps-test,update-deps-doc

.. warning::

   This shall be run on linux only otherwise windows specific packages will be included!

The dependencies for the ``check`` and ``dist`` environments of ``tox``
are defined in:

- check.in: for checking the source files.
- dist.in: for creating the distribution.

Update the requirements for the those environments of ``tox``:

.. code-block:: shell

    tox -e update-deps-check,update-deps-dist

Testing pypi packages
---------------------

Run

.. code-block:: shell

   tox -e pypi-pyX

for all the supported Python versions ``X``, e.g. ``tox -e pypi-py39``.

Testing conda-forge packages
----------------------------

Run

.. code-block:: shell

   tox -e conda-forge-pyX

for all the supported Python versions ``X``, e.g. ``tox -e conda-forge-py39``.

Updating the changelog
----------------------

To avoid rebase and merge conflicts,
the changelog is not directly updated in a branch
but updated once a release is ready from changelog fragments.
Changelog fragment is a file that contains the part of the changelog of a branch,
named with :file:`<issue number>.<change kind>.rst`
and stored under :file:`changelog/fragments`.
The update is done with `towncrier <https://github.com/twisted/towncrier>`_:

.. code-block:: shell

   towncrier build --version <version number>

Publishing process
------------------

The publishing of the distribution archives of a package at the version X.Y.Z
(where Z may contain a rcW suffix)
is done automatically by the CI on the following conditions:

- a CI variable with a PyPI token has be set,
- a branch named release-X.Y.Z is merged to the master branch.

A tag named X.Y.Z is also automatically created on the master branch.

Making a new release
--------------------

#. Create a release branch named release-X.Y.Z.
#. For plugins only:

   #. Update the required gemseo version in :file:`pyproject.toml`.

#. Update the changelog.
#. Push the branch.
#. Create a MR to master.
#. Make sure the full test suite passes.
#. Merge master to develop so the last tag is a parent commit for defining the dev versions.
#. Push develop.
#. For |g| only:

   #. Update the recipe for conda-forge once the update bot sends the PR.
   #. Test the conda-forge packages.
   #. Create the anaconda stand alone distribution.

Mirroring to github
-------------------

To mirror a project from gitlab to github:

- Clone the repository on github,
- Enable push mirroring on the gitlab repository setting page.
