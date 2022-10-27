..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _pypi: https://pypi.org
.. _anaconda: https://anaconda.org
.. _conda-forge: https://conda-forge.org
.. _pip-tools: https://github.com/jazzband/pip-tools
.. _pre-commit: https://pre-commit.com

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

All the dependencies of |g| are defined in :file:`setup.cfg`,
this files does not tell where the packages will be pulled from.
The dependencies could be provided by the packages repositories
`pypi`_, `anaconda`_ or `conda-forge`_.

Getting |g| to work with
a set of packages versions common to several platforms
and python versions is tricky and challenging.
This kind of work is mostly done by trials and errors.

In addition to the dependencies of |g|,
:file:`setup.cfg` also defines optional dependencies
used for running the tests or building the documentation.
These are defined in the ``[options.extras_require]`` section.

Dependencies for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependencies used for development shall be fully controlled
so all developers are provided
with reproducible and working environments.
The dependencies shall be updated
at least once in a while (couple months)
to benefit from packages improvements,
security and bug fixes.

The dependencies update is done with `pip-tools`_
and eventually from input requirements files.
These input requirements files contain
the minimum pinning requirements
and are intended to be modified by maintainers.
The `pip-tools`_ package provides the :command:`pip-compile`
which can process an input requirements file
to produce a fully pinned requirements file.
The actual call to `pip-tools`_ is done via ``tox`` (see below).

.. warning::

   All environments and tools shall be checked
   whenever dependencies have been changed.

Documentation files like :file:`CREDITS.rst`
and :file:`dependencies.rst` shall also be updated accordingly.

Whenever a dependency defined in :file:`setup.cfg` is changed,
update the requirements for the testing and ``doc`` environments of ``tox``:

.. _update-deps:

.. code-block:: shell

    tox -e update-deps-test-py37,update-deps-test-py38,update-deps-test-py39,update-deps-test-py310,update-deps-doc

The dependencies for the ``check`` and ``dist`` environments of ``tox``
are defined in:

- check.in: for checking the source files.
- dist.in: for creating the distribution.

Update the requirements for the those environments of ``tox``:

.. code-block:: shell

    tox -e update-deps-check
    tox -e update-deps-dist

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

Making a new release
--------------------

#. Create a release branch.
#. Make sure the full test suite passes.
#. Update the changelog.
#. Push the branch.
#. Build the docs for this branch on rtd, check the version and changelog.
#. Merge to master.
#. Tag.
#. Run :command:`tox -e dist` to create the distribution archives.
#. Run :command:`twine upload dist/* -u <your login>` to upload to pypi.org.
#. Test the pypi packages.
#. Merge master to develop so the last tag is a parent commit for defining the dev versions.
#. Push develop.
#. Update the recipe for conda-forge once the update bot sends the PR.
#. Test the conda-forge packages.
#. Create the anaconda stand alone distribution.

Making a new release for plugins
--------------------------------

#. Create a release branch.
#. Update the required gemseo version in :file:`setup.cfg`.
#. Update the environments dependencies (:ref:`update-deps`)
   while setting the environment variable :command:`GEMSEO_PIP_REQ_SPEC="gemseo"`.
#. Update the changelog.
#. Push the branch.
#. Make sure the full test suite passes.
#. Merge to master.
#. Tag.
#. Run :command:`tox -e dist` to create the distribution archives.
#. Run :command:`twine upload dist/* -u <your login>` to upload to pypi.org.
#. Test the pypi packages.
#. Merge master to develop so the last tag is a parent commit for defining the dev versions.
#. Update the environments dependencies (:ref:`update-deps`)
   **without** setting the environment variable ``GEMSEO_PIP_REQ_SPEC``.
#. Push develop.

Mirroring to github
-------------------

To mirror a project from gitlab to github:

- Clone the repository on github,
- Enable push mirroring on the gitlab repository setting page.
