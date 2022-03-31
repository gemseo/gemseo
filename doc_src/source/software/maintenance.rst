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

All the dependencies of |g| shall be defined in :file:`setup.cfg`,
this files does not tell where the packages will be pulled from.
The dependencies could be provided by the packages repositories
`pypi`_, `anaconda`_ or `conda-forge`_.

Getting |g| to work with
a set of packages versions common to several platforms
and python versions is tricky and challenging.
This kind of work is mostly done by trials and errors.

As opposed to the dependencies of |g|,
the development dependencies shall be fully controlled.
Thus their versions are pinned
so all developers are provided
with reproducible and working environments.
The dependencies shall be updated
at least once in a while (couple months)
to benefit from packages improvements and bug fixes.

The dependencies update is done with `pip-tools`_
and from input requirements files.
These input requirements files contain
the minimum pinning requirements
and are intended to be modified by maintainers.
The `pip-tools`_ package provides the :command:`pip-compile`
which can process an input requirements file
to produce a fully pinned requirements file.
The actual call to `pip-tools`_ is done via ``tox`` (see below).

To reduce maintenance and complexity,
our testing environments shall have the same packages providers
for all the platforms and all the python versions.
Furthermore it shall be identical to
the references end-user environments
under the same constraints.

When a dependency is changed,
:file:`setup.cfg` shall always be modified.

.. warning::

   All environments and tools shall be checked
   whenever dependencies have been changed.

Documentation files like :file:`CREDITS.rst`
and :file:`dependencies.rst` shall also be updated accordingly.

Test dependencies
-----------------

The test dependencies are defined in :file:`setup.cfg`
so a end-user can easily run the |g| tests.

To add or constrain them,
if needed,
change the contents of the ``test`` key in the
``[options.extras_require]`` section
of :file:`setup.cfg`,
then execute:

.. code-block:: shell

    tox -e check

This will call a pre-commit hook that will update
:file:`requirements/test.in`.
Using a tool prevents human copy/paste errors.

Update the actual test requirements with:

.. code-block:: shell

    tox -e update-deps-test
    tox -e update-deps-test-py27

.. note::

   To reduce discrepancy among the environments,
   :file:`requirements/test.txt`,
   produced from :file:`requirements/test.in`,
   shall be working for all the python 3 testing environments.

Other dependencies
~~~~~~~~~~~~~~~~~~

We have the following input requirements files:

- doc.in: for building the documentation.
- dist.in: for creating the distribution.
- check.in: for checking the source files.

To update them:

.. code-block:: shell

    tox -e update-deps-doc
    tox -e update-deps-dist
    tox -e update-deps-check

Testing pypi packages
---------------------

Run

.. code-block:: shell

   tox -e pyX-pypi

For all the supported Python versions ``X``.

Testing conda-forge packages
----------------------------

Run

.. code-block:: shell

   tox -e pyX-conda-forge

For all the supported Python versions ``X``.

Testing anaconda environment file
---------------------------------

Run

.. code-block:: shell

   tox -e anaconda-env-file


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

   towncrier build

Making a new release
--------------------

#. Create a release branch.
#. Make sure the full test suite passes.
#. Replace ``Unreleased`` by the new version in :file:`CHANGELOG.rst`.
#. Hardcode the version number in :file:`conf.py`.
#. Update the changelog.
#. Push the branch.
#. Build the docs for this branch on rtd, check the version and changelog.
#. Merge to master.
#. Tag.
#. Run :command:`tox -e create-dist` to create the distribution archives.
#. Run :command:`twine upload dist/* -u <your login>` to upload to pypi.org.
#. Test the pypi packages.
#. Update the recipe for conda-forge once the update bot sends the PR.
#. Test the conda-forge packages.
#. Merge master to develop so the last tag is a parent commit for defining the dev versions.
#. Remove the hardcoded version number in :file:`conf.py`.
