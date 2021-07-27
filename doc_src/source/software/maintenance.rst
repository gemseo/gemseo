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
Thus the versions of its dependencies cannot be pinned\s,
but a range of compatible versions shall be defined.

All the dependencies shall be defined in :file:`setup.cfg`,
this files does not tell where the packages will be pulled from.
The dependencies could be provided by the packages repositories
`pypi`_, `anaconda`_ or `conda-forge`_.

Getting |g| to work with
a common set of packages versions on several platforms
and python versions is tricky and challenging.
This kind of work is mostly done by trials and errors.

To reduce maintenance and complexity,
our testing environments shall have the same packages providers
for all the platforms and all the python versions.
Furthermore it shall be identical to
the references end-user environments
under the same constraints.

In the context of **tox**,
the versions of the dependencies
that shall be installed with :command:`conda`
are defined in :file:`requirements/gemseo-conda-python{2,3}.txt`,
they are pulled from `conda-forge`_.
All other dependencies are installed with :command:`pip`,
they are pulled from `pypi`_.

When a dependency is changed,
:file:`setup.cfg` shall always be modified.
If the changed dependency is installed with :command:`conda`,
then both :file:`gemseo-conda-python{2,3}.txt`
and :file:`environment-py{2,3}.yml` shall be modified.

Documentation files like :file:`CREDITS.rst`
and :file:`dependencies.rst` shall also be updated accordingly.

CI cache
~~~~~~~~

To optimize the usage of the CI cache with gitlab,
the cache shall match closely the contents of the **tox** environments used for testing.
This is currently done by defining a cache affinity with
the dependencies of the environments and the environment name.
The dependencies are stored in the file
:file:`requirements/gemseo.in` which is created automatically by a custom
**pre-commit** hook defined in
:py:`tools/extract_req_in_from_setup_cfg.py`.

Development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

As opposed to the dependencies of |g|,
the development dependencies can be fully controlled.
Thus their versions are pinned
so all developers are provided
with reproducible and working environments.
The dependencies shall be updated
at least once in a while (couple months)
to benefit from packages improvements and bug fixes.

This is done with `pip-tools`_
and from input requirements files.
These input requirements files contain
the minimum pinning requirements
and are intended to be modified by maintainers.
The `pip-tools`_ package provides the :command:`pip-compile`
which can process an input requirements file
to produce a fully pinned requirements file.

We have the following input requirements files:

- dev.in: with development tools with python 3 only,
- doc.in: for building the documentation with python 3 only.

To update them:

.. code-block:: shell

   conda run -p .tox/dev pip-compile -U requirements/dev.in
   conda run -p .tox/doc pip-compile -U requirements/doc.in

.. note::

    Append ``-win`` to the environment names under windows.

.. note::

   To reduce discrepancy among the environments,
   :file:`requirements/test-python3.txt`
   shall be working for all the python 3 testing environments.

Git hooks are defined and run with `pre-commit`_.
It relies on packages that are managed
with `pre-commit`_ instead of `pip-tools`_.
To update them:

.. code-block:: shell

   conda run -p .tox/dev pre-commit autoupdate

.. note::

    Append ``-win`` to the environment names under windows.

.. warning::

   All environments and tools shall be checked
   whenever dependencies have been changed.

Test dependencies
-----------------

The test dependencies are defined in :file:`setup.cfg`
so a end-user can easily run the |g| tests.

To update them,
change the ``test`` key of the
``[options.extras_require]`` section
in :file:`setup.cfg`,
then execute

.. code-block:: shell

    tox -e style

This will call a pre-commit hook that will update
:file:`requirements/test.in`.
Then update the actual test requirements with:

.. code-block:: shell

   conda run -p .tox/dev pip-compile -U requirements/test.in -o requirements/test-python3.txt
   conda run -p .tox/py27 pip-compile -U requirements/test.in -o requirements/test-python2.txt

.. note::

    Append ``-win`` to the environment names under windows.

.. warning::

   All environments and tools shall be checked
   whenever dependencies have been changed.

Testing pypi packages
---------------------

Run (append ``-win`` on windows)

.. code-block:: shell

   tox -e pyX-pypi

For all the supported Python versions ``X``.

Testing conda-forge packages
----------------------------

Run (append ``-win`` on windows)

.. code-block:: shell

   tox -e pyX-conda-forge

For all the supported Python versions ``X``.

Testing anaconda environment file
---------------------------------

Run (append ``-win`` on windows)

.. code-block:: shell

   tox -e anaconda-env-file

Making a new release
--------------------

#. Create a release branch.
#. Make sure the full test suite passes.
#. Replace ``Unreleased`` by the new version in :file:`CHANGELOG.rst`.
#. Hardcode the version number in :file:`conf.py`.
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
