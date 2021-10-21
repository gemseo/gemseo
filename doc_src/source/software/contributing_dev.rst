..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
      INITIAL AUTHORS - initial API and implementation and/or
                        initial documentation
          :author:  Francois Gallard

.. _pytest: https://docs.pytest.org
.. _tox: https://tox.readthedocs.io
.. _tox-conda: https://github.com/tox-dev/tox-conda
.. _anaconda: https://docs.anaconda.com/anaconda/install
.. _sphinx: https://www.sphinx-doc.org
.. _gitflow: https://nvie.com/posts/a-successful-git-branching-model
.. _pylint: https://pylint.readthedocs.io
.. _pep8: https://pep8.org
.. _flake8: https://flake8.pycqa.org
.. _black: https://black.readthedocs.io
.. _isort: https://timothycrosley.github.io/isort
.. _conventional commits: https://www.conventionalcommits.org
.. _commitizen: https://commitizen-tools.github.io/commitizen
.. _semantic versioning: https://semver.org
.. _editable mode: https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs
.. _semantic linefeeds: https://rhodesmill.org/brandon/2012/one-sentence-per-line
.. _mypy: http://mypy-lang.org
.. _standard duck typing: https://mypy.readthedocs.io/en/stable/cheat_sheet.html?highlight=Sequence#standard-duck-types
.. _pytest-cov: https://pytest-cov.readthedocs.io
.. _gitlab: https://gitlab.com/gemseo/dev/gemseo
.. _pyperf: https://pyperf.readthedocs.io
.. _profiler: https://docs.python.org/3/library/profile.html
.. _develop branch: https://gitlab.com/gemseo/dev/gemseo/-/tree/develop
.. _develop documentation: https://gemseo.readthedocs.io/en/develop/index.html

.. _dev:

Developer information
=====================

This page contains information about |g| development and how to contribute to it.
The source code of |g| is available on `gitlab`_,
this is the place where code contributions shall be submitted.
Note also that it is required to accompany any contribution with a Developer Certificate of Origin,
certifying that the contribution is compatible with |g| software licence.

We aim to have industrial quality standards for software,
in order to:

* have a good and running software,
* be confident about it,
* facilitate collaborative work with the team,
* facilitate distribution to our partners.

To meet these goals,
we use best practices described below,
these practices are not optional,
they are fully part of the development job.

Quick start
-----------

First time setup:

* Create a main environment with `tox-conda`_, see :ref:`requirements`.
* Then on Linux

  .. code-block:: console

     tox -e dev

* or, on Windows

  .. code-block:: console

     tox -e dev-win

Run the tests

* on Linux

   .. code-block:: console

      tox -e py27,py37

* on Windows

   .. code-block:: console

      tox -e py27-win,py37-win

Environments
------------

We use `tox`_ for handling the environments related to the development,
be it for coding,
testing,
documenting,
checking ...
This tool offers a simplified
and high level interface to many ingredients used in development,
while providing reproducible
and isolated outcomes that are
as much independent as possible of the platform
and environment from which it is used.

All the settings of `tox`_ are defined in the file :file:`tox.ini`.
It contains the descriptions of the environments:

* version of Python to use,
* packages to install,
* environment variables to set or pass from the current environment,
* commands to execute.

All the directories created by `tox`_
are stored under :file:`.tox` next to :file:`tox.ini`.
In particular,
:file:`.tox` contains the environments
in directories named after the environments.

.. _requirements:

Requirements
++++++++++++

We use `tox`_ and `tox-conda`_ along with `anaconda`_,
you need to have them installed before moving along.
Create an Anaconda environment with `tox-conda`_:

.. code-block:: console

   conda create -n tox python=3.8 pip
   conda activate tox
   pip install tox-conda
   conda deactivate
   conda activate tox

The last two commands are necessary
to have the :command:`tox` executable available
in the just created environment.

.. _matlab_requirements:

MATLAB requirements
~~~~~~~~~~~~~~~~~~~

The MATLAB Python API is not defined as a dependency of |g|,
it has to be installed manually in the Anaconda environment.
The Python API usually needs to be built
and installed since it is not done by default during the MATLAB installation.

For testing with `tox`_,
set the environment variable :envvar:`MATLAB_PYTHON_WRAPPER`
to point to the path to a ``pip`` installable version of the MATLAB Python API,
with eventually a conditionnal dependency on the Python version:

.. code-block:: console

   export MATLAB_PYTHON_WRAPPER="<path or URL to MATLAB Python API package> ; python_version<'3.9'"


pSeven requirements
~~~~~~~~~~~~~~~~~~~

Like the MATLAB Python API, the pSeven one shall be installed manually in the Anaconda environment.

For testing with `tox`_,
set the environment variable :envvar:`PSEVEN_PYTHON_WRAPPER`
to point to the path to a ``pip`` installable pSeven Python API.
Set the environment variable :envvar:`DATADVD_LICENSE_FILE`
for the pSeven license.

How to use tox
++++++++++++++

The environments created by `tox`_
and their usage are described in the different sections below.
In this section we give the common command line usages and tips.

Create the environment named *env* and run its commands with:

.. code-block:: console

   tox -e env

The first invocation of this command line may take some time to proceed,
further invocations will be faster because `tox`_ shall not create a new
environment from scratch unless,
for instance,
some of the dependencies have been modified.

You may run (sequentially) more than one environment with:

.. code-block:: console

   tox -e env,env2,env3

Recreate an existing environment with:

.. code-block:: console

   tox -e env -r

This may be necessary
if an environment is broken
or if `tox`_ cannot figure out
that a dependency has been updated
(for instance with dependencies defined by a git branch).

We use `tox`_ with `anaconda`_ environments,
activate the `tox`_ environment named *env* with:

.. code-block:: console

   conda activate .tox/env

.. note::

  An Anaconda environment created by `tox`_ has no Anaconda name,
  thus :command:`conda` cannot activate it by its name as usual.

Activating environments may be useful for instance
to investigate a particular issue that happens
in a specific environment and not others.
You may modify an activated environment
just like any other `anaconda`_ environment,
in case of trouble just recreate it.
Be aware that the environment variables defined in :file:`tox.ini`
will not be set with a manually activated environment.

Show available environments with:

.. code-block:: console

   tox -a

Use a double ``--`` to pass options to an underlying command,
for example:

.. code-block:: console

   tox -e env -- ARG1 --opt1

Not all the environments allow this feature,
see the specific topics below for more information.

.. note::

  On Windows,
  the environment names shall be suffixed with *-win*.
  This is a limitation of `tox`_.

Coding
------

Coding environment
++++++++++++++++++

Create the development environment:

* On Linux

  .. code-block:: console

     tox -e dev

* On Windows

  .. code-block:: console

     tox -e dev-win

This will create an environment with:

* |g| installed in `editable mode`_,
* all the |g| dependencies,
* tools used for development
  (debugging,
  code checking
  and formatting)
* git settings (see :ref:`git`)

With an editable installation,
|g| appears installed in the development environment created by `tox`_,
but yet is still editable in the source tree.

.. note::

  You do not need to activate this environment for contributing to |g|.

.. _coding-style:

Coding Style
++++++++++++

We use the `pep8`_ convention.
The formatting of the source code is done
with `isort`_ and `black`_.
The code is systematically checked with `flake8`_
and on demand with `pylint`_.
A git commit shall have no flake8 violations.

Except for *pylint*,
all these tools are used:

* either automatically by the git hooks when creating a commit,
* or manually by running :command:`tox -e style`.

Use :command:`tox -e pylint` to run `pylint`_.

Coding guidelines
+++++++++++++++++

String formatting
  Do not format strings with **+**
  or with the old `printf-style
  <https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`_
  formatting:
  format strings with :func:`format` (`documentation
  <https://docs.python.org/3/library/stdtypes.html#str.format>`_).

Logging
  Loggers shall be defined at module level and named after the module with::

    LOGGER = logging.getLogger(__name__)

  This means that logger names track the package/module hierarchy,
  and it’s intuitively obvious where events are logged
  just from the logger name.

Error messages
  Error messages will be read by humans:
  they shall be explicit and valid sentences.

.. _git:

Git
---

Workflow
++++++++

We use the `gitflow`_ for managing git branches.
For the daily work,
this basically means that evolutions of |g|
are done in feature branches created from the `develop branch`_
and merged back into it when finished.

Git hooks
+++++++++

When a commit is being created,
git will perform predefined actions:

* remove the trailing whitespaces,
* fix the end of files,
* check toml, yaml and json files are well formed,
* check that no big file is committed,
* check bad symbolic links,
* check or fix some of the python docstrings formatting,
* fix the Python import order,
* fix the Python code formatting,
* check for Python coding issues (see :ref:`coding-style`),
* check the commit message (see :ref:`commit-msg`),
* check for forbidden :func:`print` usage,
* check for misused :mod:`logging` formatting,
* check for :file:`.rst` files issues.
* check or fix license headers

Those actions will eventually modify the files about to be committed.
In this case your commit is denied
and you have to check that the modifications are OK,
then add the modifications to the commit staged files
before creating the commit again.

.. _commit-msg:

Commit message
++++++++++++++

We use `conventional commits`_ for writing clear
and useful git commit messages.
The commit message should be structured as follows:

.. code-block:: shell

  <type>(optional scope): <description>

  [optional body]

  [optional footer(s)]

Where:

* *<type>* defines the type of change you are committing

    * feat: A new feature
    * fix: A bug fix
    * docs: Documentation only changes
    * style: Changes that do not affect the meaning of the code
    * refactor: A code change that neither fixes a bug nor adds a feature
    * perf: A code change that improves performance
    * test: Adding missing tests or correcting existing tests
    * build: Changes that affect the build system or external dependencies
    * ci: Changes to our CI configuration files and scripts
* *(optional scope)* provide additional contextual information and is contained
  within parentheses
* *<description>* is a concise description of the changes,
  imperative,
  lower case
  and no final dot
* *[optional body]* with the motivation for the change and contrast this with
  previous behavior
* *[optional footer(s)]* with information about Breaking Changes and reference
  issues that this commit closes

From the ``.tox/dev`` environment,
you may use `commitizen`_ to easily create commits that follow `conventional commits`_.
Run it and and let it drive you through with:

.. code-block:: console

   cz commit

Commit message examples:

.. code-block:: shell

  feat(study): open browser when generating XDSM

.. code-block:: shell

  fix(scenario): xdsm put back filename arg

Commit best practices
+++++++++++++++++++++

The purpose of these best practices is to ease
the code reviews,
commit reverting (rollback changes)
bisecting (find regressions),
branch merging or rebasing.

Write atomic commits
  Commits should be logical,
  atomic units of change that represent a specific idea
  as well as its tests.
  Do not rename and modify a file in a single commit.
  Do not combine cosmetic and functional changes in a single commit.

Commits history
   Try to keep the commit history as linear as possible
   by avoiding unnecessary merge commit.
   When possible, prefer rebasing over merging,
   git can help to achieve this with:

   .. code-block:: console

      git config pull.rebase true
      git config rerere.enabled true

Rework commit history
  You may reorder, split or combine the commits of a branch.
  Such history modifications shall be done
  before the branch has been pushed to the main repository.

Tests
    Avoid commits that break tests,
    only push a branch that passes all the tests
    for py27 and py38 on your machine.

Testing
-------

Testing is mandatory in any engineering activity,
which is based on trial and error.
All developments shall be tested:

* this gives confidence to the code,
* this enables code refactoring with mastered consequences: tests must pass!

Tests writing guidelines
++++++++++++++++++++++++

We use `pytest`_ for writing and executing all the |g| tests.
Older tests were written with the unittest module from the Python standard library
but newer tests shall be written with `pytest`_.

Logic
    Follow the
    `Arrange, Act, Assert, Cleanup <https://docs.pytest.org/en/stable/fixture.html#what-fixtures-are>`_
    steps by splitting the testing code accordingly.
    Limit the number of assertions per test functions in a consistent manner
    by writing more test functions.
    Use the
    `pytest fixtures <https://docs.pytest.org/en/stable/fixture.html>`_
    features.
    Tests shall be independent,
    any test function shall be executable alone.

Logging
    Do no create loggers in the tests,
    instead let `pytest`_ manage the logging
    and use its builtin `features <https://docs.pytest.org/en/stable/logging.html>`_.
    Some pytest logging settings are already defined in :file:`pyproject.toml`.

Messages
    The information provided to the user by the error
    and logging messages
    shall be correct.
    Use the
    `caplog fixture <https://docs.pytest.org/en/stable/logging.html#caplog-fixture>`_
    for checking the logging messages.
    Use
    `pytest.raises <https://docs.pytest.org/en/stable/assert.html#assertraises>`_
    for checking the error messages.

Skipping under Windows
    Use the `pytest`_ marker like:

    .. code-block:: python

        @pytest.mark.skip_under_windows
        def test_foo():

Validation of images
    For images generated by matplotlib,
    use the :func:`image_comparison` decorator provided by the
    `matplotlib testing tools <https://matplotlib.org/stable/devel/testing.html#writing-an-image-comparison-test>`_.
    See :file:`tests/post/dataset/test_surfaces.py` for an example.

Executing tests
+++++++++++++++

For Python 2.7,
run the tests with:

.. code-block:: console

   tox -e py27

Replace py27 by py38 for testing with Python 3.8,
you may run the tests accordingly with Python 2.7,
3.6,
3.7,
3.8
and 3.9.
With `tox`_,
you can pass options to `pytest`_ after ``--``,
for instance:

.. code-block:: console

   tox -e py38 -- --last-failed --step-wise

Run the tests for several Python versions with for instance (on Linux):

.. code-block:: console

   tox -e py27,py38

Under Windows,
append ``-win`` to the names of the test environments,
for instance:

.. code-block:: console

   tox -e py27-win,py38-win

Tests coverage
++++++++++++++

For a selected python version (for instance Python 3.8),
get the coverage information with:

.. code-block:: console

   tox -e py38 -- --cov --cov-report=term

See `pytest-cov`_ for more information.

Documentation
-------------

The documentation of the `develop branch`_
is available online: `develop documentation`_.

Generating the doc
++++++++++++++++++

The documentation is written with `sphinx`_.
On Linux, generate the documentation with:

.. code-block:: console

   tox -e doc

Under Windows,
append ``-win`` to the names of the this environment.

Pass options to ``sphinx-build`` after ``--``,
for instance:

.. code-block:: console

   tox -e doc -- -vv -j2

Check the links in the generated documentation with:

.. code-block:: console

   tox -e doc-linkchecker

.. note::

   doc-linkchecker does not work on windows.

Writing guidelines
++++++++++++++++++

Documenting classes, functions, methods, attributes, modules, etc... is mandatory.
End users and developers shall not have to guess the purpose of an API
and how to use it.

Style
~~~~~

Use the Google Style Docstrings format for documenting the code.
This :ref:`example module` shows how to write such docstrings.
Older docstrings use the legacy *epydoc* docstrings format
which is visually dense and hard to read.
They will be overhauled progressively.

Type hints
~~~~~~~~~~

For functions and methods,
write type hints with inlined comments as shown in :ref:`example module`
(this is compatible with both Python 2.7 and 3.6+).
The type hints are used when generating the functions and methods documentation,
they will also be used gradually to check and improved the code quality
with the help of a type checker like `mypy`_.

Functions and methods arguments shall use `standard duck typing`_.
In practice, use :class:`Iterable` or :class:`Sequence`
instead of :class:`List` when appropriate,
similarly for :class:`Mapping` instead of :class:`Dict`.
For ``*args`` and ``**kwargs`` arguments,
use only the value types with no container.

Return types shall match exactly the type of the returned object.

Type hinting may cause circular imports,
if so, use the special constant :attr:`TYPE_CHECKING`
that's :code:`False` by default
and :code:`True` when type checking:

.. code::

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from gemseo.api import create_discipline

Linefeeds
~~~~~~~~~

Use `semantic linefeeds`_
by starting a new line at the end of each sentence,
and splitting sentences themselves at natural breaks between clauses,
a text file becomes far easier to edit and version control.
You can have a look at the current page's source for instance.

Example
~~~~~~~

Have a look to the uncertainty module
for an example of proper code documentation.

Versioning
----------

We use `semantic versioning`_ for defining the version numbers of |g|.
Given a version number MAJOR.MINOR.PATCH,
we increment the:

1. MAJOR version when we make incompatible API changes,
2. MINOR version when we add functionality in a backwards compatible manner, and
3. PATCH version when we make backwards compatible bug fixes.

Benchmarking
------------

Use `pyperf`_ to create valid benchmark,
mind properly tuning the system for the benchmark (see the docs).

Profiling
---------

The Python standard library provides a `profiler`_,
mind using it with controlled system like for benchmarking.
The profiling data could be analyzed with one of these tools:

- `snakeviz <https://jiffyclub.github.io/snakeviz>`_
- `kcachegrind <https://kcachegrind.github.io/html/Home.html>`_,
  after having converted the profiling data with
  `pyprof2calltree <https://github.com/pwaller/pyprof2calltree/>`_
