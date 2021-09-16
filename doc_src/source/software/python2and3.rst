..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard, Jean-Christophe Giret

.. _python2and3:

Python 2 and Python 3 migration and cross-compatibility
=======================================================

Objective and scope
-------------------

This section describes the migration of Python 2 code using |g| 1.3.2 and lower to |g| 2.0.0, as well as the strategy for the cross compatibility with Python 2 and Python 3 in |g| and for the code using |g|.

.. _py2_user_migration:

If you use |g| with Python 2
----------------------------

If you are a |g| user with Python 2, please read carefully the next warning to
migrate your existing code to be compliant with the |g| 2.0.0 API.

.. warning::
    If your current code is aimed to be run only with a Python 2 interpreter, you can
    migrate your code by adding the :code:`from __future__ import unicode_literals`
    as a first import on all the files defining strings which will be latter
    passed in |g| objects. All the strings created after this statement will
    be encoded in ``unicode``. Note that the migrated code will **not** be
    compatible with |g| 1.3.2 and lower.

A common error message that you may encounter is a grammar error when passing
string options to a |g| object due to an incorrect string encoding. For
instance, the following error message is obtained when string options with an
incorrect encoding are passed as options for the execution of an MDO scenario:


.. code:: console

  ERROR - 17:11:09 : Invalid data in : MDOScenario_input
  ', error : data.algo must be string
  Traceback (most recent call last):
    File "plot_mdo_scenario.py", line 85, in <module>
      scenario.execute({"algo": "L-BFGS-B", "max_iter": 100})
    File "/home/distracted_user/workspace/gemseo/gemseo/core/discipline.py", line 586, in execute
      self.check_input_data(input_data)
    File "/home/distracted_user/workspace/gemseo/gemseo/core/discipline.py", line 1243, in check_input_data
      raise InvalidDataException("Invalid input data for: " + self.name)
  gemseo.core.grammar.InvalidDataException: Invalid input data for: MDOScenario


In this case, the "L-BFGS-B" string is not recognized as a correct input, as it
is not encoded in unicode. The addition of :code:`from __future__ import
unicode_literals` at the beginning of the script would correct the issue.


.. _py23_migration_py2_code:

Migration of Python 2 code using |g|
------------------------------------

Starting with |g| 2.0.0, and whatever the version of the Python interpreter
used (2.7 or 3.x), all the strings must be encoded in ``unicode``. As
mentionned in :ref:`py23_string_unicode`, in Python 3, strings are by default
encoded in ``unicode``, while they are encoded in ``str`` in Python 2.

It implies that the code using |g| (i.e. all the code using the |g| API or
the |g| core objects) written for |g| 1.3.2 and lower running under a Python
2 interpreter **will likely fail** with |g| 2.0.0 and higher. This code must
consequently be migrated, and three options may be considered:

    - If the code is aimed to be run only with a Python 2 interpreter, the user
      can migrate its code by adding the :code:`from __future__ import
      unicode_literals` as a first import on all the files defining strings
      which will be latter passed in |g| objects. All the strings created
      after this statement will be encoded in ``unicode``. Note that the
      migrated code will **not** be compatible with |g| 1.3.2 and lower.
    - If the code is aimed to be run only with Python 3 interpreter, then the
      user has only to migrate its existing code from Python 2 to Python 3.
      Migration information can be found in :ref:`py3_migration`.
    - If the code is aimed to be run with a Python 2 **and** a Python 3
      interpreter, first the code must be migrated to Python 3. Then, the
      migrated code must be pasteurized and specific imports must be added to
      have a Python 2 and 3 cross-compatibility. Specific information are given
      in :ref:`py23_cross_compat`.

.. _py23_string_unicode:

Strings
-------
One of the main differences between python 2 and 3 is the string encoding, see `Strings <https://portingguide.readthedocs.io/en/latest/strings.html>`_.
The following paragraph is a quote from this page:

From a developer's point of view, the largest change in Python 3
is the handling of strings.
In Python 2, the ``str`` type was used for two different kinds of values *text* and *bytes*, whereas in Python 3, these are separate and incompatible types.

*

    **Text** contains human-readable messages, represented as a sequence of
    Unicode codepoints.
    Usually, it does not contain unprintable control characters such as ``\0``.

    This type is available as ``str`` in Python 3, and ``unicode``
    in Python 2.

    In code, we will refer to this type as ``unicode`` a short, unambiguous
    name, although one that is not built-in in Python 3.
    Some projects refer to it as ``six.text_type``
    (from the `six library <https://github.com/benjaminp/six>`_).

*

    **Bytes** or *bytestring* is a binary serialization format suitable for
    storing data on disk or sending it over the wire. It is a sequence of
    integers between 0 and 255.
    Most data images, sound, configuration info, or *text* can be
    serialized (encoded) to bytes and deserialized (decoded) from
    bytes, using an appropriate protocol such as PNG, VAW, JSON
    or UTF-8.

    In both Python 2.6+ and 3, this type is available as ``bytes``.

Ideally, every "stringy" value will explicitly and unambiguously be one of
these types (or the native string, below).
This means that you need to go through the entire codebase, and decide
which value is what type.
Unfortunately, this process generally cannot be automated.

We recommend replacing the word "string" in developer documentation
(including docstrings and comments) with either "text"/"text string" or
"bytes"/"byte string", as appropriate.



.. _py3_migration:

Migration from Python 2 to Python 3
-----------------------------------
First, please read the official documentation `Porting Python 2 Code to Python 3 <https://docs.python.org/3/howto/pyporting.html>`_.

The first requirement is to have a good code coverage for the code, otherwise, detecting the portability issues is very hard.

The `Futurize <http://python-future.org/automatic_conversion.html>`_ utility should be used to automatically make Python 3 compatible code.

Then, manual modifications of the code are usually required to make all the unit tests work.

A series of changes were made to builtin types, which classically impact most of the code. Here is a non exhaustive list:

dict
~~~~
dict.iteritems() has changed to dict.items().
dict.iterkeys() has changed to dict.keys().

dict.keys() were doing a copy of the keys, which is needed to iterate on a dict and delete some elements.
To do this, use now :code:`list(iter(dict.keys()))`.

Dictionary keys are now a specific :code:`dict_keys` objects.

OrderedDict
~~~~~~~~~~~

Since python 3.6, dict is ordered, as the former OrderedDict, but much faster. To automatically switch, use  :meth:`~gemseo.utils.py23_compat.OrderedDict`, which points to dict from python 3.6.

str
~~~
ascii strings in Python 2 refered to either bytes of a string. Now in Python 3, str are unicode strings and bytes are separate types. See the first section for details.

xrange
~~~~~~
xrange was moved to range, which is now an iterator.
To create a list, use list(range).
To use xrange in Python 3 like in Python 2, use :meth:`~gemseo.utils.py23_compat.xrange`

long
~~~~
In python 3, int are long. long do not exist any more.

next
~~~~
next is now a special method __next__ in Python 3, because it is not recommended to manually get the next item of an iterator, instead of performing a loop.
For cross compatibility, you may use :meth:`~gemseo.utils.py23_compat.next`

.. _py23_cross_compat:

Generating cross compatible code Python 2 and Python 3
------------------------------------------------------

Then, the generated code must work also in Python 2, the Pasteurize utility is used `Pasteurize <http://python-future.org/automatic_conversion.html#pasteurize-py3-to-py2-3>`_ .

Also, see the useful `six <https://six.readthedocs.io>`_ library that contains compatible Python 2 and 3 functions.

Remaining incompatibilities
---------------------------

Specific incompatibilities are handled in the :mod:`~gemseo.utils.py23_compat`   module, please see the methods.

.. automodule:: gemseo.utils.py23_compat
   :noindex:


Known issues
------------

H5py
~~~~

The h5py library handles ASCII characters, while strings are encoded in unicode in Python3.
Therefore, specific lines are needed to encode or decode strings when using h5py.

Also, when passing a string array to h5py, use the :meth:`~gemseo.utils.py23_compat.string_array` to handle string types appropriately.


Raw strings
~~~~~~~~~~~

see `Escaping <http://python-future.org/automatic_conversion.html#known-limitations>`_

Strings containing \\U produce a SyntaxError on Python 3.
:code:`s = 'C:\Users'`.
Python 2 expands this to :code:`s = 'C:\\Users'`, but Python 3 requires a raw prefix (r'...'). This also applies to multi-line strings (including multi-line docstrings).
