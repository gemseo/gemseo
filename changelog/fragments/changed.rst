For performance and maintainability reasons,
``DisciplineData`` no longer handles ``DataFrame`` data in a special way.
Please refer to the ``DataFrame`` example in the documentation for handling ``DataFrame`` data.
API changes:
    - ``DisciplineData.__init__`` no longer has the arguments ``input_to_namespaced ``and ``output_to_namespaced``.
    - When the name ``x`` does not exist in a ``DisciplineData`` object, ``DisciplineData['x']`` no longer tries to return the value bound to the key prefixed with a namespace like ``ns:x``.
    - ``Defaults`` removed the method ``rename``
    - ``DisciplineData`` and ``Defaults``:

        - removed the method ``restrict``
        - remove the argument ``exclude`` from the method ``update``
        - remove the argument ``with_namespace`` from the method ``copy``
