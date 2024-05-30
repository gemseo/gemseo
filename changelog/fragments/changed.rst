API changes:
    - ``DisciplineData.__init__`` no longer has the arguments ``input_to_namespaced ``and ``output_to_namespaced``.
    - When the name ``x`` does not exist in a ``DisciplineData`` object, ``DisciplineData['x']`` no longer tries to return the value bound to the key prefixed with a namespace like ``ns:x``.
