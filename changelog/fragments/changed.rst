For performance and maintainability reasons,
``DisciplineData`` no longer handles ``DataFrame`` data in a special way.
Please refer to the ``DataFrame`` example in the documentation for handling ``DataFrame`` data.
API changes:
    - ``DisciplineData.__init__`` no longer has the arguments ``input_to_namespaced ``and ``output_to_namespaced``.
    - When the name ``x`` does not exist in a ``DisciplineData`` object, ``DisciplineData['x']`` no longer tries to return the value bound to the key prefixed with a namespace like ``ns:x``.
    - The execution status ``LINEARIZE`` has been renamed to ``LINEARIZING``.
API changes from old to new:
  - set_cache_policy: set_cache with arguments passed to CacheFactory.create
  - ``set_linear_relationships`` now takes ``input_names`` first before ``output_names``.
  - ``Scenario._run_algorithm``: ``Scenario._execute``
  - The classes in the module ``execution_sequences`` has been split into modules.
  - ``ExecSequence``: ``BaseExecSequence``
  - ``ExecSequence.START_STR``: ``BaseExecSequence._PREFIX``
  - ``ExecSequence.END_STR``: ``BaseExecSequence._SUFFIX``
  - ``ExecSequence.enabled()``: ``BaseExecSequence.is_enabled``
  - ``CompositeExecSequence``: ``BaseCompositeExecSequence``
  - ``ExtendableExecSequence``: ``BaseExtendableExecSequence``
  - ``SerialExecSequence``: ``SequentialExecSequence``
  - ``initialize_grammars``: ``_initialize_grammars``
API changes removed:
  - The ``VIRTUAL`` execution status.
  - The ``RE_EXEC_POLICY`` for rexecuting disciplines.
  - ``create_scenario_result``: use ``Scenario.get_results`` instead.
  - ``ExecSequenceFactory``: import and instantiate directly instead.
    - ``Defaults`` removed the method ``rename``
    - ``DisciplineData`` and ``Defaults``:

        - removed the method ``restrict``
        - remove the argument ``exclude`` from the method ``update``
        - remove the argument ``with_namespace`` from the method ``copy``
  - The type of the grammar of a discipline is now set from the class attribute ``Discipline.default_grammar_type``,
    instead of being set a initialization.
  - The automatic search for a ``JSONGrammar`` file a discipline is now set from the class attribute ``Discipline.auto_detect_grammar_files``,
    instead of being set a initialization.
  - The type of the cache of a ``Discipline ``is now set from the class attribute ``Discipline.default_cache_type``,
    instead of being set a initialization. It can be changed after initialization with the method
    ``Discipline.set_cache``. The initialization argument ``cache_file_path`` has been removed too.
  - ``activate_cache`` has been removed: use ``default_cache_type = Discipline.CacheType.NONE`` to deactivate the cache.
  - The initialization arguments ``input_grammar_path`` and ``output_grammar_path`` of a ``Discipline`` have been removed,
    use ``GRAMMAR_DIRECTORY`` instead.
    - ``ScipyLinalgAlgos``:

        - The option ``tol`` has been renamed to ``rtol``.
        - The default value of the option ``atol`` is ``1e-12`` instead of ``None``.

Different import paths due to minor packaging adjustments (old to new):
  - ``from gemseo.core.chain import MDOChain`` to ``from gemseo.core.chains.chain import MDOChain``

The class ``MDODiscipline`` is renamed ``Discipline``.
