- ``ExecutionStatistics.record``: now use ``ExecutionStatistics.record_execution`` and ``ExecutionStatistics.record_linearization``.
- ``ExecutionStatistics.n_calls``:  now use ``ExecutionStatistics.n_executions``.
- ``ExecutionStatistics.n_calls_linearize``: now use ``ExecutionStatistics.n_linearizations``.
- ``ExecutionStatus.run``: now use  ``ExecutionStatistics.handle``.
- ``ExecutionStatus.linearize``: now use ``ExecutionStatistics.handle``.
- The integer type of a design variable is now ``int64`` instead of ``int32``,
  this matches the type of python ``int`` and the default type of integer in NumPy 2 which we will support later.
- ``base_cache.DATA_COMPARATOR``: now use ``BaseCache.compare_dict_of_arrays``.
