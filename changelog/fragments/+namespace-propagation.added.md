- `gemseo.discipline.propagate_namespace` propagates a namespace forward along the discipline coupling graph, namespacing every
  input and output affected by the seed variables while leaving untouched the variables that are neither seeds nor
  produced inside the reached set. Its `excluded_names` argument marks variables considered identical to those of the
  original, non-namespaced group: the propagation is not carried out along the coupling edges that only carry such
  variables, so a discipline reached only through them is left completely untouched and must be instantiated only
  once, shared by both groups. A discipline whose outputs are all excluded is left out and untouched for the same
  reason, even when it is otherwise reached.
