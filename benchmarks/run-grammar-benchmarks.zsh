#! /usr/bin/env zsh

benchmark_suffix=${1:?"The suffix of the grammar benchmark"}
grammar_type=${2:?"The grammar type is required"}

for with_ndarrays_only in "--with-ndarrays-only" ""; do
for keys_nb in 1 10 100; do
for items_nb in 1 10 100; do
for depth in 0; do
#for depth in 0 2; do
	python \
		benchmarks/grammar_$benchmark_suffix.py \
		--append grammar_$benchmark_suffix.$grammar_type.json \
		--items_nb $items_nb \
		--keys_nb $keys_nb \
		--depth $depth \
		--grammar-type $grammar_type \
		$with_ndarrays_only
done
done
done
done
