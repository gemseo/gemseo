#! /usr/bin/env zsh

suffix=${1:?"The output suffix is required"}

for tolerance in 0 1; do
for keys_nb in 1 10 100; do
for items_nb in 1 10 100; do
for depth in 0 2; do
	python \
		benchmarks/compare_data.py \
		--append compare.json.$suffix \
		--items_nb $items_nb \
		--keys_nb $keys_nb \
		--depth $depth \
		--tolerance $tolerance
done
done
done
done
