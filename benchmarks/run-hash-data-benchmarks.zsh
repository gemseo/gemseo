#! /usr/bin/env zsh

suffix=${1:?"The output suffix is required"}

for keys_nb in 1 10 100; do
for items_nb in 1 10 100; do
	python \
		benchmarks/hash_data.py \
		--append hash-data.json.$suffix \
		--items_nb $items_nb \
		--keys_nb $keys_nb
done
done
