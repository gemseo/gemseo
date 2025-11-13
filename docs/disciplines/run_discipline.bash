#!/bin/bash
set -e

echo "Parsed inputs.txt file"
source "inputs.txt"
echo "a="$a
echo "b="$b

echo "executing simulation..."
c=$(perl -e "print $a*$a+$b*$b")

echo "Done."
echo "Computed output : c = a**2+b**2 = "$c
echo "c="$c>"outputs.txt"

echo "Wrote output file 'outputs.txt'"
