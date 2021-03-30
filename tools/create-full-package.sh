#! /usr/bin/env bash

set -euox pipefail

# this script creates a redistributable archive

# do not include tests cruft
git clean -fdx tests

version=$(git describe --tags --dirty --always)

tar -cjv \
	doc \
	tests \
	environment-py?.yml \
	README.rst \
	-C .tox/dist . \
	-f gemseo-${version}-full-package.tbz
