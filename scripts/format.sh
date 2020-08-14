#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place pytest_dl tests --exclude=__init__.py
black pytest_dl tests
isort pytest_dl tests