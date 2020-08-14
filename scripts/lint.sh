#!/usr/bin/env bash

set -e
set -x

flake8 pytest_dl tests
black pytest_dl tests --check
isort pytest_dl tests --check-only