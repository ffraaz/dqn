#!/usr/bin/env bash

set -eux

black --check .
pylint dqn.py test_dqn.py
mypy dqn.py test_dqn.py
pytest
