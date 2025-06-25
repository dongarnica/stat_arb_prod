#!/usr/bin/env bash

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade black isort mypy pytest
