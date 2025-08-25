#!/bin/bash
# Upgrade the dependencies including those from Git sources
uv lock -U
# Run tests using pytest
uv run --group test pytest tests/
