#!/bin/bash
# Run tests using pytest
uv run --group test pytest tests/ -vvv --capture=tee-sys $@
