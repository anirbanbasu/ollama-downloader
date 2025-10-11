#!/bin/bash
NATIVE_FILE="od-native"
PRODUCT_NAME="ollama-downloader"
# Remove the existing native file
rm $NATIVE_FILE
uv run python -m nuitka \
		--onefile \
		--standalone \
		--clean-cache="all" \
		--disable-cache="all" \
		--remove-output \
		--static-libpython="yes" \
		--noinclude-pytest-mode="nofollow" \
		--product-name=$PRODUCT_NAME \
		# Extract version from pyproject.toml but only the numerical parts, e.g., "1.2.3.rc1" -> "1.2.3"
		--product-version=$(sed -n 's/^version = "\([0-9.]*\).*/\1/p' pyproject.toml | head -n 1) \
		--output-file=$NATIVE_FILE \
		--macos-prohibit-multiple-instances \
		--macos-app-name=$PRODUCT_NAME \
		src/ollama_downloader/cli.py
