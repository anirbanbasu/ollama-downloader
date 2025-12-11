conf_dir := quote(invocation_dir() / "conf")
conf_backup_dir := quote(invocation_dir() / "conf_backup")
native_file := "od-native"
product_name := "ollama-downloader"

# Install minimal project dependencies in a virtual environment
install:
    @echo "Installing project dependencies in a virtual environment..."
    @uv sync --no-dev
    @echo "Project dependencies installed."

# Install all project dependencies in a virtual environment
install-all:
    @echo "Installing all project dependencies in a virtual environment..."
    @uv sync --all-groups
    @echo "All project dependencies installed."

# Install pre-commit hooks using 'prek'
install-pre-commit-hooks:
    @echo "Installing pre-commit hooks using prek..."
    @prek install
    @echo "Pre-commit hooks installed."

# Update pre-commit hooks using 'prek'
pre-commit-update:
    @echo "Updating pre-commit hooks using prek..."
    @prek auto-update
    @echo "Pre-commit hooks updated."

# Upgrade project dependencies using 'uv'
upgrade-dependencies:
    @echo "Upgrading project dependencies..."
    @uv lock -U
    @echo "Dependencies upgraded."

# Bump the patch version of the project using 'uv'
bump-patch:
    @echo "Updating current project version: $(uv version --short)"
    @uv version --bump patch
    @echo "Updated project to: $(uv version --short)"

# Format the code
format:
    @echo "Formatting code..."
    @uv run ruff format
    @uv run ruff check --fix --fix-only
    @echo "Code formatted."

# Run the type checker
type-check:
    @echo "Running type checker..."
    @uv run ty check
    @echo "Type checking complete."

export MCP_SERVER_TRANSPORT := "streamable-http"
# Run tests with coverage reporting
test-coverage:
    @echo "Running tests with coverage..."
    # Check if configuration directory exists and back it up, if it does
    @rm -fR {{conf_backup_dir}}
    @[ -d {{conf_dir}} ] && mv {{conf_dir}} {{conf_backup_dir}} && echo "Configuration backed up." || echo "Configuration does not exist, skipping backup."
    @uv run --group test coverage run -m pytest --capture=tee-sys -vvv tests/
    # Remove the configuration directory to simulate a fresh environment
    @rm -fR {{conf_dir}}
    # Check if configuration backup directory exists and restore from it, if it does
    @[ -d {{conf_backup_dir}} ] && mv {{conf_backup_dir}} {{conf_dir}} && echo "Configuration restored." || echo "Configuration did not exist, skipping restore."
    @uv run coverage report -m
    @echo "Test coverage complete."

# Run tests with profiling but no coverage
test-with-profile:
    @echo "Running tests with profiling..."
    @uv run --group test pytest --capture=tee-sys -vvv tests/ --profile --profile-svg
    @echo "Tests with profiling complete."

# Compile native binary with Nuitka
build-native-nuitka:
    @echo "Building native binary with Nuitka..."
    # Remove the existing native file
    @rm -f {{native_file}} || true
    @uv run python -m nuitka \
            --onefile \
            --standalone \
            --clean-cache="all" \
            --disable-cache="all" \
            --remove-output \
            --static-libpython="yes" \
            --noinclude-pytest-mode="nofollow" \
            --product-name={{product_name}} \
            --product-version=$(sed -n 's/^version = "\([0-9.]*\).*/\1/p' pyproject.toml | head -n 1) \
            --output-file={{native_file}} \
            --macos-prohibit-multiple-instances \
            --macos-app-name={{product_name}} \
            src/ollama_downloader/cli.py
    @echo "Native binary built."
