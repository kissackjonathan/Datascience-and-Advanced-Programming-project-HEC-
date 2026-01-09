# Pre-commit Configuration

This folder contains all configuration files and scripts for the pre-commit system.

## Structure

```
pre-commit/
├── README.md                    # This file
├── .pre-commit-config.yaml      # Active configuration
├── hooks/
│   └── pre-commit              # Active git hook
└── scripts/
    └── help-message.sh         # Help script for errors
```

## How it Works

Pre-commit runs automatically before each `git commit` and verifies:

1. **Black**: Automatic Python code formatting
2. **isort**: Import sorting
3. **flake8**: Code style verification
4. **Basic hooks**: Trailing whitespace, EOF, YAML, etc.

## Configuration

All active files are now centralized in the `pre-commit/` folder:
- `/pre-commit/.pre-commit-config.yaml` - Active configuration
- `/pre-commit/hooks/pre-commit` - Active git hook
- Git is configured with `core.hooksPath = pre-commit/hooks`

## Useful Commands

```bash
# Run all checks manually
pre-commit run --all-files --config pre-commit/.pre-commit-config.yaml

# Reinstall hooks
pre-commit install --config pre-commit/.pre-commit-config.yaml

# Skip pre-commit (not recommended)
git commit --no-verify
```

## Documentation

https://pre-commit.com/
