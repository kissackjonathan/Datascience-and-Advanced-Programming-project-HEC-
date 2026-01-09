#!/bin/bash
# Custom pre-commit failure helper
# Displays user-friendly troubleshooting tips when hooks fail

cat << 'EOF'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-COMMIT HOOK DETECTED ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Common fixes based on error type:

  Black (formatting):
     Files were auto-formatted. Re-stage and commit:
     git add . && git commit

  Flake8 (code quality):
     Check specific errors above
     Fix code style issues, then retry commit

  Large file blocked (>500KB):
     Move to data/ directory (gitignored)
     Or add download link to README.md

  YAML syntax error:
     Check environment.yml or .pre-commit-config.yaml
     Validate at: https://www.yamllint.com

  Merge conflict markers:
     Resolve conflicts before committing
     Remove <<<<<<<, =======, >>>>>>> markers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manual run: pre-commit run --all-files
Skip hooks (not recommended): git commit --no-verify
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF

exit 0
