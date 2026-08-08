#!/usr/bin/env bash
# Installs the tool-independent commit-msg guard. Called by `just setup`,
# so a cold clone is protected immediately.
set -euo pipefail
hook="$(git rev-parse --git-path hooks)/commit-msg"
cat > "$hook" <<'EOF'
#!/usr/bin/env bash
if grep -qiE 'co-authored-by:[[:space:]]*claude|generated with \[?claude code|🤖' "$1"; then
  echo "commit-msg: Claude attribution is not permitted in this repository." >&2
  exit 1
fi
EOF
chmod +x "$hook"
echo "installed: $hook"
