#!/usr/bin/env bash
# Blocks any `git commit` carrying Claude attribution trailers.
# Wired as a PreToolUse hook on Bash. Exit 2 = block and tell Claude why.
set -euo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')"

# Only inspect git commits; everything else passes straight through.
case "$cmd" in
  *git*commit*) ;;
  *) exit 0 ;;
esac

if printf '%s' "$cmd" \
   | grep -qiE 'co-authored-by:[[:space:]]*claude|generated with \[?claude code|🤖'; then
  echo "BLOCKED: commit message contains Claude attribution. This repository never credits \
Claude as co-author. Remove the trailer and retry." >&2
  exit 2
fi
exit 0
