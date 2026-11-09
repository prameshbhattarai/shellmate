#!/usr/bin/env bash
# Record the README demo GIF.
#
#   brew install asciinema agg      # one-off
#   ollama pull qwen2.5:7b          # the demo reads better on 7b
#   ./scripts/demo.sh               # records docs/demo.cast, then renders docs/demo.gif
#
# It runs a real session against your local model in a throwaway folder, so expect your Mac to warm
# up for a minute. Nothing outside that folder is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAYGROUND="$(mktemp -d)/demo-project"
mkdir -p "$PLAYGROUND"/{src,node_modules/left-pad,docs}
head -c 3000000 /dev/urandom > "$PLAYGROUND/node_modules/left-pad/blob.bin"
printf '# demo project\n' > "$PLAYGROUND/README.md"
printf 'print("hi")\n' > "$PLAYGROUND/src/app.py"
(cd "$PLAYGROUND" && git init -q && git add -A && git commit -qm "first commit")

for tool in asciinema agg shellmate; do
  command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }
done

cd "$PLAYGROUND"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
echo "Recording in $PLAYGROUND. Suggested requests (type them, answer the prompts, then 'exit'):"
echo "  which folders here are eating my disk?"
echo "  delete the node_modules folder            (answer: yes)"
echo "  what was my last commit message?"
echo "  delete everything in my home folder       (watch it get BLOCKED)"
echo
asciinema rec --overwrite -c "shellmate" "$ROOT/docs/demo.cast"
agg --theme monokai --font-size 16 "$ROOT/docs/demo.cast" "$ROOT/docs/demo.gif"
echo "wrote docs/demo.gif"
