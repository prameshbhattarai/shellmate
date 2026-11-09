#!/usr/bin/env bash
# Record the README demo GIF.
#
#   brew install asciinema agg      # one-off
#   ./scripts/demo.sh               # replay: scripted model answers, no Ollama, Mac stays cool
#   ./scripts/demo.sh --live        # real session: you type, against your local model
#
# Both write docs/demo.cast and docs/demo.gif. They run in a throwaway folder with their own
# SHELLMATE_HOME, so nothing outside it is touched and your history stays clean.
#
# In replay mode only the model is scripted (scripts/demo_replay.json); the risk rules, prompts,
# commands and their output are real, and the banner says "model=replay (scripted answers)".
set -euo pipefail

MODE="${1:-replay}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
PLAYGROUND="$WORK/demo-project"
export SHELLMATE_HOME="$WORK/shellmate-home"

for tool in asciinema agg shellmate; do
  command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }
done

mkdir -p "$PLAYGROUND"/{src,node_modules/left-pad,docs}
head -c 3000000 /dev/urandom > "$PLAYGROUND/node_modules/left-pad/blob.bin"
printf '# demo project\n' > "$PLAYGROUND/README.md"
printf 'print("hi")\n' > "$PLAYGROUND/src/app.py"
printf 'notes\n' > "$PLAYGROUND/docs/notes.md"
(cd "$PLAYGROUND" && git init -q && git add -A \
  && git -c user.name=demo -c user.email=demo@example.com commit -qm "first commit")

cd "$PLAYGROUND"
if [[ "$MODE" == "--live" ]]; then
  export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
  echo "Recording a live session in $PLAYGROUND. Suggested requests (then type 'exit'):"
  echo "  which folders here are eating my disk?"
  echo "  delete the node_modules folder            (answer: yes)"
  echo "  what was my last commit message?"
  echo "  delete everything in my home folder       (watch it get BLOCKED)"
  asciinema rec --overwrite --idle-time-limit 2 -c "shellmate" "$ROOT/docs/demo.cast"
else
  command -v expect >/dev/null || { echo "missing: expect"; exit 1; }
  export SHELLMATE_REPLAY="$ROOT/scripts/demo_replay.json" SHELLMATE_REPLAY_DELAY=0.9
  asciinema rec --overwrite --headless --quiet --window-size 96x34 --idle-time-limit 2 \
    -c "expect -f $ROOT/scripts/demo.exp" "$ROOT/docs/demo.cast"
fi

agg --theme monokai --font-size 16 --last-frame-duration 4 "$ROOT/docs/demo.cast" "$ROOT/docs/demo.gif"
rm -rf "$WORK"
echo "wrote docs/demo.cast and docs/demo.gif"
