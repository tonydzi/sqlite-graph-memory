#!/usr/bin/env bash
# Renders docs/demo.gif, the clip at the top of the README. Real run, real numbers.
#
#   pip install 'sqlite-graph-memory[lite]'            # sgm-index and sgm-ask on PATH
#   uvx --python 3.12 --from 'asciinema<3' asciinema rec --cols 100 --rows 22 -c 'bash docs/demo.sh' demo.cast
#   agg --idle-time-limit 2 --last-frame-duration 12 --font-size 16 --theme monokai demo.cast docs/demo.gif
#
# Trimmed from the display so the result fits one screen: the indexer's progress lines,
# the (Direct=...) counts line and ranks 6-12 of each list. Green marks the graph-only hit.
# Trimmed from the time: agg's idle-time limit skips the silence while the index builds.
# The query is the one from "Three commands" in the README.
set -eo pipefail
for tool in sgm-index sgm-ask; do
  command -v "$tool" > /dev/null || { echo "$tool must be on PATH" >&2; exit 1; }
done
export TQDM_DISABLE=1 SGM_NO_STAR_PROMPT=1 TOKENIZERS_PARALLELISM=false
export TERM="${TERM:-xterm-256color}"   # asciinema rec may start without one, and clear needs it
demo=$(mktemp -d)
cp -R "$(dirname "$0")/../bench/vault" "$demo/notes"
cd "$demo"

top5() {   # keep the title, the list headers and ranks 1-5; green on the graph hit
  awk '/^\(Direct=/ { next }
       /^---|^A\/B/ || /^ [1-5]\. / || /^$/ {
         if (/<== \+graph/) printf "\033[1;32m%s\033[0m\n", $0; else print }'
}

say() {   # print a command as if typed, then run it; $2 = optional display filter
  printf '\033[1;32m$\033[0m '
  for ((i = 0; i < ${#1}; i++)); do printf '%s' "${1:i:1}"; sleep 0.03; done
  echo; sleep 0.4
  eval "$1 ${2:+| $2}" 2>/dev/null   # model-loading warnings go to stderr
}

clear
printf '\033[2m# 202 markdown notes that link to each other with [[wikilinks]]\033[0m\n'
sleep 1.5
say 'sgm-index ./notes' "grep -v '^  encoded '"
sleep 2
echo
say "sgm-ask --ab 'In which programming language is the tool behind Project Ouzel written?'" top5
sleep 1.5
printf '\033[2m# the answer is in tool-lanternfish; only the graph hop from project-ouzel reaches it\033[0m\n'
sleep 2
