#!/usr/bin/env bash
# Renders docs/demo.gif, the clip at the top of the README. Real run, output not edited.
#
#   pip install 'sqlite-graph-memory[embeddings]'      # sgm-index and sgm-ask on PATH
#   uvx --from 'asciinema<3' asciinema rec --cols 104 --rows 36 -c 'bash docs/demo.sh' demo.cast
#   agg --idle-time-limit 2 --last-frame-duration 10 --font-size 16 --theme monokai demo.cast docs/demo.gif
#
# The only cut is agg's idle-time limit: on an x86_64 Mac CPU the embedding pass
# takes about a minute and prints nothing (the progress bar is off), so the gif
# skips that silence. The query is the one from "Three commands" in the README.
set -e
export TQDM_DISABLE=1 SGM_NO_STAR_PROMPT=1 TOKENIZERS_PARALLELISM=false
demo=$(mktemp -d)
cp -R "$(dirname "$0")/../bench/vault" "$demo/notes"
cd "$demo"

say() {   # print a command as if typed, then run it
  printf '\033[1;32m$\033[0m '
  for ((i = 0; i < ${#1}; i++)); do printf '%s' "${1:i:1}"; sleep 0.045; done
  echo; sleep 0.4
  eval "$1" 2>/dev/null   # model-loading warnings go to stderr
}

clear
printf '\033[2m# 202 markdown notes that link to each other with [[wikilinks]]\033[0m\n'
sleep 1.5
say 'sgm-index ./notes'
sleep 2
echo
say "sgm-ask --ab 'In which programming language is the tool behind Project Ouzel written?'"
sleep 1.5
printf '\n\033[2m# the answer is in tool-lanternfish; only the graph hop from project-ouzel reaches it\033[0m\n'
sleep 2
