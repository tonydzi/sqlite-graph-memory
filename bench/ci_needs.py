# -*- coding: utf-8 -*-
"""ci_needs.py <need> -> the systems in ci_reproduced.txt whose line says `needs: <need>`.

    python bench/ci_needs.py ollama      # fidelis mem0-local

The bench workflow uses it to pick which interpreter reruns each system. The badge
(bench_core.ci_reproduced) ignores comments, so a `needs:` note never changes who is listed.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEDS_RX = re.compile(r'^\s*needs:\s*([\w-]+(?:\s*,\s*[\w-]+)*)')   # the comment must START with it


def needing(need, path=HERE / 'ci_reproduced.txt'):
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        name, _, comment = line.partition('#')
        m = NEEDS_RX.search(comment)
        if name.strip() and m and need in [x.strip() for x in m.group(1).split(',')]:
            out.append(name.strip())
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python bench/ci_needs.py <need>   (e.g. ollama)')
    print(' '.join(needing(sys.argv[1])))
