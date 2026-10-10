# -*- coding: utf-8 -*-
"""Adapter template — copy this file to bench/adapters/<your_system>.py and fill in two methods.

The harness (bench/run.py) does everything else: loads the questions, times each call,
scores the outputs, writes bench/results/<name>.json.

    python bench/run.py --system my-system            # full run
    python bench/run.py --system my-system --limit 5  # smoke run, nothing written

Rules that keep the board honest:
- Index the vault in setup(), not the questions. The questions file is the test set.
- Return note ids exactly as the vault names them: the filename without `.md`.
- To abstain, return answer=None. Do not answer "unknown" as a string: that counts as an
  answer, and a wrong one.
- Quote the source word for word. citation_faithful checks every quote against the note.
- Anything you tuned (prompt, threshold, top-k) goes into config, so a reader sees it.
"""

# One module can expose several systems (e.g. two modes of the same library).
# Key = the name that goes on the leaderboard; value = kwargs passed to make().
SYSTEMS = {'my-system': {}}


class Adapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def info(self):
        """Shown on the leaderboard. version and url are required."""
        return {'version': '0.0.0', 'url': 'https://example.com/my-system',
                'config': dict(self.kwargs)}

    def setup(self, vault_dir, work_dir):
        """Ingest every .md file under vault_dir (a flat folder, ~200 notes).
        work_dir is an empty scratch folder you own for this run."""
        raise NotImplementedError('load bench/vault into your memory system here')

    def ask(self, question):
        """-> {'retrieved': [note ids, best first],
               'answer': str or None,
               'citations': [{'note': note id, 'quote': exact text from that note}]}"""
        raise NotImplementedError('query your memory system here')


def make(**kwargs):
    return Adapter(**kwargs)
