# Releasing

Two states to keep apart: the package *builds and installs* (automated, proven on every
push) and the package is *on PyPI* (one manual registration away, as of 2026-10-07).

## What is already automated

`.github/workflows/tests.yml` has an `installable` job that, on every push and pull request:

- builds the sdist and the wheel,
- runs `twine check` on both,
- installs the wheel into a venv with nothing else in it,
- asserts all five commands (`sgm-index`, `sgm-ask`, `sgm-mcp`, `sgm-turnstate-hook`,
  `sgm-turnstate-show`) are present and start,
- asserts `schema.sql` travelled inside the wheel,
- asserts a base install (no `[embeddings]`) prints install instructions rather than a
  traceback.

That job exists because a wheel can break without a single test failing: a bad entry point,
a file left out of the build, a dependency that only exists on the author's machine.

## The one-time step that is NOT done yet

`.github/workflows/publish.yml` publishes to PyPI on every published GitHub release using
Trusted Publishing (OIDC) — no API token is stored in a repository secret or on a laptop.
But OIDC only works once PyPI has been told to expect this workflow:

1. Create or sign in to a PyPI account.
2. Go to <https://pypi.org/manage/account/publishing/> → **Add a new pending publisher**:

   | field       | value                 |
   |-------------|-----------------------|
   | PyPI name   | `sqlite-graph-memory` |
   | Owner       | `tonydzi`             |
   | Repository  | `sqlite-graph-memory` |
   | Workflow    | `publish.yml`         |
   | Environment | `pypi`                |

3. Publish a GitHub release. The workflow builds, re-checks installability, and uploads.

**Skip step 2 and the job fails with `invalid-publisher` while still reporting a valid OIDC
token** — a confusing failure that has already cost us one release: `secondop-panel v0.1.1`
on 2026-08-29 sat at 404 on PyPI while its release notes said it had shipped. The name
`sqlite-graph-memory` was verified free on 2026-10-07 by two independent rails (the PyPI
JSON API returned 404, and `pip download` reported *no matching distribution*), with a
control run against a name that is taken to prove the check can say "yes". Note that a
plain `curl` against `pypi.org/project/...` is **not** a valid check: PyPI serves
non-browser clients a bot challenge with HTTP 200 for any path, including paths that do not
exist.

## Cutting a release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/sqlite_graph_memory/__init__.py` (they must match), and the `version` in
   `SERVER_INFO` in `mcp_server.py` if the MCP surface changed.
2. Add a CHANGELOG entry that says what changed for a *user*, not what commits landed.
3. Tag and publish a GitHub release. `publish.yml` does the rest.
