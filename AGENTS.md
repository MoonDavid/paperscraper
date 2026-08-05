# AGENTS.md

## Cursor Cloud specific instructions

`paperscraper` is a single Python library (no server, database, or GUI). Development
uses [`uv`](https://docs.astral.sh/uv/); dependencies are defined in `pyproject.toml`
(dev extras in the `dev` dependency group) and locked in `uv.lock`. Standard
build/lint/test/run commands live in `CONTRIBUTING.md` and `.github/workflows/test_tip.yml`.

Notes for working in this environment:

- `uv` installs to `~/.local/bin`. The startup update script installs it there and
  runs `uv sync --group dev`. Interactive shells pick it up via `~/.bashrc`; if `uv`
  is ever not found, use `~/.local/bin/uv` or run `source ~/.local/bin/env`.
- Many tests and all "scrape" workflows hit live external scholarly APIs (arXiv,
  PubMed, bioRxiv/medRxiv/chemRxiv, Semantic Scholar, Google Scholar). They are
  therefore network-dependent, slow, and can be flaky/rate-limited. arXiv in
  particular returns HTTP 429 if you fire multiple queries back-to-back; space out
  arXiv calls (a few seconds) when running demos or tests. GitHub Actions is the
  source of truth for release readiness.
- Optional API keys/credentials (`SS_API_KEY`, publisher tokens, AWS, Kaggle) only
  improve rate limits and PDF/dump fallbacks; the core library and most tests run
  without them.
- `paperscraper/server_dumps/*.jsonl` (downloaded preprint dumps), `dist/`, `build/`,
  and `*.egg-info` are generated artifacts and are gitignored — do not commit them.
- Run a single test module to avoid the slow full network suite, e.g.
  `uv run pytest paperscraper/tests/test_dump.py`.
