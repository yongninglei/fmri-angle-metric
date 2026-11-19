# Changelog

All notable changes to fmri-angle-metric. Every change adds an entry under
`## [Unreleased]`: what changed, the problem it solves, and why this way.
History before this file: `git log`.

## [Unreleased]

### Removed
- Sphinx docs: the CI `build-docs` job (build + Pages deploy; it failed since the first
  commit) is replaced by a smoke job (every CLI answers `--help`). Sphinx files for the
  maintainer to delete: `docs/source/`, `docs/Makefile`, `docs/requirements.txt`,
  `docs/_static/`, `docs/_templates/` (that is all of `docs/`). *Why:* maintainer decision
  2026-11-19 — no Sphinx in any repo.

### Changed
- `## Upstream / downstream` section in the repo's Claude.md: which folders it reads (made by
  which repo) and writes (read by which repo), with their space. *Problem:* written in a
  parallel session on 2026-11-19 but never committed, and the repo had no CHANGELOG, so it
  was left out of that day's MRI-hub updates. *Why:* the MRI-hub map
  (`~/soft/docs/repo_map.dot`) is drawn from these sections; every change goes through
  branch + CHANGELOG + PR (`~/soft/CLAUDE.md`).
