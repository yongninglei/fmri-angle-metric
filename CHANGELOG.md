# Changelog

All notable changes to fmri-angle-metric. Every change adds an entry under
`## [Unreleased]`: what changed, the problem it solves, and why this way.
History before this file: `git log`.

## [Unreleased]

### Changed
- `## Upstream / downstream` section in the repo's Claude.md: which folders it reads (made by
  which repo) and writes (read by which repo), with their space. *Problem:* written in a
  parallel session on 2026-11-19 but never committed, and the repo had no CHANGELOG, so it
  was left out of that day's MRI-hub updates. *Why:* the MRI-hub map
  (`~/soft/docs/repo_map.dot`) is drawn from these sections; every change goes through
  branch + CHANGELOG + PR (`~/soft/CLAUDE.md`).
