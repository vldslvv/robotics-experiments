# ADR 0007: Tests assert public behavior; enable Ruff SLF001

- Status: Accepted
- Date: 2026-06-28
- Related: ADR 0002 (layering), ADR 0006 (contract-seam testing)

## Context

Unit tests across `tests/` reached into internals -- reading a unit's private
state (`port._sensor_send_count`, `bridge._tilt_cutoff_count`,
`extension._switches_spawned`) and asserting on mock/interaction logs even where a
public observable existed. These tests pin implementation, not behavior, and break
on harmless refactors. `tests/test_bridge_host.py` had already been reworked to
the better pattern and serves as the exemplar.

## Decision

Rework the suite to assert on **public behavior**, then lock it in:

- **FIX** any external private-member access where a public property, return
  value, or raised error exposes the same fact.
- **KEEP** (documented inline) assertions on a fake's recorded interaction only
  when it mirrors a **documented contract seam** -- constructor params a
  collaborator received, commands sent to a port, callbacks invoked on an env,
  forces/dynamics applied to the articulation IO (an adapter's observable
  output); prefer to supplement the seam assertion with the public observable.
- Two no-behavior-change techniques avoid `obj._x`: import private functions by
  name (`from module import _private_fn`) rather than attribute-calling them, and
  drive Kit-extension events by firing the fake event stream's registered
  callbacks rather than calling private `_on_*` handlers.
- **Capstone:** enable Ruff `SLF001` repo-wide so the win cannot regress. The only
  allowed exceptions are justified `per-file-ignores` (production seams against
  vendored Pegasus / Isaac Lab privates, fixture injection, focused private-helper
  unit tests, deliberate fault injection); no new external private access.

## Consequences

- Tests survive internal refactors; AGENTS.md records the public-behavior rule and
  the `SLF001` policy with its narrow exception list.
- The refactor doc is a resumable per-file checklist, kept until every file is
  ticked.
