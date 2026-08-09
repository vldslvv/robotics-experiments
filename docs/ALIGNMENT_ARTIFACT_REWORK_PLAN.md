# Alignment artifact rework: implemented v1 contract

The artifact rework is complete in the current design. This document records
the resulting contract rather than a migration procedure.

## Result

- `robotics_experiments.controller` is the shared, host-safe controller module.
- Alignment publishes one strict schema-1 controller artifact.
- RL consumes the controller module and artifact without importing alignment.
- The full candidate population is stored separately as compressed selection
  evidence; `controller.json` stays a compact downstream contract.
- Run evidence uses relative paths and an exact manifest with byte sizes,
  compression metadata, and SHA-256 hashes.
- Old controller/alignment schemas, migrations, aliases, physical-force paths,
  and policy compatibility loaders are removed.

## Bundle layout

An accepted run contains:

```text
<run-id>/
  controller.json
  selection.json.zst
  manifest.json
  evidence/
    reference_traces.json
    reference.ulg
    bridge.log
    target_traces.json.zst
  diagnostics/
    actuator_audit.json
    controller_boundary_replay.json
    gpu_utilization.json
    stages/
      capture_manifest.json
      preflight-alignment.json
      search_realization.json
    acceptance/
```

`controller.json` contains identity, comparison fingerprints, complete selected
settings, certified control modes, quality result, source provenance, and
backend/image identity.
`selection.json.zst` reconstructs every candidate from base settings plus exact
overrides and stores its metrics, ranking, usability, and failure. The manifest
inventories every payload exactly once.

## Acceptance and publication

Finalization verifies controller/selection consistency, hard controller-quality
gates, the actuator audit, the per-profile controller-boundary replay, compressed
payload integrity, and the complete manifest. A failed diagnostic cannot be
published. Policy training independently rejects a quality-failed controller.

`latest` and comparable `best` are relative symlinks to immutable run
directories. Binary ULogs and generated fit artifacts remain under `.build/`
and are not committed.

Historical `.build/alignment` directories may be retained as inert experimental
evidence. They are not input to v1 reproduction and are deliberately rejected
by the strict loader.
