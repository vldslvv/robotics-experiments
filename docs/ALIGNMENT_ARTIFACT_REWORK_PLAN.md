# Alignment Artifact Rework Plan

Status: IMPLEMENTED

This document is the implementation specification for the controller-v1 bundle
and training-provenance rework.

## Summary

The current `alignment.json` combines two unrelated purposes:

- a compact controller contract needed by Isaac Lab; and
- the complete search history and alignment evidence needed for auditing.

The accepted v6 artifact is 36,157,576 bytes and 1,096,363 lines because it
repeats complete `ControllerSettings` for 8,300 candidates. Training ultimately
passes only the selected settings into the Isaac Lab environment, together with
provenance and advisory-quality information.

Replace the monolithic artifact with a compact training contract and separate,
single-purpose evidence:

| Artifact | Purpose | Read by training |
| --- | --- | --- |
| `controller.json` | Selected controller settings, compatibility, quality, and provenance | Yes |
| `selection.json.zst` | Candidate definitions, scores, ranking, and selection decision | No |
| `manifest.json` | Bundle inventory, hashes, sizes, and compression metadata | No |
| `evidence/` | Reference data and expensive target traces required for replay or rescoring | No |
| `diagnostics/` | Derived audits, replay summaries, telemetry, and stage logs | No |

No controller behavior, parameter search, metric definition, threshold, probe,
or selected setting changes as part of this refactor.

## Artifact and API contracts

The finalized run layout will be:

```text
<run>/
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
    acceptance/
```

`controller.json` is a new schema-v1 artifact and the only alignment output
consumed by policy training, Isaac Lab evaluation, controller replay, or policy
loading. It contains:

- alignment and run identity;
- target/reference controller identities and backends;
- vehicle, airframe, command cadence, and command interface;
- complete selected `ControllerSettings`;
- a deterministic settings fingerprint;
- target, reference, probe-suite, experiment, and comparison fingerprints;
- source commit, dirty state, and image provenance; and
- advisory quality result and reasons.

It must not contain candidate rows, parameter ranges, ranking vectors, or metric
maps. Serialization is deterministic compact JSON and the representative
artifact must remain below 16 KiB.

`selection.json.zst` is the authoritative search and metric record. It contains
the base settings and declared `ParameterSpec` values once. Each ordered
candidate row contains only:

- candidate index;
- settings overrides relative to the shared base settings;
- registered fit and validation metrics;
- ranking values; and
- usability and failure reason.

The selected index identifies the winning row. Separate top-level copies of the
selected settings, ranking values, fit metrics, and validation metrics are
removed. Loading the file must reconstruct every complete candidate exactly and
verify that the selected candidate reconstructs to the settings in
`controller.json`.

The remaining artifacts have one purpose each:

- Rename `reference.json` to `evidence/reference_traces.json`.
- Rename `target_candidates.json.zst` to
  `evidence/target_traces.json.zst`; it contains traces, not candidate scores.
- Reduce `actuator_audit.json` to thresholds, pass/fail results, sample
  coverage, timing, aggregate errors, saturation, and attribution. Remove
  time-series arrays recoverable from the ULog, bridge log, or reference
  traces.
- Keep controller-boundary replay and GPU utilization as summary diagnostics.
- Store retained execution and acceptance logs under their respective
  diagnostic directories.
- Make `manifest.json` the sole inventory and checksum authority. Record
  logical role, relative path, byte size, SHA-256, compression, and
  uncompressed size for every payload. The manifest does not hash itself, and
  new bundles do not contain `SHA256SUMS`.

Replace the public alignment boundary as follows:

- `AlignmentArtifact` becomes `ControllerArtifact`.
- `load_alignment_artifact(...)` becomes `load_controller_artifact(...)`.
- `align(...)` returns a new immutable `AlignmentResult` containing the
  selected result and in-memory candidate scores.
- Candidate scores never become part of `ControllerArtifact`.
- Do not retain legacy production loaders or compatibility aliases.

Rename all training-facing surfaces consistently:

- Hydra: `rl.training.controller_artifact`
- CLI: `--controller-artifact`
- Environment: `CONTROLLER_ARTIFACT`
- Policy metadata section: `controller`
- Policy fields: `controller_digest` and
  `controller_comparison_fingerprint`

Training resolves a file directly or appends `controller.json` to a run
directory or publication-link path. It validates and hashes that file, warns on
advisory-quality failure, embeds the complete settings, digest, and
fingerprints in the policy artifact, and passes only `ControllerSettings` into
Isaac Lab environment construction.

## Generation and publication

Use one lifecycle for all new runs:

1. Capture PX4 reference traces, ULog, bridge log, and stage diagnostics.
2. Produce the summary-only actuator audit.
3. Sweep Isaac Lab candidates into a temporary uncompressed target-trace file.
4. Fit and return `AlignmentResult`; write `controller.json` and
   `selection.json.zst`.
5. Run controller-boundary replay from `controller.json`.
6. Compress and verify target traces, then remove only the verified temporary
   uncompressed file.
7. Finalize and validate the manifest.
8. Publish `latest` and, when applicable, `best` atomically.

`alignment-fit` must stop publishing by itself. `align-controllers` must include
replay, compression, final validation, and publication. A missing or malformed
required output blocks publication.

Controller-boundary replay remains diagnostic rather than a quality gate.
Structurally valid poor-quality alignment results remain consumable, warn
training, advance `latest`, and participate in `best` only under the existing
comparable-fingerprint rules.

Publication loads ranking data from `selection.json.zst`; ranking data is not
copied into the training contract. Reuse the existing `zstd` executable and
fail explicitly when it is unavailable or compression verification fails.

Once implementation lands, update the current-behavior documentation in
`SIM_TO_SIM_ALIGNMENT.md`, `RL_WORKFLOW.md`, experiment configuration
documentation, and affected standalone-script documentation. This plan remains
the historical implementation specification.

## Accepted-v6 migration

Extend the existing host-safe alignment-fit entry point with an explicit
one-time migration mode rather than adding another command:

```console
PYTHONPATH=src uv run python isaac/runtime/standalone/alignment_fit.py \
  --migrate-v6-source .build/alignment/x500-velocity-px4-v6/20260731T185547Z \
  --run-dir .build/alignment/x500-velocity-px4-v6/20260731T185547Z-controller-v1 \
  --run-id 20260731T185547Z-controller-v1
```

Migration must:

- verify the source bundle with its existing `SHA256SUMS` before copying;
- leave the accepted source directory byte-for-byte unchanged;
- build a self-contained sibling run and fail if the destination already
  exists;
- split `alignment.json` into `controller.json` and compressed selection data;
- reorganize retained raw evidence and diagnostics without keeping copied
  legacy manifests, checksums, or `alignment.json`;
- preserve all candidate order, settings, metrics, ranking values, usability,
  failures, selected settings, and advisory-quality results;
- record the source alignment ID, run ID, schema, artifact digest,
  checksum-file digest, and legacy comparison fingerprint in migration
  provenance inside the new manifest;
- generate a new comparison fingerprint because the artifact schema changed,
  while preserving the underlying definition, experiment, probe-suite, and
  settings fingerprints; and
- initialize publication for the new comparison fingerprint and atomically
  update `best` and `latest` only after complete validation.

The source and migrated bundles will temporarily duplicate the retained 1.52
GiB compressed target traces. This is intentional: the source is immutable and
the migrated run must remain self-contained.

## Verification and acceptance

Add tests covering:

- strict `controller.json` field whitelist, finite values, complete settings,
  settings fingerprint, supported backends, and file/directory/link resolution;
- rejection of legacy `alignment.json`, malformed controller artifacts,
  unknown fields, and inconsistent fingerprints;
- a representative controller contract below 16 KiB with no candidates,
  parameter declarations, ranking vectors, or metric maps;
- deterministic selection serialization and compression;
- exact reconstruction of all v6 candidate settings from base settings plus
  overrides;
- preservation of candidate order, metrics, rankings, failures, selected
  result, and advisory-quality outcome;
- selection output smaller than the legacy compact candidate payload while
  retaining all 8,300 candidate records;
- summary-only actuator audit with no retained recoverable time-series arrays;
- finalization failures for missing outputs, hash mismatches, compression
  failures, or controller/selection disagreement;
- `latest` and comparable `best` behavior for passing and advisory-failing
  results;
- migration checksum verification, immutable source, self-contained
  destination, absence of legacy files, new comparison fingerprint, and atomic
  publication;
- training bootstrap using only `controller.json`, with selection, evidence,
  and diagnostic files absent or inaccessible; and
- policy metadata validation using the renamed controller-oriented contract.

Run the host unit suite, `make check`, `uv run pre-commit run --all-files`, the
alignment integration target, and a small Isaac Lab training smoke test.

Old policy artifacts containing legacy alignment metadata are unsupported and
must be retrained. No compatibility shim or dual-read transition is planned.
