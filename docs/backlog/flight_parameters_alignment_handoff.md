# Consume independent flight suites in alignment

Flight capture publishes an immutable unsplit `FlightDataset` containing repeated
complete 24-stage flights. It no longer produces alignment dataset membership or
training bounds. Existing alignment and RL readers require qualified
`FlightParameters` and reject the raw dataset until this follow-up is implemented.

1. Read `FlightDataset` through its reader, resolve `latest` once, and pin the
   canonical producer run, artifact digest and fingerprint. Reference producer
   files directly; do not copy, rewrite or migrate evidence.
2. Assign complete independent flights to fitting, validation and frozen test
   sets in alignment. Record membership and seed identity before candidate
   search. Never split correlated time samples or stages into independent sets.
3. Fit against multiple complete flights so their distributions influence
   candidate fitting and selection. Select with validation flights, then test
   frozen settings on untouched flights. Keep qualification criteria explicit.
4. Derive hover bounds and policy reset populations from fitting flights only.
   Record these as consumer-owned qualified outputs with exact dataset lineage.
   Validation and test extremes must not enlarge fitting bounds.
5. Reuse the canonical physical stage definitions under `flight_parameters`.
   Remove duplicated physical commands from controller definitions when their
   consumers can read the suite. Alignment owns objective/profile selection,
   error weights and acceptance limits; capture owns physical instructions.
6. Diagnose requested-command, successful transmission, PX4 activation and
   physical-response timing separately. Use the retained simulation timestamps,
   generation IDs, stage boundaries, ULog and bridge dynamics. Do not change
   gains to compensate for a shifted measurement window or command-delivery delay.
7. Validate trained hover policy behavior in PX4 after trajectory calibration.
   Passing calibration probes alone is insufficient deployment validation.
   Training sensor noise and domain randomization follow after calibration and
   remain separate from noise-free flight-parameter capture.

The capture count is independent of splitting policy. A ten-flight dataset
supports the interface work; statistical qualification still requires adequate
independent repeated-seed and disjoint-seed campaigns. Establish a new baseline
only after the capture mechanism passes those stability checks.
