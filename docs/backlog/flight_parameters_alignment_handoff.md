# Remaining flight-suite consumers

Alignment consumes unsplit `FlightDataset` evidence, owns whole-flight
train/validation/test membership, fits against all assigned flights and derives
hover bounds from native training-flight windows. See
[controller alignment](../SIM_TO_SIM_ALIGNMENT.md).

Remaining work:

1. Adapt RL's qualified flight-parameter and reset-population reader to the raw
   dataset and alignment-owned bounds, preserving exact source lineage and using
   only training flights. RL currently requires its existing qualified artifact.
2. Remove duplicated physical commands from controller definitions by reading the
   canonical flight suite. Capture owns physical instructions; alignment owns
   profile interpretation and acceptance criteria.
3. Validate the trained hover policy in PX4 after trajectory calibration. Keep
   training noise and domain randomization separate from noise-free capture.
4. Establish a new statistical baseline only after adequate repeated-seed and
   disjoint-seed capture campaigns pass stability checks. A three-flight
   integration run is not evidence of distribution stability.

Retained timestamps, generation IDs, stage boundaries, ULog and bridge dynamics
support separate diagnosis of command delivery and physical response. Controller
gains must not compensate for a shifted measurement window.
