# Isaac Lab runtime area

This directory holds runtime examples and files copied or mounted into Isaac Lab.
Task implementations live in `src/robotics_experiments/workflows/isaac_lab/`;
consumers use that package's lazy public boundary, not task submodules.

See [the RL workflow](../../../docs/RL_WORKFLOW.md) for task registration,
controller/observation contracts, training, evaluation, and deployment. The
`isaac-lab` Docker stage is built with `make build-isaac-lab`; training and Lab
integration use `ISAAC_LAB_IMAGE`. App startup/import ordering is documented in
[standalone workflows](../standalone/README.md).
