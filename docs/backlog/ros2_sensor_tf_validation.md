# ROS 2 Sensor and TF Validation

## Idea

Preserve ROS 2 publishing checks and Isaac graph questions for a future sensor
pipeline validation workflow.

## Useful Notes

- A ROS 2 Publish Transform Tree graph should exist when TF publishing is
  expected.
- Validate TF publication with:
  - `ros2 topic echo /tf`
  - `ros2 run tf2_tools view_frames`
- In RViz2, set the fixed frame to `world` or `map`, depending on the workflow.
- For camera publishing, consider different publication rates by stream:
  - Camera info every frame.
  - RGB image every 5 frames.
  - Depth image every 60 frames.
- For Isaac action graphs, investigate `pipelineStage` and why some graphs use
  `pipelineStageOnDemand`.
- For reading simulation time, consider whether `Reset on Stop` should be set.

## Follow-Up

- Turn these checks into a repeatable validation script only after the project
  has an accepted ROS 2 sensor pipeline.
- If adopted, document the final workflow in `docs/RUNTIME_OPERATIONS.md` or a
  ROS-specific document instead of leaving it under `docs/backlog/`.
