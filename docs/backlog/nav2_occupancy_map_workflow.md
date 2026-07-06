# Nav2 Occupancy Map Workflow

## Idea

Record ground-robot navigation notes that may become useful if the project adds
Nav2-based workflows.

## Useful Notes

- Isaac Sim occupancy maps can export both YAML and PNG map artifacts for ROS 2
  navigation.
- Lidar scan height must be consistent with the occupancy map `Z` bounds.
- In RViz2, use the Nav2 Goal tool to select a goal pose on the map.

## Follow-Up

- Confirm whether ground-robot navigation is in scope before wiring this into
  project runtime docs.
- If adopted, define the generated map artifact location, frame conventions,
  and launch sequence for Nav2.
