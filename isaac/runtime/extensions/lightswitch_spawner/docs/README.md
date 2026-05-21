# Light Switch Spawner

Spawns configured light switches and publishes switch state changes through
ROS 2 while the Isaac timeline is playing.

Enable it from Isaac Sim with:

```bash
--ext-folder {path_to_ext_folder} --enable lightswitch_spawner
```

The extension reads scene configuration through `robotics_experiments.config`,
owns the ROS 2 node lifecycle, and despawns runtime prims when simulation stops.
