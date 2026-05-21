# Drone Spawner

Spawns and despawns the configured drone on stage and timeline events.

Enable it from Isaac Sim with:

```bash
--ext-folder {path_to_ext_folder} --enable drone_spawner
```

The extension depends on `experiment_config` for drone and PX4 settings. When
PX4 is enabled, it owns the bridge host lifecycle around timeline playback.
