# Drone Spawner

Spawns the configured drone when the timeline starts playing and despawns it
when playback stops or pauses.

Enable it from Isaac Sim with:

```bash
--ext-folder {path_to_ext_folder} --enable drone_spawner
```

The extension depends on `experiment_config` for drone and PX4 settings. It
does not spawn a preview drone when the extension is enabled or when a stage is
opened. When PX4 is enabled, it owns the bridge host lifecycle around timeline
playback.
