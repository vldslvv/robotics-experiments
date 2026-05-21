# PX4 SITL Bridge

Provides the PX4 bridge package and no-op Isaac Kit entrypoint used by host
extensions.

Enable it from Isaac Sim with:

```bash
--ext-folder {path_to_ext_folder} --enable px4_sitl_brigde
```

The extension does not spawn drones or own physics callbacks. Host extensions
construct `Px4Bridge`, select the MAVLink port, and call the bridge during
timeline playback. See `docs/PX4_BRIDGE_CONTRACTS.md` for layer contracts.
