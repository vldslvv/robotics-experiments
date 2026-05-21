# Usage

To enable this extension, run Isaac Sim with:

```bash
--ext-folder {path_to_ext_folder} --enable sim_settings
```

The extension applies simulation settings (physics Hz, render Hz, solver, gravity)
from the Hydra-backed experiment config, then delegates the stage mutation to
`robotics_experiments.simulation.apply_settings`.

Stage loading is the host workflow's responsibility. Use Isaac Sim `--exec`,
standalone scripts, or Isaac Lab scene configuration to choose the USD scene.
