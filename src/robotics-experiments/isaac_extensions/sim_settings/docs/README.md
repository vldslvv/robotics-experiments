# Usage

To enable this extension, run Isaac Sim with the flags --ext-folder {path_to_ext_folder} --enable sim_settings

The extension applies simulation settings (physics Hz, render Hz, solver, gravity) from sim_config.config.SIM_SETTINGS to the active stage on every stage-open event. Stage loading is the host's responsibility: use Isaac Sim's --exec open_stage.py ... or Isaac Lab's InteractiveSceneCfg to choose the USD scene.
