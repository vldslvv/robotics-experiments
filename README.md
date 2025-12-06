# Running

## Run isaac sim:
On host:
`make run-isaac`

Inside docker:
`./runapp.sh` -- launches sim GUI

## Run ROS2
On host:
`make run-ros`

Both containers "should" be integrated to message each other

## Sometimes permissions are messed after creating something inside a container:

`sudo chown -R "$(whoami):" isaac-sim`

# Development

There's no good way to get autocomplete, neither locally nor while connected to running container.

The workflow is the following:
1. Create script using isaacsim
2. Mount its folder to a isaacsim container
3. Run script using `./python.sh ...` inside a container
4. For reference, use web documentation

VS Code can be used to connect to container, make sure to set interpreter path to the one used by `./python.sh` script.

To find out the interpreter, use:

```bash
./python.sh -c "import sys; print(sys.executable); print(sys.path)"
```

The output is likely to be the following:

```bash
/isaac-sim /isaac-sim/kit/python/bin/python3
['', '/isaac-sim', '/isaac-sim/kit/python/lib/python3.11', '/isaac-sim/kit/python/lib/python3.11/site-packages', '/isaac-sim/python_packages', '/isaac-sim/exts/isaacsim.simulation_app', '/isaac-sim/extsDeprecated/omni.isaac.kit', '/isaac-sim/kit/kernel/py', '/isaac-sim/kit/plugins/bindings-python', '/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle', '/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle', '/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle', '/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle', '/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle', '/isaac-sim/exts/omni.pip.compute/pip_prebundle', '/isaac-sim/exts/omni.pip.cloud/pip_prebundle', '/isaac-sim/kit/python/lib/python311.zip', '/isaac-sim/kit/python/lib/python3.11/lib-dynload']
```

Tickets:
`https://forums.developer.nvidia.com/t/python-autocomplete-not-working-in-isaacsim-vscode-extension/329010`

