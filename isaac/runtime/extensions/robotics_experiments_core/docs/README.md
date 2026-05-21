# Robotics Experiments Core

This extension exposes the shared `src/robotics_experiments` Python package to
other Isaac Sim extensions.

Enable it as a dependency from extension metadata:

```toml
[dependencies]
"robotics_experiments_core" = {}
```

After Isaac Kit resolves the dependency, dependent extensions can import the
core package normally:

```python
import robotics_experiments
```

Keep reusable experiment logic in `src/robotics_experiments`. Keep Isaac Sim
event wiring, UI, and stage integration in the consuming extension.
