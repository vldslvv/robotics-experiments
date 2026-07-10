# Binary asset policy

New binary files are not accepted in this repository. Generated drone assets
use text USDA, and runtime caches, logs, models, and training artifacts stay
untracked.

Three pre-existing LightSwitchWarehouse USD crates are intentional,
grandfathered source assets:

- `isaac/assets/LightSwitchWarehouse/Lamp.usd`
- `isaac/assets/LightSwitchWarehouse/LightSwitch.usd`
- `isaac/assets/LightSwitchWarehouse/Warehouse.usd`

Their exact paths are the complete `.binary-asset-allowlist`. Do not add another
entry or modify/regenerate these crates unless a task explicitly requests an
asset-policy change. Prefer text USDA for any authored project asset.

The `binary-asset-policy` pre-commit hook inspects every changed file and rejects
binary content outside the allowlist. Its all-files run also verifies the tracked
tree against the same policy.
