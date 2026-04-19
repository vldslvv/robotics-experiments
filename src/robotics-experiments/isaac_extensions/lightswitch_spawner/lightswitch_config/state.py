from dataclasses import dataclass, field

from isaacsim.core.prims import SingleArticulation


@dataclass
class SwitchState:
    prim_path: str
    lamp_prim_path: str = ""
    articulation: SingleArticulation | None = field(default=None, repr=False)
    state: bool = False
    last_button_state: bool = False
