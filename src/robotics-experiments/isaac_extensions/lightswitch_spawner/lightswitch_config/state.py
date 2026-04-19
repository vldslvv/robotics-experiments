from dataclasses import dataclass, field

from isaacsim.core.prims import SingleArticulation


@dataclass
class SwitchState:
    """Runtime state for a single spawned light switch.

    Attributes:
        prim_path: Absolute USD path to the switch root prim.
        lamp_prim_path: Absolute USD path to the paired lamp root prim.
        articulation: PhysX articulation handle, set after initialization.
        state: Current on/off state of the lamp.
        last_button_state: Previous-tick press state, used for edge detection.
    """

    prim_path: str
    lamp_prim_path: str = ""
    articulation: SingleArticulation | None = field(default=None, repr=False)
    state: bool = False
    last_button_state: bool = False
