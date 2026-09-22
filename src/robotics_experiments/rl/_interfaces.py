"""Private policy-interface identities and observation contracts."""

from enum import Enum

__all__: list[str] = []


class PolicyInterface(str, Enum):
    """Supported runtime observation/adapter contracts."""

    HOVER = "hover"
    PRESS = "press"


HOVER_OBSERVATION_SCHEMA_VERSION = 2
HOVER_OBSERVATION_DIM = 12
PRESS_OBSERVATION_SCHEMA_VERSION = 3
PRESS_OBSERVATION_DIM = 35

INTERFACE_OBSERVATIONS = {
    PolicyInterface.HOVER: (
        HOVER_OBSERVATION_SCHEMA_VERSION,
        HOVER_OBSERVATION_DIM,
    ),
    PolicyInterface.PRESS: (
        PRESS_OBSERVATION_SCHEMA_VERSION,
        PRESS_OBSERVATION_DIM,
    ),
}
