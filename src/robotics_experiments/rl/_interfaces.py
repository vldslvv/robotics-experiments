"""Private policy-interface identities and observation contracts."""

from enum import Enum

__all__: list[str] = []


class PolicyInterface(str, Enum):
    """Supported runtime observation/adapter contracts."""

    HOVER_V1 = "hover-v1"
    PRESS_V1 = "press-v1"


HOVER_OBSERVATION_SCHEMA_VERSION = 1
HOVER_OBSERVATION_DIM = 13
PRESS_OBSERVATION_SCHEMA_VERSION = 1
PRESS_OBSERVATION_DIM = 34

INTERFACE_OBSERVATIONS = {
    PolicyInterface.HOVER_V1: (
        HOVER_OBSERVATION_SCHEMA_VERSION,
        HOVER_OBSERVATION_DIM,
    ),
    PolicyInterface.PRESS_V1: (
        PRESS_OBSERVATION_SCHEMA_VERSION,
        PRESS_OBSERVATION_DIM,
    ),
}
