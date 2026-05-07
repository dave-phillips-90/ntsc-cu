from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt


_Channels = Literal[3]
BGRImageArray = Annotated[npt.NDArray[np.uint8], tuple[int, int, _Channels]]
YIQImageArray = Annotated[npt.NDArray[np.int32], tuple[_Channels, int, int]]
YUVImageArray = Annotated[npt.NDArray[np.int32], tuple[_Channels, int, int]]
ChannelComponent = Annotated[npt.NDArray[np.int32], tuple[int, int]]


class ScanField(IntEnum):
    EVEN = 0  # Even lines
    ODD = 1  # Odd lines


@dataclass(frozen=True)
class VHSSpeed:
    luma_cut: float
    chroma_cut: float
    chroma_delay: int


class VHSSpeedPreset(Enum):
    VHS_SP = (2_400_000.0, 320_000.0, 9)
    VHS_LP = (1_900_000.0, 300_000.0, 12)
    VHS_EP = (1_400_000.0, 280_000.0, 14)

    def __init__(self, luma_cut: float, chroma_cut: float, chroma_delay: int):
        self.luma_cut = luma_cut
        self.chroma_cut = chroma_cut
        self.chroma_delay = chroma_delay


@dataclass(frozen=True)
class LFilterConfig:
    """
    This class is used to set the cutoff point for the low-pass filter function
    """

    cutoff: float


class LFilterPreset(Enum):
    """
    Some commonly used values for the low-pass filter cutoff, defined by the approximate
    bandwith available to each channel
    """

    NTSC_CHROMA_I = 1_300_000.0
    NTSC_CHROMA_Q = 600_000.0
    PAL_CHROMA_UV = 1_300_000.0

    def __init__(self, cutoff: float):
        self.cutoff = cutoff
