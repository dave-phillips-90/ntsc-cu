import numpy as np
import numpy.typing as npt
from numpy import int32, uint32
from numpy.random.mtrand import RandomState


INT_MIN_VALUE = np.iinfo(np.int32).min
INT_MAX_VALUE = np.iinfo(np.int32).max


def fmod(x: float, y: float) -> float:
    """
    Calculates the modulo of two floating point numbers
    """
    return x % y


class NumpyRandom:
    def __init__(
        self, seed: npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None = None
    ):
        self._rnd: RandomState = np.random.RandomState(seed)

    def next_int(self, _from: int = INT_MIN_VALUE, until: int = INT_MAX_VALUE) -> int:
        return self._rnd.randint(_from, until)

    def next_int_array(
        self, size: int, _from: int = INT_MIN_VALUE, until: int = INT_MAX_VALUE
    ) -> np.ndarray:
        return self._rnd.randint(_from, until, size, dtype=np.int32)


class XorWowRandom:
    def __init__(self, seed1: int, seed2: int):
        self.x: int32 = int32(seed1)
        self.y: int32 = int32(seed2)
        self.z: int32 = int32(0)
        self.w: int32 = int32(0)
        self.v: int32 = -int32(seed1) - 1
        self.addend: int32 = int32((int32(seed1) << 10) ^ (uint32(seed2) >> 4))
        [self._next_int() for _ in range(0, 64)]

    def _next_int(self) -> int32:
        t = self.x
        t: int32 = np.int32(t ^ (np.uint32(t) >> 2))
        self.x = int32(self.y)
        self.y = int32(self.z)
        self.z = int32(self.w)
        v0 = int32(self.v)
        self.w = int32(v0)
        t = (t ^ (t << 1)) ^ v0 ^ (v0 << 4)
        self.v = int32(t)
        self.addend += 362437
        return t + int32(self.addend)

    def next_int(self, start: int = INT_MIN_VALUE, end: int = INT_MAX_VALUE) -> int32:
        _range = end - start
        if _range > 0 or _range == INT_MIN_VALUE:
            if (_range & -_range) == _range:
                assert False, "not implemented"
            else:
                v: int = 0
                while True:
                    bits = int(self._next_int()) >> 1
                    v = bits % _range
                    if bits - v + (_range - 1) >= 0:
                        break
                return int32(start + v)
        else:
            r = range(start, end)
            while True:
                rnd = self._next_int()
                if rnd in r:
                    return int32(rnd)

    def next_int_array(
        self, size: int, _from: int = INT_MIN_VALUE, until: int = INT_MAX_VALUE
    ) -> np.ndarray:
        zeros = np.zeros(size, dtype=np.int32)
        for i in range(0, size):
            zeros[i] = self.next_int(start=_from, end=until)
        return zeros
