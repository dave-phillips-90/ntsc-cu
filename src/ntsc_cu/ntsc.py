import math
import random
from enum import Enum, IntEnum
from importlib.resources import as_file, files

import cv2
import numpy as np
import numpy.typing as npt
import scipy
from scipy.ndimage import shift  # pyright: ignore[reportUnknownVariableType]
from scipy.signal import lfilter  # pyright: ignore[reportUnknownVariableType]


class ScanField(IntEnum):
    TOP = 0  # Even lines: 0, 2, 4...
    BOTTOM = 1  # Odd lines: 1, 3, 5...


M_PI = math.pi

INT_MIN_VALUE = np.iinfo(np.int32).min
INT_MAX_VALUE = np.iinfo(np.int32).max


pattern_file = files("ntsc_cu").joinpath("data", "ring_pattern.npy")
with as_file(pattern_file) as local_path:
    RING_PATTERN = np.load(local_path)


def ringing(
    img2d: np.ndarray,
    alpha: float = 0.5,
    noise_size: float = 0,
    noise_value: float = 2,
    clip: bool = True,
    seed: npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None = None,
):
    """
    Simulates NTSC/VHS ringing artifacts using a frequency-domain band-pass filter.

    This function transforms a 2D image into the frequency domain via a Discrete
    Fourier Transform (DFT), applies a vertical mask to limit horizontal bandwidth,
    and optionally introduces stochastic noise to the filter edges to simulate
    analog signal instability. The result is the characteristic "ghosting" or
    "ringing" ripples seen near sharp vertical edges in analog video.

    Parameters
    ----------
    img2d : ndarray
        Input 2D image array (grayscale).
    alpha : float, optional
        Reconstruction quality (0.0 to 1.0). Controls the width of the horizontal
        frequency pass-band. Lower values result in narrower bandwidth and more
        pronounced ringing/blurring. Optimal values: 0.3 to 0.99. Default is 0.5.
    noiseSize : float, optional
        The extent of the noise interference on the filter edges (0.0 to 1.0).
        If 0, no noise is applied. Optimal values: 0.5 to 0.99. Default is 0.
    noiseValue : float, optional
        The amplitude of the noise applied to the frequency mask. Higher values
        create more erratic ringing artifacts. Optimal values: 0.5 to 5.0.
        Default is 2.
    clip : bool, optional
        Whether to clip the output pixel values to the range of the original
        input image. Prevents out-of-bounds intensity values caused by the
        Gibbs phenomenon. Default is True.
    seed : int, optional
        Seed for the random number generator to ensure reproducible noise
        patterns. Default is None.

    Returns
    -------
    ndarray
        The processed 2D image containing simulated ringing artifacts.

    Notes
    -----
    The function uses a dual-channel mask (shape: rows x cols x 2) to interact
    with the complex output (Real and Imaginary) of the OpenCV DFT. By zeroing
    out high horizontal frequencies (the outer columns of the shifted DFT), it
    replicates the physical bandwidth limitations of VHS tape heads.
    """
    dft = cv2.dft(img2d.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img2d.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)

    mask_h = min(crow, int(1 + alpha * crow))
    mask[:, ccol - mask_h : ccol + mask_h] = 1

    if noise_size > 0:
        noise = (
            np.ones((mask.shape[0], mask.shape[1], mask.shape[2])) * noise_value
            - noise_value / 2.0
        )
        start = int(ccol - ((1 - noise_size) * ccol))
        stop = int(ccol + ((1 - noise_size) * ccol))
        noise[:, start:stop, :] = 0
        rnd = np.random.RandomState(seed)
        mask = (
            mask.astype(float)
            + rnd.rand(mask.shape[0], mask.shape[1], mask.shape[2]) * noise
            - noise / 2.0
        )

    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask), flags=cv2.DFT_SCALE)
    if clip:
        _min, _max = img2d.min(), img2d.max()
        return np.clip(img_back[:, :, 0], _min, _max)
    else:
        return img_back[:, :, 0]


def ringing2(img2d: np.ndarray, power: int = 4, shift: int = 0, clip: bool = True):
    """
    Simulates ringing artifacts using a precomputed impulse response pattern.

    Unlike the stochastic method, this function uses a fixed frequency-response
    curve (`RING_PATTERN`) to filter horizontal frequencies. This more
    accurately models the deterministic behavior of analog video circuitry
    and bandwidth-limited transmission cables.

    Parameters
    ----------
    img2d : ndarray
        Input 2D image array (grayscale).
    power : int, optional
        The filter order/steepness. Higher values narrow the pass-band and
        sharpen the frequency cutoff, leading to more aggressive ringing.
        Typical values are 2 to 6. Default is 4.
    shift : int, optional
        Horizontal frequency shift/scaling factor. Adjusts the effective
        bandwidth by stretching or shrinking the filter mask. Default is 0.
    clip : bool, optional
        Whether to constrain output pixels to the original input's intensity
        range. Necessary to handle overshoot from the Gibbs phenomenon.
        Default is True.

    Returns
    -------
    ndarray
        The processed 2D image containing simulated ringing artifacts.

    Notes
    -----
    This function performs a 1D horizontal filtration across a 2D DFT by
    broadcasting a 1D mask across all rows and both complex channels of
    the frequency-shifted image.
    """
    dft = cv2.dft(img2d.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    _, cols = img2d.shape

    scalecols = int(cols * (1 + shift))
    mask = cv2.resize(  # pyright: ignore[reportUnknownVariableType]
        RING_PATTERN[np.newaxis, :], (scalecols, 1), interpolation=cv2.INTER_LINEAR
    )[0]

    mask = mask[(scalecols // 2) - (cols // 2) : (scalecols // 2) + (cols // 2)]  # pyright: ignore[reportUnknownVariableType]
    mask = mask**power  # pyright: ignore[reportUnknownVariableType]
    img_back = cv2.idft(
        np.fft.ifftshift(dft_shift * mask[None, :, None]),  # pyright: ignore[reportUnknownArgumentType]
        flags=cv2.DFT_SCALE,
    )
    if clip:
        _min, _max = img2d.min(), img2d.max()
        return np.clip(img_back[:, :, 0], _min, _max)
    else:
        return img_back[:, :, 0]


def bgr2yiq(bgrimg: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
    """Convert a BGR image to YIQ colour space using the standard NTSC matrix.

    The result is scaled by 256 and stored as int32 to preserve sub-unit
    precision in a fixed-point representation.

    Args:
        bgrimg: Input image in BGR channel order with shape (H, W, 3) and
            dtype uint8.

    Returns:
        YIQ image with shape (3, H, W) and dtype int32, where each channel
        value is scaled by 256 (i.e. true value = result / 256).
    """
    # Standard NTSC RGB -> YIQ Matrix (assuming BGR input)
    matrix = np.array(
        [
            [0.114, 0.587, 0.299],  # Coefficients for B, G, R -> Y
            [-0.322, -0.274, 0.596],  # Coefficients for B, G, R -> I
            [0.312, -0.523, 0.211],  # Coefficients for B, G, R -> Q
        ]
    )

    # Convert to float32, apply transformation and transpose
    yiq = (bgrimg.astype(np.float32) @ matrix.T).transpose(2, 0, 1)

    # Scale the result by 256 and cast to int32
    return (yiq * 256).astype(np.int32)


def yiq2bgr(
    yiq: npt.NDArray[np.int32], field: int | ScanField = ScanField.TOP
) -> npt.NDArray[np.uint8]:
    """Convert a fixed-point YIQ image back to BGR, extracting a single scan field.

    Reverses the scaling applied by ``bgr2yiq`` (divides by 256) and applies
    the inverse NTSC matrix to recover BGR values, then clips the result to
    the valid [0, 255] range.

    Args:
        yiq: YIQ image with shape (3, H, W) and dtype int32, as produced by
            ``bgr2yiq``. Values are expected to be scaled by 256.
        field: The interlaced scan field to extract. Either a ``ScanField``
            enum member or a raw integer row offset. Rows are selected as
            ``field, field+2, field+4, ...`` along the height axis.
            Defaults to ``ScanField.TOP``.

    Returns:
        Reconstructed BGR image with shape (H_field, W, 3) and dtype uint8,
        where H_field is half the height of the input (one interlaced field).
    """
    # Select the field (rows)
    # yiq is (3, H, W)
    y_luma, i_modulation, q_uad = yiq[:, field::2, :]

    # Inverse YIQ -> BGR Matrix with standard NTSC coefficients
    inv_matrix = np.array(
        [
            [1.0, -1.106, 1.703],  # B
            [1.0, -0.272, -0.647],  # G
            [1.0, 0.956, 0.621],  # R
        ]
    )

    # Stack channels for dot product: (H_field, W, 3)
    yiq_stack = np.stack([y_luma, i_modulation, q_uad], axis=-1).astype(np.float32)

    # Apply inverse transform and the /256 scaling
    # We use 256.0 to match the original fixed-point scale
    bgr_float = (yiq_stack @ inv_matrix.T) / 256.0

    # Clip and convert to uint8
    return np.clip(bgr_float, 0, 255).astype(np.uint8)


class LowpassFilter:
    def __init__(self, rate: float, hz: float, value: float = 0.0):
        self.timeInterval: float = 1.0 / rate
        self.tau: float = 1 / (hz * 2.0 * M_PI)
        self.alpha: float = self.timeInterval / (self.tau + self.timeInterval)
        self.prev: float = value

    def lowpass(self, sample: float) -> float:
        stage1 = sample * self.alpha
        stage2 = self.prev - self.prev * self.alpha
        self.prev = stage1 + stage2
        return self.prev

    def highpass(self, sample: float) -> float:
        stage1 = sample * self.alpha
        stage2 = self.prev - self.prev * self.alpha
        self.prev = stage1 + stage2
        return sample - self.prev

    def lowpass_array(self, samples: np.ndarray) -> np.ndarray:
        if self.prev == 0.0:
            return lfilter([self.alpha], [1, -(1.0 - self.alpha)], samples)  # pyright: ignore[reportReturnType]
        else:
            ic = scipy.signal.lfiltic(
                [self.alpha], [1, -(1.0 - self.alpha)], [self.prev]
            )
            return lfilter([self.alpha], [1, -(1.0 - self.alpha)], samples, zi=ic)[0]

    def highpass_array(self, samples: np.ndarray) -> np.ndarray:
        f = self.lowpass_array(samples)
        return samples - f


def composite_lowpass(yiq: np.ndarray, field: int, fieldno: int):
    _, height, width = yiq.shape
    fY, fI, fQ = yiq
    for p in range(1, 3):
        cutoff = 1300000.0 if p == 1 else 600000.0
        delay = 2 if (p == 1) else 4
        P = fI if (p == 1) else fQ
        P = P[field::2]
        lp = lowpassFilters(cutoff, reset=0.0)
        for i, f in enumerate(P):
            f = lp[0].lowpass_array(f)
            f = lp[1].lowpass_array(f)
            f = lp[2].lowpass_array(f)
            P[i, 0 : width - delay] = f.astype(np.int32)[delay:]


# lighter-weight filtering, probably what your old CRT does to reduce color fringes a bit
def composite_lowpass_tv(yiq: np.ndarray, field: int, fieldno: int):
    _, height, width = yiq.shape
    fY, fI, fQ = yiq
    for p in range(1, 3):
        delay = 1
        P = fI if (p == 1) else fQ
        P = P[field::2]
        lp = lowpassFilters(2600000.0, reset=0.0)
        for i, f in enumerate(P):
            f = lp[0].lowpass_array(f)
            f = lp[1].lowpass_array(f)
            f = lp[2].lowpass_array(f)
            P[i, 0 : width - delay] = f.astype(np.int32)[delay:]


def composite_preemphasis(
    yiq: np.ndarray,
    field: int,
    composite_preemphasis: float,
    composite_preemphasis_cut: float,
):
    fY, fI, fQ = yiq
    pre = LowpassFilter(Ntsc.NTSC_RATE, composite_preemphasis_cut, 16.0)
    fields = fY[field::2]
    for i, samples in enumerate(fields):
        filtered = samples + pre.highpass_array(samples) * composite_preemphasis
        fields[i] = filtered.astype(np.int32)


class VHSSpeed(Enum):
    VHS_SP = (2400000.0, 320000.0, 9)
    VHS_LP = (1900000.0, 300000.0, 12)
    VHS_EP = (1400000.0, 280000.0, 14)

    def __init__(self, luma_cut: float, chroma_cut: float, chroma_delay: int):
        self.luma_cut = luma_cut
        self.chroma_cut = chroma_cut
        self.chroma_delay = chroma_delay


class Ntsc:
    # https://en.wikipedia.org/wiki/NTSC
    NTSC_RATE = 315000000.00 / 88 * 4  # 315/88 Mhz rate * 4

    def __init__(self, precise=False, random=None):
        self.precise = precise
        self.random = random if random is not None else XorWowRandom(31374242, 0)
        self._composite_preemphasis_cut = 1000000.0
        # analog artifacts related to anything that affects the raw composite signal i.e. CATV modulation
        self._composite_preemphasis = 0.0  # values 0..8 look realistic

        self._vhs_out_sharpen = 1.5  # 1.0..5.0

        self._vhs_edge_wave = 0  # 0..10

        self._vhs_head_switching = (
            False  # turn this on only on frames height 486 pixels or more
        )
        self._vhs_head_switching_point = (
            1.0 - (4.5 + 0.01) / 262.5
        )  # 4 scanlines NTSC up from vsync
        self._vhs_head_switching_phase = (
            1.0 - 0.01
        ) / 262.5  # 4 scanlines NTSC up from vsync
        self._vhs_head_switching_phase_noise = (
            1.0 / 500 / 262.5
        )  # 1/500th of a scanline

        self._color_bleed_before = True  # color bleed comes before other degradations if True or after otherwise
        self._color_bleed_horiz = (
            0  # horizontal color bleeding 0 = no color bleed, 1..10 sane values
        )
        self._color_bleed_vert = (
            0  # vertical color bleeding  0 = no color bleed, 1..10 sane values
        )
        self._ringing = 1.0  # 1 = no ringing, 0.3..0.99 = sane values
        self._enable_ringing2 = False
        self._ringing_power = 2
        self._ringing_shift = 0
        self._freq_noise_size = (
            0  # (0-1) optimal values  is 0.5..0.99 if noiseSize=0 - no noise
        )
        self._freq_noise_amplitude = (
            2  # noise amplitude  (0-5) optimal values  is 0.5-2
        )
        self._composite_in_chroma_lowpass = (
            True  # apply chroma lowpass before composite encode
        )
        self._composite_out_chroma_lowpass = True
        self._composite_out_chroma_lowpass_lite = True

        self._video_chroma_noise = 0  # 0..16384
        self._video_chroma_phase_noise = 0  # 0..50
        self._video_chroma_loss = 0  # 0..100_000
        self._video_noise = 2  # 0..4200
        self._subcarrier_amplitude = 50
        self._subcarrier_amplitude_back = 50
        self._emulating_vhs = False
        self._nocolor_subcarrier = (
            False  # if set, emulate subcarrier but do not decode back to color (debug)
        )
        self._vhs_chroma_vert_blend = True  # if set, and VHS, blend vertically the chroma scanlines (as the VHS format does)
        self._vhs_svideo_out = (
            False  # if not set, and VHS, video is recombined as if composite out on VCR
        )

        self._output_ntsc = True  # NTSC color subcarrier emulation
        self._video_scanline_phase_shift = 180
        self._video_scanline_phase_shift_offset = 0  # 0..4
        self._output_vhs_tape_speed = VHSSpeed.VHS_SP

    def rand(self) -> np.int32:
        return self.random.nextInt(_from=0)

    def rand_array(self, size: int) -> np.ndarray:
        return self.random.nextIntArray(size, 0, INT_MAX_VALUE)

    def video_noise(self, yiq: np.ndarray, field: int, video_noise: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        noise_mod = video_noise * 2 + 1
        fields = fY[field::2]
        fh, fw = fields.shape
        if not self.precise:  # this one works FAST
            lp = LowpassFilter(1, 1, 0)
            lp.alpha = 0.5
            rnds = self.rand_array(fw * fh) % noise_mod - video_noise
            noises = shift(lp.lowpass_array(rnds).astype(np.int32), 1)
            fields += noises.reshape(fields.shape)
        else:  # this one works EXACTLY like original code
            noise = 0
            for field1 in fields:
                rnds = self.rand_array(fw) % noise_mod - video_noise
                for x in range(0, fw):
                    field1[x] += noise
                    noise += rnds[x]
                    noise = int(noise / 2)

    # https://bavc.github.io/avaa/artifacts/chrominance_noise.html
    def video_chroma_noise(self, yiq: np.ndarray, field: int, video_chroma_noise: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq

        noise_mod = video_chroma_noise * 2 + 1
        U = fI[field::2]
        V = fQ[field::2]
        fh, fw = U.shape
        if not self.precise:
            lp = LowpassFilter(1, 1, 0)
            lp.alpha = 0.5
            rndsU = self.rand_array(fw * fh) % noise_mod - video_chroma_noise
            noisesU = shift(lp.lowpass_array(rndsU).astype(np.int32), 1)

            rndsV = self.rand_array(fw * fh) % noise_mod - video_chroma_noise
            noisesV = shift(lp.lowpass_array(rndsV).astype(np.int32), 1)

            U += noisesU.reshape(U.shape)
            V += noisesV.reshape(V.shape)
        else:
            noiseU = 0
            noiseV = 0
            for y in range(0, fh):
                for x in range(0, fw):
                    U[y][x] += noiseU
                    noiseU += self.rand() % noise_mod - video_chroma_noise
                    noiseU = int(noiseU / 2)

                    V[y][x] += noiseV
                    noiseV += self.rand() % noise_mod - video_chroma_noise
                    noiseV = int(noiseV / 2)

    def video_chroma_phase_noise(
        self, yiq: np.ndarray, field: int, video_chroma_phase_noise: int
    ):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        noise_mod = video_chroma_phase_noise * 2 + 1
        U = fI[field::2]
        V = fQ[field::2]
        fh, fw = U.shape
        noise = 0
        for y in range(0, fh):
            noise += self.rand() % noise_mod - video_chroma_phase_noise
            noise = int(noise / 2)
            pi = noise * M_PI / 100
            sinpi = math.sin(pi)
            cospi = math.cos(pi)
            u = U[y] * cospi - V[y] * sinpi
            v = U[y] * sinpi + V[y] * cospi
            U[y, :] = u
            V[y, :] = v

    def vhs_head_switching(self, yiq: np.ndarray, field: int = 0):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        twidth = width + width // 10
        shy = 0
        noise = 0.0
        if self._vhs_head_switching_phase_noise != 0.0:
            x = np.int32(
                (
                    int(self.rand())
                    * int(self.rand())
                    * int(self.rand())
                    * int(self.rand())
                )
                % (2**31)
            )
            x %= 2000000000
            noise = x / 1000000000.0 - 1.0
            noise *= self._vhs_head_switching_phase_noise

        t = twidth * (262.5 if self._output_ntsc else 312.5)
        p = int(fmod(self._vhs_head_switching_point + noise, 1.0) * t)
        y = int(p // twidth * 2) + field
        p = int(fmod(self._vhs_head_switching_phase + noise, 1.0) * t)
        x = p % twidth
        y -= (262 - 240) * 2 if self._output_ntsc else (312 - 288) * 2
        tx = x
        ishif = x - twidth if x >= twidth // 2 else x
        shif = 0
        while y < height:
            if y >= 0:
                Y = fY[y]
                if shif != 0:
                    tmp = np.zeros(twidth)
                    x2 = (tx + twidth + shif) % twidth
                    tmp[:width] = Y

                    x = tx
                    while x < width:
                        Y[x] = tmp[x2]
                        x2 += 1
                        if x2 == twidth:
                            x2 = 0
                        x += 1

            shif = ishif if shy == 0 else int(shif * 7 / 8)
            tx = 0
            y += 2
            shy += 1

    _Umult = np.array([1, 0, -1, 0], dtype=np.int32)
    _Vmult = np.array([0, 1, 0, -1], dtype=np.int32)

    def _chroma_luma_xi(self, fieldno: int, y: int):
        if self._video_scanline_phase_shift == 90:
            return int(fieldno + self._video_scanline_phase_shift_offset + (y >> 1)) & 3
        elif self._video_scanline_phase_shift == 180:
            return int(
                (((fieldno + y) & 2) + self._video_scanline_phase_shift_offset) & 3
            )
        elif self._video_scanline_phase_shift == 270:
            return int((fieldno + self._video_scanline_phase_shift_offset) & 3)
        else:
            return int(self._video_scanline_phase_shift_offset & 3)

    def chroma_into_luma(
        self, yiq: np.ndarray, field: int, fieldno: int, subcarrier_amplitude: int
    ):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        y = field
        umult = np.tile(Ntsc._Umult, int((width / 4) + 1))
        vmult = np.tile(Ntsc._Vmult, int((width / 4) + 1))
        while y < height:
            Y = fY[y]
            I = fI[y]
            Q = fQ[y]
            xi = self._chroma_luma_xi(fieldno, y)

            chroma = I * subcarrier_amplitude * umult[xi : xi + width]
            chroma += Q * subcarrier_amplitude * vmult[xi : xi + width]
            Y[:] = Y + chroma.astype(np.int32) // 50
            I[:] = 0
            Q[:] = 0
            y += 2

    def chroma_from_luma(
        self, yiq: np.ndarray, field: int, fieldno: int, subcarrier_amplitude: int
    ):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        chroma = np.zeros(width, dtype=np.int32)
        for y in range(field, height, 2):
            Y = fY[y]
            I = fI[y]
            Q = fQ[y]
            sum: int = Y[0] + Y[1]
            y2 = np.pad(Y[2:], (0, 2))
            yd4 = np.pad(Y[:-2], (2, 0))
            sums = y2 - yd4
            sums0 = np.concatenate([np.array([sum], dtype=np.int32), sums])
            acc = np.add.accumulate(sums0, dtype=np.int32)[1:]
            acc4 = acc // 4
            chroma = y2 - acc4
            Y[:] = acc4

            xi = self._chroma_luma_xi(fieldno, y)

            x = 4 - xi & 3
            # // flip the part of the sine wave that would correspond to negative U and V values
            chroma[x + 2 :: 4] = -chroma[x + 2 :: 4]
            chroma[x + 3 :: 4] = -chroma[x + 3 :: 4]

            chroma = chroma * 50 / subcarrier_amplitude

            # decode the color right back out from the subcarrier we generated
            cxi = -chroma[xi::2]
            cxi1 = -chroma[xi + 1 :: 2]
            I[::2] = np.pad(cxi, (0, width // 2 - cxi.shape[0]))
            Q[::2] = np.pad(cxi1, (0, width // 2 - cxi1.shape[0]))

            I[1 : width - 2 : 2] = (I[: width - 2 : 2] + I[2::2]) >> 1
            Q[1 : width - 2 : 2] = (Q[: width - 2 : 2] + Q[2::2]) >> 1
            I[width - 2 :] = 0
            Q[width - 2 :] = 0

    def vhs_luma_lowpass(self, yiq: np.ndarray, field: int, luma_cut: float):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        for Y in fY[field::2]:
            pre = LowpassFilter(Ntsc.NTSC_RATE, luma_cut, 16.0)
            lp = lowpassFilters(cutoff=luma_cut, reset=16.0)
            f0 = lp[0].lowpass_array(Y)
            f1 = lp[1].lowpass_array(f0)
            f2 = lp[2].lowpass_array(f1)
            f3 = f2 + pre.highpass_array(f2) * 1.6
            Y[:] = f3

    def vhs_chroma_lowpass(
        self, yiq: np.ndarray, field: int, chroma_cut: float, chroma_delay: int
    ):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        for U in fI[field::2]:
            lpU = lowpassFilters(cutoff=chroma_cut, reset=0.0)
            f0 = lpU[0].lowpass_array(U)
            f1 = lpU[1].lowpass_array(f0)
            f2 = lpU[2].lowpass_array(f1)
            U[: width - chroma_delay] = f2[chroma_delay:]

        for V in fQ[field::2]:
            lpV = lowpassFilters(cutoff=chroma_cut, reset=0.0)
            f0 = lpV[0].lowpass_array(V)
            f1 = lpV[1].lowpass_array(f0)
            f2 = lpV[2].lowpass_array(f1)
            V[: width - chroma_delay] = f2[chroma_delay:]

    # VHS decks also vertically smear the chroma subcarrier using a delay line
    # to add the previous line's color subcarrier to the current line's color subcarrier.
    # note that phase changes in NTSC are compensated for by the VHS deck to make the
    # phase line up per scanline (else summing the previous line's carrier would
    # cancel it out).
    def vhs_chroma_vert_blend(self, yiq: np.ndarray, field: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        U2 = fI[field + 2 :: 2,]
        V2 = fQ[field + 2 :: 2,]
        delayU = np.pad(U2[:-1,], [[1, 0], [0, 0]])
        delayV = np.pad(V2[:-1,], [[1, 0], [0, 0]])
        fI[field + 2 :: 2,] = (delayU + U2 + 1) >> 1
        fQ[field + 2 :: 2,] = (delayV + V2 + 1) >> 1

    def vhs_sharpen(self, yiq: np.ndarray, field: int, luma_cut: float):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        for Y in fY[field::2]:
            lp = lowpassFilters(cutoff=luma_cut * 4, reset=0.0)
            s = Y
            ts = lp[0].lowpass_array(Y)
            ts = lp[1].lowpass_array(ts)
            ts = lp[2].lowpass_array(ts)
            Y[:] = s + (s - ts) * self._vhs_out_sharpen * 2.0

    # http://www.michaeldvd.com.au/Articles/VideoArtefacts/VideoArtefactsColourBleeding.html
    # https://bavc.github.io/avaa/artifacts/yc_delay_error.html
    def color_bleed(self, yiq: np.ndarray, field: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq

        field_ = fI[field::2]
        h, w = field_.shape
        fI[field::2] = np.pad(
            field_, ((self._color_bleed_vert, 0), (self._color_bleed_horiz, 0))
        )[0:h, 0:w]

        field_ = fQ[field::2]
        h, w = field_.shape
        fQ[field::2] = np.pad(
            field_, ((self._color_bleed_vert, 0), (self._color_bleed_horiz, 0))
        )[0:h, 0:w]

    def vhs_edge_wave(self, yiq: np.ndarray, field: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        rnds = self.random.nextIntArray(height // 2, 0, self._vhs_edge_wave)
        lp = LowpassFilter(
            Ntsc.NTSC_RATE, self._output_vhs_tape_speed.luma_cut, 0
        )  # no real purpose to initialize it with ntsc values
        rnds = lp.lowpass_array(rnds).astype(np.int32)

        for y, Y in enumerate(fY[field::2]):
            if rnds[y] != 0:
                shift = rnds[y]
                Y[:] = np.pad(Y, (shift, 0))[:-shift]
        for y, I in enumerate(fI[field::2]):
            if rnds[y] != 0:
                shift = rnds[y]
                I[:] = np.pad(I, (shift, 0))[:-shift]
        for y, Q in enumerate(fQ[field::2]):
            if rnds[y] != 0:
                shift = rnds[y]
                Q[:] = np.pad(Q, (shift, 0))[:-shift]

    def vhs_chroma_loss(self, yiq: np.ndarray, field: int, video_chroma_loss: int):
        _, height, width = yiq.shape
        fY, fI, fQ = yiq
        for y in range(field, height, 2):
            U = fI[y]
            V = fQ[y]
            if self.rand() % 100000 < video_chroma_loss:
                U[:] = 0
                V[:] = 0

    def emulate_vhs(self, yiq: np.ndarray, field: int, fieldno: int):
        vhs_speed = self._output_vhs_tape_speed
        if self._vhs_edge_wave != 0:
            self.vhs_edge_wave(yiq, field)

        self.vhs_luma_lowpass(yiq, field, vhs_speed.luma_cut)

        self.vhs_chroma_lowpass(
            yiq, field, vhs_speed.chroma_cut, vhs_speed.chroma_delay
        )

        if self._vhs_chroma_vert_blend and self._output_ntsc:
            self.vhs_chroma_vert_blend(yiq, field)

        if True:  # TODO: make option
            self.vhs_sharpen(yiq, field, vhs_speed.luma_cut)

        if not self._vhs_svideo_out:
            self.chroma_into_luma(yiq, field, fieldno, self._subcarrier_amplitude)
            self.chroma_from_luma(yiq, field, fieldno, self._subcarrier_amplitude)

    def composite_layer(
        self, dst: np.ndarray, src: np.ndarray, field: int, fieldno: int
    ):
        assert dst.shape == src.shape, "dst and src images must be of same shape"
        yiq = bgr2yiq(src)

        if self._color_bleed_before and (
            self._color_bleed_vert != 0 or self._color_bleed_horiz != 0
        ):
            self.color_bleed(yiq, field)

        if self._composite_in_chroma_lowpass:
            composite_lowpass(yiq, field, fieldno)

        if self._ringing != 1.0:
            self.ringing(yiq, field)

        self.chroma_into_luma(yiq, field, fieldno, self._subcarrier_amplitude)

        if self._composite_preemphasis != 0.0 and self._composite_preemphasis_cut > 0:
            composite_preemphasis(
                yiq, field, self._composite_preemphasis, self._composite_preemphasis_cut
            )

        if self._video_noise != 0:
            self.video_noise(yiq, field, self._video_noise)

        if self._vhs_head_switching:
            self.vhs_head_switching(yiq, field)

        if not self._nocolor_subcarrier:
            self.chroma_from_luma(yiq, field, fieldno, self._subcarrier_amplitude_back)

        if self._video_chroma_noise != 0:
            self.video_chroma_noise(yiq, field, self._video_chroma_noise)

        if self._video_chroma_phase_noise != 0:
            self.video_chroma_phase_noise(yiq, field, self._video_chroma_phase_noise)

        if self._emulating_vhs:
            self.emulate_vhs(yiq, field, fieldno)

        if self._video_chroma_loss != 0:
            self.vhs_chroma_loss(yiq, field, self._video_chroma_loss)

        if self._composite_out_chroma_lowpass:
            if self._composite_out_chroma_lowpass_lite:
                composite_lowpass_tv(yiq, field, fieldno)
            else:
                composite_lowpass(yiq, field, fieldno)

        if not self._color_bleed_before and (
            self._color_bleed_vert != 0 or self._color_bleed_horiz != 0
        ):
            self.color_bleed(yiq, field)

        # if self._ringing != 1.0:
        #     self.ringing(yiq, field)

        Y, I, Q = yiq

        # simulate 2x less bandwidth for chroma components, just like yuv420
        I[field::2] = self._blur_chroma(I[field::2])
        Q[field::2] = self._blur_chroma(Q[field::2])

        yiq2bgr(yiq, dst, field)

    def _blur_chroma(self, chroma: np.ndarray) -> np.ndarray:
        h, w = chroma.shape
        down2 = cv2.resize(
            chroma.astype(np.float32),
            (w // 2, h // 2),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return cv2.resize(down2, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(
            np.int32
        )

    def ringing(self, yiq: np.ndarray, field: int):
        Y, I, Q = yiq
        sz = self._freq_noise_size
        amp = self._freq_noise_amplitude
        shift = self._ringing_shift
        if not self._enable_ringing2:
            Y[field::2] = ringing(
                Y[field::2], self._ringing, noise_size=sz, noise_value=amp, clip=False
            )
            I[field::2] = ringing(
                I[field::2], self._ringing, noise_size=sz, noise_value=amp, clip=False
            )
            Q[field::2] = ringing(
                Q[field::2], self._ringing, noise_size=sz, noise_value=amp, clip=False
            )
        else:
            Y[field::2] = ringing2(
                Y[field::2], power=self._ringing_power, shift=shift, clip=False
            )
            I[field::2] = ringing2(
                I[field::2], power=self._ringing_power, shift=shift, clip=False
            )
            Q[field::2] = ringing2(
                Q[field::2], power=self._ringing_power, shift=shift, clip=False
            )


def random_ntsc(seed: int | float | str | bytes | bytearray | None = None) -> Ntsc:
    rnd = random.Random(seed)  # noqa: S311
    ntsc = Ntsc(random=npRandom(seed))
    ntsc._composite_preemphasis = rnd.triangular(0, 8, 0)
    ntsc._vhs_out_sharpen = rnd.triangular(1, 5, 1.5)
    ntsc._composite_in_chroma_lowpass = rnd.random() < 0.8  # lean towards default value
    ntsc._composite_out_chroma_lowpass = (
        rnd.random() < 0.8
    )  # lean towards default value
    ntsc._composite_out_chroma_lowpass_lite = (
        rnd.random() < 0.8
    )  # lean towards default value
    ntsc._video_chroma_noise = int(rnd.triangular(0, 16384, 2))
    ntsc._video_chroma_phase_noise = int(rnd.triangular(0, 50, 2))
    ntsc._video_chroma_loss = int(rnd.triangular(0, 50000, 10))
    ntsc._video_noise = int(rnd.triangular(0, 4200, 2))
    ntsc._emulating_vhs = rnd.random() < 0.2  # lean towards default value
    ntsc._vhs_edge_wave = int(rnd.triangular(0, 5, 0))
    ntsc._video_scanline_phase_shift = rnd.choice([0, 90, 180, 270])
    ntsc._video_scanline_phase_shift_offset = rnd.randint(0, 3)
    ntsc._output_vhs_tape_speed = rnd.choice(
        [VHSSpeed.VHS_SP, VHSSpeed.VHS_LP, VHSSpeed.VHS_EP]
    )
    enable_ringing = rnd.random() < 0.8
    if enable_ringing:
        ntsc._ringing = rnd.uniform(0.3, 0.7)
        enable_freq_noise = rnd.random() < 0.8
        if enable_freq_noise:
            ntsc._freq_noise_size = rnd.uniform(0.5, 0.99)
            ntsc._freq_noise_amplitude = rnd.uniform(0.5, 2.0)
        ntsc._enable_ringing2 = rnd.random() < 0.5
        ntsc._ringing_power = rnd.randint(2, 7)
    ntsc._color_bleed_before = rnd.randint(0, 1) == 1
    ntsc._color_bleed_horiz = int(rnd.triangular(0, 8, 0))
    ntsc._color_bleed_vert = int(rnd.triangular(0, 8, 0))
    return ntsc


def lowpassFilters(
    cutoff: float, reset: float, rate: float = Ntsc.NTSC_RATE
) -> list[LowpassFilter]:
    return [LowpassFilter(rate, cutoff, reset) for x in range(0, 3)]
