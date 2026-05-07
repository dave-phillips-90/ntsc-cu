from importlib.resources import as_file, files

import cv2
import numpy as np
import numpy.typing as npt

from ntsc_cu.typing import (
    ChannelComponent,
)


_pattern_file = files("ntsc_cu").joinpath("data", "ring_pattern.npy")
with as_file(_pattern_file) as local_path:
    _RING_PATTERN = np.load(local_path)


def ringing(
    luma_channel: ChannelComponent,
    alpha: float = 0.5,
    noise_size: float = 0,
    noise_value: float = 2,
    clip: bool = True,
    seed: npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None = None,
) -> ChannelComponent:
    """
    Simulates ringing artifacts using a frequency-domain band-pass filter.

    This function transforms a 2D image into the frequency domain via a Discrete
    Fourier Transform (DFT), applies a vertical mask to limit horizontal bandwidth,
    and optionally introduces stochastic noise to the filter edges to simulate
    analog signal instability. The result is the characteristic ghosting or
    ringing ripples seen near sharp vertical edges in analog video.

    Parameters
    ----------
    luma_channel : ChannelComponent
        Input 2D image array (grayscale).
    alpha : float, optional
        Reconstruction quality (0.0 to 1.0). Controls the width of the horizontal
        frequency pass-band. Lower values result in narrower bandwidth and more
        pronounced ringing/blurring. Optimal values: 0.3 to 0.99. Default is 0.5.
    noise_size : float, optional
        The extent of the noise interference on the filter edges (0.0 to 1.0).
        If 0, no noise is applied. Optimal values: 0.5 to 0.99. Default is 0.
    noise_value : float, optional
        The amplitude of the noise applied to the frequency mask. Higher values
        create more erratic ringing artifacts. Optimal values: 0.5 to 5.0.
        Default is 2.
    clip : bool, optional
        Whether to clip the output pixel values to the range of the original
        input image. Prevents out-of-bounds intensity values caused by the
        Gibbs phenomenon. Default is True.
    seed : npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None, optional
        Seed for the random number generator to ensure reproducible noise
        patterns. Default is None.

    Returns
    -------
    ChannelComponent
        The processed 2D image channel simulated ringing artifacts.

    Notes
    -----
    The function uses a dual-channel mask (shape: rows x cols x 2) to interact
    with the complex output (Real and Imaginary) of the OpenCV DFT. By zeroing
    out high horizontal frequencies (the outer columns of the shifted DFT), it
    replicates the physical bandwidth limitations of period hardware.
    """
    # We use DFT_COMPLEX_OUTPUT because signals have phase and magnitude.
    dft = cv2.dft(luma_channel.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center.
    # Low frequencies (broad shapes) are now in the middle;
    # High frequencies (sharp edges) are at the edges.
    dft_shift = np.fft.fftshift(dft)

    rows, cols = luma_channel.shape
    crow, ccol = rows // 2, cols // 2

    # We create a mask that only allows a narrow vertical band of frequencies. This
    # filter behaves like a brick wall, with a clean cut at the limits.
    # TODO: Butterworth Filter
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask_h = min(crow, int(1 + alpha * crow))
    mask[:, ccol - mask_h : ccol + mask_h] = 1

    # If noise_size is greater than 0
    if noise_size > 0:
        # Generate the random jitter directly centered around 0
        rnd = np.random.RandomState(seed)
        jitter = rnd.uniform(-noise_value / 2.0, noise_value / 2.0, size=mask.shape)

        # Create a boolean mask for the protected center region
        start = int(ccol - ((1 - noise_size) * ccol))
        stop = int(ccol + ((1 - noise_size) * ccol))

        # Zero out the jitter in the protected frequency band
        jitter[:, start:stop, :] = 0.0

        # Apply it to the mask
        mask = mask.astype(float) + jitter

    # Multiplying dft_shift and mask deletes high horizontal frequencies
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask), flags=cv2.DFT_SCALE)

    # The Gibbs Phenomenon creates values > 255 and < 0.
    # Real TVs just 'clipped' these or let them bleed into the sync pulse.
    if clip:
        _min, _max = luma_channel.min(), luma_channel.max()
        return np.clip(img_back[:, :, 0], _min, _max).astype(np.int32)
    else:
        return img_back[:, :, 0].astype(np.int32)


def ringing2(
    luma_channel: ChannelComponent, power: int = 4, shift: int = 0, clip: bool = True
):
    """
    Simulates ringing artifacts using a precomputed impulse response pattern.

    Unlike the stochastic method, this function uses a fixed frequency-response
    curve to filter horizontal frequencies. This more accurately models the
    deterministic behavior of analog video circuitry and bandwidth-limited transmission
    cables.

    Parameters
    ----------
    luma_channel : ChannelComponent
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
    ChannelComponent
        The processed 2D image containing simulated ringing artifacts.

    Notes
    -----
    This function performs a 1D horizontal filtration across a 2D DFT by
    broadcasting a 1D mask across all rows and both complex channels of
    the frequency-shifted image.
    """
    if power < 1:
        raise ValueError(f"power must be >= 1, got {power}")

    dft = cv2.dft(luma_channel.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    _, cols = luma_channel.shape

    scalecols = int(cols * (1 + shift))
    mask = cv2.resize(  # pyright: ignore[reportUnknownVariableType]
        _RING_PATTERN[np.newaxis, :], (scalecols, 1), interpolation=cv2.INTER_LINEAR
    )[0]

    mask = mask[(scalecols // 2) - (cols // 2) : (scalecols // 2) + (cols // 2)]  # pyright: ignore[reportUnknownVariableType]
    mask = mask**power  # pyright: ignore[reportUnknownVariableType]
    img_back = cv2.idft(
        np.fft.ifftshift(dft_shift * mask[None, :, None]),  # pyright: ignore[reportUnknownArgumentType]
        flags=cv2.DFT_SCALE,
    )
    if clip:
        _min, _max = luma_channel.min(), luma_channel.max()
        return np.clip(img_back[:, :, 0], _min, _max).astype(np.int32)
    else:
        return img_back[:, :, 0].astype(np.int32)
