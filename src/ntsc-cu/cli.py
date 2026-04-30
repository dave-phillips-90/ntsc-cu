#!/usr/bin/env python
"""NTSC/VHS video artifact emulation CLI."""

import argparse
import os
import sys

import cv2

from .ntsc import Ntsc, NumpyRandom, VHSSpeed, random_ntsc


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply NTSC/VHS video artifact effects to images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s input.png -o output.png
  %(prog)s input.jpg --random --seed 42
  %(prog)s input.png --vhs --tape-speed EP --video-noise 1000
  %(prog)s input.png --composite-preemphasis 4 --ringing 0.5
  %(prog)s input.png --video-noise 500 --chroma-noise 2000 --chroma-phase-noise 20
  %(prog)s input.png --color-bleed-horiz 4 --color-bleed-vert 3
  %(prog)s input.png --passes 2 -o retro.png
""",
    )

    p.add_argument("input", help="Input image file (png, jpg, bmp, tiff, etc.)")
    p.add_argument(
        "-o", "--output", help="Output image file (default: <input>_ntsc.<ext>)"
    )
    p.add_argument(
        "--passes", type=int, default=1, help="Number of passes to apply (default: 1)"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible results"
    )
    p.add_argument(
        "--precise", action="store_true", help="Use precise (slower) emulation mode"
    )
    p.add_argument(
        "--resize",
        type=str,
        default=None,
        metavar="WxH",
        help="Resize before processing (e.g. 640x480, 320x240). Result is scaled back to original size.",
    )
    p.add_argument(
        "--resize-width",
        type=int,
        default=None,
        metavar="W",
        help="Resize width before processing (height auto-calculated to keep aspect ratio). Result is scaled back.",
    )
    p.add_argument(
        "--resize-height",
        type=int,
        default=None,
        metavar="H",
        help="Resize height before processing (width auto-calculated to keep aspect ratio). Result is scaled back.",
    )
    p.add_argument(
        "--output-size",
        type=str,
        default=None,
        metavar="WxH",
        help="Output resolution (e.g. 1920x1080). Overrides scaling back to original size.",
    )
    p.add_argument(
        "--output-width",
        type=int,
        default=None,
        metavar="W",
        help="Output width (height auto-calculated to keep aspect ratio).",
    )
    p.add_argument(
        "--output-height",
        type=int,
        default=None,
        metavar="H",
        help="Output height (width auto-calculated to keep aspect ratio).",
    )
    p.add_argument(
        "--crop",
        type=str,
        default=None,
        metavar="W:H",
        help="Crop to aspect ratio before processing (e.g. 4:3, 16:9, 3:2). Center crop.",
    )

    rand_group = p.add_argument_group("random preset")
    rand_group.add_argument(
        "--random",
        action="store_true",
        help="Use randomized parameters (ignores other options)",
    )

    comp = p.add_argument_group("composite signal")
    comp.add_argument(
        "--composite-preemphasis",
        type=float,
        default=None,
        metavar="V",
        help="Composite preemphasis (0..8, default: 0)",
    )
    comp.add_argument(
        "--composite-preemphasis-cut",
        type=float,
        default=None,
        metavar="HZ",
        help="Preemphasis cutoff frequency (default: 1000000)",
    )
    comp.add_argument(
        "--chroma-lowpass-in",
        type=int,
        choices=[0, 1],
        default=None,
        help="Chroma lowpass before composite encode (0/1, default: 1)",
    )
    comp.add_argument(
        "--chroma-lowpass-out",
        type=int,
        choices=[0, 1],
        default=None,
        help="Chroma lowpass after composite decode (0/1, default: 1)",
    )
    comp.add_argument(
        "--chroma-lowpass-out-lite",
        type=int,
        choices=[0, 1],
        default=None,
        help="Use lighter chroma lowpass (0/1, default: 1)",
    )

    noise = p.add_argument_group("noise")
    noise.add_argument(
        "--video-noise",
        type=int,
        default=None,
        metavar="V",
        help="Video luma noise (0..4200, default: 2)",
    )
    noise.add_argument(
        "--chroma-noise",
        type=int,
        default=None,
        metavar="V",
        help="Chroma noise (0..16384, default: 0)",
    )
    noise.add_argument(
        "--chroma-phase-noise",
        type=int,
        default=None,
        metavar="V",
        help="Chroma phase noise (0..50, default: 0)",
    )
    noise.add_argument(
        "--chroma-loss",
        type=int,
        default=None,
        metavar="V",
        help="Chroma loss probability (0..100000, default: 0)",
    )

    ring = p.add_argument_group("ringing")
    ring.add_argument(
        "--ringing",
        type=float,
        default=None,
        metavar="V",
        help="Ringing amount (0.3..0.99, 1.0=off, default: 1.0)",
    )
    ring.add_argument(
        "--ringing2",
        action="store_true",
        default=None,
        help="Use alternative ringing algorithm",
    )
    ring.add_argument(
        "--ringing-power",
        type=int,
        default=None,
        metavar="V",
        help="Ringing power for ringing2 (2..7, default: 2)",
    )
    ring.add_argument(
        "--ringing-shift",
        type=float,
        default=None,
        metavar="V",
        help="Ringing shift for ringing2 (default: 0)",
    )
    ring.add_argument(
        "--freq-noise-size",
        type=float,
        default=None,
        metavar="V",
        help="Frequency noise size (0..0.99, default: 0)",
    )
    ring.add_argument(
        "--freq-noise-amplitude",
        type=float,
        default=None,
        metavar="V",
        help="Frequency noise amplitude (0..5, default: 2)",
    )

    bleed = p.add_argument_group("color bleed")
    bleed.add_argument(
        "--color-bleed-horiz",
        type=int,
        default=None,
        metavar="V",
        help="Horizontal color bleed (0..10, default: 0)",
    )
    bleed.add_argument(
        "--color-bleed-vert",
        type=int,
        default=None,
        metavar="V",
        help="Vertical color bleed (0..10, default: 0)",
    )
    bleed.add_argument(
        "--color-bleed-before",
        type=int,
        choices=[0, 1],
        default=None,
        help="Apply color bleed before other effects (0/1, default: 1)",
    )

    sc = p.add_argument_group("subcarrier")
    sc.add_argument(
        "--subcarrier-amplitude",
        type=int,
        default=None,
        metavar="V",
        help="Subcarrier amplitude (default: 50)",
    )
    sc.add_argument(
        "--subcarrier-amplitude-back",
        type=int,
        default=None,
        metavar="V",
        help="Subcarrier decode amplitude (default: 50)",
    )
    sc.add_argument(
        "--scanline-phase-shift",
        type=int,
        choices=[0, 90, 180, 270],
        default=None,
        help="Scanline phase shift (default: 180)",
    )
    sc.add_argument(
        "--scanline-phase-shift-offset",
        type=int,
        default=None,
        metavar="V",
        help="Scanline phase shift offset (0..3, default: 0)",
    )

    vhs = p.add_argument_group("VHS emulation")
    vhs.add_argument("--vhs", action="store_true", help="Enable VHS emulation")
    vhs.add_argument(
        "--tape-speed",
        choices=["SP", "LP", "EP"],
        default=None,
        help="VHS tape speed (default: SP)",
    )
    vhs.add_argument(
        "--vhs-sharpen",
        type=float,
        default=None,
        metavar="V",
        help="VHS output sharpening (1..5, default: 1.5)",
    )
    vhs.add_argument(
        "--vhs-edge-wave",
        type=int,
        default=None,
        metavar="V",
        help="VHS edge wave distortion (0..10, default: 0)",
    )
    vhs.add_argument(
        "--vhs-svideo",
        action="store_true",
        default=None,
        help="VHS S-Video output (skip composite re-encode)",
    )
    vhs.add_argument(
        "--vhs-no-chroma-vert-blend",
        action="store_true",
        default=None,
        help="Disable VHS chroma vertical blend",
    )
    vhs.add_argument(
        "--vhs-head-switching",
        action="store_true",
        default=None,
        help="Enable VHS head switching noise (needs height >= 486)",
    )

    return p.parse_args()


def build_ntsc(args) -> Ntsc:
    if args.random:
        return random_ntsc(seed=args.seed)

    ntsc = Ntsc(precise=args.precise, random=NumpyRandom(args.seed))

    if args.composite_preemphasis is not None:
        ntsc._composite_preemphasis = args.composite_preemphasis
    if args.composite_preemphasis_cut is not None:
        ntsc._composite_preemphasis_cut = args.composite_preemphasis_cut
    if args.chroma_lowpass_in is not None:
        ntsc._composite_in_chroma_lowpass = bool(args.chroma_lowpass_in)
    if args.chroma_lowpass_out is not None:
        ntsc._composite_out_chroma_lowpass = bool(args.chroma_lowpass_out)
    if args.chroma_lowpass_out_lite is not None:
        ntsc._composite_out_chroma_lowpass_lite = bool(args.chroma_lowpass_out_lite)

    if args.video_noise is not None:
        ntsc._video_noise = args.video_noise
    if args.chroma_noise is not None:
        ntsc._video_chroma_noise = args.chroma_noise
    if args.chroma_phase_noise is not None:
        ntsc._video_chroma_phase_noise = args.chroma_phase_noise
    if args.chroma_loss is not None:
        ntsc._video_chroma_loss = args.chroma_loss

    if args.ringing is not None:
        ntsc._ringing = args.ringing
    if args.ringing2 is not None:
        ntsc._enable_ringing2 = args.ringing2
    if args.ringing_power is not None:
        ntsc._ringing_power = args.ringing_power
    if args.ringing_shift is not None:
        ntsc._ringing_shift = args.ringing_shift
    if args.freq_noise_size is not None:
        ntsc._freq_noise_size = args.freq_noise_size
    if args.freq_noise_amplitude is not None:
        ntsc._freq_noise_amplitude = args.freq_noise_amplitude

    if args.color_bleed_horiz is not None:
        ntsc._color_bleed_horiz = args.color_bleed_horiz
    if args.color_bleed_vert is not None:
        ntsc._color_bleed_vert = args.color_bleed_vert
    if args.color_bleed_before is not None:
        ntsc._color_bleed_before = bool(args.color_bleed_before)

    if args.subcarrier_amplitude is not None:
        ntsc._subcarrier_amplitude = args.subcarrier_amplitude
    if args.subcarrier_amplitude_back is not None:
        ntsc._subcarrier_amplitude_back = args.subcarrier_amplitude_back
    if args.scanline_phase_shift is not None:
        ntsc._video_scanline_phase_shift = args.scanline_phase_shift
    if args.scanline_phase_shift_offset is not None:
        ntsc._video_scanline_phase_shift_offset = args.scanline_phase_shift_offset

    if args.vhs:
        ntsc._emulating_vhs = True
    if args.tape_speed is not None:
        ntsc._emulating_vhs = True
        ntsc._output_vhs_tape_speed = {
            "SP": VHSSpeed.VHS_SP,
            "LP": VHSSpeed.VHS_LP,
            "EP": VHSSpeed.VHS_EP,
        }[args.tape_speed]
    if args.vhs_sharpen is not None:
        ntsc._vhs_out_sharpen = args.vhs_sharpen
    if args.vhs_edge_wave is not None:
        ntsc._vhs_edge_wave = args.vhs_edge_wave
    if args.vhs_svideo is not None:
        ntsc._vhs_svideo_out = args.vhs_svideo
    if args.vhs_no_chroma_vert_blend is not None:
        ntsc._vhs_chroma_vert_blend = not args.vhs_no_chroma_vert_blend
    if args.vhs_head_switching is not None:
        ntsc._vhs_head_switching = args.vhs_head_switching

    return ntsc


def make_output_path(input_path, output_path):
    if output_path:
        return output_path
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".png"
    return f"{base}_ntsc{ext}"


def main():
    args = parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    orig_h, orig_w = img.shape[:2]
    print(f"Input: {args.input} ({orig_w}x{orig_h})")

    # Crop to aspect ratio
    if args.crop:
        cw, ch = [int(x) for x in args.crop.split(":")]
        target_ratio = cw / ch
        current_ratio = orig_w / orig_h
        if current_ratio > target_ratio:
            # Too wide, crop horizontally
            new_w = int(orig_h * target_ratio)
            x_off = (orig_w - new_w) // 2
            img = img[:, x_off : x_off + new_w]
        else:
            # Too tall, crop vertically
            new_h = int(orig_w / target_ratio)
            y_off = (orig_h - new_h) // 2
            img = img[y_off : y_off + new_h, :]
        orig_h, orig_w = img.shape[:2]
        print(f"Cropped to {cw}:{ch} -> {orig_w}x{orig_h}")

    # Determine processing size
    proc_w, proc_h = orig_w, orig_h
    if args.resize:
        parts = args.resize.lower().split("x")
        proc_w, proc_h = int(parts[0]), int(parts[1])
    elif args.resize_width:
        proc_w = args.resize_width
        proc_h = int(orig_h * proc_w / orig_w)
    elif args.resize_height:
        proc_h = args.resize_height
        proc_w = int(orig_w * proc_h / orig_h)

    # Ensure even dimensions (required by chroma subsampling)
    proc_w = proc_w // 2 * 2
    proc_h = proc_h // 2 * 2

    needs_resize = proc_w != orig_w or proc_h != orig_h
    if needs_resize:
        print(f"Processing at: {proc_w}x{proc_h}")
        dst = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        dst = img.copy()

    ntsc = build_ntsc(args)

    for i in range(args.passes):
        if args.passes > 1:
            print(f"Pass {i + 1}/{args.passes}...")
        for field in range(2):
            ntsc.composite_layer(dst, dst.copy(), field=field, fieldno=field)

    # Determine output size
    out_w, out_h = orig_w, orig_h
    if args.output_size:
        parts = args.output_size.lower().split("x")
        out_w, out_h = int(parts[0]), int(parts[1])
    elif args.output_width:
        out_w = args.output_width
        out_h = int(orig_h * out_w / orig_w)
    elif args.output_height:
        out_h = args.output_height
        out_w = int(orig_w * out_h / orig_h)

    if dst.shape[1] != out_w or dst.shape[0] != out_h:
        dst = cv2.resize(dst, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    output_path = make_output_path(args.input, args.output)
    cv2.imwrite(output_path, dst)
    print(f"Output: {output_path} ({out_w}x{out_h})")


if __name__ == "__main__":
    main()
