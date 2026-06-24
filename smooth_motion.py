"""
smooth_motion.py — Apply a moving-average smooth to translation and
rotation_6d fields of a motion JSON (as produced by object-mimic-demo.py).

Usage:
  python smooth_motion.py input/motion.json output/motion_smooth.json
  python smooth_motion.py input/motion.json output/motion_smooth.json --window 5

The script writes a new JSON file with the same structure as the input,
with "translation" and "rotation_6d" values replaced by their
moving-average-smoothed counterparts.  All other fields (scale, index,
timestamp, video_info, …) are copied unchanged.
"""

import argparse
import json
import sys
from pathlib import Path


def moving_average(sequences: list[list[float]], window: int) -> list[list[float]]:
    """Return a new list of vectors smoothed with a centred moving average.

    Boundary frames use a smaller, asymmetric window (same as 'valid'
    padding on each side) so the output length equals the input length.

    Args:
        sequences: List of N vectors, each of length D.
        window:    Number of frames to average (must be >= 1).

    Returns:
        Smoothed list of N vectors of length D.
    """
    n = len(sequences)
    if n == 0 or window <= 1:
        return [list(v) for v in sequences]

    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        bucket = sequences[lo:hi]
        dim = len(bucket[0])
        avg = [sum(v[d] for v in bucket) / len(bucket) for d in range(dim)]
        smoothed.append(avg)
    return smoothed


def smooth_motion(data: dict, window: int) -> dict:
    """Return a deep-copied motion dict with smoothed translation and rotation_6d."""
    import copy

    out = copy.deepcopy(data)
    frames = out["frames"]

    translations = [f["translation"] for f in frames]
    rotations = [f["rotation_6d"] for f in frames]

    smooth_t = moving_average(translations, window)
    smooth_r = moving_average(rotations, window)

    for i, frame in enumerate(frames):
        frame["translation"] = smooth_t[i]
        frame["rotation_6d"] = smooth_r[i]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smooth translation and rotation_6d in a motion JSON.",
    )
    parser.add_argument("input", type=Path, help="Input motion.json path.")
    parser.add_argument("output", type=Path, help="Output (smoothed) JSON path.")
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        metavar="N",
        help="Moving-average window size (default: 3).",
    )
    args = parser.parse_args()

    if args.window < 1:
        print("error: --window must be >= 1", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    smoothed = smooth_motion(data, args.window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(smoothed, f, indent=2)

    n = len(data["frames"])
    print(
        f"Smoothed {n} frames with window={args.window} → {args.output}"
    )


if __name__ == "__main__":
    main()
