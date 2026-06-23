# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
object-mimic-demo.py — Motion capture from video via SAM 3D Objects.

Takes a video + per-frame masks, runs SAM 3D Objects on every frame,
and outputs a JSON with per-frame pose data (translation, 6D rotation,
scale) suitable for motion transfer to a 3D scene platform.

Optionally saves per-frame .ply files (--save-ply) and a fixed-camera
debug video showing the reconstructed trajectory (--save-video).

6D rotation convention (Zhou et al. 2019):
  rotation_6d = [r1x, r1y, r1z, r2x, r2y, r2z]
  where r1 = X-axis, r2 = Y-axis of the object frame.
  Z-axis = r1 × r2 (not stored).

Usage examples:

  # Masks as PNGs in a directory (0.png, 1.png, ...):
  python object-mimic-demo.py \\
    --video input.mp4 \\
    --mask-dir masks/ \\
    --output results/

  # Masks as a grayscale video:
  python object-mimic-demo.py \\
    --video input.mp4 \\
    --mask-video masks.mp4 \\
    --output results/ \\
    --save-ply --save-video

  # Process only frames 10-50:
  python object-mimic-demo.py \\
    --video input.mp4 \\
    --mask-dir masks/ \\
    --output results/ \\
    --frame-start 10 --frame-end 50

  # Resample to 10 fps:
  python object-mimic-demo.py \\
    --video input.mp4 \\
    --mask-dir masks/ \\
    --output results/ \\
    --fps 10
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Environment setup (must happen before any sam3d_objects import)
# ---------------------------------------------------------------------------
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

sys.path.append("notebook")

import cv2  # noqa: E402  (after env setup)
import torch  # noqa: E402

from inference import (  # noqa: E402
  Inference,
  _yaw_pitch_r_fov_to_extrinsics_intrinsics,
)
from sam3d_objects.model.backbone.tdfy_dit.utils import (  # noqa: E402
  render_utils,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="SAM 3D Objects — video motion capture demo",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )

  # --- required ---
  p.add_argument(
    "--video",
    required=True,
    metavar="PATH",
    help="Input video file (MP4, WebM, MOV, or any OpenCV-compatible format).",
  )
  mask_group = p.add_mutually_exclusive_group(required=True)
  mask_group.add_argument(
    "--mask-dir",
    metavar="DIR",
    help=(
      "Directory containing per-frame grayscale PNG masks named "
      "0.png, 1.png, … (matching extracted frame indices)."
    ),
  )
  mask_group.add_argument(
    "--mask-video",
    metavar="PATH",
    help="Grayscale video whose pixel intensity encodes the mask.",
  )

  # --- output ---
  p.add_argument(
    "--output",
    required=True,
    metavar="DIR",
    help="Output directory. Created if it does not exist.",
  )

  # --- model ---
  p.add_argument(
    "--config",
    default="checkpoints/hf/pipeline.yaml",
    metavar="PATH",
    help="Path to pipeline.yaml (default: checkpoints/hf/pipeline.yaml).",
  )
  p.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed passed to the inference pipeline (default: 42).",
  )
  p.add_argument(
    "--compile",
    action="store_true",
    help="Enable torch.compile (slower first run, faster subsequent).",
  )

  # --- frame selection (mutually exclusive) ---
  range_group = p.add_argument_group(
    "frame selection",
    "Use --frame-start/--frame-end OR --fps, not both.",
  )
  range_group.add_argument(
    "--frame-start",
    type=int,
    default=None,
    metavar="N",
    help="First frame index to process (0-based, inclusive).",
  )
  range_group.add_argument(
    "--frame-end",
    type=int,
    default=None,
    metavar="N",
    help="Last frame index to process (0-based, inclusive).",
  )
  range_group.add_argument(
    "--fps",
    type=float,
    default=None,
    metavar="F",
    help=(
      "Resample video to this frame rate before processing. "
      "Incompatible with --frame-start / --frame-end."
    ),
  )

  # --- optional outputs ---
  p.add_argument(
    "--save-ply",
    action="store_true",
    help=(
      "Save each frame's Gaussian splat as frame_NNNN.ply "
      "inside --output."
    ),
  )
  p.add_argument(
    "--save-video",
    action="store_true",
    help=(
      "Render a fixed-camera composite video (debug_video.mp4) "
      "showing the reconstructed object trajectory."
    ),
  )

  # --- debug video camera ---
  p.add_argument(
    "--video-r",
    type=float,
    default=2.0,
    metavar="R",
    help="Camera distance for the debug video (default: 2.0).",
  )
  p.add_argument(
    "--video-fov",
    type=float,
    default=40.0,
    metavar="DEG",
    help="Camera field-of-view (degrees) for the debug video (default: 40).",
  )
  p.add_argument(
    "--video-pitch-deg",
    type=float,
    default=0.0,
    metavar="DEG",
    help="Camera pitch (degrees) for the debug video (default: 0).",
  )
  p.add_argument(
    "--video-yaw-deg",
    type=float,
    default=-90.0,
    metavar="DEG",
    help="Camera yaw (degrees) for the debug video (default: -90).",
  )
  p.add_argument(
    "--video-resolution",
    type=int,
    default=512,
    metavar="PX",
    help="Pixel resolution of the debug video (default: 512).",
  )

  return p


def validate_args(args: argparse.Namespace) -> None:
  """Fail fast on mutually-exclusive frame-selection options."""
  using_range = (
    args.frame_start is not None or args.frame_end is not None
  )
  using_fps = args.fps is not None
  if using_range and using_fps:
    raise SystemExit(
      "error: --fps cannot be combined with "
      "--frame-start / --frame-end"
    )


# ---------------------------------------------------------------------------
# Video / mask I/O helpers
# ---------------------------------------------------------------------------

def open_video(path: str) -> cv2.VideoCapture:
  cap = cv2.VideoCapture(path)
  if not cap.isOpened():
    raise RuntimeError(
      f"Cannot open video '{path}'. "
      "Check that the file exists and the codec is supported by OpenCV."
    )
  return cap


def video_metadata(cap: cv2.VideoCapture) -> dict:
  fps = cap.get(cv2.CAP_PROP_FPS)
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  return {
    "fps": fps,
    "num_frames": n_frames,
    "resolution": [width, height],
  }


def compute_frame_indices(
  total_frames: int,
  source_fps: float,
  frame_start: int | None,
  frame_end: int | None,
  target_fps: float | None,
) -> list[int]:
  """Return the sorted list of 0-based frame indices to process."""
  if target_fps is not None:
    step = max(1, round(source_fps / target_fps))
    return list(range(0, total_frames, step))
  start = frame_start if frame_start is not None else 0
  end = (frame_end + 1) if frame_end is not None else total_frames
  end = min(end, total_frames)
  if start >= end:
    raise ValueError(
      f"--frame-start ({start}) must be less than "
      f"--frame-end ({frame_end}) and within [0, {total_frames})."
    )
  return list(range(start, end))


def extract_frames(
  cap: cv2.VideoCapture, indices: list[int]
) -> list[np.ndarray]:
  """
  Extract specific frames from an open VideoCapture.
  Returns RGB uint8 arrays.
  """
  frames = []
  index_set = set(indices)
  # Sort so we can seek forward efficiently
  max_idx = max(indices)
  for i in range(max_idx + 1):
    ret, frame = cap.read()
    if not ret:
      break
    if i in index_set:
      # cv2 returns BGR; convert to RGB
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  if len(frames) != len(indices):
    raise RuntimeError(
      f"Expected {len(indices)} frames but could only read "
      f"{len(frames)}. The video may be shorter than expected."
    )
  return frames


def load_masks_from_dir(
  mask_dir: str, indices: list[int]
) -> list[np.ndarray]:
  """
  Load grayscale PNG masks from a directory.
  Files must be named {index}.png.
  Returns boolean arrays of shape (H, W).
  """
  masks = []
  for idx in indices:
    path = os.path.join(mask_dir, f"{idx}.png")
    if not os.path.exists(path):
      raise FileNotFoundError(
        f"Mask not found: '{path}'. "
        "Expected files named 0.png, 1.png, … matching frame indices."
      )
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
      raise RuntimeError(f"Failed to read mask image: '{path}'.")
    masks.append(img > 0)
  return masks


def load_masks_from_video(
  mask_video_path: str, indices: list[int]
) -> list[np.ndarray]:
  """
  Extract per-frame masks from a grayscale mask video.
  Returns boolean arrays of shape (H, W).
  """
  cap = open_video(mask_video_path)
  mask_frames = []
  index_set = set(indices)
  max_idx = max(indices)
  for i in range(max_idx + 1):
    ret, frame = cap.read()
    if not ret:
      break
    if i in index_set:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      mask_frames.append(gray > 0)
  cap.release()
  if len(mask_frames) != len(indices):
    raise RuntimeError(
      f"Mask video is shorter than input video. "
      f"Expected {len(indices)} frames, got {len(mask_frames)}."
    )
  return mask_frames


# ---------------------------------------------------------------------------
# Rotation conversion helpers
# ---------------------------------------------------------------------------

def quaternion_wxyz_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
  """
  Convert a (4,) wxyz quaternion to a (3, 3) rotation matrix.
  Uses pytorch3d convention (w, x, y, z).
  """
  from pytorch3d.transforms import quaternion_to_matrix
  # quaternion_to_matrix expects (..., 4) in wxyz
  q = quat.float()
  if q.dim() == 1:
    q = q.unsqueeze(0)  # (1, 4)
  R = quaternion_to_matrix(q)  # (1, 3, 3)
  return R.squeeze(0)  # (3, 3)


def rotation_matrix_to_6d(R: torch.Tensor) -> list[float]:
  """
  Convert a (3, 3) rotation matrix to a 6D representation.
  Returns [r1x, r1y, r1z, r2x, r2y, r2z] where
    r1 = first column (X axis), r2 = second column (Y axis).
  Z = r1 × r2 (recoverable, not stored).
  """
  r1 = R[:, 0].tolist()  # X axis
  r2 = R[:, 1].tolist()  # Y axis
  return r1 + r2


# ---------------------------------------------------------------------------
# Fixed-camera rendering helper
# ---------------------------------------------------------------------------

def render_single_frame(
  gs,
  r: float,
  fov: float,
  pitch_deg: float,
  yaw_deg: float,
  resolution: int,
) -> np.ndarray:
  """
  Render one frame of a Gaussian splat from a fixed camera viewpoint.
  Returns a uint8 RGB array of shape (resolution, resolution, 3).
  """
  yaw_rad = math.radians(yaw_deg)
  pitch_rad = math.radians(pitch_deg)
  extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaw_rad, pitch_rad, r, fov
  )
  frames = render_utils.render_frames(
    gs,
    [extr],
    [intr],
    {"resolution": resolution, "bg_color": (0, 0, 0), "backend": "gsplat"},
  )
  # render_frames returns a dict with keys "color" and "depth"; we want the color image
  assert isinstance(frames, dict), f"Expected render_frames to return a dict, got {type(frames)}"
  assert "color" in frames, f"Expected 'color' key in render_frames output, got keys: {list(frames.keys())}"
  assert isinstance(frames["color"], list), f"Expected 'color' to be a list, got {type(frames['color'])}"
  assert len(frames["color"]) >= 1, f"Expected at least one frame in 'color', got {len(frames['color'])}"
  assert isinstance(frames["color"][0], np.ndarray), f"Expected 'color'[0] to be a numpy array, got {type(frames['color'][0])}"
  return frames["color"][0]

def write_video(
  frames: list[np.ndarray],
  output_path: str,
  fps: float,
) -> None:
  """Write a list of RGB uint8 frames to an MP4 file."""
  if not frames:
    return
  h, w = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
  for frame in frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
  writer.release()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  validate_args(args)

  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)

  # --- open video & compute frame indices ---
  cap = open_video(args.video)
  meta = video_metadata(cap)
  print(
    f"Video: {args.video} | "
    f"{meta['num_frames']} frames @ {meta['fps']:.2f} fps | "
    f"{meta['resolution'][0]}×{meta['resolution'][1]}"
  )

  indices = compute_frame_indices(
    total_frames=meta["num_frames"],
    source_fps=meta["fps"],
    frame_start=args.frame_start,
    frame_end=args.frame_end,
    target_fps=args.fps,
  )
  print(f"Processing {len(indices)} frames: {indices[0]}…{indices[-1]}")

  # --- extract frames ---
  frames_rgb = extract_frames(cap, indices)
  cap.release()

  # --- load masks ---
  if args.mask_dir is not None:
    masks = load_masks_from_dir(args.mask_dir, indices)
  else:
    masks = load_masks_from_video(args.mask_video, indices)
  print(f"Loaded {len(masks)} masks.")

  # --- load model ---
  print(f"Loading model from '{args.config}' …")
  inference = Inference(args.config, compile=args.compile)
  print("Model loaded.")

  # --- per-frame inference ---
  json_frames = []
  rendered_frames = []  # only populated if --save-video

  for i, (frame_idx, image, mask) in enumerate(
    zip(indices, frames_rgb, masks)
  ):
    timestamp = frame_idx / meta["fps"]
    print(
      f"  [{i + 1}/{len(indices)}] frame {frame_idx} "
      f"(t={timestamp:.3f}s) …",
      end=" ",
      flush=True,
    )

    output = inference(image, mask, seed=args.seed)

    # --- extract pose ---
    # translation: tensor of shape (3,) or (1, 3)
    trans = output["translation"]
    if trans.dim() > 1:
      trans = trans.squeeze(0)
    translation = trans.cpu().float().tolist()

    # rotation: quaternion tensor (wxyz) of various shapes
    rot_q = output["rotation"]
    rot_q = rot_q.reshape(-1)[:4].cpu().float()
    R = quaternion_wxyz_to_rotation_matrix(rot_q)
    rotation_6d = rotation_matrix_to_6d(R)

    # scale: tensor of shape (3,) or (1, 3)
    sc = output["scale"]
    if sc.dim() > 1:
      sc = sc.squeeze(0)
    scale = sc.cpu().float().tolist()

    json_frames.append({
      "index": frame_idx,
      "timestamp": round(timestamp, 6),
      "translation": translation,
      "rotation_6d": rotation_6d,
      "scale": scale,
    })

    print("done.")

    # --- optional: save .ply ---
    if args.save_ply:
      ply_path = out_dir / f"frame_{frame_idx:04d}.ply"
      output["gs"].save_ply(str(ply_path))
      print(f"    → saved {ply_path.name}")

    # --- optional: render debug frame ---
    if args.save_video:
      rendered = render_single_frame(
        output["gs"],
        r=args.video_r,
        fov=args.video_fov,
        pitch_deg=args.video_pitch_deg,
        yaw_deg=args.video_yaw_deg,
        resolution=args.video_resolution,
      )
      rendered_frames.append(rendered)

  # --- write JSON ---
  motion_json = {
    "video_info": {
      "source": str(args.video),
      "fps": meta["fps"],
      "num_frames": meta["num_frames"],
      "resolution": meta["resolution"],
      "processed_frames": len(indices),
    },
    "frames": json_frames,
  }
  json_path = out_dir / "motion.json"
  with open(json_path, "w") as f:
    json.dump(motion_json, f, indent=2)
  print(f"Motion data saved to '{json_path}'.")

  # --- write debug video ---
  if args.save_video:
    video_fps = (
      args.fps if args.fps is not None else meta["fps"]
    )
    video_path = str(out_dir / "debug_video.mp4")
    write_video(rendered_frames, video_path, fps=video_fps)
    print(f"Debug video saved to '{video_path}'.")

  print("Done.")


if __name__ == "__main__":
  main()
