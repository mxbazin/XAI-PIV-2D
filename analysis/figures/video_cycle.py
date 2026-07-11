"""
Video: sweep a range of snapshots and assemble them into an MP4 (or GIF).

Each frame is the same 3-row layout as Figure A (coincidence + |u'| + ||SHAP||)
built from `instantaneous.make_figure` with a single index.

Examples (run from analysis/):

    # Two cycles starting at t=2100 (60 frames at T=30), 5 fps => 12 s video
    python -m figures.video_cycle --start 2100 --end 2160 --fps 5

    # One cycle, GIF output
    python -m figures.video_cycle --start 2100 --cycles 1 --fps 5 --out cycle.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_THIS.parents[1]))

from figures._common import OUT_DIR, PERIOD_SNAP
from figures._structures import SHAP_DIR
from figures.instantaneous import make_figure


def _available_shap_indices() -> set[int]:
    """Indices with a SHAP file present on disk."""
    import re
    pat = re.compile(r"\.(\d+)\.h5\.shap$")
    out: set[int] = set()
    for p in Path(SHAP_DIR).iterdir():
        m = pat.search(p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    # ARGB -> RGB
    return buf[..., [1, 2, 3]].copy()


def build_video(indices: list[int], out_path: Path, fps: int,
                save_frames_dir: Path | None = None) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(
            "imageio is required. Install with: pip install imageio imageio-ffmpeg"
        ) from exc

    is_gif = out_path.suffix.lower() == ".gif"
    writer_kwargs = {"fps": fps}
    if not is_gif:
        writer_kwargs["codec"] = "libx264"
        writer_kwargs["quality"] = 8

    print(f"Writing {len(indices)} frames to {out_path} at {fps} fps...")
    with imageio.get_writer(out_path, **writer_kwargs) as writer:
        for k, idx in enumerate(indices):
            fig = make_figure(index=idx, save=False)
            frame = _fig_to_rgb(fig)
            # Ensure even dimensions for libx264.
            if not is_gif:
                h, w = frame.shape[:2]
                if h % 2 or w % 2:
                    frame = frame[: h - (h % 2), : w - (w % 2)]
            writer.append_data(frame)

            if save_frames_dir is not None:
                save_frames_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(save_frames_dir / f"frame_{k:04d}_t{idx:05d}.png", frame)

            plt.close(fig)
            print(f"  [{k + 1:3d}/{len(indices)}] t = {idx}")

    print(f"Done: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble a video sweeping snapshot indices.")
    p.add_argument("--start", type=int, required=True, help="First snapshot index.")
    p.add_argument("--end", type=int, default=None,
                   help="Last snapshot index (exclusive). Mutually exclusive with --cycles.")
    p.add_argument("--cycles", type=float, default=None,
                   help=f"Number of forcing cycles from --start (1 cycle = {PERIOD_SNAP} frames).")
    p.add_argument("--step", type=int, default=5,
                   help="Snapshot stride (default 5, matches SHAP sampling).")
    p.add_argument("--fps", type=int, default=3,
                   help="Frames per second (default 3; 1 cycle = 6 SHAP frames).")
    p.add_argument("--out", type=str, default=None,
                   help="Output path. Default: results/figures/video_cycle_t{start}_t{end}.mp4")
    p.add_argument("--save-frames", action="store_true",
                   help="Also save individual PNG frames next to the video.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.end is None and args.cycles is None:
        raise SystemExit("Specify either --end or --cycles.")
    if args.end is not None and args.cycles is not None:
        raise SystemExit("--end and --cycles are mutually exclusive.")

    end = args.end if args.end is not None else args.start + int(args.cycles * PERIOD_SNAP)
    requested = list(range(args.start, end, args.step))

    available = _available_shap_indices()
    indices = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]
    if missing:
        print(f"Skipping {len(missing)} indices without SHAP files "
              f"(first few: {missing[:5]}). SHAP stride appears to be "
              f"{sorted(available)[1] - sorted(available)[0] if len(available) > 1 else '?'}.")
    if not indices:
        raise SystemExit(
            f"No SHAP files found in range [{args.start}, {end}). "
            f"Try --step 5 (SHAP is computed every 5 snapshots)."
        )

    if args.out is None:
        out_path = OUT_DIR / f"video_cycle_t{args.start:05d}_t{end:05d}.mp4"
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = OUT_DIR / out_path

    frames_dir = out_path.with_suffix("") if args.save_frames else None
    build_video(indices, out_path, fps=args.fps, save_frames_dir=frames_dir)
