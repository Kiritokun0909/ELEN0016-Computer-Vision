"""
Videos must be in the same folder as this script, inside a folder "videos"

Usage:
    python phase1.py path/to/group_X_phase1_static_indoor.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# ------------------------- Parameters -------------------------
SCALE = 0.5           # process frames at half size (960x540): 4x less memory and time
N_FRAMES = 100        # number of frames sampled evenly over the video for the median
DIFF_THRESHOLD = 30   # pixel is foreground if it differs by more than this (0-255) from the background
N_GHOSTS = 8          # number of copies of the moving objects in the ghost image
FEATHER = 21          # size of the blur that softens the mask borders (must be odd)
# --------------------------------------------------------------


def read_frames(path):
    """Read about N_FRAMES frames, evenly spaced over the whole video, resized by SCALE."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open the video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // N_FRAMES)

    frames = []
    i = 0
    while cap.grab():                 # move to the next frame
        if i % step == 0:             # only convert and keep one frame every `step`
            _, frame = cap.retrieve()
            frames.append(cv2.resize(frame, None, fx=SCALE, fy=SCALE,
                                     interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    return frames


def median_background(frames):
    """Temporal median: for each pixel, take the median value over time.
    A moving person covers a given pixel only for a short time, so the
    median picks the background value seen most of the time."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def foreground_mask(frame, background):
    """Binary mask (0 or 255) of the pixels that differ from the background."""
    diff = cv2.absdiff(frame, background).max(axis=2)   # largest difference over B, G, R
    mask = np.uint8(diff > DIFF_THRESHOLD) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # remove noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) # fill holes in objects
    return mask


def ghost_image(frames, background):
    """Paste N_GHOSTS copies of the moving objects on the background,
    from oldest (faint) to newest (fully opaque): alpha_i = i / N."""
    result = background.astype(np.float32)
    picks = np.linspace(0, len(frames) - 1, N_GHOSTS).astype(int)   # evenly spaced in time
    for i, idx in enumerate(picks, start=1):
        mask = foreground_mask(frames[idx], background).astype(np.float32) / 255
        mask = cv2.GaussianBlur(mask, (FEATHER, FEATHER), 0)        # feathering: soft edges
        alpha = (i / N_GHOSTS) * mask[..., None]                     # opacity grows with time
        result = (1 - alpha) * result + alpha * frames[idx].astype(np.float32)
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: clean background + ghost image")
    parser.add_argument("video", help="path to the input .mp4 video")
    parser.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.video).stem

    print("Reading frames...")
    frames = read_frames(args.video)
    print(f"  kept {len(frames)} frames")

    print("Computing the median background...")
    background = median_background(frames)
    cv2.imwrite(str(out_dir / f"{name}_background.png"), background)

    print("Building the ghost image...")
    ghosts = ghost_image(frames, background)
    cv2.imwrite(str(out_dir / f"{name}_ghosts.png"), ghosts)

    print(f"Done. Results saved in {out_dir}/")


if __name__ == "__main__":
    main()
