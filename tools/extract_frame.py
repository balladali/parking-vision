"""Extract the first video frame to use when measuring polygon coordinates."""

import argparse
from pathlib import Path

import cv2

from parking_vision.video import open_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with open_video(args.source) as video:
        _, frame = next(video.frames())
        if not cv2.imwrite(str(args.output), frame):
            raise OSError(f"Unable to write {args.output}")


if __name__ == "__main__":
    main()
