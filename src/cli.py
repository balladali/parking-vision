"""The primary user interface; help never loads a model or opens a camera."""

import argparse
from contextlib import redirect_stdout
import logging
from pathlib import Path
import sys


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parking-vision", description="Local fixed-camera parking occupancy PoC")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate YAML without loading YOLO")
    validate.add_argument("--config", type=Path, required=True)
    process = commands.add_parser("run", help="Process local video or a camera into JSONL")
    process.add_argument("--config", type=Path, required=True)
    process.add_argument("--max-frames", type=positive_int)
    process.add_argument("--output", type=Path, help="New JSONL file (default: stdout)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True)
    logger = logging.getLogger("parking_vision")
    try:
        from parking_vision.config import load_config

        config = load_config(args.config)
        logging.getLogger().setLevel(config.log_level)
        if args.command == "validate":
            logger.info("Valid configuration: %d parking spots", len(config.spots))
            return 0
        if isinstance(config.source, str) and not Path(config.source).is_file():
            raise ValueError(f"Video file does not exist: {config.source}")
        from parking_vision.detection.yolo import YoloDetector
        from parking_vision.pipeline import run

        # Keep third-party initialization/download output away from JSONL stdout.
        with redirect_stdout(sys.stderr):
            detector = YoloDetector(config.detector)
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                with redirect_stdout(sys.stderr):
                    run(config, detector, output, args.max_frames)
        else:
            output = sys.stdout
            with redirect_stdout(sys.stderr):
                run(config, detector, output, args.max_frames)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as exc:
        logger.error("%s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1
