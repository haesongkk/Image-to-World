import argparse
from src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Image-to-World pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "segmentation", "generation", "placement"],
        default="all",
        help="Select a single stage or run the full pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed stage outputs when possible.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Input image path. If omitted, uses data/raw_image.jpg.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(stage=args.stage, resume=args.resume, image_path=args.image_path)
