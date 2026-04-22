from __future__ import annotations

import argparse
import json
from typing import Sequence

from .core.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spam phone MLOps pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline config YAML (default: configs/pipeline.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--report-source", default=None)
    pipeline_parser.add_argument("--history-source", default=None)

    infer_parser = subparsers.add_parser("infer", help="Run inference only")
    infer_parser.add_argument("--history-source", default=None)

    label_parser = subparsers.add_parser("label", help="Run feedback labeling only")
    label_parser.add_argument("--report-source", default=None)

    subparsers.add_parser("dataset", help="Build training dataset only")

    train_parser = subparsers.add_parser("train", help="Train and promote model")
    train_parser.add_argument("--train-data", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .pipeline import SpamDetectionPipeline

    pipeline = SpamDetectionPipeline(load_config(args.config))

    if args.command == "pipeline":
        result = pipeline.run_full(
            report_source=args.report_source,
            history_source=args.history_source,
        )
    elif args.command == "infer":
        result = pipeline.run_inference(history_source=args.history_source)
    elif args.command == "label":
        result = pipeline.run_feedback_labeling(report_source=args.report_source)
    elif args.command == "dataset":
        result = pipeline.build_training_dataset()
    elif args.command == "train":
        result = pipeline.train_and_promote(train_data_path=args.train_data)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
