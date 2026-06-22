"""
PoincaréBall: Hyperbolic Prototypical Networks for Open-Set NIDS.

CLI entrypoint for training and testing.

Usage:
    python main.py --train --config configs/config.yaml
    python main.py --test  --method poincare --data path/to/test.csv
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="PoincaréBall: Hyperbolic Prototypical Networks for Open-Set NIDS",
        add_help=True,
    )

    # Modes
    mode_group = parser.add_argument_group("Modes")
    mode_group.add_argument("--train", action="store_true", help="Train model")
    mode_group.add_argument("--test", action="store_true", help="Test model")
    mode_group.add_argument(
        "--recalculate", nargs=2, metavar=("METHOD", "PERCENTILE"),
        help="Recalculate thresholds for a method at a new percentile (e.g., --recalculate euclidean 99.0)",
    )

    # Training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config (.yaml)",
    )
    
    # Testing options
    test_group = parser.add_argument_group("Testing Options")
    test_group.add_argument(
        "--method", type=str, choices=["euclidean", "poincare"], default=None,
        help="Method to test",
    )
    test_group.add_argument(
        "--data", type=str, default=None,
        help="Path to test data CSV/Parquet",
    )

    args = parser.parse_args()

    # Validate mode
    modes = sum([args.train, args.test, bool(args.recalculate)])
    if modes == 0:
        print("[ERROR] Specify a mode: --train, --test, or --recalculate")
        sys.exit(1)
    if modes > 1:
        print("[ERROR] Only one mode may be active at a time.")
        sys.exit(1)

    # Load config if provided or required
    config = {}
    if args.train:
        if not args.config:
            print("[ERROR] --config is required for --train")
            sys.exit(1)
            
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"\nLoaded config from: {config_path}")
        print(f"Method: {config.get('method', 'NOT SET')}")

    try:
        if args.train:
            from modules.train import run_training
            run_training(config)

        elif args.recalculate:
            if args.config:
                print("[ERROR] --config should not be used with --recalculate. It strictly uses the embedded checkpoint config.")
                sys.exit(1)
            method, percentile = args.recalculate
            from modules.recalculate import run_recalculate
            run_recalculate(method, float(percentile))

        elif args.test:
            if args.config:
                print("[ERROR] --config should not be used with --test.")
                sys.exit(1)
                
            if not args.method:
                print("[ERROR] Must specify --method for testing.")
                sys.exit(1)
            
            if not args.data:
                print("[ERROR] Must specify --data for testing.")
                sys.exit(1)

            from modules.test import run_testing
            run_testing(args.method, args.data)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
