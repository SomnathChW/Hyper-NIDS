"""
PoincaréBall: Hyperbolic Prototypical Networks for Open-Set NIDS.

CLI entrypoint for training and testing.

Usage:
    python main.py --train --config configs/config.yaml
    python main.py --test  --method poincare --data path/to/test.csv
    python main.py --test  --config configs/config.yaml --data path/to/test.csv
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
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Test model")

    # Config is required for training, optional for testing
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config (.yaml) (Required for training)",
    )
    
    # Testing options
    parser.add_argument(
        "--method", type=str, choices=["euclidean", "poincare"], default=None,
        help="Method to test (Required if no config provided)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to test data CSV/Parquet",
    )

    args = parser.parse_args()

    # Validate mode
    modes = sum([args.train, args.test])
    if modes == 0:
        print("[ERROR] Specify a mode: --train or --test")
        sys.exit(1)
    if modes > 1:
        print("[ERROR] Only one mode may be active at a time.")
        sys.exit(1)

    # Load config if provided
    config = {}
    if args.config:
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
            if not config:
                print("[ERROR] --config is required for --train")
                sys.exit(1)
            from modules.train import run_training
            run_training(config)

        elif args.test:
            if args.method:
                config["method"] = args.method
            if args.data:
                config["test_data_path"] = args.data
            
            if "method" not in config:
                print("[ERROR] Must specify --method or provide a --config with a method for testing.")
                sys.exit(1)
                
            test_path = config.get("test_data_path") or config.get("data", {}).get("data_path")
            if not test_path:
                print("[ERROR] Must specify --data or provide a --config with a data path for testing.")
                sys.exit(1)

            from modules.test import run_testing
            run_testing(config)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
