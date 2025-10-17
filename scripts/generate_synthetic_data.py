#!/usr/bin/env python3
"""Generate synthetic training data

Usage:
    python scripts/generate_synthetic_data.py \
        --output-dir /mnt/data/project_lacuna/synthetic \
        --n-examples 10000
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-examples", type=int, required=True)
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--n-bins", type=int, required=True)
    
    args = parser.parse_args()
    
    # TODO: Implement generation
    # See spec section 5.1
    print(f"Generating {args.n_examples} examples...")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
