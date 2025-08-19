#!/usr/bin/env python3
"""
Generate Training Dataset Script

This script uses the synthetic light curve generator to create .tbl files
for training machine learning models on astronomical data.

Usage:
    python backend/generate_training_data.py --help
    python backend/generate_training_data.py --n-stars 50 --output-dir training_dataset
"""

import argparse
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from generator import generate_training_dataset_tbl


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic astronomical training dataset as .tbl files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", 
        default="training_dataset",
        help="Output directory for generated .tbl files"
    )
    
    parser.add_argument(
        "--n-stars", 
        type=int, 
        default=50,
        help="Number of synthetic stars to generate"
    )
    
    parser.add_argument(
        "--surveys", 
        nargs="+", 
        default=["hubble", "kepler", "tess"],
        help="Survey names to simulate (space-separated)"
    )
    
    parser.add_argument(
        "--max-days", 
        type=float, 
        default=50.0,
        help="Maximum observation duration in days"
    )
    
    parser.add_argument(
        "--min-days", 
        type=float, 
        default=10.0,
        help="Minimum observation duration in days"
    )
    
    parser.add_argument(
        "--noise-level", 
        type=float, 
        default=0.02,
        help="Observational noise level (standard deviation)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=== Synthetic Astronomical Training Dataset Generator ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of stars: {args.n_stars}")
    print(f"Surveys: {args.surveys}")
    print(f"Observation duration: {args.min_days}-{args.max_days} days")
    print(f"Noise level: {args.noise_level}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    print()
    
    # Set random seed if specified
    if args.seed:
        import numpy as np
        np.random.seed(args.seed)
    
    # Generate training dataset
    summary = generate_training_dataset_tbl(
        output_dir=args.output_dir,
        n_stars=args.n_stars,
        surveys=args.surveys,
        max_days=args.max_days,
        min_days=args.min_days,
        noise_level=args.noise_level
    )
    
    print("\n=== Generation Complete ===")
    print(f"✓ {summary['n_files']} .tbl files generated")
    print(f"✓ {summary['n_stars']} synthetic stars created")
    print(f"✓ Metadata saved to {summary['csv_file']}")
    
    # Show class distribution
    class_counts = {}
    for file_info in summary['files']:
        cls = file_info['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"\nClass distribution (files per class):")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:30s}: {count:3d}")
    
    print(f"\nTraining dataset ready at: {Path(args.output_dir).absolute()}")
    print(f"Use these .tbl files with your machine learning training pipeline.")


if __name__ == "__main__":
    main()