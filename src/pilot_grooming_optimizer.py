"""Pilot Grooming Optimizer - Script 1"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict


def parse_arguments(args: List[str] = None) -> Dict:
    """
    Parse and validate command line arguments for pilot optimizer.
    
    Args:
        args: Command line arguments (default: sys.argv[1:])
    
    Returns:
        Dictionary containing parsed configuration with keys:
        - data_dir: Path to data directory
        - output: Output JSON path
        - expected_frames: Expected frame count
        - alpha: Significance level
        - power: Target power
    
    Raises:
        SystemExit: If required arguments are missing or argparse fails
        ValueError: If validation fails for alpha, power, or expected_frames
        FileNotFoundError: If data_dir does not exist
    
    Example:
        >>> config = parse_arguments(['--data-dir', 'data', '--output', 'results.json'])
        >>> print(config['expected_frames'])
        9000
    """
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description='Pilot Grooming Optimizer - Analyze grooming behavior and calculate sample sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing genotype subdirectories with CSV files'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for optimization_results.json'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--expected-frames',
        type=int,
        default=9000,
        help='Expected frame count per video'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (must be between 0 and 1, exclusive)'
    )
    
    parser.add_argument(
        '--power',
        type=float,
        default=0.8,
        help='Target statistical power (must be between 0 and 1, exclusive)'
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Validation: expected_frames must be positive
    if parsed_args.expected_frames <= 0:
        raise ValueError(f"expected_frames must be a positive integer, got: {parsed_args.expected_frames}")
    
    # Validation: alpha must be between 0 and 1 (exclusive)
    if parsed_args.alpha <= 0 or parsed_args.alpha >= 1:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got: {parsed_args.alpha}")
    
    # Validation: power must be between 0 and 1 (exclusive)
    if parsed_args.power <= 0 or parsed_args.power >= 1:
        raise ValueError(f"power must be between 0 and 1 (exclusive), got: {parsed_args.power}")
    
    # Validation: data_dir must exist
    data_dir_path = Path(parsed_args.data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {parsed_args.data_dir}")
    
    # Build and return config dictionary
    config = {
        'data_dir': parsed_args.data_dir,
        'output': parsed_args.output,
        'expected_frames': parsed_args.expected_frames,
        'alpha': parsed_args.alpha,
        'power': parsed_args.power
    }
    
    return config
