"""Pilot Grooming Optimizer - Script 1"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from common_library import load_event_csv, process_directory_structure


# =============================================================================
# Unit 1: Parse Command Line Arguments
# =============================================================================

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


# =============================================================================
# Unit 2: Load Pilot Data from Multiple Genotypes
# =============================================================================

def load_pilot_data(data_dir: str) -> Tuple[Dict[str, List[List[Tuple[int, int]]]], Dict[str, List[int]]]:
    """
    Load all pilot CSV files organized by genotype.
    
    Processes a directory structure containing genotype subdirectories, each
    with multiple CSV files representing individual flies. Loads grooming
    events and extracts frame counts for each fly.
    
    Directory structure expected:
        data_dir/
          ├── genotype_A/
          │   ├── fly1.csv
          │   ├── fly2.csv
          │   └── ...
          ├── genotype_B/
          │   └── ...
    
    Args:
        data_dir: Directory containing genotype subdirectories
    
    Returns:
        Tuple containing:
        - pilot_data: Nested structure mapping genotype → flies → events
          Example: {'WT': [[(100, 150)], [(200, 250)]]}
        - frame_counts: Actual frame counts per genotype and fly
          Example: {'WT': [9000, 9010], 'KO': [9005, 9000]}
    
    Raises:
        ValueError: If fewer than 2 genotypes found
        FileNotFoundError: If data_dir does not exist
    
    Notes:
        - Empty CSV files are allowed (represent flies with no grooming events)
        - Frame count for empty CSVs is recorded as 0 (unknown/no data)
        - Warnings from directory processing are printed to stdout
        - Files are processed in sorted alphabetical order for consistency
    
    Example:
        >>> pilot_data, frame_counts = load_pilot_data('data/pilot/')
        >>> print(len(pilot_data))
        2
        >>> print(pilot_data['WT'][0])
        [(100, 150), (200, 250)]
    """
    # Get directory structure using common_library function
    structure, warnings = process_directory_structure(data_dir, file_extension='.csv')
    
    # Print warnings if any
    for warning in warnings:
        print(f"Warning: {warning}")
    
    # Validation: require at least 2 genotypes for comparison
    if len(structure) < 2:
        raise ValueError(
            f"At least 2 genotypes required for analysis, found {len(structure)}. "
            f"Ensure data_dir contains at least 2 genotype subdirectories with CSV files."
        )
    
    # Initialize output dictionaries
    pilot_data: Dict[str, List[List[Tuple[int, int]]]] = {}
    frame_counts: Dict[str, List[int]] = {}
    
    # Process each genotype
    for genotype_name, file_paths in structure.items():
        genotype_events = []
        genotype_frames = []
        
        # Sort file paths for consistent ordering across filesystems
        sorted_file_paths = sorted(file_paths)
        
        # Process each CSV file for this genotype
        for csv_path in sorted_file_paths:
            # Load events from CSV using common_library function
            events = load_event_csv(csv_path, validate=True)
            
            # Extract frame count from events
            if len(events) > 0:
                # Get maximum end frame from all events
                frame_count = max([end for start, end in events])
            else:
                # Empty CSV: no events, frame count unknown (use 0 as sentinel)
                frame_count = 0
            
            # Store events and frame count for this fly
            genotype_events.append(events)
            genotype_frames.append(frame_count)
        
        # Store results for this genotype
        pilot_data[genotype_name] = genotype_events
        frame_counts[genotype_name] = genotype_frames
    
    return pilot_data, frame_counts


# =============================================================================
# Unit 3: Generate Parameter Space
# =============================================================================

def generate_parameter_space(frame_counts: Dict[str, List[int]]) -> Dict:
    """
    Generate valid parameter combinations based on shortest video.
    
    Creates a comprehensive parameter space for optimization by determining
    valid window sizes from the shortest video's frame count using hierarchical
    fallback logic, then combining with fixed sampling parameters.
    
    Args:
        frame_counts: Dictionary mapping genotype names to lists of frame counts
                     Example: {'WT': [9000, 9010], 'KO': [8995]}
    
    Returns:
        Dictionary containing parameter space with keys:
        - 'window_sizes': List of valid window sizes (based on hierarchical selection)
        - 'sampling_rates': Fixed list of sampling rates
        - 'strategies': Fixed list of sampling strategies
        - 'edge_thresholds': Fixed list of edge thresholds (in frames)
    
    Window Size Selection (Hierarchical Fallback):
        1. Find shortest_frames across all videos
        2. Find all divisors of shortest_frames
        3. Apply selection tiers:
           - Primary: divisors ≥ 100 AND multiples of 25
           - Fallback: divisors ≥ 100 (if primary yields none)
        4. Raise ValueError if no valid window sizes found
    
    Raises:
        ValueError: If no valid window sizes can be generated (shortest_frames < 100)
    
    Example:
        >>> frame_counts = {'WT': [9000, 9010], 'KO': [8995]}
        >>> params = generate_parameter_space(frame_counts)
        >>> print(params['window_sizes'])
        [100, 125, 150, 175, 200, 225, ...]
        >>> print(params['sampling_rates'])
        [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
    """
    # Fixed parameters
    sampling_rates = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    strategies = ['uniform', 'stratified', 'systematic']
    edge_thresholds = [5, 10, 15, 20, 25, 30]
    
    # Extract all frame counts from nested structure
    all_frame_counts = []
    for genotype_counts in frame_counts.values():
        all_frame_counts.extend(genotype_counts)
    
    # Find shortest frame count
    shortest_frames = min(all_frame_counts)
    
    # Check if shortest video is too short for valid analysis
    if shortest_frames < 100:
        raise ValueError(
            f"Shortest video has {shortest_frames} frames, which is below the "
            f"minimum of 100 frames required for parameter space generation."
        )
    
    # Find all divisors of shortest_frames efficiently
    divisors = []
    for i in range(1, int(shortest_frames**0.5) + 1):
        if shortest_frames % i == 0:
            divisors.append(i)
            if i != shortest_frames // i:
                divisors.append(shortest_frames // i)
    
    # Sort divisors for consistent ordering
    divisors.sort()
    
    # Filter to divisors >= 100
    valid_divisors = [d for d in divisors if d >= 100]
    
    # Apply hierarchical selection
    # Primary: divisors ≥ 100 AND multiples of 25
    primary_candidates = [d for d in valid_divisors if d % 25 == 0]
    
    if primary_candidates:
        window_sizes = primary_candidates
    else:
        # Fallback: all divisors ≥ 100
        window_sizes = valid_divisors
    
    # Return parameter space dictionary
    return {
        'window_sizes': window_sizes,
        'sampling_rates': sampling_rates,
        'strategies': strategies,
        'edge_thresholds': edge_thresholds
    }


# =============================================================================
# Unit 4: Bootstrap Sample Generator
# =============================================================================

def generate_bootstrap_samples(
    data: List[List[Tuple[int, int]]], 
    n_samples: int, 
    seed: int
) -> List[List[List[Tuple[int, int]]]]:
    """
    Generate bootstrap samples by resampling flies with replacement.
    
    Bootstrap sampling is a statistical technique for estimating the sampling
    distribution by repeatedly resampling from the original dataset with 
    replacement. In this context, we resample entire flies (not individual 
    events) to maintain the correlation structure within each fly's behavior.
    
    Args:
        data: List of flies, where each fly is a list of (start_frame, end_frame) events
        n_samples: Number of bootstrap samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of bootstrap samples, where each sample is a list of flies
        
    Example:
        >>> data = [[(10, 20), (30, 40)], [(50, 60)], [(100, 120)]]
        >>> samples = generate_bootstrap_samples(data, 2, seed=42)
        >>> len(samples)
        2
        >>> len(samples[0])
        3
        
    Notes:
        - Each bootstrap sample has the same size as the original dataset
        - Flies can appear multiple times in a single bootstrap sample
        - Empty data returns empty samples (n_samples empty lists)
    """
    import numpy as np
    
    # Initialize random number generator with seed
    rng = np.random.RandomState(seed)
    
    # Handle empty data edge case
    if len(data) == 0:
        return [[] for _ in range(n_samples)]
    
    # Generate n_samples bootstrap samples
    samples = []
    for _ in range(n_samples):
        # Sample indices with replacement
        indices = rng.choice(range(len(data)), size=len(data), replace=True)
        
        # Create bootstrap sample by selecting flies at sampled indices
        bootstrap_sample = [data[i] for i in indices]
        
        samples.append(bootstrap_sample)
    
    return samples


# =============================================================================
# Unit 5: Simulate Window Sampling
# =============================================================================

def simulate_window_sampling(
    events: List[Tuple[int, int]],
    total_frames: int,
    window_size: int,
    sampling_rate: float,
    strategy: str,
    seed: int
) -> Tuple[List[Tuple[int, int]], Dict]:
    """
    Simulate the effect of window sampling on event detection.
    
    Divides video into non-overlapping windows, selects a subset based on
    the specified strategy, and extracts events that overlap with sampled
    windows. Events are truncated at window boundaries to reflect partial
    observation within sampled windows.
    
    Args:
        events: List of (start_frame, end_frame) tuples representing grooming events
        total_frames: Total number of frames in video
        window_size: Size of sampling window in frames
        sampling_rate: Proportion of windows to sample (0-1)
        strategy: Sampling strategy ('uniform', 'stratified', 'systematic')
        seed: Random seed for reproducibility
    
    Returns:
        Tuple containing:
        - sampled_events: Events within sampled windows (truncated at boundaries)
        - edge_info: Dictionary with edge event statistics
    
    Edge Info Structure:
        {
            'total_events': int,           # Events in sampled windows
            'edge_events': int,            # Events touching boundaries
            'edge_percentage': float       # Percentage that are edge events
        }
    
    Event Truncation Examples:
        - Event (100, 200) in window (150, 250) → becomes (150, 200)
        - Event (200, 300) in window (150, 250) → becomes (200, 250)
        - Event (150, 200) in window (150, 250) → unchanged (150, 200)
    
    Sampling Strategies:
        - 'uniform': Random sampling without structure
        - 'stratified': Ensures even distribution across video
        - 'systematic': Every nth window (deterministic)
    
    Edge Cases:
        - Empty events list: Returns empty list with zero statistics
        - No events in sampled windows: Returns empty list with zero statistics
        - Sampling rate rounds to at least 1 window
        - Events spanning multiple windows appear once per window (truncated)
    
    Example:
        >>> events = [(100, 200), (300, 400)]
        >>> sampled, info = simulate_window_sampling(
        ...     events, 1000, 100, 0.5, 'uniform', 42
        ... )
        >>> info['total_events'] >= 0
        True
        >>> info['edge_percentage'] >= 0.0
        True
    """
    import numpy as np
    
    # Initialize random number generator
    rng = np.random.RandomState(seed)
    
    # Generate non-overlapping windows
    windows = [(i, min(i + window_size - 1, total_frames - 1)) 
               for i in range(0, total_frames, window_size)]
    
    # Calculate number of windows to sample (at least 1)
    n_windows = len(windows)
    n_sample = max(1, int(n_windows * sampling_rate))
    
    # Select windows based on strategy
    if strategy == 'uniform':
        # Random sampling without replacement
        sampled_indices = rng.choice(n_windows, size=n_sample, replace=False)
        sampled_windows = [windows[i] for i in sorted(sampled_indices)]
    
    elif strategy == 'stratified':
        # Divide windows into strata and sample from each
        # Ensures even distribution across the video
        stride = n_windows / n_sample
        sampled_indices = [int(i * stride) for i in range(n_sample)]
        sampled_windows = [windows[i] for i in sampled_indices]
    
    elif strategy == 'systematic':
        # Every nth window (deterministic)
        stride = max(1, n_windows // n_sample)
        sampled_indices = list(range(0, n_windows, stride))[:n_sample]
        sampled_windows = [windows[i] for i in sampled_indices]
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. "
                        f"Must be 'uniform', 'stratified', or 'systematic'.")
    
    # Extract and truncate events from sampled windows
    sampled_events = []
    edge_events_count = 0
    
    for window_start, window_end in sampled_windows:
        for event_start, event_end in events:
            # Check if event overlaps with window
            if event_end >= window_start and event_start <= window_end:
                # Truncate event at window boundaries
                truncated_start = max(event_start, window_start)
                truncated_end = min(event_end, window_end)
                
                truncated_event = (truncated_start, truncated_end)
                sampled_events.append(truncated_event)
                
                # Check if event touches window boundaries (edge event)
                touches_start = event_start < window_start
                touches_end = event_end > window_end
                if touches_start or touches_end:
                    edge_events_count += 1
    
    # Calculate edge statistics
    total_events = len(sampled_events)
    edge_percentage = (edge_events_count / total_events * 100.0) if total_events > 0 else 0.0
    
    edge_info = {
        'total_events': total_events,
        'edge_events': edge_events_count,
        'edge_percentage': edge_percentage
    }
    
    return sampled_events, edge_info
