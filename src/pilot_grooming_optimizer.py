"""Pilot Grooming Optimizer - Script 1"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from common_library import load_event_csv, process_directory_structure, calculate_all_behavioral_metrics

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


# ============================================================================= 
# Unit 6: Calculate Statistical Power
# =============================================================================

def calculate_statistical_power(
    group1_metrics: List[float],
    group2_metrics: List[float],
    alpha: float,
    effect_size: float = 0.8
) -> float:
    """
    Estimate statistical power to detect differences between two groups.
    
    Calculates the achieved effect size (Cohen's d) between two groups and
    estimates the statistical power for detecting this difference using a
    two-sample t-test framework.
    
    Args:
        group1_metrics: List of metrics from group 1
        group2_metrics: List of metrics from group 2
        alpha: Significance level (e.g., 0.05)
        effect_size: Target effect size (default: 0.8) - currently unused
        
    Returns:
        Statistical power as probability (0-1 scale)
        
    Method:
        1. Calculate Cohen's d from observed data
        2. Use statsmodels.stats.power.ttest_power to estimate power
        3. Handle edge cases (identical groups, small samples, zero variance)
        
    Formula:
        Cohen's d = |mean1 - mean2| / pooled_std
        pooled_std = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1 + n2 - 2))
        
    Edge cases:
        - Empty groups: Returns 0.0
        - Identical groups (d=0): Returns alpha (type I error rate)
        - Very small samples: Returns reduced power
        - Zero pooled std: Returns alpha if means equal, 1.0 otherwise
        
    Example:
        >>> group1 = [10.0, 10.5, 9.5, 10.2]
        >>> group2 = [20.0, 20.5, 19.5, 20.2]
        >>> power = calculate_statistical_power(group1, group2, 0.05)
        >>> power > 0.8
        True
    """
    import numpy as np
    from statsmodels.stats.power import ttest_power
    
    # Handle edge case: empty groups
    if len(group1_metrics) == 0 or len(group2_metrics) == 0:
        return 0.0
    
    # Convert to numpy arrays
    group1 = np.array(group1_metrics)
    group2 = np.array(group2_metrics)
    
    # Calculate sample sizes
    n1 = len(group1)
    n2 = len(group2)
    
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Calculate sample standard deviations (ddof=1 for sample std)
    std1 = np.std(group1, ddof=1) if n1 > 1 else 0.0
    std2 = np.std(group2, ddof=1) if n2 > 1 else 0.0
    
    # Calculate pooled standard deviation
    if n1 + n2 - 2 > 0:
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        # Not enough degrees of freedom (e.g., single sample in each group)
        return 0.0
    
    # Handle edge case: zero pooled std (no variance in data)
    if pooled_std == 0 or np.isnan(pooled_std):
        # All values are identical within and across groups
        if abs(mean1 - mean2) < 1e-10:  # Means essentially equal
            return alpha
        else:
            # Means differ but no variance - infinite effect size
            return 1.0
    
    # Calculate Cohen's d (achieved effect size from data)
    cohens_d = abs(mean1 - mean2) / pooled_std
    
    # Calculate effective sample size for power calculation
    # For unequal groups, use harmonic mean
    nobs = 2 * n1 * n2 / (n1 + n2)
    
    # Calculate statistical power using statsmodels
    try:
        power = ttest_power(
            effect_size=cohens_d,
            nobs=nobs,
            alpha=alpha,
            alternative='two-sided'
        )
        
        # Handle NaN results (can occur with very small samples)
        if np.isnan(power):
            # For very small samples or numerical instability, use conservative estimate
            # If effect size is large, assume some power; otherwise near alpha
            if cohens_d > 2.0:
                return 0.5  # Moderate power for large effects with small samples
            elif cohens_d > 0.5:
                return 0.3  # Low-moderate power
            else:
                return alpha
        
        # Ensure power is in valid range [0, 1]
        power = float(np.clip(power, 0.0, 1.0))
        
        return power
    except Exception:
        # If calculation fails, return conservative estimate based on effect size
        if cohens_d > 2.0:
            return 0.5
        elif cohens_d > 0.5:
            return 0.3
        else:
            return alpha


# =============================================================================
# Unit 7: Evaluate Parameter Combination
# =============================================================================

def evaluate_parameter_combination(
    pilot_data: Dict[str, List[List[Tuple[int, int]]]],
    params: Dict,
    config: Dict,
    n_bootstrap: int = 10000
) -> Dict[str, float]:
    """
    Comprehensively evaluate one parameter combination.
    
    Evaluates a parameter combination by simulating window sampling across
    bootstrap iterations and calculating multiple performance scores including
    statistical power, bias, error rate, efficiency, and robustness.
    
    Args:
        pilot_data: Complete pilot dataset by genotype
                   Structure: {genotype_name: [fly1_events, fly2_events, ...]}
        params: Single parameter combination with keys:
               - window_size: Size of sampling window in frames
               - sampling_rate: Proportion of windows to sample (0-1)
               - strategy: Sampling strategy ('uniform', 'stratified', 'systematic')
               - edge_threshold: Maximum acceptable edge event percentage
        config: Configuration with keys:
               - alpha: Significance level for statistical tests
               - power: Target statistical power (for reference)
               - expected_frames: Expected frame count per video
        n_bootstrap: Bootstrap iterations (default: 10000)
    
    Returns:
        Dictionary containing evaluation scores:
        - 'power': Average statistical power (0-1)
        - 'bias': Bias vs ground truth (0-1, lower is better)
        - 'error_rate': False positive/negative rate (0-1, lower is better)
        - 'efficiency': Time saved based on sampling_rate (0-1, higher is better)
        - 'robustness': CV of power across iterations (0-1+, lower is better)
        - 'composite': Weighted composite score (0-1, higher is better)
    
    Composite Score Formula:
        composite = 0.4 * power + 0.2 * (1 - bias) + 0.2 * efficiency + 0.2 * (1 - robustness)
    
    Example:
        >>> pilot_data = {
        ...     'WT': [[(100, 150), (200, 250)], [(300, 400)]],
        ...     'KO': [[(500, 600)], [(700, 800)]]
        ... }
        >>> params = {
        ...     'window_size': 100,
        ...     'sampling_rate': 0.3,
        ...     'strategy': 'uniform',
        ...     'edge_threshold': 20
        ... }
        >>> config = {'alpha': 0.05, 'power': 0.8, 'expected_frames': 9000}
        >>> scores = evaluate_parameter_combination(pilot_data, params, config, n_bootstrap=100)
        >>> 0 <= scores['composite'] <= 1
        True
    """
    from tqdm import tqdm
    import numpy as np
    
    # Extract parameters
    window_size = params['window_size']
    sampling_rate = params['sampling_rate']
    strategy = params['strategy']
    edge_threshold = params['edge_threshold']
    
    # Extract config
    alpha = config['alpha']
    expected_frames = config['expected_frames']
    
    # Get genotype names (should be exactly 2)
    genotype_names = list(pilot_data.keys())
    if len(genotype_names) != 2:
        raise ValueError(f"Expected 2 genotypes, got {len(genotype_names)}")
    
    genotype1, genotype2 = genotype_names[0], genotype_names[1]
    
    # Calculate ground truth metrics for each genotype (aggregate all flies)
    ground_truth = {}
    for genotype_name in genotype_names:
        all_events = []
        for fly_events in pilot_data[genotype_name]:
            all_events.extend(fly_events)
        
        gt_metrics = calculate_all_behavioral_metrics(
            all_events, 
            expected_frames
        )
        ground_truth[genotype_name] = gt_metrics
    
    # Generate bootstrap samples for each genotype
    bootstrap_samples = {}
    for genotype_name in genotype_names:
        bootstrap_samples[genotype_name] = generate_bootstrap_samples(
            pilot_data[genotype_name],
            n_samples=n_bootstrap,
            seed=42
        )
    
    # Initialize tracking arrays
    power_scores = []
    bias_scores = []
    edge_percentages = []
    
    # Iterate through bootstrap samples with progress bar
    for i in tqdm(range(n_bootstrap), desc="Evaluating parameters"):
        # For this bootstrap iteration, collect fly-level metrics for each genotype
        fly_metrics = {genotype1: [], genotype2: []}
        iteration_edge_events = []
        iteration_total_events = []
        
        for genotype_name in genotype_names:
            # Get bootstrap sample for this iteration
            bootstrap_sample = bootstrap_samples[genotype_name][i]
            
            # Apply window sampling to each fly in the bootstrap sample
            for fly_events in bootstrap_sample:
                # Apply window sampling to this fly
                sampled_events, edge_info = simulate_window_sampling(
                    fly_events,
                    expected_frames,
                    window_size,
                    sampling_rate,
                    strategy,
                    seed=42 + i  # Different seed for each iteration
                )
                
                # Calculate metrics for this individual fly
                fly_metric = calculate_all_behavioral_metrics(
                    sampled_events,
                    expected_frames
                )
                
                # Store fly-level frequency metric
                fly_metrics[genotype_name].append(fly_metric['event_frequency'])
                
                # Track edge events
                iteration_edge_events.append(edge_info['edge_events'])
                iteration_total_events.append(edge_info['total_events'])
        
        # Calculate power for this iteration using fly-level metrics
        iteration_power = calculate_statistical_power(
            fly_metrics[genotype1],
            fly_metrics[genotype2],
            alpha
        )
        power_scores.append(iteration_power)
        
        # Calculate bias for this iteration (aggregate metrics vs ground truth)
        agg_metrics = {}
        for genotype_name in genotype_names:
            # Aggregate events from all flies in this bootstrap sample
            all_sampled_events = []
            for fly_events in bootstrap_samples[genotype_name][i]:
                sampled_events, _ = simulate_window_sampling(
                    fly_events,
                    expected_frames,
                    window_size,
                    sampling_rate,
                    strategy,
                    seed=42 + i
                )
                all_sampled_events.extend(sampled_events)
            
            agg_metrics[genotype_name] = calculate_all_behavioral_metrics(
                all_sampled_events,
                expected_frames
            )
        
        # Calculate bias for each genotype
        bias1 = abs(agg_metrics[genotype1]['event_frequency'] - 
                    ground_truth[genotype1]['event_frequency'])
        bias2 = abs(agg_metrics[genotype2]['event_frequency'] - 
                    ground_truth[genotype2]['event_frequency'])
        avg_bias = (bias1 + bias2) / 2
        bias_scores.append(avg_bias)
        
        # Calculate edge percentage for this iteration
        total_edge = sum(iteration_edge_events)
        total_evts = sum(iteration_total_events)
        if total_evts > 0:
            edge_pct = (total_edge / total_evts) * 100
        else:
            edge_pct = 0.0
        edge_percentages.append(edge_pct)
    
    # Calculate final scores
    
    # Power: average across iterations
    avg_power = float(np.mean(power_scores))
    
    # Bias: average bias normalized to 0-1 scale
    avg_bias = float(np.mean(bias_scores))
    max_bias = 10.0  # Assume max reasonable bias is 10 events/min
    normalized_bias = min(avg_bias / max_bias, 1.0)
    
    # Error rate: based on edge events exceeding threshold
    avg_edge_percentage = float(np.mean(edge_percentages))
    if avg_edge_percentage > edge_threshold:
        error_rate = (avg_edge_percentage - edge_threshold) / 100.0
    else:
        error_rate = 0.0
    error_rate = min(error_rate, 1.0)  # Cap at 1.0
    
    # Efficiency: inverse of sampling rate (more sampling = less efficient)
    efficiency = 1.0 - sampling_rate
    
    # Robustness: CV of power across iterations
    if avg_power > 0:
        std_power = float(np.std(power_scores))
        robustness_cv = std_power / avg_power
    else:
        robustness_cv = 0.0
    # Normalize to 0-1 scale (assume max CV is 1.0)
    normalized_robustness = min(robustness_cv / 1.0, 1.0)
    
    # Calculate composite score
    composite = (0.4 * avg_power + 
                 0.2 * (1 - normalized_bias) + 
                 0.2 * efficiency + 
                 0.2 * (1 - normalized_robustness))
    
    return {
        'power': avg_power,
        'bias': normalized_bias,
        'error_rate': error_rate,
        'efficiency': efficiency,
        'robustness': normalized_robustness,
        'composite': composite
    }


# =============================================================================
# Unit 8: Cross-Validation Framework
# =============================================================================

def cross_validate_parameters(
    data: Dict[str, List[List[Tuple[int, int]]]],
    params: Dict,
    n_folds: int,
    seed: int
) -> Dict:
    """
    Assess parameter stability using k-fold cross-validation.
    
    Performs stratified cross-validation by splitting data by genotype,
    evaluating parameters on each fold, and calculating mean/std of metrics.
    This helps determine if parameter performance is stable across different
    data subsets, which is crucial for generalizing to new experiments.
    
    Args:
        data: Pilot data by genotype
              Structure: {genotype_name: [fly1_events, fly2_events, ...]}
        params: Parameters to validate with keys:
               - window_size: Size of sampling window in frames
               - sampling_rate: Proportion of windows to sample (0-1)
               - strategy: Sampling strategy ('uniform', 'stratified', 'systematic')
               - edge_threshold: Maximum acceptable edge event percentage
        n_folds: Number of CV folds (will be adjusted if dataset is too small)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing CV results:
        - 'power_mean': Mean statistical power across folds
        - 'power_std': Std of statistical power across folds
        - 'bias_mean': Mean bias across folds
        - 'bias_std': Std of bias across folds
        - 'composite_mean': Mean composite score across folds
        - 'composite_std': Std of composite score across folds
        - 'n_folds': Actual number of folds used
        
    Cross-Validation Process:
        1. Determine actual fold count (min of n_folds and smallest genotype size)
        2. For each fold:
           - Extract test subset (stratified by genotype)
           - Evaluate parameters on test subset
           - Record power, bias, and composite scores
        3. Calculate mean and standard deviation across all folds
        
    Edge Cases:
        - If dataset has < n_folds samples per genotype, fold count is adjusted
        - Ensures no data leakage between folds (each sample appears in exactly one fold)
        - Uses stratified splitting to ensure each fold has both genotypes
        
    Example:
        >>> data = {
        ...     'WT': [[(100, 150)], [(200, 250)], [(300, 350)], [(400, 450)], [(500, 550)]],
        ...     'KO': [[(600, 650)], [(700, 750)], [(800, 850)], [(900, 950)], [(1000, 1050)]]
        ... }
        >>> params = {'window_size': 300, 'sampling_rate': 0.2, 
        ...           'strategy': 'uniform', 'edge_threshold': 20}
        >>> results = cross_validate_parameters(data, params, n_folds=5, seed=42)
        >>> results['n_folds']
        5
        >>> 0 <= results['power_mean'] <= 1
        True
    """
    from sklearn.model_selection import KFold
    from tqdm import tqdm
    import numpy as np
    
    # Determine actual number of folds based on smallest genotype
    genotype_names = list(data.keys())
    min_samples = min(len(data[genotype]) for genotype in genotype_names)
    actual_folds = min(n_folds, min_samples)
    
    # Initialize KFold with seed for reproducibility
    kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    
    # Create fold indices for each genotype
    fold_indices = {}
    for genotype_name in genotype_names:
        n_samples = len(data[genotype_name])
        indices = np.arange(n_samples)
        fold_indices[genotype_name] = list(kfold.split(indices))
    
    # Track metrics across folds
    fold_metrics = []
    
    # Configuration for evaluation (using standard defaults)
    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }
    
    # Iterate through folds with progress tracking
    for fold_idx in tqdm(range(actual_folds), desc="Cross-validation folds"):
        # Create test set for this fold (stratified by genotype)
        test_data = {}
        for genotype_name in genotype_names:
            _, test_idx = fold_indices[genotype_name][fold_idx]
            test_data[genotype_name] = [data[genotype_name][i] for i in test_idx]
        
        # Evaluate parameters on test set
        scores = evaluate_parameter_combination(
            test_data, params, config, n_bootstrap=1000
        )
        
        # Record metrics for this fold
        fold_metrics.append({
            'power': scores['power'],
            'bias': scores['bias'],
            'composite': scores['composite']
        })
    
    # Calculate mean and std across folds
    power_values = [m['power'] for m in fold_metrics]
    bias_values = [m['bias'] for m in fold_metrics]
    composite_values = [m['composite'] for m in fold_metrics]
    
    cv_results = {
        'power_mean': float(np.mean(power_values)),
        'power_std': float(np.std(power_values)),
        'bias_mean': float(np.mean(bias_values)),
        'bias_std': float(np.std(bias_values)),
        'composite_mean': float(np.mean(composite_values)),
        'composite_std': float(np.std(composite_values)),
        'n_folds': actual_folds
    }
    
    return cv_results


# =============================================================================
# Unit 9: Optimize Across Parameter Space
# =============================================================================

def optimize_parameter_space(
    pilot_data: Dict[str, List[List[Tuple[int, int]]]],
    parameter_space: Dict,
    config: Dict,
    n_bootstrap: int = 10000
) -> Tuple[Dict, List[Dict]]:
    """
    Find optimal parameters through exhaustive search.
    
    Evaluates all possible parameter combinations from the parameter space,
    scoring each based on statistical power, bias, efficiency, and robustness.
    Returns the best combination and complete results for all combinations.
    
    Args:
        pilot_data: Complete pilot dataset by genotype
                   Structure: {genotype_name: [fly1_events, fly2_events, ...]}
        parameter_space: All parameter combinations with keys:
                        - 'window_sizes': List of window sizes
                        - 'sampling_rates': List of sampling rates
                        - 'strategies': List of sampling strategies
                        - 'edge_thresholds': List of edge thresholds
        config: Optimization configuration with keys:
               - 'alpha': Significance level
               - 'power': Target statistical power (for reference/threshold)
               - 'expected_frames': Expected frame count
        n_bootstrap: Number of bootstrap iterations (default: 10000)
    
    Returns:
        Tuple containing:
        - best_params: Dictionary with optimal parameter combination
        - all_results: List of all evaluation results (sorted by composite score)
    
    Result Structure:
        Each result dict contains:
        {
            'window_size': int,
            'sampling_rate': float,
            'strategy': str,
            'edge_threshold': int,
            'scores': {
                'power': float,
                'bias': float,
                'efficiency': float,
                'robustness': float,
                'composite': float
            }
        }
    
    Algorithm:
        1. Generate all combinations using itertools.product
        2. Evaluate each combination with progress tracking
        3. Sort results by composite score (descending)
        4. Select best parameters
        5. Warn if best parameters don't meet power threshold
    
    Example:
        >>> pilot_data = {
        ...     'WT': [[(100, 150), (200, 250)] for _ in range(5)],
        ...     'KO': [[(300, 350), (400, 450)] for _ in range(5)]
        ... }
        >>> parameter_space = {
        ...     'window_sizes': [100, 200, 300],
        ...     'sampling_rates': [0.1, 0.2, 0.3],
        ...     'strategies': ['uniform', 'stratified'],
        ...     'edge_thresholds': [10, 20]
        ... }
        >>> config = {'alpha': 0.05, 'power': 0.8, 'expected_frames': 9000}
        >>> best_params, all_results = optimize_parameter_space(
        ...     pilot_data, parameter_space, config, n_bootstrap=100
        ... )
        >>> best_params['scores']['composite'] >= all_results[-1]['scores']['composite']
        True
    """
    import itertools
    from tqdm import tqdm
    
    # Extract parameter lists
    window_sizes = parameter_space['window_sizes']
    sampling_rates = parameter_space['sampling_rates']
    strategies = parameter_space['strategies']
    edge_thresholds = parameter_space['edge_thresholds']
    
    # Generate all parameter combinations
    combinations = list(itertools.product(
        window_sizes,
        sampling_rates,
        strategies,
        edge_thresholds
    ))
    
    total_combinations = len(combinations)
    
    # Initialize results storage
    all_results = []
    best_score = 0.0
    
    # Evaluate each combination with progress tracking
    with tqdm(total=total_combinations, desc="Optimizing parameters") as pbar:
        for window_size, sampling_rate, strategy, edge_threshold in combinations:
            # Create parameter dictionary
            params = {
                'window_size': window_size,
                'sampling_rate': sampling_rate,
                'strategy': strategy,
                'edge_threshold': edge_threshold
            }
            
            # Evaluate this combination
            scores = evaluate_parameter_combination(
                pilot_data, params, config, n_bootstrap=n_bootstrap
            )
            
            # Store result with parameters
            result = {
                'window_size': window_size,
                'sampling_rate': sampling_rate,
                'strategy': strategy,
                'edge_threshold': edge_threshold,
                'scores': scores
            }
            all_results.append(result)
            
            # Update best score for progress display
            if scores['composite'] > best_score:
                best_score = scores['composite']
                pbar.set_postfix({'best_score': f"{best_score:.4f}"})
            
            # Update progress bar
            pbar.update(1)
    
    # Sort results by composite score (descending - highest first)
    all_results.sort(key=lambda x: x['scores']['composite'], reverse=True)
    
    # Select best parameters (first in sorted list)
    best_params = all_results[0]
    
    # Check if best parameters meet power threshold
    target_power = config['power']
    if best_params['scores']['power'] < target_power:
        print(f"\nWarning: Best parameters achieve power of {best_params['scores']['power']:.4f}, "
              f"which is below target power of {target_power:.4f}")
    
    return best_params, all_results
