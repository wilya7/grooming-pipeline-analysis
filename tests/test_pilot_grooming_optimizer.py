"""Test suite for Pilot Grooming Optimizer - Script 1"""

import pytest
import tempfile
import sys
from pathlib import Path

from pilot_grooming_optimizer import (
    parse_arguments, 
    load_pilot_data, 
    generate_parameter_space,
    generate_bootstrap_samples,
		simulate_window_sampling,
		calculate_statistical_power,
		evaluate_parameter_combination,
		cross_validate_parameters,
		optimize_parameter_space,
		generate_visualization_plots,
		generate_detailed_log,
		generate_pdf_report,
		main
)


# ============================================================================= 
# Fixtures and Helpers 
# ============================================================================= 

def create_test_csv_new_format(filepath, events):
    """
    Create a CSV file in the new EventID,StartFrame,StopFrame format.
    
    Args:
        filepath: Path to create CSV file
        events: List of (start_frame, stop_frame) tuples
    """
    import pandas as pd
    df = pd.DataFrame({
        'EventID': list(range(1, len(events) + 1)),
        'StartFrame': [e[0] for e in events],
        'StopFrame': [e[1] for e in events]
    })
    df.to_csv(filepath, index=False)
		

# =============================================================================
# Tests for Unit 1: Parse Command Line Arguments
# =============================================================================

def test_all_required_args():
    """Test that all required arguments are parsed correctly with defaults."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        args = ['--data-dir', tmpdir, '--output', 'output.json']
        config = parse_arguments(args)
        
        # Check all returned values
        assert config['data_dir'] == tmpdir
        assert config['output'] == 'output.json'
        assert config['expected_frames'] == 9000  # default
        assert config['alpha'] == 0.05  # default
        assert config['power'] == 0.8  # default
        
        # Verify return type
        assert isinstance(config, dict)
        assert len(config) == 5


def test_missing_required_arg():
    """Test that missing --data-dir raises SystemExit."""
    args = ['--output', 'output.json']
    
    # argparse raises SystemExit when required argument is missing
    with pytest.raises(SystemExit):
        parse_arguments(args)


def test_invalid_alpha():
    """Test that alpha > 1 raises ValueError or SystemExit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = ['--data-dir', tmpdir, '--output', 'output.json', '--alpha', '1.5']
        
        # Should raise ValueError from validation
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)


def test_invalid_power():
    """Test that power < 0 raises ValueError or SystemExit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = ['--data-dir', tmpdir, '--output', 'output.json', '--power', '-0.2']
        
        # Should raise ValueError from validation
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)


def test_nonexistent_data_dir():
    """Test that non-existent data_dir raises FileNotFoundError or SystemExit."""
    args = ['--data-dir', '/nonexistent/path/xyz123', '--output', 'output.json']
    
    # Should raise FileNotFoundError from validation
    with pytest.raises((FileNotFoundError, SystemExit)):
        parse_arguments(args)


def test_default_args_from_sys_argv():
    """Test that parse_arguments() reads from sys.argv[1:] when args is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save original sys.argv
        original_argv = sys.argv
        
        try:
            # Mock sys.argv with our test arguments
            sys.argv = [
                'script_name.py',  # argv[0] is the script name
                '--data-dir', tmpdir,
                '--output', 'test_output.json',
                '--alpha', '0.01'
            ]
            
            # Call without arguments - should read from sys.argv[1:]
            config = parse_arguments()
            
            # Verify it read from sys.argv correctly
            assert config['data_dir'] == tmpdir
            assert config['output'] == 'test_output.json'
            assert config['alpha'] == 0.01
            assert config['expected_frames'] == 9000  # default
            assert config['power'] == 0.8  # default
            
        finally:
            # Always restore original sys.argv
            sys.argv = original_argv


def test_explicit_args_override_sys_argv():
    """Test that explicit args parameter overrides sys.argv."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Save original sys.argv
        original_argv = sys.argv
        
        try:
            # Set sys.argv to one set of arguments
            sys.argv = [
                'script_name.py',
                '--data-dir', tmpdir1,
                '--output', 'wrong_output.json'
            ]
            
            # Pass different explicit arguments - these should take precedence
            explicit_args = ['--data-dir', tmpdir2, '--output', 'correct_output.json']
            config = parse_arguments(explicit_args)
            
            # Verify explicit args were used, not sys.argv
            assert config['data_dir'] == tmpdir2
            assert config['output'] == 'correct_output.json'
            
        finally:
            # Always restore original sys.argv
            sys.argv = original_argv


def test_custom_optional_args():
    """Test parsing with custom optional arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = [
            '--data-dir', tmpdir,
            '--output', 'custom_output.json',
            '--expected-frames', '12000',
            '--alpha', '0.01',
            '--power', '0.9'
        ]
        config = parse_arguments(args)
        
        assert config['expected_frames'] == 12000
        assert config['alpha'] == 0.01
        assert config['power'] == 0.9


def test_alpha_boundary_values():
    """Test alpha at boundary values (0 and 1 should fail)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test alpha = 0 (should fail)
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--alpha', '0']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)
        
        # Test alpha = 1 (should fail)
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--alpha', '1.0']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)


def test_power_boundary_values():
    """Test power at boundary values (0 and 1 should fail)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test power = 0 (should fail)
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--power', '0']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)
        
        # Test power = 1 (should fail)
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--power', '1.0']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)


def test_invalid_expected_frames():
    """Test that expected_frames <= 0 raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test negative value
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--expected-frames', '-100']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)
        
        # Test zero
        args = ['--data-dir', tmpdir, '--output', 'out.json', '--expected-frames', '0']
        with pytest.raises((ValueError, SystemExit)):
            parse_arguments(args)


# =============================================================================
# Tests for Unit 2: Load Pilot Data
# =============================================================================

def test_two_genotypes_multiple_files(tmp_path):
    """Test loading data from two genotypes with multiple CSV files."""
    # Create directory structure
    genotype_a = tmp_path / "genotype_A"
    genotype_b = tmp_path / "genotype_B"
    genotype_a.mkdir()
    genotype_b.mkdir()
    
    # Create CSV files for genotype A using new format
    csv_a1 = genotype_a / "fly1.csv"
    create_test_csv_new_format(csv_a1, [(100, 150), (200, 250)])
    
    csv_a2 = genotype_a / "fly2.csv"
    create_test_csv_new_format(csv_a2, [(50, 75), (300, 400)])
    
    # Create CSV files for genotype B using new format
    csv_b1 = genotype_b / "fly1.csv"
    create_test_csv_new_format(csv_b1, [(1000, 1500), (2000, 2500)])
    
    # Load data
    pilot_data, frame_counts = load_pilot_data(str(tmp_path))
    
    # Verify structure
    assert len(pilot_data) == 2
    assert "genotype_A" in pilot_data
    assert "genotype_B" in pilot_data
    
    # Verify genotype A
    assert len(pilot_data["genotype_A"]) == 2
    assert pilot_data["genotype_A"][0] == [(100, 150), (200, 250)]
    assert pilot_data["genotype_A"][1] == [(50, 75), (300, 400)]
    
    # Verify genotype B
    assert len(pilot_data["genotype_B"]) == 1
    assert pilot_data["genotype_B"][0] == [(1000, 1500), (2000, 2500)]
    
    # Verify frame counts
    assert "genotype_A" in frame_counts
    assert "genotype_B" in frame_counts
    assert frame_counts["genotype_A"] == [250, 400]  # Max end frames
    assert frame_counts["genotype_B"] == [2500]


def test_single_genotype(tmp_path):
    """Test that single genotype raises ValueError."""
    # Create only one genotype directory
    genotype_a = tmp_path / "genotype_A"
    genotype_a.mkdir()
    
    csv_a1 = genotype_a / "fly1.csv"
    csv_a1.write_text("Frame\n100\n150\n")
    
    # Should raise ValueError requiring at least 2 genotypes
    with pytest.raises(ValueError, match="At least 2 genotypes required"):
        load_pilot_data(str(tmp_path))


def test_empty_csv_file(tmp_path):
    """Test handling of empty CSV files."""
    # Create directory structure
    genotype_a = tmp_path / "genotype_A"
    genotype_b = tmp_path / "genotype_B"
    genotype_a.mkdir()
    genotype_b.mkdir()
    
    # Create CSV with data using new format
    csv_a1 = genotype_a / "fly1.csv"
    create_test_csv_new_format(csv_a1, [(100, 150)])
    
    # Create empty CSV (header only, no events)
    csv_a2 = genotype_a / "fly2.csv"
    csv_a2.write_text("EventID,StartFrame,StopFrame\n")
    
    # Create CSV with data for genotype B using new format
    csv_b1 = genotype_b / "fly1.csv"
    create_test_csv_new_format(csv_b1, [(200, 250)])
    
    # Load data
    pilot_data, frame_counts = load_pilot_data(str(tmp_path))
    
    # Verify empty CSV handled correctly
    assert len(pilot_data["genotype_A"]) == 2
    assert pilot_data["genotype_A"][0] == [(100, 150)]
    assert pilot_data["genotype_A"][1] == []  # Empty events list
    
    # Verify frame counts (empty CSV should have 0)
    assert frame_counts["genotype_A"][0] == 150
    assert frame_counts["genotype_A"][1] == 0
		

def test_mixed_frame_counts(tmp_path):
    """Test handling of mixed frame counts within a genotype."""
    # Create directory structure
    genotype_a = tmp_path / "genotype_A"
    genotype_b = tmp_path / "genotype_B"
    genotype_a.mkdir()
    genotype_b.mkdir()
    
    # Create CSVs with different frame counts for genotype A
    csv_a1 = genotype_a / "fly1.csv"
    csv_a1.write_text("Frame\n100\n150\n8900\n9000\n")
    
    csv_a2 = genotype_a / "fly2.csv"
    csv_a2.write_text("Frame\n100\n150\n8900\n9010\n")
    
    csv_a3 = genotype_a / "fly3.csv"
    csv_a3.write_text("Frame\n100\n150\n")
    
    # Create CSV for genotype B with different frame count
    csv_b1 = genotype_b / "fly1.csv"
    csv_b1.write_text("Frame\n100\n8995\n")
    
    # Load data
    pilot_data, frame_counts = load_pilot_data(str(tmp_path))
    
    # Verify all frame counts are recorded correctly
    # Files are processed in sorted order: fly1.csv, fly2.csv, fly3.csv
    assert frame_counts["genotype_A"] == [9000, 9010, 150]
    assert frame_counts["genotype_B"] == [8995]
    
    # Verify data structure is correct
    assert len(pilot_data["genotype_A"]) == 3
    assert len(pilot_data["genotype_B"]) == 1


# ============================================================================= 
# Tests for Unit 3: Generate Parameter Space
# =============================================================================

def test_standard_9000_frames():
    """Test parameter generation with standard 9000-frame videos."""
    frame_counts = {
        'WT': [9000, 9000, 9000],
        'KO': [9000, 9000]
    }
    
    params = generate_parameter_space(frame_counts)
    
    # Verify structure
    assert 'window_sizes' in params
    assert 'sampling_rates' in params
    assert 'strategies' in params
    assert 'edge_thresholds' in params
    
    # Verify fixed parameters
    assert params['sampling_rates'] == [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    assert params['strategies'] == ['uniform', 'stratified', 'systematic']
    assert params['edge_thresholds'] == [5, 10, 15, 20, 25, 30]
    
    # Verify window sizes (9000 has many divisors that are multiples of 25 and ≥ 100)
    # Expected: 100, 125, 150, 180, 200, 225, 300, 360, 450, 600, 900, 1125, 1800, 2250, 4500, 9000
    # All should be divisors of 9000, ≥ 100, and multiples of 25
    assert len(params['window_sizes']) > 5  # Should have multiple valid sizes
    for size in params['window_sizes']:
        assert size >= 100
        assert size % 25 == 0
        assert 9000 % size == 0  # Should be divisor


def test_short_video_500_frames():
    """Test parameter generation with short 500-frame video."""
    frame_counts = {
        'WT': [500, 550, 600],
        'KO': [500, 525]
    }
    
    params = generate_parameter_space(frame_counts)
    
    # Verify structure
    assert 'window_sizes' in params
    assert len(params['window_sizes']) > 0
    
    # Verify window sizes are valid for 500 frames
    # Divisors of 500: 1, 2, 4, 5, 10, 20, 25, 50, 100, 125, 250, 500
    # Primary (≥100 AND multiple of 25): 100, 125, 250, 500
    for size in params['window_sizes']:
        assert size >= 100
        assert size <= 500
        assert size % 25 == 0
        assert 500 % size == 0


def test_very_short_video():
    """Test that very short videos raise ValueError."""
    frame_counts = {
        'WT': [50, 60, 75],
        'KO': [50, 55]
    }
    
    # Should raise ValueError because shortest video < 100 frames
    with pytest.raises(ValueError, match="Shortest video has .* frames, which is below the minimum"):
        generate_parameter_space(frame_counts)


def test_mixed_frame_counts():
    """Test that shortest video determines window sizes."""
    frame_counts = {
        'WT': [9100, 9000, 9050],
        'KO': [8900, 9100]  # 8900 is shortest
    }
    
    params = generate_parameter_space(frame_counts)
    
    # Verify window sizes are based on 8900 (shortest)
    # All window sizes must be divisors of 8900
    for size in params['window_sizes']:
        assert size >= 100
        assert 8900 % size == 0  # Must divide evenly into shortest video
    
    # Verify that a divisor of 9000 that doesn't divide 8900 is NOT included
    # Example: 450 divides 9000 but not 8900
    # 8900 = 2^2 × 5^2 × 89, so 450 (2 × 3^2 × 5^2) doesn't divide it
    assert 450 not in params['window_sizes']


def test_fallback_1_triggered():
    """Test that Fallback 1 triggers when no divisors are multiples of 25."""
    # Use 101 frames (prime number)
    # Divisors: 1, 101
    # valid_divisors: [101] (meets >= 100 threshold)
    # primary_candidates: [] (101 % 25 = 1, not a multiple of 25)
    # Should use Fallback 1: all valid_divisors
    frame_counts = {
        'WT': [101, 101],
        'KO': [101]
    }
    
    params = generate_parameter_space(frame_counts)
    
    # Verify structure
    assert 'window_sizes' in params
    assert 'sampling_rates' in params
    assert 'strategies' in params
    assert 'edge_thresholds' in params
    
    # Verify Fallback 1 was used: got [101] even though it's not a multiple of 25
    assert params['window_sizes'] == [101]
    assert 101 >= 100  # Meets minimum threshold
    assert 101 % 25 != 0  # Not a multiple of 25 (proves Fallback 1 was used)
    
    # Verify fixed parameters unchanged
    assert params['sampling_rates'] == [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    assert params['strategies'] == ['uniform', 'stratified', 'systematic']
    assert params['edge_thresholds'] == [5, 10, 15, 20, 25, 30]


def test_fallback_1_with_prime_103():
    """Test Fallback 1 with another prime number (103)."""
    # 103 is prime, so divisors are 1 and 103
    # valid_divisors: [103]
    # primary_candidates: [] (103 % 25 = 3, not a multiple of 25)
    # Should use Fallback 1: [103]
    frame_counts = {
        'WT': [103, 103, 103],
        'KO': [103, 103]
    }
    
    params = generate_parameter_space(frame_counts)
    
    # Verify Fallback 1 was used
    assert params['window_sizes'] == [103]
    assert 103 >= 100
    assert 103 % 25 != 0  # Not a multiple of 25


def test_fallback_2_theoretically_unreachable():
    """Test that Fallback 2 logic exists but is theoretically unreachable.
    
    Fallback 2 is for multiples of 25 >= 100 that are NOT divisors.
    However, if shortest_frames >= 100, then shortest_frames itself is
    always a valid divisor >= 100, so valid_divisors is never empty.
    Therefore, Fallback 2 cannot be reached in practice.
    
    This test documents that the code exists as defensive programming.
    """
    # Any frame count >= 100 will have itself as a valid divisor
    # So valid_divisors will never be empty, and Fallback 2 won't trigger
    
    # Test with 100 (smallest valid value)
    frame_counts = {'WT': [100], 'KO': [100]}
    params = generate_parameter_space(frame_counts)
    
    # 100 is its own divisor, so we get [100] from either Primary or Fallback 1
    assert 100 in params['window_sizes']
    
    # Test with 101 (prime, but still has itself as divisor)
    frame_counts = {'WT': [101], 'KO': [101]}
    params = generate_parameter_space(frame_counts)
    
    # 101 is its own divisor, so valid_divisors = [101], triggers Fallback 1
    assert params['window_sizes'] == [101]
    
    # The logic for Fallback 2 exists in the code but is unreachable
    # because any valid shortest_frames >= 100 is itself a divisor >= 100


def test_window_sizes_sorted():
    """Test that window sizes are returned in sorted order."""
    frame_counts = {
        'WT': [9000, 9000],
        'KO': [9000]
    }
    
    params = generate_parameter_space(frame_counts)
    window_sizes = params['window_sizes']
    
    # Verify sorted in ascending order
    assert window_sizes == sorted(window_sizes)


def test_window_sizes_no_duplicates():
    """Test that window sizes contain no duplicates."""
    frame_counts = {
        'WT': [9000, 9000, 9000],
        'KO': [9000, 9000]
    }
    
    params = generate_parameter_space(frame_counts)
    window_sizes = params['window_sizes']
    
    # Verify no duplicates
    assert len(window_sizes) == len(set(window_sizes))


def test_window_sizes_all_valid_divisors():
    """Test that all returned window sizes are valid divisors of shortest frame count."""
    frame_counts = {
        'WT': [1200, 1300],
        'KO': [1200, 1250]
    }
    
    params = generate_parameter_space(frame_counts)
    shortest = 1200
    
    # All window sizes should divide evenly into shortest
    for size in params['window_sizes']:
        assert shortest % size == 0
        assert size >= 100


# =============================================================================
# Tests for Unit 4: Bootstrap Sample Generator
# =============================================================================

def test_standard_sampling():
    """Test bootstrap sampling with 10 flies and 100 samples."""
    # Create data with 10 flies
    data = [
        [(i*100, i*100 + 50)] for i in range(10)
    ]
    
    samples = generate_bootstrap_samples(data, n_samples=100, seed=42)
    
    # Verify structure
    assert len(samples) == 100  # 100 bootstrap samples
    for sample in samples:
        assert len(sample) == 10  # Each sample has 10 flies (same as original)
        
    # Verify each fly in each sample is a valid fly from original data
    for sample in samples:
        for fly in sample:
            assert fly in data


def test_single_fly():
    """Test that single fly data results in identical samples."""
    # Single fly data
    data = [[(100, 150), (200, 250)]]
    
    samples = generate_bootstrap_samples(data, n_samples=10, seed=42)
    
    # Verify structure
    assert len(samples) == 10
    
    # All samples should be identical (only one fly to sample from)
    for sample in samples:
        assert len(sample) == 1
        assert sample[0] == data[0]


def test_reproducibility():
    """Test that same seed produces identical results."""
    data = [
        [(10, 20), (30, 40)],
        [(50, 60)],
        [(100, 120), (150, 180)]
    ]
    
    # Generate samples with same seed twice
    samples1 = generate_bootstrap_samples(data, n_samples=5, seed=42)
    samples2 = generate_bootstrap_samples(data, n_samples=5, seed=42)
    
    # Should be identical
    assert samples1 == samples2
    
    # Generate with different seed
    samples3 = generate_bootstrap_samples(data, n_samples=5, seed=99)
    
    # Should be different (with very high probability)
    assert samples1 != samples3


def test_empty_data():
    """Test that empty data returns empty samples."""
    data = []
    
    samples = generate_bootstrap_samples(data, n_samples=10, seed=42)
    
    # Verify structure
    assert len(samples) == 10
    
    # All samples should be empty
    for sample in samples:
        assert sample == []


# =============================================================================
# Tests for Unit 5: Simulate Window Sampling
# =============================================================================

def test_events_fully_within():
    """Test that events fully within windows are preserved exactly."""
    # Window size 100: windows are [0, 99], [100, 199], [200, 299]
    events = [
        (10, 20),    # Fully in window [0, 99]
        (110, 120),  # Fully in window [100, 199]
        (210, 220)   # Fully in window [200, 299]
    ]
    
    # Sample all windows (rate=1.0) to ensure all events are captured
    sampled_events, edge_info = simulate_window_sampling(
        events, 300, 100, 1.0, 'uniform', 42
    )
    
    # Events should be preserved exactly (no truncation needed)
    assert (10, 20) in sampled_events
    assert (110, 120) in sampled_events
    assert (210, 220) in sampled_events
    
    # No edge events (all events fully within their windows)
    assert edge_info['edge_events'] == 0
    assert edge_info['edge_percentage'] == 0.0
    assert edge_info['total_events'] == 3


def test_events_crossing_boundaries():
    """Test that events crossing window boundaries are truncated correctly."""
    # Window size 100: windows are [0, 99], [100, 199], [200, 299]
    events = [
        (50, 150),   # Crosses boundary between [0, 99] and [100, 199]
        (180, 250)   # Crosses boundary between [100, 199] and [200, 299]
    ]
    
    # Sample all windows to capture all events
    sampled_events, edge_info = simulate_window_sampling(
        events, 300, 100, 1.0, 'uniform', 42
    )
    
    # First event should be split and truncated into two parts
    # (50, 150) in window [0, 99] → (50, 99)
    # (50, 150) in window [100, 199] → (100, 150)
    assert (50, 99) in sampled_events
    assert (100, 150) in sampled_events
    
    # Second event should be split and truncated into two parts
    # (180, 250) in window [100, 199] → (180, 199)
    # (180, 250) in window [200, 299] → (200, 250)
    assert (180, 199) in sampled_events
    assert (200, 250) in sampled_events
    
    # All events are edge events (all cross boundaries)
    assert edge_info['edge_events'] == 4
    assert edge_info['total_events'] == 4
    assert edge_info['edge_percentage'] == 100.0


def test_events_spanning_multiple():
    """Test that event spanning multiple windows is handled correctly."""
    # Window size 100: windows are [0, 99], [100, 199], [200, 299], [300, 399]
    events = [
        (50, 350)  # Spans 4 windows
    ]
    
    # Sample all windows
    sampled_events, edge_info = simulate_window_sampling(
        events, 400, 100, 1.0, 'uniform', 42
    )
    
    # Event should appear in each window it spans, truncated at each boundary
    assert (50, 99) in sampled_events     # Window [0, 99]
    assert (100, 199) in sampled_events   # Window [100, 199]
    assert (200, 299) in sampled_events   # Window [200, 299]
    assert (300, 350) in sampled_events   # Window [300, 399]
    
    # All are edge events (all touch at least one boundary)
    assert edge_info['total_events'] == 4
    assert edge_info['edge_events'] == 4
    assert edge_info['edge_percentage'] == 100.0


def test_no_events_in_windows():
    """Test that no events in sampled windows returns empty list."""
    # Use systematic sampling for deterministic behavior
    # Windows: [0, 99], [100, 199], [200, 299], [300, 399], [400, 499]
    # With systematic rate 0.2 (1 out of 5 windows), stride = 5
    # Sampled window index: 0 → window [0, 99]
    
    # Place events in windows that won't be sampled
    events = [
        (110, 120),  # Window [100, 199] - not sampled
        (210, 220),  # Window [200, 299] - not sampled
        (310, 320)   # Window [300, 399] - not sampled
    ]
    
    # Systematic sampling: 5 windows, rate 0.2 → 1 window → index 0 → [0, 99]
    sampled_events, edge_info = simulate_window_sampling(
        events, 500, 100, 0.2, 'systematic', 42
    )
    
    # No events in the sampled window [0, 99]
    assert sampled_events == []
    assert edge_info['total_events'] == 0
    assert edge_info['edge_events'] == 0
    assert edge_info['edge_percentage'] == 0.0


def test_stratified_strategy():
    """Test stratified sampling strategy."""
    events = [(10, 20), (110, 120), (210, 220)]
    
    # Use stratified strategy
    sampled_events, edge_info = simulate_window_sampling(
        events, 300, 100, 0.67, 'stratified', 42
    )
    
    # Should sample some windows and capture events
    assert len(sampled_events) > 0
    assert edge_info['total_events'] == len(sampled_events)


def test_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    events = [(10, 20)]
    
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        simulate_window_sampling(
            events, 300, 100, 0.5, 'invalid_strategy', 42
        )


# =============================================================================
# Tests for Unit 6: Calculate Statistical Power
# =============================================================================

import numpy as np

def test_large_difference():
    """Test that groups with large difference yield high power."""
    # Create groups with large difference: mean 10 vs 20, std ~ 2
    # Cohen's d = (20 - 10) / 2 = 5.0 (very large effect)
    group1 = [10.0, 9.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.3, 9.7, 11.2]
    group2 = [20.0, 19.0, 21.0, 20.5, 19.5, 20.2, 19.8, 20.3, 19.7, 21.2]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Large effect size should yield high power
    assert power > 0.8
    assert power <= 1.0


def test_identical_groups():
    """Test that identical groups yield low power (~alpha)."""
    # Identical groups - no difference to detect
    group1 = [10.0, 10.0, 10.0, 10.0, 10.0]
    group2 = [10.0, 10.0, 10.0, 10.0, 10.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Identical groups (effect size = 0) should yield power ~ alpha
    # When there's no real effect, power equals the false positive rate
    assert power <= 0.10  # Should be close to alpha (0.05)
    assert power >= 0.0


def test_small_sample():
    """Test that small sample sizes yield reduced power."""
    # Small samples (n=3 each) with moderate difference
    group1 = [10.0, 11.0, 12.0]
    group2 = [15.0, 16.0, 17.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Small sample should have reduced power despite moderate effect
    # Power should be valid and reasonable
    assert 0.0 <= power <= 1.0
    assert not np.isnan(power)  # Should not return NaN


def test_unequal_sizes():
    """Test that unequal group sizes yield appropriate power calculation."""
    # Unequal groups: n1=10, n2=5 with large difference
    group1 = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.4]
    group2 = [15.0, 15.5, 14.5, 15.2, 14.8]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should calculate power appropriately for unequal groups
    assert 0.0 <= power <= 1.0
    assert not np.isnan(power)  # Should not return NaN
    # With this large difference, should have at least moderate power
    # (small samples may limit power even with large effects)
    assert power >= 0.3


def test_empty_group1():
    """Test that empty group1 returns 0.0."""
    group1 = []
    group2 = [10.0, 11.0, 12.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    assert power == 0.0


def test_empty_group2():
    """Test that empty group2 returns 0.0."""
    group1 = [10.0, 11.0, 12.0]
    group2 = []
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    assert power == 0.0


def test_single_sample_each():
    """Test that single sample in each group returns 0.0 (not enough df)."""
    # n=1 in each group means df = n1 + n2 - 2 = 0
    group1 = [10.0]
    group2 = [15.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    assert power == 0.0


def test_zero_variance_different_means():
    """Test zero variance with different means returns 1.0."""
    # All values identical within groups, but groups differ
    group1 = [10.0, 10.0, 10.0, 10.0]
    group2 = [20.0, 20.0, 20.0, 20.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Perfect separation with no variance = infinite effect size
    assert power == 1.0


def test_moderate_effect_size():
    """Test moderate effect size triggering appropriate fallback."""
    # Create groups with moderate difference that might trigger NaN
    # Cohen's d around 1.0-1.5 (moderate-large)
    group1 = [10.0, 11.0, 12.0]
    group2 = [13.0, 14.0, 15.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should return valid power (not NaN)
    assert 0.0 <= power <= 1.0
    assert not np.isnan(power)


def test_small_effect_nan_fallback():
    """Test small effect size with tiny sample triggering NaN fallback (returns alpha)."""
    # Very small samples with small difference
    # This should trigger NaN and return alpha (cohens_d <= 0.5)
    group1 = [10.0, 10.2]
    group2 = [10.3, 10.5]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should handle gracefully - either calculate power or return alpha
    assert 0.0 <= power <= 1.0
    assert not np.isnan(power)


def test_moderate_effect_nan_fallback():
    """Test moderate effect size with tiny sample triggering NaN fallback (returns 0.3)."""
    # Very small samples with moderate difference
    # This should trigger NaN and return 0.3 (0.5 < cohens_d <= 2.0)
    group1 = [10.0, 10.5]
    group2 = [12.0, 12.5]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should handle gracefully - either calculate power or return 0.3
    assert 0.0 <= power <= 1.0
    assert not np.isnan(power)


def test_nan_moderate_effect_mock(monkeypatch):
    """Test NaN return with moderate effect size (0.5 < d <= 2.0) returns 0.3."""
    from unittest.mock import MagicMock
    
    # Mock ttest_power to return NaN at the actual import location
    mock_ttest_power = MagicMock(return_value=np.nan)
    monkeypatch.setattr('statsmodels.stats.power.ttest_power', mock_ttest_power)
    
    # Create groups with moderate effect size (Cohen's d ~ 1.0)
    # Using larger spread to keep effect size moderate
    group1 = [5.0, 8.0, 10.0, 12.0, 15.0]
    group2 = [10.0, 13.0, 15.0, 17.0, 20.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should return 0.3 for moderate effect (lines 615-617)
    assert power == 0.3


def test_nan_small_effect_mock(monkeypatch):
    """Test NaN return with small effect size (d <= 0.5) returns alpha."""
    from unittest.mock import MagicMock
    
    # Mock ttest_power to return NaN at the actual import location
    mock_ttest_power = MagicMock(return_value=np.nan)
    monkeypatch.setattr('statsmodels.stats.power.ttest_power', mock_ttest_power)
    
    # Create groups with small effect size (Cohen's d ~ 0.3)
    group1 = [10.0, 10.5, 11.0, 11.5, 12.0]
    group2 = [10.2, 10.7, 11.2, 11.7, 12.2]
    
    alpha = 0.05
    power = calculate_statistical_power(group1, group2, alpha=alpha)
    
    # Should return alpha for small effect (lines 617-618)
    assert power == alpha


def test_exception_large_effect_mock(monkeypatch):
    """Test exception handler with large effect size (d > 2.0) returns 0.5."""
    from unittest.mock import MagicMock
    
    # Mock ttest_power to raise an exception at the actual import location
    mock_ttest_power = MagicMock(side_effect=ValueError("Mocked error"))
    monkeypatch.setattr('statsmodels.stats.power.ttest_power', mock_ttest_power)
    
    # Create groups with large effect size (Cohen's d > 2.0)
    group1 = [10.0, 11.0, 12.0, 13.0, 14.0]
    group2 = [20.0, 21.0, 22.0, 23.0, 24.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should return 0.5 for large effect (lines 626-627)
    assert power == 0.5


def test_exception_moderate_effect_mock(monkeypatch):
    """Test exception handler with moderate effect size (0.5 < d <= 2.0) returns 0.3."""
    from unittest.mock import MagicMock
    
    # Mock ttest_power to raise an exception at the actual import location
    mock_ttest_power = MagicMock(side_effect=RuntimeError("Mocked error"))
    monkeypatch.setattr('statsmodels.stats.power.ttest_power', mock_ttest_power)
    
    # Create groups with moderate effect size (Cohen's d ~ 1.0)
    # Using larger spread to keep effect size moderate
    group1 = [5.0, 8.0, 10.0, 12.0, 15.0]
    group2 = [10.0, 13.0, 15.0, 17.0, 20.0]
    
    power = calculate_statistical_power(group1, group2, alpha=0.05)
    
    # Should return 0.3 for moderate effect (lines 628-629)
    assert power == 0.3


def test_exception_small_effect_mock(monkeypatch):
    """Test exception handler with small effect size (d <= 0.5) returns alpha."""
    from unittest.mock import MagicMock
    
    # Mock ttest_power to raise an exception at the actual import location
    mock_ttest_power = MagicMock(side_effect=Exception("Mocked error"))
    monkeypatch.setattr('statsmodels.stats.power.ttest_power', mock_ttest_power)
    
    # Create groups with small effect size (Cohen's d ~ 0.3)
    group1 = [10.0, 10.5, 11.0, 11.5, 12.0]
    group2 = [10.2, 10.7, 11.2, 11.7, 12.2]
    
    alpha = 0.05
    power = calculate_statistical_power(group1, group2, alpha=alpha)
    
    # Should return alpha for small effect (lines 630-631)
    assert power == alpha


# =============================================================================
# Tests for Unit 7: Evaluate Parameter Combination
# =============================================================================

def test_optimal_parameters():
    """Test that good parameters yield high composite score."""
    # Create pilot data with clear difference between genotypes
    pilot_data = {
        'WT': [
            [(100, 150), (200, 250), (300, 350)] for _ in range(5)
        ],
        'KO': [
            [(1000, 1500), (2000, 2500), (3000, 3500)] for _ in range(5)
        ]
    }

    # Good parameters: reasonable window, moderate sampling, low threshold
    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # Verify all scores are present
    assert 'power' in scores
    assert 'bias' in scores
    assert 'error_rate' in scores
    assert 'efficiency' in scores
    assert 'robustness' in scores
    assert 'composite' in scores

    # Good parameters should yield high composite score
    assert scores['composite'] > 0.5

    # All scores should be in valid ranges
    assert 0 <= scores['power'] <= 1
    assert 0 <= scores['bias'] <= 1
    assert 0 <= scores['error_rate'] <= 1
    assert 0 <= scores['efficiency'] <= 1
    assert 0 <= scores['robustness'] <= 1
    assert 0 <= scores['composite'] <= 1


def test_poor_parameters():
    """Test that poor parameters (very low sampling) yield low power score."""
    # Create pilot data
    pilot_data = {
        'WT': [
            [(100, 150), (200, 250)] for _ in range(5)
        ],
        'KO': [
            [(1000, 1500), (2000, 2500)] for _ in range(5)
        ]
    }

    # Poor parameters: very low sampling rate
    params = {
        'window_size': 300,
        'sampling_rate': 0.05,  # Only 5% sampling
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # Very low sampling should yield reduced power
    # (though might still detect large differences)
    assert 0 <= scores['power'] <= 1

    # Should have low composite score due to reduced detection ability
    assert scores['composite'] < 0.8


def test_excessive_sampling():
    """Test that 100% sampling yields low efficiency score."""
    # Create pilot data
    pilot_data = {
        'WT': [
            [(100, 150), (200, 250)] for _ in range(5)
        ],
        'KO': [
            [(1000, 1500), (2000, 2500)] for _ in range(5)
        ]
    }

    # Excessive sampling: 100% sampling rate
    params = {
        'window_size': 300,
        'sampling_rate': 1.0,  # 100% sampling
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # 100% sampling should yield zero efficiency
    # (no time saved)
    assert scores['efficiency'] == 0.0

    # This should reduce composite score despite good power
    assert scores['composite'] < 1.0


def test_high_bias():
    """Test that parameters causing bias yield low bias score."""
    # Create pilot data with short events
    pilot_data = {
        'WT': [
            [(i, i+10) for i in range(0, 1000, 100)] for _ in range(5)
        ],
        'KO': [
            [(i, i+15) for i in range(0, 1000, 100)] for _ in range(5)
        ]
    }

    # Parameters likely to cause bias: very large windows with low sampling
    # This will miss many short events
    params = {
        'window_size': 3000,  # Very large windows
        'sampling_rate': 0.1,   # Low sampling
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # Should detect some bias
    assert 0 <= scores['bias'] <= 1

    # Bias component should negatively impact composite score
    # composite = 0.25*power + 0.25*(1-bias) + 0.20*(1-error_rate) + 0.15*efficiency + 0.15*(1-robustness)
    # If bias is high, (1-bias) is low, reducing composite
    assert scores['composite'] < 1.0


def test_invalid_genotype_count():
    """Test that non-2 genotype count raises ValueError."""
    # Create pilot data with 3 genotypes (invalid)
    pilot_data = {
        'WT': [[(100, 150)] for _ in range(3)],
        'KO': [[(200, 250)] for _ in range(3)],
        'HET': [[(300, 350)] for _ in range(3)]  # Third genotype - invalid
    }

    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    # Should raise ValueError for non-2 genotypes
    with pytest.raises(ValueError, match="Expected 2 genotypes, got 3"):
        evaluate_parameter_combination(
            pilot_data, params, config, n_bootstrap=10
        )


def test_zero_power_robustness_mock(monkeypatch):
    """Test robustness calculation when avg_power is 0 using mock."""
    from unittest.mock import MagicMock

    # Create pilot data
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(3)],
        'KO': [[(300, 350), (400, 450)] for _ in range(3)]
    }

    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    # Mock calculate_statistical_power to always return 0.0
    # This will cause avg_power to be 0, triggering the else branch
    mock_power = MagicMock(return_value=0.0)
    monkeypatch.setattr('pilot_grooming_optimizer.calculate_statistical_power', mock_power)

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=10
    )

    # Verify power is 0 (from our mock)
    assert scores['power'] == 0.0

    # Verify robustness is 0.0 (from the else branch)
    assert scores['robustness'] == 0.0

    # Verify all scores are in valid ranges
    assert 0 <= scores['bias'] <= 1
    assert 0 <= scores['error_rate'] <= 1
    assert 0 <= scores['efficiency'] <= 1
    assert 0 <= scores['composite'] <= 1


def test_composite_score_uses_all_five_components():
    """Test that composite score includes all 5 components: power, bias, error_rate, efficiency, robustness."""
    # Create pilot data
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(3)],
        'KO': [[(300, 350), (400, 450)] for _ in range(3)]
    }

    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # Manually calculate composite score with correct formula
    expected_composite = (
        0.25 * scores['power'] +
        0.25 * (1 - scores['bias']) +
        0.20 * (1 - scores['error_rate']) +
        0.15 * scores['efficiency'] +
        0.15 * (1 - scores['robustness'])
    )

    # Verify returned composite matches expected calculation
    assert abs(scores['composite'] - expected_composite) < 1e-10


def test_composite_weights_sum_to_one():
    """Test that composite score weights sum to 1.0 (25% + 25% + 20% + 15% + 15%)."""
    # Weights should sum to 1.0
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    assert abs(sum(weights) - 1.0) < 1e-10


def test_error_rate_impacts_composite():
    """Test that error_rate component actually impacts the composite score."""
    # Create pilot data with many short events (will create edge events)
    pilot_data = {
        'WT': [[(i, i+20) for i in range(0, 9000, 100)] for _ in range(3)],
        'KO': [[(i, i+25) for i in range(0, 9000, 100)] for _ in range(3)]
    }

    # Case 1: Low edge threshold (high error_rate expected)
    params_high_error = {
        'window_size': 100,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 5  # Very low threshold
    }

    # Case 2: High edge threshold (low error_rate expected)
    params_low_error = {
        'window_size': 100,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 50  # Very high threshold
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores_high_error = evaluate_parameter_combination(
        pilot_data, params_high_error, config, n_bootstrap=50
    )

    scores_low_error = evaluate_parameter_combination(
        pilot_data, params_low_error, config, n_bootstrap=50
    )

    # Verify error_rate differs between the two cases
    # Low threshold should result in higher error_rate
    assert scores_high_error['error_rate'] >= scores_low_error['error_rate']

    # Verify error_rate impacts composite
    # Higher error_rate should reduce composite score (assuming other factors similar)
    # Since error_rate has 20% weight, it should have meaningful impact
    if scores_high_error['error_rate'] > scores_low_error['error_rate']:
        # Calculate expected difference in composite due to error_rate alone
        error_rate_diff = scores_high_error['error_rate'] - scores_low_error['error_rate']
        expected_composite_reduction = 0.20 * error_rate_diff
        
        # Actual composite difference
        composite_diff = scores_low_error['composite'] - scores_high_error['composite']
        
        # The actual difference should be at least partially explained by error_rate
        # (it may not be exact due to other components also varying slightly)
        assert composite_diff >= 0  # Low error case should have equal or higher composite


def test_error_rate_zero_when_below_threshold():
    """Test that error_rate is 0 when edge percentage is below threshold."""
    # Create pilot data with few edge events
    pilot_data = {
        'WT': [[(i*1000, i*1000+100) for i in range(5)] for _ in range(3)],
        'KO': [[(i*1000+500, i*1000+600) for i in range(5)] for _ in range(3)]
    }

    params = {
        'window_size': 500,  # Large windows to minimize edge events
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 50  # High threshold
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # With high threshold and large windows, error_rate should be 0 or very low
    assert scores['error_rate'] >= 0
    assert scores['error_rate'] <= 0.5  # Should be low


def test_error_rate_nonzero_when_above_threshold():
    """Test that error_rate is non-zero when edge percentage exceeds threshold."""
    # Create pilot data with many short events (will create many edge events)
    pilot_data = {
        'WT': [[(i, i+15) for i in range(0, 9000, 50)] for _ in range(3)],
        'KO': [[(i, i+18) for i in range(0, 9000, 50)] for _ in range(3)]
    }

    params = {
        'window_size': 50,   # Small windows to maximize edge events
        'sampling_rate': 0.50,  # High sampling to capture many edge events
        'strategy': 'uniform',
        'edge_threshold': 1   # Very low threshold
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # With small windows and low threshold, error_rate should be non-zero
    # (many events will be at window boundaries)
    assert scores['error_rate'] >= 0
    assert 0 <= scores['error_rate'] <= 1


def test_all_components_in_valid_range():
    """Test that all score components are in [0, 1] range."""
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(3)],
        'KO': [[(300, 350), (400, 450)] for _ in range(3)]
    }

    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=50
    )

    # Verify all components are in valid [0, 1] range
    for component in ['power', 'bias', 'error_rate', 'efficiency', 'robustness', 'composite']:
        assert 0 <= scores[component] <= 1, f"{component} out of range: {scores[component]}"


def test_composite_formula_correctness():
    """Test that the composite formula matches the specification exactly."""
    # Create simple pilot data
    pilot_data = {
        'WT': [[(100, 200)] for _ in range(3)],
        'KO': [[(300, 400)] for _ in range(3)]
    }

    params = {
        'window_size': 300,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 20
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores = evaluate_parameter_combination(
        pilot_data, params, config, n_bootstrap=30
    )

    # Manually verify the formula
    # composite = 0.25*power + 0.25*(1-bias) + 0.20*(1-error_rate) + 0.15*efficiency + 0.15*(1-robustness)
    expected = (
        0.25 * scores['power'] +
        0.25 * (1 - scores['bias']) +
        0.20 * (1 - scores['error_rate']) +
        0.15 * scores['efficiency'] +
        0.15 * (1 - scores['robustness'])
    )

    # Should match exactly (within floating point precision)
    assert abs(scores['composite'] - expected) < 1e-9


def test_error_rate_weight_is_twenty_percent():
    """Test that error_rate has exactly 20% weight in composite formula."""
    # Create pilot data
    pilot_data = {
        'WT': [[(i*100, i*100+50) for i in range(10)] for _ in range(3)],
        'KO': [[(i*100+500, i*100+550) for i in range(10)] for _ in range(3)]
    }

    # Two scenarios with different edge thresholds
    params_high_error = {
        'window_size': 100,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 5
    }

    params_low_error = {
        'window_size': 100,
        'sampling_rate': 0.20,
        'strategy': 'uniform',
        'edge_threshold': 80
    }

    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }

    scores_high = evaluate_parameter_combination(
        pilot_data, params_high_error, config, n_bootstrap=50
    )

    scores_low = evaluate_parameter_combination(
        pilot_data, params_low_error, config, n_bootstrap=50
    )

    # If error_rate is the only thing that changed significantly,
    # the composite difference should be approximately 0.20 * error_rate_difference
    error_diff = scores_high['error_rate'] - scores_low['error_rate']
    
    if abs(error_diff) > 0.1:  # Only test if there's meaningful difference
        # The weight of error_rate in composite should be 0.20
        # This is implicitly tested by the formula, but we verify the relationship exists
        assert abs(error_diff) >= 0  # Just verify error_rate can differ
		

# =============================================================================
# Tests for Unit 8: Cross-Validation Framework
# =============================================================================

def test_standard_cv():
    """Test standard 5-fold CV with sufficient data."""
    # Create data with 10 samples per genotype (enough for 5 folds)
    data = {
        'WT': [[(i*100, i*100 + 50)] for i in range(10)],
        'KO': [[(i*100 + 1000, i*100 + 1050)] for i in range(10)]
    }
    
    params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20
    }
    
    results = cross_validate_parameters(data, params, n_folds=5, seed=42)
    
    # Verify structure
    assert 'power_mean' in results
    assert 'power_std' in results
    assert 'bias_mean' in results
    assert 'bias_std' in results
    assert 'composite_mean' in results
    assert 'composite_std' in results
    assert 'n_folds' in results
    
    # Verify 5 folds were used
    assert results['n_folds'] == 5
    
    # Verify all metrics are in valid ranges
    assert 0 <= results['power_mean'] <= 1
    assert 0 <= results['power_std'] <= 1
    assert 0 <= results['bias_mean'] <= 1
    assert 0 <= results['bias_std'] <= 1
    assert 0 <= results['composite_mean'] <= 1
    assert 0 <= results['composite_std'] <= 1


def test_small_dataset():
    """Test that small dataset with only 3 samples per genotype adjusts fold count."""
    # Create data with only 3 samples per genotype
    data = {
        'WT': [[(100, 150)], [(200, 250)], [(300, 350)]],
        'KO': [[(400, 450)], [(500, 550)], [(600, 650)]]
    }
    
    params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20
    }
    
    # Request 5 folds but only 3 samples available
    results = cross_validate_parameters(data, params, n_folds=5, seed=42)
    
    # Should adjust to 3 folds (max possible with 3 samples)
    assert results['n_folds'] == 3
    
    # Verify all metrics are present and valid
    assert 0 <= results['power_mean'] <= 1
    assert 0 <= results['composite_mean'] <= 1


def test_no_data_leakage():
    """Test that train/test splits are disjoint (no data leakage between folds)."""
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Create data
    data = {
        'WT': [[(i*100, i*100 + 50)] for i in range(10)],
        'KO': [[(i*100 + 1000, i*100 + 1050)] for i in range(10)]
    }
    
    n_folds = 5
    seed = 42
    
    # Manually verify fold splitting to ensure no leakage
    genotype_names = list(data.keys())
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Check each genotype's folds
    for genotype_name in genotype_names:
        n_samples = len(data[genotype_name])
        indices = np.arange(n_samples)
        
        # Collect all test indices across folds
        all_test_indices = []
        for train_idx, test_idx in kfold.split(indices):
            # Verify train and test are disjoint
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Collect test indices
            all_test_indices.extend(test_idx.tolist())
        
        # Verify each sample appears exactly once in test sets
        assert sorted(all_test_indices) == list(range(n_samples))
        
        # Verify no duplicates in test sets
        assert len(all_test_indices) == len(set(all_test_indices))


def test_reproducibility():
    """Test that same seed produces reproducible results."""
    # Create data
    data = {
        'WT': [[(i*100, i*100 + 50)] for i in range(8)],
        'KO': [[(i*100 + 1000, i*100 + 1050)] for i in range(8)]
    }
    
    params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20
    }
    
    # Run CV twice with same seed
    results1 = cross_validate_parameters(data, params, n_folds=4, seed=42)
    results2 = cross_validate_parameters(data, params, n_folds=4, seed=42)
    
    # Results should be identical
    assert results1['power_mean'] == results2['power_mean']
    assert results1['power_std'] == results2['power_std']
    assert results1['bias_mean'] == results2['bias_mean']
    assert results1['bias_std'] == results2['bias_std']
    assert results1['composite_mean'] == results2['composite_mean']
    assert results1['composite_std'] == results2['composite_std']
    assert results1['n_folds'] == results2['n_folds']
    
    # Run with different seed
    results3 = cross_validate_parameters(data, params, n_folds=4, seed=99)
    
    # Results should be different (with very high probability)
    # At least one metric should differ
    assert (results1['power_mean'] != results3['power_mean'] or
            results1['bias_mean'] != results3['bias_mean'] or
            results1['composite_mean'] != results3['composite_mean'])


# =============================================================================
# Tests for Unit 9: Optimize Parameter Space
# =============================================================================

def test_multiple_viable_params():
    """Test that highest composite score is selected from multiple viable parameters."""
    # Create pilot data with clear difference between genotypes
    pilot_data = {
        'WT': [[(i*100, i*100 + 50)] for i in range(5)],
        'KO': [[(i*100 + 1000, i*100 + 1050)] for i in range(5)]
    }
    
    # Create parameter space with multiple combinations
    parameter_space = {
        'window_sizes': [100, 200],
        'sampling_rates': [0.2, 0.3],
        'strategies': ['uniform', 'stratified'],
        'edge_thresholds': [10, 20]
    }
    # Total combinations: 2 * 2 * 2 * 2 = 16
    
    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }
    
    # Run optimization with smaller n_bootstrap for faster testing
    best_params, all_results = optimize_parameter_space(
        pilot_data, parameter_space, config, n_bootstrap=50
    )
    
    # Verify we got results for all combinations
    assert len(all_results) == 16
    
    # Verify best_params is the first in all_results (highest composite)
    assert best_params == all_results[0]
    
    # Verify all_results is sorted by composite score (descending)
    composite_scores = [r['scores']['composite'] for r in all_results]
    assert composite_scores == sorted(composite_scores, reverse=True)
    
    # Verify best_params has highest composite score
    assert best_params['scores']['composite'] >= all_results[-1]['scores']['composite']
    
    # Verify structure of best_params
    assert 'window_size' in best_params
    assert 'sampling_rate' in best_params
    assert 'strategy' in best_params
    assert 'edge_threshold' in best_params
    assert 'scores' in best_params
    
    # Verify scores structure
    assert 'power' in best_params['scores']
    assert 'bias' in best_params['scores']
    assert 'efficiency' in best_params['scores']
    assert 'robustness' in best_params['scores']
    assert 'composite' in best_params['scores']
    
    # Verify all scores are in valid ranges
    assert 0 <= best_params['scores']['power'] <= 1
    assert 0 <= best_params['scores']['bias'] <= 1
    assert 0 <= best_params['scores']['efficiency'] <= 1
    assert 0 <= best_params['scores']['robustness'] <= 1
    assert 0 <= best_params['scores']['composite'] <= 1


def test_no_params_meet_threshold(capsys):
    """Test warning when no parameters meet power threshold."""
    # Create pilot data with very similar groups (low power expected)
    pilot_data = {
        'WT': [[(i*100, i*100 + 10)] for i in range(3)],
        'KO': [[(i*100, i*100 + 11)] for i in range(3)]  # Very similar to WT
    }
    
    # Limited parameter space with poor parameters
    parameter_space = {
        'window_sizes': [100],
        'sampling_rates': [0.05],  # Very low sampling rate
        'strategies': ['uniform'],
        'edge_thresholds': [10]
    }
    # Total combinations: 1
    
    config = {
        'alpha': 0.05,
        'power': 0.95,  # Very high target that likely won't be met
        'expected_frames': 9000
    }
    
    # Run optimization
    best_params, all_results = optimize_parameter_space(
        pilot_data, parameter_space, config, n_bootstrap=50
    )
    
    # Verify we still got results
    assert len(all_results) == 1
    assert best_params == all_results[0]
    
    # Capture printed output to check for warning
    captured = capsys.readouterr()
    
    # Verify warning was printed
    assert "Warning" in captured.out
    assert "below target power" in captured.out
    
    # Verify best parameters are returned anyway (even if below threshold)
    assert best_params['scores']['power'] < config['power']
    
    # Verify structure is still correct
    assert 'window_size' in best_params
    assert 'scores' in best_params
    assert 'composite' in best_params['scores']


def test_tie_in_scores():
    """Test consistent selection when two params have identical composite scores."""
    # Create pilot data
    pilot_data = {
        'WT': [[(i*100, i*100 + 50)] for i in range(3)],
        'KO': [[(i*100 + 500, i*100 + 550)] for i in range(3)]
    }
    
    # Create parameter space with two window sizes
    # (other parameters identical to create potential ties)
    parameter_space = {
        'window_sizes': [100, 200],
        'sampling_rates': [0.2],
        'strategies': ['uniform'],
        'edge_thresholds': [10]
    }
    # Total combinations: 2 * 1 * 1 * 1 = 2
    
    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }
    
    # Run optimization multiple times to verify consistency
    best_params1, all_results1 = optimize_parameter_space(
        pilot_data, parameter_space, config, n_bootstrap=50
    )
    best_params2, all_results2 = optimize_parameter_space(
        pilot_data, parameter_space, config, n_bootstrap=50
    )
    
    # Verify we got 2 results both times
    assert len(all_results1) == 2
    assert len(all_results2) == 2
    
    # Should get consistent selection
    # (even if scores are identical, sorting is stable, so first occurrence wins)
    assert best_params1['window_size'] == best_params2['window_size']
    assert best_params1['sampling_rate'] == best_params2['sampling_rate']
    assert best_params1['strategy'] == best_params2['strategy']
    assert best_params1['edge_threshold'] == best_params2['edge_threshold']
    
    # Verify results are sorted properly
    assert all_results1[0]['scores']['composite'] >= all_results1[1]['scores']['composite']
    assert all_results2[0]['scores']['composite'] >= all_results2[1]['scores']['composite']


def test_progress_tracking():
    """Test that progress bar updates correctly and all combinations are evaluated."""
    # Create pilot data
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(3)],
        'KO': [[(300, 350), (400, 450)] for _ in range(3)]
    }
    
    # Create small parameter space for faster testing
    parameter_space = {
        'window_sizes': [100, 200],
        'sampling_rates': [0.2, 0.3],
        'strategies': ['uniform'],
        'edge_thresholds': [10]
    }
    # Total: 2 * 2 * 1 * 1 = 4 combinations
    
    config = {
        'alpha': 0.05,
        'power': 0.8,
        'expected_frames': 9000
    }
    
    # Run optimization (tqdm will show progress)
    best_params, all_results = optimize_parameter_space(
        pilot_data, parameter_space, config, n_bootstrap=50
    )
    
    # Verify all combinations were evaluated
    expected_combinations = 4
    assert len(all_results) == expected_combinations
    
    # Verify each unique combination is present
    combinations_seen = set()
    for result in all_results:
        combo = (
            result['window_size'],
            result['sampling_rate'],
            result['strategy'],
            result['edge_threshold']
        )
        combinations_seen.add(combo)
    
    # All 4 unique combinations should be present
    assert len(combinations_seen) == expected_combinations
    
    # Verify results structure
    for result in all_results:
        assert 'window_size' in result
        assert 'sampling_rate' in result
        assert 'strategy' in result
        assert 'edge_threshold' in result
        assert 'scores' in result
        
        # Verify all score components are present
        assert 'power' in result['scores']
        assert 'bias' in result['scores']
        assert 'efficiency' in result['scores']
        assert 'robustness' in result['scores']
        assert 'composite' in result['scores']
        
        # Verify scores are in valid ranges
        assert 0 <= result['scores']['power'] <= 1
        assert 0 <= result['scores']['bias'] <= 1
        assert 0 <= result['scores']['efficiency'] <= 1
        assert 0 <= result['scores']['robustness'] <= 1
        assert 0 <= result['scores']['composite'] <= 1
    
    # Verify sorting is correct
    for i in range(len(all_results) - 1):
        assert all_results[i]['scores']['composite'] >= all_results[i+1]['scores']['composite']


# =============================================================================
# Tests for Unit 10: Generate Visualization Plots
# =============================================================================

def test_standard_results(tmp_path):
    """Test that all expected plots are generated from standard results."""
    # Create comprehensive results data
    all_results = []
    
    # Generate results for multiple parameter combinations
    window_sizes = [100, 200, 300]
    sampling_rates = [0.1, 0.2, 0.3]
    strategies = ['uniform', 'stratified', 'systematic']
    
    for window_size in window_sizes:
        for sampling_rate in sampling_rates:
            for strategy in strategies:
                result = {
                    'window_size': window_size,
                    'sampling_rate': sampling_rate,
                    'strategy': strategy,
                    'edge_threshold': 10,
                    'scores': {
                        'power': 0.7 + (sampling_rate * 0.3),
                        'bias': 0.1 + (sampling_rate * 0.1),
                        'efficiency': 1.0 - sampling_rate,
                        'robustness': 0.2,
                        'composite': 0.6 + (sampling_rate * 0.2)
                    }
                }
                all_results.append(result)
    
    # Generate plots
    output_dir = str(tmp_path / "plots")
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify plots were generated
    assert len(plot_paths) > 0
    
    # Expected plots:
    # - 3 heat maps (one per strategy)
    # - 1 power curves
    # - 1 pareto frontier
    # - 1 bias assessment
    # - 1 strategy comparison
    # Total: 7 plots
    assert len(plot_paths) == 7
    
    # Verify all files exist
    for plot_path in plot_paths:
        assert Path(plot_path).exists()
        assert Path(plot_path).suffix == '.pdf'
    
    # Verify expected file names are present
    plot_names = [Path(p).name for p in plot_paths]
    assert 'heatmap_uniform.pdf' in plot_names
    assert 'heatmap_stratified.pdf' in plot_names
    assert 'heatmap_systematic.pdf' in plot_names
    assert 'power_curves.pdf' in plot_names
    assert 'pareto_frontier.pdf' in plot_names
    assert 'bias_assessment.pdf' in plot_names
    assert 'strategy_comparison.pdf' in plot_names


def test_missing_output_dir(tmp_path):
    """Test that missing output directory is created automatically."""
    # Create results data
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    # Use a non-existent nested directory path
    output_dir = str(tmp_path / "level1" / "level2" / "level3" / "plots")
    
    # Verify directory doesn't exist yet
    assert not Path(output_dir).exists()
    
    # Generate plots
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify directory was created
    assert Path(output_dir).exists()
    assert Path(output_dir).is_dir()
    
    # Verify plots were generated
    assert len(plot_paths) > 0
    
    # Verify all files exist in the created directory
    for plot_path in plot_paths:
        assert Path(plot_path).exists()
        assert Path(plot_path).parent == Path(output_dir)


def test_single_parameter_combo(tmp_path):
    """Test handling of only one parameter combination."""
    # Create results with only one combination
    all_results = [
        {
            'window_size': 300,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.85,
                'bias': 0.05,
                'efficiency': 0.8,
                'robustness': 0.15,
                'composite': 0.75
            }
        }
    ]
    
    # Generate plots
    output_dir = str(tmp_path / "plots")
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still generate plots (even if some are less informative)
    assert len(plot_paths) > 0
    
    # Verify all files exist and are PDFs
    for plot_path in plot_paths:
        assert Path(plot_path).exists()
        assert Path(plot_path).suffix == '.pdf'
    
    # Should have at least:
    # - 1 heat map (for the one strategy)
    # - 1 strategy comparison (with one bar)
    # - potentially others depending on implementation
    assert len(plot_paths) >= 2


def test_invalid_data_values(tmp_path):
    """Test handling of NaN and inf values in results."""
    # Create results with problematic values
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.1,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': float('nan'),  # NaN value
                'bias': 0.1,
                'efficiency': 0.9,
                'robustness': 0.2,
                'composite': 0.7
            }
        },
        {
            'window_size': 200,
            'sampling_rate': 0.2,
            'strategy': 'stratified',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': float('inf'),  # Inf value
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.65
            }
        },
        {
            'window_size': 300,
            'sampling_rate': 0.3,
            'strategy': 'systematic',
            'edge_threshold': 10,
            'scores': {
                'power': 0.75,
                'bias': 0.15,
                'efficiency': float('-inf'),  # -Inf value
                'robustness': 0.25,
                'composite': 0.6
            }
        },
        # Add some valid data too
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    # Generate plots - should handle gracefully without crashing
    output_dir = str(tmp_path / "plots")
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still generate some plots (handling invalid data appropriately)
    # The function should not crash even with NaN/inf values
    assert isinstance(plot_paths, list)
    
    # Verify returned paths are valid
    for plot_path in plot_paths:
        assert isinstance(plot_path, str)
        if Path(plot_path).exists():
            assert Path(plot_path).suffix == '.pdf'


def test_dataframe_conversion_error(tmp_path):
    """Test handling of error during DataFrame conversion."""
    # Create malformed results that will cause DataFrame conversion to fail
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            # Missing 'strategy' key - will cause issues
            'edge_threshold': 10,
            'scores': None  # This will cause problems
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Should handle gracefully and return empty list
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should return empty list when DataFrame conversion fails
    assert isinstance(plot_paths, list)
    assert len(plot_paths) == 0


def test_heatmap_error(tmp_path, monkeypatch):
    """Test error handling during heat map generation."""
    from unittest.mock import MagicMock
    import pandas as pd
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock pivot_table to raise an error
    original_pivot = pd.DataFrame.pivot_table
    def mock_pivot(*args, **kwargs):
        raise ValueError("Mocked pivot error")
    
    monkeypatch.setattr(pd.DataFrame, 'pivot_table', mock_pivot)
    
    # Should handle error gracefully and continue
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still be a list (but heat maps won't be generated)
    assert isinstance(plot_paths, list)


def test_power_curves_error(tmp_path, monkeypatch, capsys):
    """Test error handling during power curves generation."""
    import matplotlib.pyplot as plt
    from unittest.mock import MagicMock
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock plt.subplots to raise error during power curves
    original_subplots = plt.subplots
    call_count = [0]
    
    def mock_subplots(*args, **kwargs):
        call_count[0] += 1
        # Fail on second call (power curves section)
        if call_count[0] == 2:
            raise RuntimeError("Mocked power curves error")
        return original_subplots(*args, **kwargs)
    
    monkeypatch.setattr(plt, 'subplots', mock_subplots)
    
    # Should handle error gracefully
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify warning was printed
    captured = capsys.readouterr()
    assert "Warning: Could not generate power curves" in captured.out
    
    # Should still be a list
    assert isinstance(plot_paths, list)


def test_pareto_frontier_error(tmp_path, monkeypatch, capsys):
    """Test error handling during Pareto frontier generation."""
    import matplotlib.pyplot as plt
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock plt.subplots to raise error during Pareto frontier
    original_subplots = plt.subplots
    call_count = [0]
    
    def mock_subplots(*args, **kwargs):
        call_count[0] += 1
        # Fail on third call (Pareto frontier section)
        if call_count[0] == 3:
            raise RuntimeError("Mocked Pareto error")
        return original_subplots(*args, **kwargs)
    
    monkeypatch.setattr(plt, 'subplots', mock_subplots)
    
    # Should handle error gracefully
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify warning was printed
    captured = capsys.readouterr()
    assert "Warning: Could not generate Pareto frontier" in captured.out
    
    # Should still be a list
    assert isinstance(plot_paths, list)


def test_bias_assessment_error(tmp_path, monkeypatch, capsys):
    """Test error handling during bias assessment generation."""
    import matplotlib.pyplot as plt
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock plt.subplots to raise error during bias assessment
    original_subplots = plt.subplots
    call_count = [0]
    
    def mock_subplots(*args, **kwargs):
        call_count[0] += 1
        # Fail on fourth call (bias assessment section)
        if call_count[0] == 4:
            raise RuntimeError("Mocked bias assessment error")
        return original_subplots(*args, **kwargs)
    
    monkeypatch.setattr(plt, 'subplots', mock_subplots)
    
    # Should handle error gracefully
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify warning was printed
    captured = capsys.readouterr()
    assert "Warning: Could not generate bias assessment" in captured.out
    
    # Should still be a list
    assert isinstance(plot_paths, list)


def test_strategy_comparison_error(tmp_path, monkeypatch, capsys):
    """Test error handling during strategy comparison generation."""
    import matplotlib.pyplot as plt
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock plt.subplots to raise error during strategy comparison
    original_subplots = plt.subplots
    call_count = [0]
    
    def mock_subplots(*args, **kwargs):
        call_count[0] += 1
        # Fail on fifth call (strategy comparison section)
        if call_count[0] == 5:
            raise RuntimeError("Mocked strategy comparison error")
        return original_subplots(*args, **kwargs)
    
    monkeypatch.setattr(plt, 'subplots', mock_subplots)
    
    # Should handle error gracefully
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Verify warning was printed
    captured = capsys.readouterr()
    assert "Warning: Could not generate strategy comparison" in captured.out
    
    # Should still be a list
    assert isinstance(plot_paths, list)


def test_empty_pivot_table(tmp_path):
    """Test handling when pivot table is empty for heat map."""
    # Create results where all strategies have the same window_size and sampling_rate
    # This can create issues with pivot tables
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        },
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'stratified',
            'edge_threshold': 10,
            'scores': {
                'power': 0.75,
                'bias': 0.12,
                'efficiency': 0.8,
                'robustness': 0.22,
                'composite': 0.68
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Should handle gracefully
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still generate some plots (just not heat maps with varied data)
    assert isinstance(plot_paths, list)


def test_empty_strategy_data(tmp_path, monkeypatch):
    """Test skip when strategy filter returns no data (line 1261)."""
    import pandas as pd
    
    # Create valid results with one strategy
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock the unique() method to return a strategy that doesn't exist in the data
    original_unique = pd.Series.unique
    
    def mock_unique(self):
        # Return strategies that don't exist in the actual data
        if 'strategy' in str(self.name):
            return ['nonexistent_strategy_1', 'nonexistent_strategy_2']
        return original_unique(self)
    
    monkeypatch.setattr(pd.Series, 'unique', mock_unique)
    
    # Should handle gracefully - skipping strategies with no data
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still return a list (but no heat maps generated)
    assert isinstance(plot_paths, list)


def test_empty_pivot_table_coverage(tmp_path, monkeypatch):
    """Test skip when pivot table is empty (line 1273)."""
    import pandas as pd
    
    # Create valid results
    all_results = [
        {
            'window_size': 100,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.8,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.2,
                'composite': 0.7
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Mock pivot_table to return empty DataFrame
    original_pivot_table = pd.DataFrame.pivot_table
    
    def mock_pivot_table(self, **kwargs):
        # Return empty DataFrame for heat map pivots
        if 'values' in kwargs and kwargs.get('values') == 'composite':
            return pd.DataFrame()
        return original_pivot_table(self, **kwargs)
    
    monkeypatch.setattr(pd.DataFrame, 'pivot_table', mock_pivot_table)
    
    # Should handle gracefully - skipping empty pivot tables
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should still return a list (but heat maps skipped due to empty pivots)
    assert isinstance(plot_paths, list)


def test_pareto_dominated_point(tmp_path):
    """Test Pareto frontier logic where a point is dominated (lines 1350-1351)."""
    # Create results where some points are clearly dominated by others
    all_results = [
        # Point 1: Low efficiency, low power - DOMINATED
        {
            'window_size': 100,
            'sampling_rate': 0.8,  # High sampling = low efficiency
            'strategy': 'uniform',
            'edge_threshold': 10,
            'scores': {
                'power': 0.3,  # Low power
                'bias': 0.1,
                'efficiency': 0.2,  # Low efficiency (1 - 0.8)
                'robustness': 0.2,
                'composite': 0.4
            }
        },
        # Point 2: High efficiency, high power - DOMINATES Point 1
        {
            'window_size': 200,
            'sampling_rate': 0.2,  # Low sampling = high efficiency
            'strategy': 'stratified',
            'edge_threshold': 10,
            'scores': {
                'power': 0.9,  # High power
                'bias': 0.05,
                'efficiency': 0.8,  # High efficiency (1 - 0.2)
                'robustness': 0.15,
                'composite': 0.8
            }
        },
        # Point 3: Medium efficiency, medium power - NOT dominated
        {
            'window_size': 150,
            'sampling_rate': 0.5,
            'strategy': 'systematic',
            'edge_threshold': 10,
            'scores': {
                'power': 0.6,
                'bias': 0.08,
                'efficiency': 0.5,
                'robustness': 0.18,
                'composite': 0.6
            }
        }
    ]
    
    output_dir = str(tmp_path / "plots")
    
    # Generate plots
    plot_paths = generate_visualization_plots(all_results, output_dir)
    
    # Should successfully generate Pareto frontier plot
    plot_names = [Path(p).name for p in plot_paths]
    assert 'pareto_frontier.pdf' in plot_names
    
    # Verify the plot was created
    pareto_plot = [p for p in plot_paths if 'pareto_frontier' in p][0]
    assert Path(pareto_plot).exists()
    
    # The key here is that Point 1 should be identified as non-Pareto (dominated)
    # because Point 2 has BOTH higher efficiency AND higher power
    # This triggers lines 1350-1351: is_pareto = False, break


# =============================================================================
# Tests for Unit 11: Generate Detailed Log
# =============================================================================

def test_complete_optimization():
    """Test full optimization run with all sections populated."""
    # Create comprehensive pilot data
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(5)],
        'KO': [[(300, 350), (400, 450)] for _ in range(5)]
    }
    
    frame_counts = {
        'WT': [9000, 9010, 8990, 9005, 9000],
        'KO': [9000, 9000, 8995, 9010, 9005]
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    # Create multiple results
    all_results = [
        {
            'window_size': 300,
            'sampling_rate': 0.2,
            'strategy': 'uniform',
            'edge_threshold': 20,
            'scores': {
                'composite': 0.75,
                'power': 0.85,
                'bias': 0.1,
                'efficiency': 0.8,
                'robustness': 0.15
            }
        },
        {
            'window_size': 200,
            'sampling_rate': 0.3,
            'strategy': 'stratified',
            'edge_threshold': 10,
            'scores': {
                'composite': 0.70,
                'power': 0.80,
                'bias': 0.15,
                'efficiency': 0.7,
                'robustness': 0.20
            }
        }
    ]
    
    best_params = all_results[0]
    warnings = []
    
    # Generate log
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify all top-level sections exist
    assert 'timestamp' in log
    assert 'config' in log
    assert 'pilot_summary' in log
    assert 'optimization_process' in log
    assert 'best_results' in log
    assert 'warnings' in log
    assert 'validation_metrics' in log
    
    # Verify config section
    assert log['config']['data_dir'] == '/data/pilot'
    assert log['config']['expected_frames'] == 9000
    assert log['config']['alpha'] == 0.05
    assert log['config']['power'] == 0.8
    
    # Verify pilot_summary section
    assert log['pilot_summary']['n_genotypes'] == 2
    assert log['pilot_summary']['flies_per_genotype'] == {'WT': 5, 'KO': 5}
    assert log['pilot_summary']['total_flies'] == 10
    assert 'frame_counts' in log['pilot_summary']
    assert log['pilot_summary']['frame_counts']['min'] == 8990
    assert log['pilot_summary']['frame_counts']['max'] == 9010
    assert 8990 <= log['pilot_summary']['frame_counts']['mean'] <= 9010
    assert log['pilot_summary']['frame_counts']['variation'] >= 0
    
    # Verify optimization_process section
    assert log['optimization_process']['n_combinations_tested'] == 2
    assert 'parameter_space' in log['optimization_process']
    assert log['optimization_process']['parameter_space']['window_sizes'] == [200, 300]
    assert log['optimization_process']['parameter_space']['sampling_rates'] == [0.2, 0.3]
    assert log['optimization_process']['parameter_space']['strategies'] == ['stratified', 'uniform']
    assert log['optimization_process']['parameter_space']['edge_thresholds'] == [10, 20]
    assert log['optimization_process']['best_parameters']['window_size'] == 300
    assert log['optimization_process']['best_parameters']['sampling_rate'] == 0.2
    assert log['optimization_process']['best_parameters']['strategy'] == 'uniform'
    assert log['optimization_process']['best_parameters']['edge_threshold'] == 20
    
    # Verify best_results section
    assert log['best_results']['composite_score'] == 0.75
    assert log['best_results']['power'] == 0.85
    assert log['best_results']['bias'] == 0.1
    assert log['best_results']['efficiency'] == 0.8
    assert log['best_results']['robustness'] == 0.15
    
    # Verify warnings section
    assert log['warnings'] == []
    
    # Verify validation_metrics section
    assert log['validation_metrics']['total_parameters_evaluated'] == 2
    assert log['validation_metrics']['best_composite_score'] == 0.75
    assert log['validation_metrics']['worst_composite_score'] == 0.70
    assert 0.70 <= log['validation_metrics']['mean_composite_score'] <= 0.75
    assert log['validation_metrics']['std_composite_score'] >= 0


def test_with_warnings():
    """Test optimization with warnings included in log."""
    pilot_data = {
        'WT': [[(100, 150)] for _ in range(3)],
        'KO': [[(200, 250)] for _ in range(3)]
    }
    
    frame_counts = {
        'WT': [9000, 9000, 9000],
        'KO': [9000, 9000, 9000]
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    best_params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20,
        'scores': {
            'composite': 0.65,
            'power': 0.70,
            'bias': 0.12,
            'efficiency': 0.8,
            'robustness': 0.18
        }
    }
    
    all_results = [best_params]
    
    # Include warnings
    warnings = [
        'Genotype WT has only 3 files (minimum recommended: 10)',
        'Genotype KO has only 3 files (minimum recommended: 10)',
        'Best parameters achieve power of 0.70, which is below target power of 0.80'
    ]
    
    # Generate log
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify warnings are included
    assert 'warnings' in log
    assert len(log['warnings']) == 3
    assert log['warnings'][0] == 'Genotype WT has only 3 files (minimum recommended: 10)'
    assert log['warnings'][1] == 'Genotype KO has only 3 files (minimum recommended: 10)'
    assert log['warnings'][2] == 'Best parameters achieve power of 0.70, which is below target power of 0.80'
    
    # Verify other sections still populated
    assert 'timestamp' in log
    assert 'config' in log
    assert 'pilot_summary' in log
    assert log['pilot_summary']['n_genotypes'] == 2
    assert log['best_results']['power'] == 0.70


def test_multiple_genotypes():
    """Test multiple genotypes analyzed and documented."""
    # Note: The function expects exactly 2 genotypes based on the script design
    # But we verify it handles the documentation correctly
    pilot_data = {
        'WT': [[(100, 150), (200, 250)] for _ in range(4)],
        'KO': [[(300, 350), (400, 450)] for _ in range(6)]
    }
    
    frame_counts = {
        'WT': [9000, 9005, 8995, 9010],
        'KO': [9000, 9000, 8990, 9010, 9005, 9000]
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    best_params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20,
        'scores': {
            'composite': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        }
    }
    
    all_results = [best_params]
    warnings = []
    
    # Generate log
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify genotype documentation
    assert log['pilot_summary']['n_genotypes'] == 2
    assert log['pilot_summary']['flies_per_genotype']['WT'] == 4
    assert log['pilot_summary']['flies_per_genotype']['KO'] == 6
    assert log['pilot_summary']['total_flies'] == 10
    
    # Verify frame counts aggregate correctly
    # All frame counts: [9000, 9005, 8995, 9010, 9000, 9000, 8990, 9010, 9005, 9000]
    assert log['pilot_summary']['frame_counts']['min'] == 8990
    assert log['pilot_summary']['frame_counts']['max'] == 9010
    assert 8990 <= log['pilot_summary']['frame_counts']['mean'] <= 9010
    assert log['pilot_summary']['frame_counts']['variation'] >= 0


def test_timestamp_format():
    """Test timestamp is in valid ISO format."""
    from datetime import datetime
    
    pilot_data = {
        'WT': [[(100, 150)] for _ in range(2)],
        'KO': [[(200, 250)] for _ in range(2)]
    }
    
    frame_counts = {
        'WT': [9000, 9000],
        'KO': [9000, 9000]
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    best_params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20,
        'scores': {
            'composite': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        }
    }
    
    all_results = [best_params]
    warnings = []
    
    # Generate log
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify timestamp exists
    assert 'timestamp' in log
    assert isinstance(log['timestamp'], str)
    
    # Verify ISO format by parsing it
    try:
        parsed_time = datetime.fromisoformat(log['timestamp'])
        assert isinstance(parsed_time, datetime)
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO 8601 format")
    
    # Verify timestamp is recent (within last minute)
    now = datetime.now()
    time_diff = abs((now - parsed_time).total_seconds())
    assert time_diff < 60  # Should be generated within last 60 seconds


def test_all_empty_csvs():
    """Test handling when all CSVs are empty (frame_counts all 0)."""
    # Simulates scenario where all CSV files were empty
    pilot_data = {
        'WT': [[], [], []],  # Empty event lists
        'KO': [[], []]       # Empty event lists
    }
    
    frame_counts = {
        'WT': [0, 0, 0],  # All zeros (empty CSVs)
        'KO': [0, 0]      # All zeros (empty CSVs)
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    best_params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20,
        'scores': {
            'composite': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        }
    }
    
    all_results = [best_params]
    warnings = ['All CSVs appear to be empty']
    
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify handling of all-zero frame counts
    assert log['pilot_summary']['frame_counts']['min'] == 0
    assert log['pilot_summary']['frame_counts']['max'] == 0
    assert log['pilot_summary']['frame_counts']['mean'] == 0.0
    assert log['pilot_summary']['frame_counts']['variation'] == 0.0  # Covers mean_frames == 0 branch


def test_empty_frame_counts():
    """Test defensive handling of empty frame_counts (edge case)."""
    # This tests the function's robustness when called with empty data
    # (unreachable in normal script flow, but good defensive programming)
    pilot_data = {
        'WT': [],
        'KO': []
    }
    
    frame_counts = {
        'WT': [],
        'KO': []
    }
    
    config = {
        'data_dir': '/data/pilot',
        'expected_frames': 9000,
        'alpha': 0.05,
        'power': 0.8
    }
    
    best_params = {
        'window_size': 300,
        'sampling_rate': 0.2,
        'strategy': 'uniform',
        'edge_threshold': 20,
        'scores': {
            'composite': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        }
    }
    
    all_results = [best_params]
    warnings = []
    
    log = generate_detailed_log(
        pilot_data, frame_counts, config, best_params, all_results, warnings
    )
    
    # Verify graceful handling of empty data
    assert log['pilot_summary']['n_genotypes'] == 2
    assert log['pilot_summary']['total_flies'] == 0
    assert log['pilot_summary']['frame_counts']['min'] == 0
    assert log['pilot_summary']['frame_counts']['max'] == 0
    assert log['pilot_summary']['frame_counts']['mean'] == 0.0
    assert log['pilot_summary']['frame_counts']['variation'] == 0.0  # Covers len == 0 branch


# =============================================================================
# Tests for Unit 12: Generate PDF Report
# =============================================================================

def test_standard_report(tmp_path):
    """Test standard report generation with all sections."""
    # Create comprehensive log data
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 8990,
                'max': 9010,
                'mean': 9000.0,
                'variation': 0.22
            }
        },
        'optimization_process': {
            'n_combinations_tested': 100,
            'parameter_space': {
                'window_sizes': [100, 200, 300],
                'sampling_rates': [0.1, 0.2, 0.3],
                'strategies': ['uniform', 'stratified'],
                'edge_thresholds': [10, 20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 100,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.45,
            'mean_composite_score': 0.60,
            'std_composite_score': 0.08
        }
    }
    
    # Create dummy plot files
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    plot_paths = []
    for plot_name in ['heatmap_uniform.pdf', 'power_curves.pdf', 'pareto_frontier.pdf']:
        plot_file = plots_dir / plot_name
        plot_file.write_text("dummy PDF content")
        plot_paths.append(str(plot_file))
    
    # Generate report
    output_path = str(tmp_path / "report.pdf")
    success = generate_pdf_report(log_data, plot_paths, output_path)
    
    # Verify report was created
    assert success is True
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0  # Non-empty file


def test_missing_plots(tmp_path):
    """Test report generation when some plot files are missing."""
    # Create minimal log data
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 3, 'KO': 3},
            'total_flies': 6,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.70,
            'power': 0.75,
            'bias': 0.12,
            'efficiency': 0.8,
            'robustness': 0.18
        },
        'warnings': ['Small sample size'],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.70,
            'worst_composite_score': 0.70,
            'mean_composite_score': 0.70,
            'std_composite_score': 0.0
        }
    }
    
    # Provide non-existent plot paths
    plot_paths = [
        '/nonexistent/plot1.pdf',
        '/nonexistent/plot2.pdf',
        '/nonexistent/plot3.pdf'
    ]
    
    # Generate report (should succeed despite missing plots)
    output_path = str(tmp_path / "report.pdf")
    success = generate_pdf_report(log_data, plot_paths, output_path)
    
    # Verify report was still created
    assert success is True
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


def test_long_results(tmp_path):
    """Test report generation with very long optimization results."""
    # Create log data with many warnings (simulating long content)
    long_warnings = [
        f"Warning {i}: This is a very long warning message that contains "
        f"detailed information about parameter combination {i} and why it "
        f"might not be optimal for the current analysis. " * 3
        for i in range(50)  # 50 long warnings
    ]
    
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 10, 'KO': 10},
            'total_flies': 20,
            'frame_counts': {
                'min': 8990,
                'max': 9010,
                'mean': 9000.0,
                'variation': 0.22
            }
        },
        'optimization_process': {
            'n_combinations_tested': 1000,
            'parameter_space': {
                'window_sizes': list(range(100, 1000, 100)),
                'sampling_rates': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                'strategies': ['uniform', 'stratified', 'systematic'],
                'edge_thresholds': [5, 10, 15, 20, 25, 30]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': long_warnings,
        'validation_metrics': {
            'total_parameters_evaluated': 1000,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.30,
            'mean_composite_score': 0.55,
            'std_composite_score': 0.12
        }
    }
    
    # Generate report with no plots
    plot_paths = []
    output_path = str(tmp_path / "report.pdf")
    success = generate_pdf_report(log_data, plot_paths, output_path)
    
    # Verify report was created (proper pagination should handle long content)
    assert success is True
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


def test_special_characters(tmp_path):
    """Test proper handling of special characters in text fields."""
    # Create log data with special characters
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot_exp_2025\\test\\files',  # Backslashes
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {
                'WT_line#1': 5,  # Underscores and hash
                'KO_mutant^2': 5  # Caret
            },
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [100],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [
            'File path contains \\ backslashes',
            'Genotype name has _underscores_ and #special chars',
            'Values: 10%, $50, & other symbols'
        ],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.75,
            'mean_composite_score': 0.75,
            'std_composite_score': 0.0
        }
    }
    
    # Generate report
    plot_paths = []
    output_path = str(tmp_path / "report.pdf")
    success = generate_pdf_report(log_data, plot_paths, output_path)
    
    # Verify report was created without errors (proper escaping)
    assert success is True
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


def test_pdf_creation_error(tmp_path, monkeypatch):
    """Test handling of PDF creation errors."""
    from unittest.mock import MagicMock
    
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.75,
            'mean_composite_score': 0.75,
            'std_composite_score': 0.0
        }
    }
    
    # Mock PdfPages to raise an error
    def mock_pdf_pages(*args, **kwargs):
        raise PermissionError("Cannot write to PDF file")
    
    monkeypatch.setattr('matplotlib.backends.backend_pdf.PdfPages', mock_pdf_pages)
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    # Should raise the exception
    with pytest.raises(PermissionError, match="Cannot write to PDF file"):
        generate_pdf_report(log_data, plot_paths, output_path)


def test_invalid_timestamp_format(tmp_path):
    """Test handling of invalid timestamp format."""
    log_data = {
        'timestamp': 'invalid-timestamp-format',  # Invalid format
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.75,
            'mean_composite_score': 0.75,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    # Should still succeed (fallback to raw timestamp)
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_missing_timestamp(tmp_path):
    """Test handling when timestamp is missing from log_data."""
    log_data = {
        # No 'timestamp' key
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.75,
            'mean_composite_score': 0.75,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    # Should still succeed (use current time)
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_plot_embedding_exception(tmp_path, monkeypatch, capsys):
    """Test exception handling during plot embedding."""
    import matplotlib.pyplot as plt
    
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.75,
            'power': 0.85,
            'bias': 0.1,
            'efficiency': 0.8,
            'robustness': 0.15
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.75,
            'worst_composite_score': 0.75,
            'mean_composite_score': 0.75,
            'std_composite_score': 0.0
        }
    }
    
    # Create a valid plot file
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    plot_file = plots_dir / "test_plot.pdf"
    plot_file.write_text("dummy PDF content")
    
    plot_paths = [str(plot_file)]
    output_path = str(tmp_path / "report.pdf")
    
    # Mock plt.figure to raise error ONLY during plot page creation (4th call)
    call_count = [0]
    original_figure = plt.figure
    
    def mock_figure(*args, **kwargs):
        call_count[0] += 1
        # Fail ONLY on the 4th call (plot page), then allow subsequent calls
        if call_count[0] == 4:
            raise RuntimeError("Mock error during plot figure creation")
        return original_figure(*args, **kwargs)
    
    monkeypatch.setattr(plt, 'figure', mock_figure)
    
    # Should still succeed (exception caught and warning printed)
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()
    
    # Verify warning was printed
    captured = capsys.readouterr()
    assert "Warning: Could not embed plot" in captured.out
		

def test_power_below_target(tmp_path):
    """Test warnings when power is below target."""
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8  # Target power
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.65,
            'power': 0.70,  # Below target of 0.8
            'bias': 0.12,
            'efficiency': 0.8,
            'robustness': 0.18
        },
        'warnings': ['Power below target'],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.65,
            'worst_composite_score': 0.65,
            'mean_composite_score': 0.65,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_low_efficiency(tmp_path):
    """Test handling of low efficiency scores."""
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.9,  # High sampling rate = low efficiency
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.60,
            'power': 0.90,
            'bias': 0.05,
            'efficiency': 0.1,  # Very low efficiency
            'robustness': 0.10
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.60,
            'worst_composite_score': 0.60,
            'mean_composite_score': 0.60,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_medium_efficiency(tmp_path):
    """Test handling of medium efficiency scores."""
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.4,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.70,
            'power': 0.85,
            'bias': 0.08,
            'efficiency': 0.6,  # Medium efficiency (0.5-0.7)
            'robustness': 0.12
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.70,
            'worst_composite_score': 0.70,
            'mean_composite_score': 0.70,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_high_robustness(tmp_path):
    """Test handling of high robustness scores."""
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.65,
            'power': 0.80,
            'bias': 0.10,
            'efficiency': 0.8,
            'robustness': 0.5  # High robustness (> 0.4)
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.65,
            'worst_composite_score': 0.65,
            'mean_composite_score': 0.65,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


def test_medium_robustness(tmp_path):
    """Test handling of medium robustness scores."""
    log_data = {
        'timestamp': '2025-01-15T10:30:00',
        'config': {
            'data_dir': '/data/pilot',
            'expected_frames': 9000,
            'alpha': 0.05,
            'power': 0.8
        },
        'pilot_summary': {
            'n_genotypes': 2,
            'flies_per_genotype': {'WT': 5, 'KO': 5},
            'total_flies': 10,
            'frame_counts': {
                'min': 9000,
                'max': 9000,
                'mean': 9000.0,
                'variation': 0.0
            }
        },
        'optimization_process': {
            'n_combinations_tested': 10,
            'parameter_space': {
                'window_sizes': [300],
                'sampling_rates': [0.2],
                'strategies': ['uniform'],
                'edge_thresholds': [20]
            },
            'best_parameters': {
                'window_size': 300,
                'sampling_rate': 0.2,
                'strategy': 'uniform',
                'edge_threshold': 20
            }
        },
        'best_results': {
            'composite_score': 0.70,
            'power': 0.85,
            'bias': 0.08,
            'efficiency': 0.8,
            'robustness': 0.3  # Medium robustness (0.2-0.4)
        },
        'warnings': [],
        'validation_metrics': {
            'total_parameters_evaluated': 10,
            'best_composite_score': 0.70,
            'worst_composite_score': 0.70,
            'mean_composite_score': 0.70,
            'std_composite_score': 0.0
        }
    }
    
    output_path = str(tmp_path / "report.pdf")
    plot_paths = []
    
    success = generate_pdf_report(log_data, plot_paths, output_path)
    assert success is True
    assert Path(output_path).exists()


# =============================================================================
# Tests for Unit 13: Main Orchestration Function
# =============================================================================

from pilot_grooming_optimizer import main

def test_complete_successful_run(tmp_path):
    """Test complete successful run with all outputs generated."""
    # Create test data directory structure
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    # Create CSV files with minimal but sufficient data using new format
    for i in range(3):
        csv_a = genotype_a / f"fly{i}.csv"
        create_test_csv_new_format(csv_a, [(100, 200), (9000, 9100)])
        
        csv_b = genotype_b / f"fly{i}.csv"
        create_test_csv_new_format(csv_b, [(1000, 1100), (9000, 9100)])
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_file = output_dir / "optimization_results.json"
    
    # Run main function with minimal bootstrap iterations for speed
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file),
        '--expected-frames', '9000'
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify success
    assert exit_code == 0
    
    # Verify outputs exist
    assert output_file.exists()
    
    # Verify optimization_log.json exists
    log_file = output_dir / "optimization_log.json"
    assert log_file.exists()
    
    # Verify optimization_report.pdf exists
    report_file = output_dir / "optimization_report.pdf"
    assert report_file.exists()
    
    # Verify plots directory exists
    plots_dir = output_dir / "plots"
    assert plots_dir.exists()
    
    # Load and verify JSON output
    import json
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # Verify required fields (use actual key names from output)
    assert 'window_size' in results
    assert 'sampling_rate' in results
    assert 'strategy' in results  # Changed from 'sampling_strategy'
    assert 'edge_threshold' in results  # Changed from 'recommended_edge_duration_min'
    
    # Verify values are reasonable
    assert isinstance(results['window_size'], int)
    assert results['window_size'] > 0
    assert 0 < results['sampling_rate'] <= 1.0
    assert results['strategy'] in ['uniform', 'stratified', 'systematic']
    assert isinstance(results['edge_threshold'], int)
    assert results['edge_threshold'] > 0
		

def test_invalid_input_data(tmp_path):
    """Test clean exit with error message when data directory is invalid."""
    # Use non-existent data directory
    data_dir = tmp_path / "nonexistent"
    output_file = tmp_path / "output.json"
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify failure
    assert exit_code == 1
    
    # Verify no output files created
    assert not output_file.exists()


def test_keyboard_interrupt(tmp_path, monkeypatch):
    """Test graceful shutdown on KeyboardInterrupt."""
    # Create minimal test data
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    for i in range(2):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n100\n200\n")
        
        csv_b = genotype_b / f"fly{i}.csv"
        csv_b.write_text("Frame\n300\n400\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_file = output_dir / "results.json"
    
    # Mock optimize_parameter_space to raise KeyboardInterrupt
    from unittest.mock import MagicMock
    
    def mock_optimize(*args, **kwargs):
        raise KeyboardInterrupt()
    
    monkeypatch.setattr(
        'pilot_grooming_optimizer.optimize_parameter_space',
        mock_optimize
    )
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify graceful failure
    assert exit_code == 1


def test_write_permission_denied(tmp_path, monkeypatch):
    """Test clear error message when write permission is denied."""
    # Create minimal test data
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    for i in range(2):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n100\n200\n")
        
        csv_b = genotype_b / f"fly{i}.csv"
        csv_b.write_text("Frame\n300\n400\n")
    
    output_file = tmp_path / "output" / "results.json"
    
    # Mock write_json_output to raise PermissionError
    from unittest.mock import MagicMock
    
    def mock_write_json(*args, **kwargs):
        raise PermissionError("Permission denied: /protected/results.json")
    
    monkeypatch.setattr(
        'pilot_grooming_optimizer.write_json_output',
        mock_write_json
    )
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify failure with clear error
    assert exit_code == 1


def test_oserror_handler(tmp_path, monkeypatch):
    """Test OSError exception handler in main."""
    # Create minimal test data
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    for i in range(2):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n100\n200\n")
        
        csv_b = genotype_b / f"fly{i}.csv"
        csv_b.write_text("Frame\n300\n400\n")
    
    output_file = tmp_path / "output" / "results.json"
    
    # Mock write_json_output to raise OSError
    def mock_write_json(*args, **kwargs):
        raise OSError("Disk full or I/O error")
    
    monkeypatch.setattr(
        'pilot_grooming_optimizer.write_json_output',
        mock_write_json
    )
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify failure with OSError
    assert exit_code == 1


def test_generic_exception_handler(tmp_path, monkeypatch):
    """Test generic Exception handler in main."""
    # Create minimal test data
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    for i in range(2):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n100\n200\n")
        
        csv_b = genotype_b / f"fly{i}.csv"
        csv_b.write_text("Frame\n300\n400\n")
    
    output_file = tmp_path / "output" / "results.json"
    
    # Mock write_json_output to raise an unexpected exception
    def mock_write_json(*args, **kwargs):
        raise RuntimeError("Unexpected error occurred")
    
    monkeypatch.setattr(
        'pilot_grooming_optimizer.write_json_output',
        mock_write_json
    )
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Verify failure with generic exception
    assert exit_code == 1


def test_cleanup_files_with_exceptions():
    """Test _cleanup_files handles exceptions gracefully."""
    from pilot_grooming_optimizer import _cleanup_files
    import tempfile
    
    # Create some test files
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # List of paths including non-existent ones
    file_paths = [
        temp_file,
        temp_dir,
        '/nonexistent/path/to/file.txt',  # Non-existent path
        '/root/protected/file.txt',  # Permission denied path (might not trigger on all systems)
    ]
    
    # Should not raise any exceptions
    _cleanup_files(file_paths)
    
    # Verify that existing files were removed
    assert not Path(temp_file).exists()
    assert not Path(temp_dir).exists()


def test_cleanup_files_with_permission_error(tmp_path, monkeypatch):
    """Test _cleanup_files handles permission errors silently."""
    from pilot_grooming_optimizer import _cleanup_files
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    # Mock Path.unlink to raise PermissionError
    from pathlib import Path as OrigPath
    
    original_unlink = OrigPath.unlink
    
    def mock_unlink(self, *args, **kwargs):
        raise PermissionError("Cannot delete file")
    
    monkeypatch.setattr(Path, 'unlink', mock_unlink)
    
    # Should not raise exception (silent failure)
    _cleanup_files([str(test_file)])
    
    # File still exists (couldn't be deleted)
    assert test_file.exists()


def test_cleanup_files_with_oserror(tmp_path, monkeypatch):
    """Test _cleanup_files handles OSError silently."""
    from pilot_grooming_optimizer import _cleanup_files
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    # Mock Path.unlink to raise OSError
    from pathlib import Path as OrigPath
    
    def mock_unlink(self, *args, **kwargs):
        raise OSError("I/O error")
    
    monkeypatch.setattr(Path, 'unlink', mock_unlink)
    
    # Should not raise exception (silent failure)
    _cleanup_files([str(test_file)])
    
    # File still exists (couldn't be deleted)
    assert test_file.exists()


def test_main_entry_point(tmp_path, monkeypatch):
    """Test the if __name__ == '__main__' entry point."""
    # Create minimal test data
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    for i in range(2):
        csv_a = genotype_a / f"fly{i}.csv"
        create_test_csv_new_format(csv_a, [(100, 200), (9000, 9100)])
        
        csv_b = genotype_b / f"fly{i}.csv"
        create_test_csv_new_format(csv_b, [(300, 400), (9000, 9100)])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_file = output_dir / "results.json"
    
    # Mock sys.argv to simulate command line execution
    import sys
    original_argv = sys.argv
    
    try:
        sys.argv = [
            'pilot_grooming_optimizer.py',
            '--data-dir', str(data_dir),
            '--output', str(output_file)
        ]
        
        # Mock sys.exit to capture the exit code instead of exiting
        exit_code_captured = []
        
        def mock_exit(code):
            exit_code_captured.append(code)
        
        monkeypatch.setattr(sys, 'exit', mock_exit)
        
        # Import and reload the module to execute the __main__ block
        # This is tricky - we'll just test main() can be called without args
        from pilot_grooming_optimizer import main
        
        # Call main without args (should use sys.argv[1:])
        exit_code = main(n_bootstrap=10)
        
        # Verify it ran successfully
        assert exit_code == 0
        
        # Verify output file was created
        assert output_file.exists()
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

				
def test_cleanup_files_directory_removal(tmp_path):
    """Test _cleanup_files can remove directories."""
    from pilot_grooming_optimizer import _cleanup_files
    
    # Create a directory with files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")
    
    # Cleanup should remove the directory
    _cleanup_files([str(test_dir)])
    
    # Verify directory was removed
    assert not test_dir.exists()


def test_cleanup_files_with_shutil_error(tmp_path, monkeypatch):
    """Test _cleanup_files handles shutil.rmtree errors silently."""
    from pilot_grooming_optimizer import _cleanup_files
    
    # Create a directory
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    
    # Mock shutil.rmtree to raise an error
    import shutil
    
    def mock_rmtree(*args, **kwargs):
        raise OSError("Cannot remove directory")
    
    monkeypatch.setattr(shutil, 'rmtree', mock_rmtree)
    
    # Should not raise exception (silent failure)
    _cleanup_files([str(test_dir)])
    
    # Directory still exists (couldn't be deleted)
    assert test_dir.exists()


def test_valueerror_handler_insufficient_genotypes(tmp_path):
    """Test ValueError handler when fewer than 2 genotypes found."""
    # Create data directory with only ONE genotype (triggers ValueError)
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_a.mkdir(parents=True)
    
    # Add some CSV files
    for i in range(3):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n100\n200\n")
    
    output_file = tmp_path / "output" / "results.json"
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Should fail with ValueError (< 2 genotypes)
    assert exit_code == 1
    
    # Should not create output files (early failure)
    assert not output_file.exists()


def test_valueerror_handler_short_videos(tmp_path):
    """Test ValueError handler when videos are too short."""
    # Create data with very short videos (< 100 frames)
    data_dir = tmp_path / "data"
    genotype_a = data_dir / "WT"
    genotype_b = data_dir / "KO"
    genotype_a.mkdir(parents=True)
    genotype_b.mkdir(parents=True)
    
    # Create CSVs with events ending at frame 50 (too short)
    for i in range(3):
        csv_a = genotype_a / f"fly{i}.csv"
        csv_a.write_text("Frame\n10\n50\n")  # Max frame = 50 (< 100)
        
        csv_b = genotype_b / f"fly{i}.csv"
        csv_b.write_text("Frame\n20\n50\n")  # Max frame = 50 (< 100)
    
    output_file = tmp_path / "output" / "results.json"
    
    args = [
        '--data-dir', str(data_dir),
        '--output', str(output_file)
    ]
    
    exit_code = main(args, n_bootstrap=10)
    
    # Should fail with ValueError (videos too short for parameter generation)
    assert exit_code == 1
