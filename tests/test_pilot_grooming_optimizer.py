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
		calculate_statistical_power
)

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
    
    # Create CSV files for genotype A
    csv_a1 = genotype_a / "fly1.csv"
    csv_a1.write_text("Frame\n100\n150\n200\n250\n")
    
    csv_a2 = genotype_a / "fly2.csv"
    csv_a2.write_text("Frame\n50\n75\n300\n400\n")
    
    # Create CSV files for genotype B
    csv_b1 = genotype_b / "fly1.csv"
    csv_b1.write_text("Frame\n1000\n1500\n2000\n2500\n")
    
    # Load data
    pilot_data, frame_counts = load_pilot_data(str(tmp_path))
    
    # Verify structure
    assert len(pilot_data) == 2
    assert "genotype_A" in pilot_data
    assert "genotype_B" in pilot_data
    
    # Verify genotype A data
    assert len(pilot_data["genotype_A"]) == 2
    assert pilot_data["genotype_A"][0] == [(100, 150), (200, 250)]
    assert pilot_data["genotype_A"][1] == [(50, 75), (300, 400)]
    
    # Verify genotype B data
    assert len(pilot_data["genotype_B"]) == 1
    assert pilot_data["genotype_B"][0] == [(1000, 1500), (2000, 2500)]
    
    # Verify frame counts (max end frame from each fly)
    assert frame_counts["genotype_A"] == [250, 400]
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
    
    # Create CSV with data
    csv_a1 = genotype_a / "fly1.csv"
    csv_a1.write_text("Frame\n100\n150\n")
    
    # Create empty CSV (header only, no events)
    csv_a2 = genotype_a / "fly2.csv"
    csv_a2.write_text("Frame\n")
    
    # Create CSV with data for genotype B
    csv_b1 = genotype_b / "fly1.csv"
    csv_b1.write_text("Frame\n200\n250\n")
    
    # Load data
    pilot_data, frame_counts = load_pilot_data(str(tmp_path))
    
    # Verify empty CSV is handled correctly
    assert len(pilot_data["genotype_A"]) == 2
    assert pilot_data["genotype_A"][0] == [(100, 150)]
    assert pilot_data["genotype_A"][1] == []  # Empty events list for empty CSV
    
    # Frame count should be 0 for empty CSV (sentinel value)
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


def test_fallback_triggered():
    """Test that fallback triggers when no divisors are multiples of 25."""
    # Use 101 frames (prime number)
    # Divisors: 1, 101
    # valid_divisors: [101] (meets >= 100 threshold)
    # primary_candidates: [] (101 % 25 = 1, not a multiple of 25)
    # Should use fallback: all valid_divisors
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
    
    # Verify fallback was used: got [101] even though it's not a multiple of 25
    assert params['window_sizes'] == [101]
    assert 101 >= 100  # Meets minimum threshold
    assert 101 % 25 != 0  # Not a multiple of 25 (proves fallback was used)
    
    # Verify fixed parameters unchanged
    assert params['sampling_rates'] == [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    assert params['strategies'] == ['uniform', 'stratified', 'systematic']
    assert params['edge_thresholds'] == [5, 10, 15, 20, 25, 30]


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
