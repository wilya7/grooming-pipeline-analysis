"""Test suite for Pilot Grooming Optimizer - Script 1"""

import pytest
import tempfile
import sys
from pathlib import Path

from pilot_grooming_optimizer import parse_arguments, load_pilot_data


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
