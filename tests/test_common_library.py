"""Test suite for common_library.py"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
import json
from src.common_library import (
    load_event_csv,
    calculate_event_frequency,
    calculate_mean_bout_duration,
    calculate_grooming_percentage,
    calculate_fragmentation_index,
    process_directory_structure,
    validate_frame_counts,
    write_json_output,
    calculate_all_behavioral_metrics,
    detect_video_fps
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


def create_csv_with_frames(filepath: str, frames: list, column_name: str = 'Frame'):
    """Helper function to create a CSV file with given frames."""
    df = pd.DataFrame({column_name: frames})
    df.to_csv(filepath, index=False)


# =============================================================================
# Unit 1 Tests: load_event_csv
# =============================================================================

def test_valid_csv_multiple_events(temp_csv_file):
    """Test loading a valid CSV with multiple grooming events."""
    frames = [100, 150, 200, 250, 300, 400]
    create_csv_with_frames(temp_csv_file, frames)
    
    events = load_event_csv(temp_csv_file)
    
    assert len(events) == 3
    assert events[0] == (100, 150)
    assert events[1] == (200, 250)
    assert events[2] == (300, 400)
    assert all(isinstance(event, tuple) for event in events)
    assert all(len(event) == 2 for event in events)


def test_odd_number_of_frames(temp_csv_file):
    """Test that odd number of frames raises ValueError."""
    frames = [100, 150, 200]  # 3 frames - odd number
    create_csv_with_frames(temp_csv_file, frames)
    
    with pytest.raises(ValueError) as exc_info:
        load_event_csv(temp_csv_file)
    
    assert "even number of frames" in str(exc_info.value).lower()
    assert "3 frames" in str(exc_info.value)


def test_non_chronological_frames(temp_csv_file):
    """Test that non-chronological frames raise ValueError."""
    frames = [100, 150, 140, 200]  # 140 comes after 150
    create_csv_with_frames(temp_csv_file, frames)
    
    with pytest.raises(ValueError) as exc_info:
        load_event_csv(temp_csv_file)
    
    assert "chronological order" in str(exc_info.value).lower()


def test_start_greater_than_end(temp_csv_file):
    """Test that start frame > end frame raises ValueError."""
    frames = [100, 150, 250, 200]  # Second event: start=250, end=200
    create_csv_with_frames(temp_csv_file, frames)
    
    with pytest.raises(ValueError) as exc_info:
        load_event_csv(temp_csv_file)
    
    assert "start frame must be less than end frame" in str(exc_info.value).lower()
    assert "250" in str(exc_info.value)
    assert "200" in str(exc_info.value)


def test_empty_csv(temp_csv_file):
    """Test that an empty CSV returns an empty list."""
    create_csv_with_frames(temp_csv_file, [])
    
    events = load_event_csv(temp_csv_file)
    
    assert events == []
    assert isinstance(events, list)


def test_missing_frame_column(temp_csv_file):
    """Test that missing 'Frame' column raises ValueError."""
    # Create CSV with wrong column name
    df = pd.DataFrame({'WrongColumn': [100, 150, 200, 250]})
    df.to_csv(temp_csv_file, index=False)
    
    with pytest.raises(ValueError) as exc_info:
        load_event_csv(temp_csv_file)
    
    assert "Frame" in str(exc_info.value)
    assert "column" in str(exc_info.value).lower()


def test_validation_disabled(temp_csv_file):
    """Test that validation can be disabled."""
    # Create invalid data (odd number of frames)
    frames = [100, 150, 200]
    create_csv_with_frames(temp_csv_file, frames)
    
    # Should not raise error when validate=False
    events = load_event_csv(temp_csv_file, validate=False)
    
    assert len(events) == 1
    assert events[0] == (100, 150)


def test_single_event_csv(temp_csv_file):
    """Test loading a CSV with a single grooming event."""
    frames = [100, 150]
    create_csv_with_frames(temp_csv_file, frames)
    
    events = load_event_csv(temp_csv_file)
    
    assert len(events) == 1
    assert events[0] == (100, 150)


def test_invalid_file_path(temp_csv_file):
    """Test that invalid file path raises ValueError."""
    # Delete the file to make it invalid
    os.unlink(temp_csv_file)
    
    with pytest.raises(ValueError) as exc_info:
        load_event_csv(temp_csv_file)
    
    assert "Error reading CSV file" in str(exc_info.value)


# =============================================================================
# Unit 2 Tests: calculate_event_frequency
# =============================================================================

def test_standard_frequency():
    """Test standard frequency calculation: 10 events in 9000 frames at 30fps."""
    # Create 10 events (actual frame values don't matter for frequency calculation)
    events = [(i*900, i*900+100) for i in range(10)]
    
    # 9000 frames / 30 fps / 60 = 5 minutes
    # 10 events / 5 minutes = 2.0 events/min
    frequency = calculate_event_frequency(events, total_frames=9000, fps=30.0)
    assert frequency == pytest.approx(2.0)


def test_zero_events_frequency():
    """Test with zero events returns 0.0."""
    events = []
    frequency = calculate_event_frequency(events, total_frames=9000, fps=30.0)
    assert frequency == 0.0


def test_different_fps_values():
    """Test correct normalization with different fps values."""
    # Create 5 events
    events = [(i*600, i*600+100) for i in range(5)]
    
    # fps=10: 3000 frames = 3000/10/60 = 5.0 minutes → 5 events / 5 min = 1.0 events/min
    frequency_10 = calculate_event_frequency(events, total_frames=3000, fps=10.0)
    assert frequency_10 == pytest.approx(1.0)
    
    # fps=60: 3000 frames = 3000/60/60 = 0.833... minutes → 5 events / 0.833... min = 6.0 events/min
    frequency_60 = calculate_event_frequency(events, total_frames=3000, fps=60.0)
    assert frequency_60 == pytest.approx(6.0)
    
    # fps=120: 3000 frames = 3000/120/60 = 0.4166... minutes → 5 events / 0.4166... min = 12.0 events/min
    frequency_120 = calculate_event_frequency(events, total_frames=3000, fps=120.0)
    assert frequency_120 == pytest.approx(12.0)


def test_edge_case_small_frames():
    """Test with very small frame counts handles without division error."""
    events = [(0, 1)]  # 1 event
    
    # 1 frame at 30fps = 1/30/60 = 0.000555... minutes
    # 1 event / 0.000555... min = 1800.0 events/min
    # This is mathematically correct for such a short duration
    frequency = calculate_event_frequency(events, total_frames=1, fps=30.0)
    assert frequency == pytest.approx(1800.0)
    
    # Edge case: 0 frames should return 0.0 (graceful handling)
    frequency_zero = calculate_event_frequency(events, total_frames=0, fps=30.0)
    assert frequency_zero == 0.0
    
    # Edge case: negative frames should return 0.0 (graceful handling)
    frequency_negative = calculate_event_frequency(events, total_frames=-100, fps=30.0)
    assert frequency_negative == 0.0


# =============================================================================
# Unit 3 Tests: calculate_mean_bout_duration
# =============================================================================

def test_standard_durations():
    """Test with events of durations 30, 60, 90 frames at 30fps.
    
    Event 1: frames 0-29 = 30 frames (inclusive)
    Event 2: frames 100-159 = 60 frames (inclusive)
    Event 3: frames 200-289 = 90 frames (inclusive)
    Mean: (30 + 60 + 90) / 3 = 60 frames = 2.0 seconds at 30fps
    """
    events = [(0, 29), (100, 159), (200, 289)]
    
    # Mean duration: 60 frames / 30 fps = 2.0 seconds
    result = calculate_mean_bout_duration(events, fps=30.0)
    assert result == pytest.approx(2.0)


def test_empty_events_mean_duration():
    """Test with empty events list should return 0.0."""
    events = []
    result = calculate_mean_bout_duration(events, fps=30.0)
    assert result == 0.0


def test_single_event_duration():
    """Test mean duration with a single event.
    
    Event: frames 10-39 = 30 frames (inclusive) = 1.0 second at 30fps
    """
    events = [(10, 39)]
    
    # 30 frames / 30 fps = 1.0 second
    result = calculate_mean_bout_duration(events, fps=30.0)
    assert result == pytest.approx(1.0)


def test_different_fps():
    """Test same events with different fps values.
    
    Events have mean duration of 60 frames:
    - At 10 fps: 60 / 10 = 6.0 seconds
    - At 60 fps: 60 / 60 = 1.0 seconds
    - At 120 fps: 60 / 120 = 0.5 seconds
    """
    events = [(0, 29), (100, 159), (200, 289)]
    
    # At 10 fps: mean 60 frames / 10 fps = 6.0 seconds
    result_10fps = calculate_mean_bout_duration(events, fps=10.0)
    assert result_10fps == pytest.approx(6.0)
    
    # At 60 fps: mean 60 frames / 60 fps = 1.0 seconds
    result_60fps = calculate_mean_bout_duration(events, fps=60.0)
    assert result_60fps == pytest.approx(1.0)
    
    # At 120 fps: mean 60 frames / 120 fps = 0.5 seconds
    result_120fps = calculate_mean_bout_duration(events, fps=120.0)
    assert result_120fps == pytest.approx(0.5)


def test_single_frame_event():
    """Test event with single frame (start == end).
    
    Event: frame 100-100 = 1 frame (inclusive) at 30fps
    """
    events = [(100, 100)]
    
    # 1 frame / 30 fps = 0.03333... seconds
    result = calculate_mean_bout_duration(events, fps=30.0)
    expected = 1.0 / 30.0
    assert result == pytest.approx(expected)


def test_multiple_single_frame_events():
    """Test multiple single-frame events.
    
    Three events, each 1 frame duration: mean = 1 frame at 30fps
    """
    events = [(10, 10), (20, 20), (30, 30)]
    
    # Mean of 1 frame / 30 fps = 0.03333... seconds
    result = calculate_mean_bout_duration(events, fps=30.0)
    expected = 1.0 / 30.0
    assert result == pytest.approx(expected)


def test_inclusive_duration_calculation():
    """Test that duration calculation is inclusive (end - start + 1).
    
    Verify that frame 100 to 102 is 3 frames, not 2.
    """
    events = [(100, 102)]  # Should be 3 frames (100, 101, 102)
    
    # 3 frames / 30 fps = 0.1 seconds
    result = calculate_mean_bout_duration(events, fps=30.0)
    expected = 3.0 / 30.0  # 0.1 seconds
    assert result == pytest.approx(expected)


# =============================================================================
# Unit 4 Tests: calculate_grooming_percentage
# =============================================================================

def test_standard_percentage():
    """Test standard percentage calculation: 900 grooming frames out of 9000 total.
    
    Events: [(0, 299), (500, 599), (1000, 1499)]
    - Event 1: 0-299 = 300 frames (inclusive)
    - Event 2: 500-599 = 100 frames (inclusive)
    - Event 3: 1000-1499 = 500 frames (inclusive)
    Total: 300 + 100 + 500 = 900 frames
    Percentage: 900/9000 * 100 = 10.0%
    """
    events = [(0, 299), (500, 599), (1000, 1499)]
    percentage = calculate_grooming_percentage(events, 9000)
    assert percentage == pytest.approx(10.0)


def test_no_grooming():
    """Test with no grooming events returns 0.0%."""
    events = []
    percentage = calculate_grooming_percentage(events, 9000)
    assert percentage == 0.0


def test_overlapping_events_handling():
    """Test that overlapping events are handled gracefully without crashing.
    
    Overlapping events should count each frame only once.
    Events: [(0, 100), (50, 150), (140, 200)]
    - Event 1: frames 0-100 (101 frames)
    - Event 2: frames 50-150 (overlaps with event 1, extends to 150)
    - Event 3: frames 140-200 (overlaps with event 2, extends to 200)
    After merging: unique frames 0-200 = 201 frames total
    Percentage: 201/1000 * 100 = 20.1%
    """
    events = [(0, 100), (50, 150), (140, 200)]
    percentage = calculate_grooming_percentage(events, 1000)
    
    # Should not crash and should return a valid percentage
    assert isinstance(percentage, float)
    assert 0.0 <= percentage <= 100.0
    
    # Expected: 201 unique frames out of 1000 = 20.1%
    assert percentage == pytest.approx(20.1)


def test_all_frames_grooming():
    """Test when all frames are grooming (100.0%).
    
    Single event covering entire video: frames 0-8999 in 9000 frame video
    Duration: 8999 - 0 + 1 = 9000 frames
    Percentage: 9000/9000 * 100 = 100.0%
    """
    events = [(0, 8999)]
    percentage = calculate_grooming_percentage(events, 9000)
    assert percentage == pytest.approx(100.0)


def test_zero_total_frames():
    """Test edge case where total_frames is 0 (division by zero protection)."""
    events = [(0, 100)]
    percentage = calculate_grooming_percentage(events, 0)
    assert percentage == 0.0


def test_single_frame_event_percentage():
    """Test with a single frame event (inclusive duration).
    
    Event: [(100, 100)] = 1 frame out of 1000 = 0.1%
    """
    events = [(100, 100)]
    percentage = calculate_grooming_percentage(events, 1000)
    assert percentage == pytest.approx(0.1)


def test_percentage_bounded_at_100():
    """Test that percentage is bounded at 100 even if calculation would theoretically exceed.
    
    Edge case: event extends beyond total_frames (shouldn't happen in practice)
    """
    events = [(0, 10000)]  # 10001 frames of grooming
    percentage = calculate_grooming_percentage(events, 9000)  # Only 9000 total frames
    assert percentage == 100.0  # Should be bounded at 100


# =============================================================================
# Unit 5 Tests: calculate_fragmentation_index
# =============================================================================

def test_uniform_durations():
    """Test with events that all have identical durations (CV = 0).
    
    Events: [(0, 29), (100, 129), (200, 229)]
    All events have duration of 30 frames
    Standard deviation = 0, so CV = 0
    """
    events = [(0, 29), (100, 129), (200, 229)]
    fragmentation = calculate_fragmentation_index(events)
    assert fragmentation == pytest.approx(0.0)


def test_variable_durations():
    """Test with events of varying durations.
    
    Events: [(0, 9), (100, 149), (200, 299)]
    Durations: [10, 50, 100] frames
    Mean = 53.333..., std ≈ 36.817..., CV ≈ 0.6903
    """
    events = [(0, 9), (100, 149), (200, 299)]
    fragmentation = calculate_fragmentation_index(events)
    
    # Expected calculation:
    # durations = [10, 50, 100]
    # mean = 53.333...
    # std = 36.817... (population std, ddof=0)
    # CV = 36.817 / 53.333 ≈ 0.6903
    assert fragmentation == pytest.approx(0.6903, abs=0.001)


def test_empty_events_fragmentation():
    """Test with empty events list returns 0.0."""
    events = []
    fragmentation = calculate_fragmentation_index(events)
    assert fragmentation == 0.0


def test_single_event_fragmentation():
    """Test with single event returns 0.0 (no variation possible)."""
    events = [(100, 150)]
    fragmentation = calculate_fragmentation_index(events)
    assert fragmentation == 0.0


def test_zero_mean_edge_case(monkeypatch):
    """Test the defensive programming check for mean=0 edge case.
    
    This tests line 281, which handles the theoretical case where mean_duration is 0.
    This shouldn't happen with valid events, but we test the safety check anyway.
    """
    import numpy as np
    
    # Create valid events
    events = [(0, 9), (100, 149)]
    
    # Mock np.mean to return 0 to trigger the edge case check
    original_mean = np.mean
    def mock_mean(values):
        return 0.0
    
    monkeypatch.setattr(np, 'mean', mock_mean)
    
    # Should return 0.0 due to the mean=0 safety check
    fragmentation = calculate_fragmentation_index(events)
    assert fragmentation == 0.0


# =============================================================================
# Unit 6 Tests: process_directory_structure
# =============================================================================

def test_valid_structure(tmp_path):
    """Test processing a valid directory structure with two genotypes."""
    # Create directory structure
    # base_dir/
    #   ├── WT/
    #   │   ├── file1.csv
    #   │   ├── file2.csv
    #   │   └── file3.csv
    #   └── KO/
    #       ├── file1.csv
    #       └── file2.csv
    
    wt_dir = tmp_path / "WT"
    ko_dir = tmp_path / "KO"
    wt_dir.mkdir()
    ko_dir.mkdir()
    
    # Create CSV files
    (wt_dir / "file1.csv").write_text("Frame\n100\n150")
    (wt_dir / "file2.csv").write_text("Frame\n200\n250")
    (wt_dir / "file3.csv").write_text("Frame\n300\n350")
    (ko_dir / "file1.csv").write_text("Frame\n400\n450")
    (ko_dir / "file2.csv").write_text("Frame\n500\n550")
    
    # Process directory
    structure, warnings = process_directory_structure(str(tmp_path))
    
    # Assertions
    assert len(structure) == 2
    assert "WT" in structure
    assert "KO" in structure
    assert len(structure["WT"]) == 3
    assert len(structure["KO"]) == 2
    
    # Check that paths are absolute and correct
    assert all(str(tmp_path / "WT" / f"file{i}.csv") in structure["WT"] for i in [1, 2, 3])
    assert all(str(tmp_path / "KO" / f"file{i}.csv") in structure["KO"] for i in [1, 2])
    
    # Should have warnings about insufficient files (< 10)
    assert len(warnings) == 2
    assert any("WT" in w and "3 file(s)" in w for w in warnings)
    assert any("KO" in w and "2 file(s)" in w for w in warnings)


def test_few_files_warning(tmp_path):
    """Test that genotype with fewer than min_files_warning generates warning."""
    # Create directory with only 5 files (default threshold is 10)
    genotype_dir = tmp_path / "LowCount"
    genotype_dir.mkdir()
    
    # Create 5 CSV files
    for i in range(5):
        (genotype_dir / f"file{i}.csv").write_text("Frame\n100\n150")
    
    # Process directory
    structure, warnings = process_directory_structure(str(tmp_path))
    
    # Assertions
    assert "LowCount" in structure
    assert len(structure["LowCount"]) == 5
    
    # Should have warning about insufficient files
    assert len(warnings) == 1
    assert "LowCount" in warnings[0]
    assert "5 file(s)" in warnings[0]
    assert "10" in warnings[0]  # Default minimum


def test_empty_subdirectory(tmp_path):
    """Test that empty subdirectory is excluded from results with warning."""
    # Create directory structure with one empty subdirectory
    wt_dir = tmp_path / "WT"
    empty_dir = tmp_path / "EmptyGenotype"
    wt_dir.mkdir()
    empty_dir.mkdir()
    
    # Add files only to WT
    for i in range(12):
        (wt_dir / f"file{i}.csv").write_text("Frame\n100\n150")
    
    # Process directory
    structure, warnings = process_directory_structure(str(tmp_path))
    
    # Assertions
    assert len(structure) == 1
    assert "WT" in structure
    assert "EmptyGenotype" not in structure  # Empty dir should be excluded
    
    # Should have warning about empty directory
    assert len(warnings) == 1
    assert "EmptyGenotype" in warnings[0]
    assert "empty" in warnings[0].lower()
    assert "excluding from results" in warnings[0]


def test_nonexistent_base_dir():
    """Test that non-existent base directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        process_directory_structure("/nonexistent/path/to/directory")
    
    assert "does not exist" in str(exc_info.value)


def test_no_subdirectories(tmp_path):
    """Test that base_dir with no subdirectories returns empty dict and warning."""
    # Create files directly in base_dir (not in subdirectories)
    (tmp_path / "file1.csv").write_text("Frame\n100\n150")
    (tmp_path / "file2.csv").write_text("Frame\n200\n250")
    
    # Process directory
    structure, warnings = process_directory_structure(str(tmp_path))
    
    # Assertions
    assert structure == {}
    assert len(warnings) == 1
    assert "No subdirectories found" in warnings[0]


def test_custom_file_extension(tmp_path):
    """Test with custom file extension."""
    # Create directory with .txt files
    genotype_dir = tmp_path / "TestGenotype"
    genotype_dir.mkdir()
    
    # Create both .txt and .csv files
    for i in range(5):
        (genotype_dir / f"file{i}.txt").write_text("test data")
    for i in range(3):
        (genotype_dir / f"file{i}.csv").write_text("Frame\n100\n150")
    
    # Process directory looking for .txt files
    structure, warnings = process_directory_structure(str(tmp_path), file_extension='.txt')
    
    # Should only find .txt files
    assert "TestGenotype" in structure
    assert len(structure["TestGenotype"]) == 5
    assert all(path.endswith('.txt') for path in structure["TestGenotype"])


def test_custom_min_files_threshold(tmp_path):
    """Test with custom minimum files threshold."""
    # Create directory with 8 files
    genotype_dir = tmp_path / "TestGenotype"
    genotype_dir.mkdir()
    
    for i in range(8):
        (genotype_dir / f"file{i}.csv").write_text("Frame\n100\n150")
    
    # Test with min_files_warning=5 (should not warn)
    structure, warnings = process_directory_structure(
        str(tmp_path),
        min_files_warning=5
    )
    assert len(warnings) == 0
    
    # Test with min_files_warning=10 (should warn)
    structure, warnings = process_directory_structure(
        str(tmp_path),
        min_files_warning=10
    )
    assert len(warnings) == 1
    assert "8 file(s)" in warnings[0]
    assert "10" in warnings[0]


def test_base_dir_is_file_not_directory(tmp_path):
    """Test that passing a file path instead of directory raises FileNotFoundError."""
    # Create a file instead of a directory
    file_path = tmp_path / "not_a_directory.csv"
    file_path.write_text("Frame\n100\n150")
    
    # Try to process the file as if it were a directory
    with pytest.raises(FileNotFoundError) as exc_info:
        process_directory_structure(str(file_path))
    
    assert "not a directory" in str(exc_info.value).lower()
    assert str(file_path) in str(exc_info.value)


# =============================================================================
# Unit 7 Tests: validate_frame_counts
# =============================================================================

def test_all_counts_exact():
    """Test with all frame counts exactly equal (variation = 0)."""
    frame_counts = [9000, 9000, 9000, 9000]
    result = validate_frame_counts(frame_counts, expected_frames=9000, variation_threshold=0.1)
    
    assert result['valid'] is True
    assert result['min'] == 9000
    assert result['max'] == 9000
    assert result['mean'] == 9000.0
    assert result['variation_percent'] == 0.0
    assert result['expected'] == 9000


def test_variation_within_threshold():
    """Test with variation within threshold (should be valid).
    
    Frame counts: [8900, 9000, 9100]
    Min: 8900, Max: 9100, Mean: 9000
    Variation: (9100 - 8900) / 9000 = 0.0222 (2.22%)
    This is < 10% threshold, so should be valid.
    """
    frame_counts = [8900, 9000, 9100]
    result = validate_frame_counts(frame_counts, expected_frames=9000, variation_threshold=0.1)
    
    assert result['valid'] is True
    assert result['min'] == 8900
    assert result['max'] == 9100
    assert result['mean'] == pytest.approx(9000.0)
    assert result['variation_percent'] == pytest.approx(2.222, abs=0.01)
    assert result['expected'] == 9000


def test_variation_exceeds_threshold():
    """Test with variation exceeding threshold (should be invalid).
    
    Frame counts: [8000, 9000, 10000]
    Min: 8000, Max: 10000, Mean: 9000
    Variation: (10000 - 8000) / 9000 = 0.2222 (22.22%)
    This is > 10% threshold, so should be invalid.
    """
    frame_counts = [8000, 9000, 10000]
    result = validate_frame_counts(frame_counts, expected_frames=9000, variation_threshold=0.1)
    
    assert result['valid'] is False
    assert result['min'] == 8000
    assert result['max'] == 10000
    assert result['mean'] == pytest.approx(9000.0)
    assert result['variation_percent'] == pytest.approx(22.222, abs=0.01)
    assert result['expected'] == 9000


def test_empty_list():
    """Test with empty list returns appropriate handling (valid=False)."""
    frame_counts = []
    result = validate_frame_counts(frame_counts, expected_frames=9000, variation_threshold=0.1)
    
    assert result['valid'] is False
    assert result['min'] == 0
    assert result['max'] == 0
    assert result['mean'] == 0.0
    assert result['variation_percent'] == 0.0
    assert result['expected'] == 9000


def test_zero_frame_counts_edge_case():
    """Test the defensive programming check for zero frame counts (mean=0).
    
    This tests the case where all frame counts are zero, making mean_count = 0.
    This shouldn't happen with valid video data, but we test the safety check anyway.
    """
    frame_counts = [0, 0, 0, 0]
    result = validate_frame_counts(frame_counts, expected_frames=9000, variation_threshold=0.1)
    
    assert result['valid'] is True  # variation is 0.0, within threshold
    assert result['min'] == 0
    assert result['max'] == 0
    assert result['mean'] == 0.0
    assert result['variation_percent'] == 0.0
    assert result['expected'] == 9000


# =============================================================================
# Unit 8 Tests: write_json_output
# =============================================================================

def test_valid_data_writable_path(tmp_path):
    """Test writing valid data to writable path creates file successfully."""
    # Prepare test data
    data = {
        "experiment": "grooming_analysis",
        "parameters": {
            "window_size": 300,
            "sampling_rate": 0.1
        },
        "results": [1.2, 3.4, 5.6]
    }
    
    # Define output path
    output_file = tmp_path / "output" / "results.json"
    
    # Write JSON
    success = write_json_output(data, str(output_file), indent=2)
    
    # Assertions
    assert success is True
    assert output_file.exists()
    
    # Verify content is correct
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data == data
    assert loaded_data["experiment"] == "grooming_analysis"
    assert loaded_data["parameters"]["window_size"] == 300
    assert loaded_data["results"] == [1.2, 3.4, 5.6]


def test_non_serializable_data(tmp_path):
    """Test that data with non-serializable objects raises TypeError."""
    # Create data with non-serializable object (function)
    def test_function():
        pass
    
    data = {
        "valid_key": "valid_value",
        "invalid_key": test_function  # Functions cannot be serialized to JSON
    }
    
    output_file = tmp_path / "output.json"
    
    # Should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        write_json_output(data, str(output_file))
    
    assert "non-serializable" in str(exc_info.value).lower()


def test_protected_directory(tmp_path):
    """Test that attempting to write to protected directory raises PermissionError."""
    import os
    import stat
    
    data = {"test": "data"}
    
    # Create a directory and make it read-only
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    
    # Remove write permissions from the directory
    # This makes it impossible to create files inside
    current_permissions = protected_dir.stat().st_mode
    protected_dir.chmod(current_permissions & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    
    output_file = protected_dir / "test_output.json"
    
    try:
        # Should raise PermissionError or OSError (both are acceptable for permission issues)
        with pytest.raises((PermissionError, OSError)) as exc_info:
            write_json_output(data, str(output_file))
        
        # Verify the error message indicates a permission/access problem
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in
                    ['permission', 'denied', 'read-only', 'access'])
    
    finally:
        # Restore write permissions for cleanup
        protected_dir.chmod(current_permissions)


def test_nested_dictionary(tmp_path):
    """Test that nested dictionary is formatted properly with specified indents."""
    # Create deeply nested structure
    data = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "value": 42,
                        "list": [1, 2, 3]
                    }
                },
                "another_key": "value"
            }
        },
        "experiment": "grooming",
        "results": [
            {"id": 1, "value": 10.5},
            {"id": 2, "value": 20.3}
        ]
    }
    
    output_file = tmp_path / "nested.json"
    
    # Write with indent=2
    success = write_json_output(data, str(output_file), indent=2)
    
    assert success is True
    assert output_file.exists()
    
    # Read the raw file content to verify formatting
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Verify proper indentation (should have 2-space indents)
    assert '  "level1"' in content
    assert '    "level2"' in content
    assert '      "level3"' in content
    
    # Verify data integrity
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data == data
    assert loaded_data["level1"]["level2"]["level3"]["level4"]["value"] == 42
    
    # Test with different indent level
    output_file_indent4 = tmp_path / "nested_indent4.json"
    success = write_json_output(data, str(output_file_indent4), indent=4)
    
    assert success is True
    
    with open(output_file_indent4, 'r') as f:
        content_indent4 = f.read()
    
    # Verify 4-space indents
    assert '    "level1"' in content_indent4
    assert '        "level2"' in content_indent4


def test_other_os_errors(tmp_path):
    """Test that other OS errors (not PermissionError) are raised as OSError."""
    data = {"test": "data"}
    
    # Use a filename that exceeds the OS limit (typically 255 characters)
    # This will cause an OSError when trying to create the file
    long_filename = "a" * 300 + ".json"
    invalid_filepath = str(tmp_path / long_filename)
    
    # Should raise OSError (not PermissionError or TypeError)
    with pytest.raises(OSError) as exc_info:
        write_json_output(data, invalid_filepath)
    
    # Verify it's the generic OSError handler
    assert "File system error writing to" in str(exc_info.value)
    # Verify it's not a PermissionError or TypeError
    assert not isinstance(exc_info.value, PermissionError)
    assert not isinstance(exc_info.value, TypeError)


# =============================================================================
# Unit 9 Tests: calculate_all_behavioral_metrics
# =============================================================================

def test_standard_events():
    """Test standard events list with all metrics correctly calculated.
    
    Events: [(0, 29), (100, 159), (200, 289)]
    - Event 1: 30 frames = 1.0 second
    - Event 2: 60 frames = 2.0 seconds
    - Event 3: 90 frames = 3.0 seconds
    
    Total frames: 9000 (5 minutes at 30fps)
    Expected metrics:
    - event_frequency: 3 events / 5 min = 0.6 events/min
    - mean_bout_duration: (1.0 + 2.0 + 3.0) / 3 = 2.0 seconds
    - grooming_percentage: 180 / 9000 * 100 = 2.0%
    - fragmentation_index: CV of [30, 60, 90] ≈ 0.4082
    - bout_durations: [1.0, 2.0, 3.0]
    """
    events = [(0, 29), (100, 159), (200, 289)]
    metrics = calculate_all_behavioral_metrics(events, 9000, fps=30.0)
    
    # Verify all keys are present
    assert 'event_frequency' in metrics
    assert 'mean_bout_duration' in metrics
    assert 'grooming_percentage' in metrics
    assert 'fragmentation_index' in metrics
    assert 'bout_durations' in metrics
    
    # Verify metric values
    assert metrics['event_frequency'] == pytest.approx(0.6)
    assert metrics['mean_bout_duration'] == pytest.approx(2.0)
    assert metrics['grooming_percentage'] == pytest.approx(2.0)
    assert metrics['fragmentation_index'] == pytest.approx(0.4082, abs=0.001)
    assert metrics['bout_durations'] == pytest.approx([1.0, 2.0, 3.0])


def test_empty_events():
    """Test empty events returns all numeric metrics = 0, bout_durations = []."""
    events = []
    metrics = calculate_all_behavioral_metrics(events, 9000, fps=30.0)
    
    # Verify all keys are present
    assert 'event_frequency' in metrics
    assert 'mean_bout_duration' in metrics
    assert 'grooming_percentage' in metrics
    assert 'fragmentation_index' in metrics
    assert 'bout_durations' in metrics
    
    # Verify all numeric metrics are 0
    assert metrics['event_frequency'] == 0.0
    assert metrics['mean_bout_duration'] == 0.0
    assert metrics['grooming_percentage'] == 0.0
    assert metrics['fragmentation_index'] == 0.0
    
    # Verify bout_durations is empty list
    assert metrics['bout_durations'] == []
    assert isinstance(metrics['bout_durations'], list)


def test_single_long_event():
    """Test single long event with low fragmentation and correct metrics.
    
    Single event: [(0, 899)] = 900 frames = 30 seconds at 30fps
    Total frames: 9000 (5 minutes at 30fps)
    
    Expected metrics:
    - event_frequency: 1 event / 5 min = 0.2 events/min
    - mean_bout_duration: 30.0 seconds
    - grooming_percentage: 900 / 9000 * 100 = 10.0%
    - fragmentation_index: 0.0 (single event, no variation)
    - bout_durations: [30.0]
    """
    events = [(0, 899)]
    metrics = calculate_all_behavioral_metrics(events, 9000, fps=30.0)
    
    # Verify metric values
    assert metrics['event_frequency'] == pytest.approx(0.2)
    assert metrics['mean_bout_duration'] == pytest.approx(30.0)
    assert metrics['grooming_percentage'] == pytest.approx(10.0)
    assert metrics['fragmentation_index'] == 0.0  # Single event = no variation
    assert metrics['bout_durations'] == pytest.approx([30.0])


def test_many_short_events():
    """Test many short events with high fragmentation and correct frequency.
    
    Create 20 short events with varying durations
    Events: [(i*100, i*100 + 5 + i) for i in range(20)]
    - Event 0: 6 frames = 0.2 seconds
    - Event 1: 7 frames = 0.233... seconds
    - Event 2: 8 frames = 0.267... seconds
    - ...
    - Event 19: 25 frames = 0.833... seconds
    
    Total frames: 18000 (10 minutes at 30fps)
    
    Expected metrics:
    - event_frequency: 20 events / 10 min = 2.0 events/min
    - mean_bout_duration: mean of varying durations
    - grooming_percentage: small percentage
    - fragmentation_index: high CV (varied durations)
    - bout_durations: list of 20 durations
    """
    # Create 20 events with increasing durations from 6 to 25 frames
    events = [(i*100, i*100 + 5 + i) for i in range(20)]
    metrics = calculate_all_behavioral_metrics(events, 18000, fps=30.0)
    
    # Verify we have 20 events
    assert len(metrics['bout_durations']) == 20
    
    # Verify frequency: 20 events in 18000 frames = 10 minutes = 2.0 events/min
    assert metrics['event_frequency'] == pytest.approx(2.0)
    
    # Verify high fragmentation (CV should be significant for varying durations)
    # Durations range from 6 to 25 frames, which gives high variability
    assert metrics['fragmentation_index'] > 0.3  # Should show significant fragmentation
    
    # Verify mean bout duration is positive
    assert metrics['mean_bout_duration'] > 0
    
    # Verify grooming percentage is positive but relatively small
    assert 0 < metrics['grooming_percentage'] < 100
    
    # Verify bout_durations are in ascending order (since durations increase)
    durations = metrics['bout_durations']
    for i in range(len(durations) - 1):
        assert durations[i] < durations[i + 1]


# =============================================================================
# Unit 10 Tests: detect_video_fps
# =============================================================================

def test_valid_video_standard_fps(monkeypatch, tmp_path):
    """Test valid video with standard fps (30) returns 30.0."""
    # Create a temporary file to simulate a video
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    # Mock imageio to return fps=30
    class MockReader:
        def get_meta_data(self):
            return {'fps': 30.0}
        
        def close(self):
            pass
    
    def mock_get_reader(video_path):
        return MockReader()
    
    # Import imageio in the test context
    import sys
    from types import ModuleType
    
    # Create a mock imageio module
    mock_imageio = ModuleType('imageio')
    mock_imageio.get_reader = mock_get_reader
    sys.modules['imageio'] = mock_imageio
    
    try:
        fps = detect_video_fps(str(video_file))
        assert fps == 30.0
    finally:
        # Cleanup
        if 'imageio' in sys.modules:
            del sys.modules['imageio']


def test_unusual_fps_below_bound(monkeypatch, tmp_path, capsys):
    """Test video with fps < 10 returns default with warning."""
    # Create a temporary file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    # Mock imageio to return fps < 10
    class MockReader:
        def get_meta_data(self):
            return {'fps': 5.0}  # Below minimum bound of 10
        
        def close(self):
            pass
    
    def mock_get_reader(video_path):
        return MockReader()
    
    import sys
    from types import ModuleType
    
    mock_imageio = ModuleType('imageio')
    mock_imageio.get_reader = mock_get_reader
    sys.modules['imageio'] = mock_imageio
    
    try:
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Check that default was returned
        assert fps == 30.0
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "outside reasonable bounds" in captured.out
        assert "5.0" in captured.out
    finally:
        if 'imageio' in sys.modules:
            del sys.modules['imageio']


def test_unusual_fps_above_bound(monkeypatch, tmp_path, capsys):
    """Test video with fps > 120 returns default with warning."""
    # Create a temporary file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    # Mock imageio to return fps > 120
    class MockReader:
        def get_meta_data(self):
            return {'fps': 240.0}  # Above maximum bound of 120
        
        def close(self):
            pass
    
    def mock_get_reader(video_path):
        return MockReader()
    
    import sys
    from types import ModuleType
    
    mock_imageio = ModuleType('imageio')
    mock_imageio.get_reader = mock_get_reader
    sys.modules['imageio'] = mock_imageio
    
    try:
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Check that default was returned
        assert fps == 30.0
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "outside reasonable bounds" in captured.out
        assert "240.0" in captured.out
    finally:
        if 'imageio' in sys.modules:
            del sys.modules['imageio']


def test_nonexistent_file(capsys):
    """Test non-existent file returns default with warning."""
    fps = detect_video_fps('/nonexistent/path/to/video.mp4', default_fps=25.0)
    
    # Check that default was returned
    assert fps == 25.0
    
    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "does not exist" in captured.out


def test_imageio_import_error_opencv_fallback(monkeypatch, tmp_path, capsys):
    """Test opencv fallback when imageio is not available."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio to raise ImportError
    def mock_import_imageio(name, *args, **kwargs):
        if name == 'imageio':
            raise ImportError("imageio not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock cv2 to return valid fps
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        
        def isOpened(self):
            return self.opened
        
        def get(self, prop):
            return 25.0  # Return valid fps
        
        def release(self):
            pass
    
    mock_cv2 = ModuleType('cv2')
    mock_cv2.VideoCapture = MockVideoCapture
    mock_cv2.CAP_PROP_FPS = 5  # OpenCV constant
    sys.modules['cv2'] = mock_cv2
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_imageio)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should get fps from opencv fallback
        assert fps == 25.0
    finally:
        monkeypatch.undo()
        if 'cv2' in sys.modules:
            del sys.modules['cv2']


def test_imageio_exception_opencv_fallback(monkeypatch, tmp_path):
    """Test opencv fallback when imageio raises exception (corrupted file)."""
    video_file = tmp_path / "corrupted_video.mp4"
    video_file.write_text("corrupted content")
    
    import sys
    from types import ModuleType
    
    # Mock imageio to raise exception (corrupted file)
    class MockReader:
        def get_meta_data(self):
            raise ValueError("Corrupted video file")
        
        def close(self):
            pass
    
    def mock_get_reader(video_path):
        return MockReader()
    
    mock_imageio = ModuleType('imageio')
    mock_imageio.get_reader = mock_get_reader
    sys.modules['imageio'] = mock_imageio
    
    # Mock cv2 to return valid fps
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        
        def isOpened(self):
            return self.opened
        
        def get(self, prop):
            return 60.0
        
        def release(self):
            pass
    
    mock_cv2 = ModuleType('cv2')
    mock_cv2.VideoCapture = MockVideoCapture
    mock_cv2.CAP_PROP_FPS = 5
    sys.modules['cv2'] = mock_cv2
    
    try:
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should get fps from opencv fallback
        assert fps == 60.0
    finally:
        if 'imageio' in sys.modules:
            del sys.modules['imageio']
        if 'cv2' in sys.modules:
            del sys.modules['cv2']


def test_opencv_import_error_fallback(monkeypatch, tmp_path, capsys):
    """Test fallback when both imageio and opencv are not available."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock both imports to fail
    def mock_import_fail(name, *args, **kwargs):
        if name in ['imageio', 'cv2']:
            raise ImportError(f"{name} not installed")
        return original_import(name, *args, **kwargs)
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_fail)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should return default
        assert fps == 30.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()


def test_opencv_exception_handling(monkeypatch, tmp_path, capsys):
    """Test opencv exception handling (e.g., corrupted file for opencv)."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio to fail
    def mock_import_imageio(name, *args, **kwargs):
        if name == 'imageio':
            raise ImportError("imageio not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock cv2 to raise exception
    class MockVideoCapture:
        def __init__(self, path):
            raise RuntimeError("OpenCV cannot open file")
    
    mock_cv2 = ModuleType('cv2')
    mock_cv2.VideoCapture = MockVideoCapture
    mock_cv2.CAP_PROP_FPS = 5
    sys.modules['cv2'] = mock_cv2
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_imageio)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should return default
        assert fps == 30.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'cv2' in sys.modules:
            del sys.modules['cv2']


def test_tifffile_fallback_for_tif(monkeypatch, tmp_path, capsys):
    """Test tifffile attempt for .tif files when other methods fail."""
    video_file = tmp_path / "microscopy_stack.tif"
    video_file.write_text("dummy tiff content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio and cv2 to fail
    def mock_import_fail_video_libs(name, *args, **kwargs):
        if name in ['imageio', 'cv2']:
            raise ImportError(f"{name} not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock tifffile (doesn't provide fps, but shows we tried)
    class MockTiffFile:
        def __init__(self, path):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    mock_tifffile = ModuleType('tifffile')
    mock_tifffile.TiffFile = MockTiffFile
    sys.modules['tifffile'] = mock_tifffile
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_fail_video_libs)
        
        fps = detect_video_fps(str(video_file), default_fps=25.0)
        
        # Should return default (tifffile doesn't have fps metadata)
        assert fps == 25.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'tifffile' in sys.modules:
            del sys.modules['tifffile']


def test_tifffile_import_error(monkeypatch, tmp_path, capsys):
    """Test when tifffile is not available for .tiff files."""
    video_file = tmp_path / "microscopy_stack.tiff"
    video_file.write_text("dummy tiff content")
    
    import sys
    import builtins
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock all imports to fail
    def mock_import_fail_all(name, *args, **kwargs):
        if name in ['imageio', 'cv2', 'tifffile']:
            raise ImportError(f"{name} not installed")
        return original_import(name, *args, **kwargs)
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_fail_all)
        
        fps = detect_video_fps(str(video_file), default_fps=25.0)
        
        # Should return default
        assert fps == 25.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()


def test_tifffile_exception_handling(monkeypatch, tmp_path, capsys):
    """Test tifffile exception handling (e.g., corrupted TIFF)."""
    video_file = tmp_path / "corrupted.tif"
    video_file.write_text("corrupted tiff content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio and cv2 to fail
    def mock_import_fail_video_libs(name, *args, **kwargs):
        if name in ['imageio', 'cv2']:
            raise ImportError(f"{name} not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock tifffile to raise exception
    class MockTiffFile:
        def __init__(self, path):
            raise ValueError("Corrupted TIFF file")
    
    mock_tifffile = ModuleType('tifffile')
    mock_tifffile.TiffFile = MockTiffFile
    sys.modules['tifffile'] = mock_tifffile
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_fail_video_libs)
        
        fps = detect_video_fps(str(video_file), default_fps=25.0)
        
        # Should return default
        assert fps == 25.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'tifffile' in sys.modules:
            del sys.modules['tifffile']


def test_opencv_closed_video_capture(monkeypatch, tmp_path, capsys):
    """Test when opencv VideoCapture fails to open (isOpened returns False)."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio to fail
    def mock_import_imageio(name, *args, **kwargs):
        if name == 'imageio':
            raise ImportError("imageio not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock cv2 with VideoCapture that fails to open
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = False
        
        def isOpened(self):
            return False  # Failed to open
        
        def release(self):
            pass
    
    mock_cv2 = ModuleType('cv2')
    mock_cv2.VideoCapture = MockVideoCapture
    mock_cv2.CAP_PROP_FPS = 5
    sys.modules['cv2'] = mock_cv2
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_imageio)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should return default
        assert fps == 30.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'cv2' in sys.modules:
            del sys.modules['cv2']


def test_opencv_zero_fps(monkeypatch, tmp_path, capsys):
    """Test when opencv returns 0 or negative fps."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio to fail
    def mock_import_imageio(name, *args, **kwargs):
        if name == 'imageio':
            raise ImportError("imageio not installed")
        return original_import(name, *args, **kwargs)
    
    # Mock cv2 to return 0 fps
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        
        def isOpened(self):
            return True
        
        def get(self, prop):
            return 0.0  # Invalid fps
        
        def release(self):
            pass
    
    mock_cv2 = ModuleType('cv2')
    mock_cv2.VideoCapture = MockVideoCapture
    mock_cv2.CAP_PROP_FPS = 5
    sys.modules['cv2'] = mock_cv2
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_imageio)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should return default
        assert fps == 30.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'cv2' in sys.modules:
            del sys.modules['cv2']


def test_imageio_generic_exception(monkeypatch, tmp_path, capsys):
    """Test that generic exceptions from imageio are caught and handled gracefully."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy video content")
    
    import sys
    import builtins
    from types import ModuleType
    
    # Store original import
    original_import = builtins.__import__
    
    # Mock imageio to raise a generic exception (not ImportError)
    class MockReader:
        def get_meta_data(self):
            raise RuntimeError("Unexpected error reading video metadata")
        
        def close(self):
            pass
    
    def mock_get_reader(video_path):
        return MockReader()
    
    mock_imageio = ModuleType('imageio')
    mock_imageio.get_reader = mock_get_reader
    sys.modules['imageio'] = mock_imageio
    
    # Also mock cv2 and tifffile to fail so we test the full fallback path
    def mock_import_no_cv2_tifffile(name, *args, **kwargs):
        if name in ['cv2', 'tifffile']:
            raise ImportError(f"{name} not installed")
        return original_import(name, *args, **kwargs)
    
    try:
        monkeypatch.setattr(builtins, '__import__', mock_import_no_cv2_tifffile)
        
        fps = detect_video_fps(str(video_file), default_fps=30.0)
        
        # Should return default because imageio raised exception
        assert fps == 30.0
        
        # Check warning message
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Could not detect fps" in captured.out
    finally:
        monkeypatch.undo()
        if 'imageio' in sys.modules:
            del sys.modules['imageio']
