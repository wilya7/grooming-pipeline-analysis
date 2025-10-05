"""Common library functions for neuroscience grooming analysis pipeline."""

from typing import List, Tuple, Dict
from pathlib import Path

import pandas as pd
import numpy as np
import json


# =============================================================================
# Unit 1: Load Event CSV
# =============================================================================

def load_event_csv(filepath: str, validate: bool = True) -> List[Tuple[int, int]]:
    """
    Load and validate grooming events from standard CSV format.
    
    Events are represented as consecutive pairs of frames where the first frame
    indicates the start of a grooming bout and the second frame indicates the end.
    
    Args:
        filepath: Path to the event-based CSV file
        validate: Whether to validate event integrity (default: True)
    
    Returns:
        List of (start_frame, end_frame) tuples representing grooming events
    
    Raises:
        ValueError: If validation fails or 'Frame' column is missing
    
    Example:
        >>> events = load_event_csv('grooming_events.csv')
        >>> print(events)
        [(100, 150), (200, 250)]
    """
    # Read the CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Check for 'Frame' column
    if 'Frame' not in df.columns:
        raise ValueError("CSV file must contain a 'Frame' column")
    
    # Extract frame numbers
    frames = df['Frame'].dropna().astype(int).tolist()
    
    # Handle empty CSV
    if len(frames) == 0:
        return []
    
    # Validation checks
    if validate:
        # Check for even number of frames
        if len(frames) % 2 != 0:
            raise ValueError(
                f"Invalid event pairing: found {len(frames)} frames, "
                f"but events require an even number of frames (pairs of start/end)"
            )
        
        # Check each pair: start < end (do this BEFORE chronological check)
        for i in range(0, len(frames), 2):
            start_frame = frames[i]
            end_frame = frames[i + 1]
            if start_frame >= end_frame:
                raise ValueError(
                    f"Invalid event at frames {start_frame}-{end_frame}: "
                    f"start frame must be less than end frame"
                )
        
        # Check chronological order
        for i in range(len(frames) - 1):
            if frames[i] >= frames[i + 1]:
                raise ValueError(
                    f"Frames must be in chronological order: "
                    f"frame at index {i} ({frames[i]}) >= frame at index {i+1} ({frames[i+1]})"
                )
    
    # Create pairs of (start_frame, end_frame)
    # Only pair complete pairs (handle odd numbers gracefully when validate=False)
    num_complete_pairs = len(frames) // 2
    events = [(frames[i], frames[i + 1]) for i in range(0, num_complete_pairs * 2, 2)]
    
    return events


# =============================================================================
# Unit 2: Calculate Event Frequency
# =============================================================================

def calculate_event_frequency(events: List[Tuple[int, int]], total_frames: int, fps: float = 30.0) -> float:
    """
    Calculate normalized event frequency (events per minute).
    
    Args:
        events: List of (start_frame, end_frame) tuples
        total_frames: Total number of frames analyzed
        fps: Frames per second (default: 30.0)
    
    Returns:
        Frequency in events per minute
    
    Formula:
        total_minutes = total_frames / fps / 60
        frequency = number_of_events / total_minutes
    
    Edge cases:
        - Returns 0.0 if total_frames <= 0 (avoids division by zero)
        - Returns 0.0 if no events present
    """
    # Handle edge cases
    if total_frames <= 0 or len(events) == 0:
        return 0.0
    
    # Calculate total time in minutes
    total_minutes = total_frames / fps / 60.0
    
    # Calculate frequency (events per minute)
    frequency = len(events) / total_minutes
    
    return frequency


# =============================================================================
# Unit 3: Calculate Mean Bout Duration
# =============================================================================

def calculate_mean_bout_duration(events: List[Tuple[int, int]], fps: float = 30.0) -> float:
    """
    Calculate mean duration of grooming bouts in seconds.
    
    Args:
        events: List of (start_frame, end_frame) tuples representing grooming events
        fps: Frames per second (default: 30.0)
    
    Returns:
        Mean bout duration in seconds
    
    Formula:
        For each event (start, end): duration_frames = end - start + 1
        mean_frames = mean(duration_frames)
        mean_duration_seconds = mean_frames / fps
    
    Edge cases:
        - Returns 0.0 if events list is empty
        - Duration calculation is inclusive: frame 100 to 100 = 1 frame
    
    Example:
        >>> events = [(0, 29), (100, 159), (200, 289)]
        >>> calculate_mean_bout_duration(events, fps=30.0)
        2.0
    """
    # Handle empty list edge case
    if len(events) == 0:
        return 0.0
    
    # Calculate duration for each event (inclusive: end - start + 1)
    durations = [end - start + 1 for start, end in events]
    
    # Calculate mean duration in frames
    mean_frames = np.mean(durations)
    
    # Convert to seconds
    mean_duration_seconds = mean_frames / fps
    
    return float(mean_duration_seconds)


# =============================================================================
# Unit 4: Calculate Grooming Percentage
# =============================================================================

def calculate_grooming_percentage(events: List[Tuple[int, int]], total_frames: int) -> float:
    """
    Calculate proportion of time spent grooming as percentage.
    
    Args:
        events: List of (start_frame, end_frame) tuples representing grooming events
        total_frames: Total number of frames in video
    
    Returns:
        Percentage of time spent grooming (0-100)
    
    Formula:
        For each event (start, end): grooming_frames = end - start + 1
        Total grooming frames = sum of all grooming_frames (counting unique frames)
        Percentage = (total_grooming_frames / total_frames) * 100
    
    Edge cases:
        - Returns 0.0 if total_frames <= 0 (avoids division by zero)
        - Returns 0.0 if no events present
        - Handles overlapping events by counting unique frames (merges intervals)
        - Result is bounded between 0 and 100
    
    Example:
        >>> events = [(0, 299), (500, 599), (1000, 1499)]
        >>> calculate_grooming_percentage(events, 9000)
        10.0
    """
    # Handle edge cases
    if total_frames <= 0 or len(events) == 0:
        return 0.0
    
    # Sort events by start frame to handle potential overlaps efficiently
    sorted_events = sorted(events, key=lambda x: x[0])
    
    # Merge overlapping intervals to count each frame only once
    merged_events = []
    for start, end in sorted_events:
        if merged_events and start <= merged_events[-1][1] + 1:
            # Overlapping or adjacent - merge with previous interval
            merged_events[-1] = (merged_events[-1][0], max(merged_events[-1][1], end))
        else:
            # Non-overlapping - add as new interval
            merged_events.append((start, end))
    
    # Calculate total grooming frames from merged intervals (inclusive duration)
    total_grooming_frames = sum(end - start + 1 for start, end in merged_events)
    
    # Calculate percentage
    percentage = (total_grooming_frames / total_frames) * 100
    
    # Bound the result between 0 and 100
    percentage = max(0.0, min(100.0, percentage))
    
    return percentage


# =============================================================================
# Unit 5: Calculate Fragmentation Index
# =============================================================================

def calculate_fragmentation_index(events: List[Tuple[int, int]]) -> float:
    """
    Calculate variability in grooming bout lengths as coefficient of variation (CV).
    
    The fragmentation index quantifies how variable grooming bout durations are.
    Higher values indicate more fragmented grooming behavior with inconsistent
    bout lengths, while lower values indicate more uniform grooming patterns.
    
    Args:
        events: List of (start_frame, end_frame) tuples representing grooming events
    
    Returns:
        Coefficient of variation (CV) of bout durations
    
    Formula:
        For each event (start, end): duration = end - start + 1
        CV = standard_deviation / mean
    
    Interpretation:
        - CV = 0: All bouts are identical length (no fragmentation)
        - CV < 1: Low variability (standard deviation less than mean)
        - CV ≥ 1: High variability (standard deviation exceeds mean)
    
    Edge cases:
        - Returns 0.0 if events list is empty
        - Returns 0.0 if only one event (no variation possible)
        - Returns 0.0 if all events have identical duration
        - Returns 0.0 if mean is 0 (shouldn't happen with valid events)
    
    Example:
        >>> events = [(0, 9), (100, 149), (200, 299)]
        >>> calculate_fragmentation_index(events)
        0.6903
    """
    # Handle edge case: empty events
    if len(events) == 0:
        return 0.0
    
    # Handle edge case: single event (no variation possible)
    if len(events) == 1:
        return 0.0
    
    # Calculate duration for each event (inclusive: end - start + 1)
    durations = [end - start + 1 for start, end in events]
    
    # Calculate mean and standard deviation
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)  # ddof=0 by default for population std
    
    # Handle edge case: mean is 0 (shouldn't happen with valid events)
    if mean_duration == 0:
        return 0.0
    
    # Calculate coefficient of variation
    cv = std_duration / mean_duration
    
    return float(cv)


# =============================================================================
# Unit 6: Process Directory Structure
# =============================================================================

def process_directory_structure(
    base_dir: str,
    file_extension: str = '.csv',
    min_files_warning: int = 10
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Process directory structure and identify genotypes with their associated files.
    
    Expected directory structure:
        base_dir/
          ├── genotype_1/
          │   ├── file1.csv
          │   ├── file2.csv
          │   └── ...
          ├── genotype_2/
          │   ├── file1.csv
          │   └── ...
          └── ...
    
    Args:
        base_dir: Path to directory containing genotype subdirectories
        file_extension: Extension to search for (default: '.csv')
        min_files_warning: Minimum files before warning (default: 10)
    
    Returns:
        Tuple containing:
        - structure: Dictionary mapping genotype names to list of file paths
        - warnings: List of warning messages
    
    Raises:
        FileNotFoundError: If base_dir does not exist
    
    Example:
        >>> structure, warnings = process_directory_structure('/data/grooming/')
        >>> print(structure)
        {'WT': ['/data/grooming/WT/mouse1.csv', '/data/grooming/WT/mouse2.csv'],
         'KO': ['/data/grooming/KO/mouse1.csv']}
        >>> print(warnings)
        ['KO has only 1 files (minimum recommended: 10)']
    """
    base_path = Path(base_dir)
    
    # Check if base directory exists
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    if not base_path.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {base_dir}")
    
    structure: Dict[str, List[str]] = {}
    warnings: List[str] = []
    
    # Get all immediate subdirectories
    subdirectories = [d for d in base_path.iterdir() if d.is_dir()]
    
    # Check if no subdirectories found
    if len(subdirectories) == 0:
        warnings.append(f"No subdirectories found in {base_dir}")
        return structure, warnings
    
    # Process each subdirectory
    for subdir in subdirectories:
        genotype_name = subdir.name
        
        # Find all files matching the extension
        # Use glob pattern for the specific extension
        pattern = f"*{file_extension}"
        matching_files = list(subdir.glob(pattern))
        
        # Convert to absolute path strings
        file_paths = [str(f.absolute()) for f in matching_files]
        
        # Handle empty subdirectory
        if len(file_paths) == 0:
            warnings.append(
                f"Subdirectory '{genotype_name}' is empty "
                f"(no {file_extension} files found), excluding from results"
            )
            continue
        
        # Add to structure
        structure[genotype_name] = file_paths
        
        # Check if below minimum file threshold
        if len(file_paths) < min_files_warning:
            warnings.append(
                f"Genotype '{genotype_name}' has only {len(file_paths)} file(s) "
                f"(minimum recommended: {min_files_warning})"
            )
    
    return structure, warnings


# =============================================================================
# Unit 7: Validate Frame Counts
# =============================================================================

def validate_frame_counts(
    frame_counts: List[int],
    expected_frames: int,
    variation_threshold: float = 0.1
) -> Dict:
    """
    Validate frame count consistency across videos.
    
    Calculates variation in frame counts and determines if videos have
    consistent frame counts within an acceptable threshold.
    
    Args:
        frame_counts: List of actual frame counts from videos
        expected_frames: Expected frame count (for reference)
        variation_threshold: Maximum acceptable variation (default: 0.1)
    
    Returns:
        Dictionary containing validation results with keys:
        - 'valid': Whether variation is within threshold
        - 'min': Minimum frame count
        - 'max': Maximum frame count
        - 'mean': Mean frame count
        - 'variation_percent': Variation as percentage
        - 'expected': The expected frame count (for reference)
    
    Formula:
        variation = (max_count - min_count) / mean_count
        valid = variation <= variation_threshold
    
    Edge cases:
        - Empty list: Returns valid=False with zero values
        - All counts identical: variation = 0.0, valid = True
        - Mean is zero: variation = 0.0 (defensive programming)
    
    Example:
        >>> result = validate_frame_counts([8900, 9000, 9100], 9000, 0.1)
        >>> print(result['valid'])
        True
        >>> print(result['variation_percent'])
        2.22
    """
    # Handle empty list edge case
    if len(frame_counts) == 0:
        return {
            'valid': False,
            'min': 0,
            'max': 0,
            'mean': 0.0,
            'variation_percent': 0.0,
            'expected': expected_frames
        }
    
    # Calculate statistics
    min_count = min(frame_counts)
    max_count = max(frame_counts)
    mean_count = np.mean(frame_counts)
    
    # Calculate variation
    if mean_count == 0:
        # Edge case: all counts are zero (defensive programming)
        variation = 0.0
    else:
        variation = (max_count - min_count) / mean_count
    
    # Check if valid (explicitly convert numpy bool to Python bool)
    valid = bool(variation <= variation_threshold)
    
    # Return results
    return {
        'valid': valid,
        'min': int(min_count),
        'max': int(max_count),
        'mean': float(mean_count),
        'variation_percent': float(variation * 100),
        'expected': expected_frames
    }


# =============================================================================
# Unit 8: Write JSON Output
# =============================================================================

def write_json_output(data: Dict, filepath: str, indent: int = 2) -> bool:
    """
    Write dictionary data to JSON file with error handling.
    
    Creates parent directories if they don't exist and writes the data
    with proper formatting. Handles serialization errors and file system
    permissions gracefully.
    
    Args:
        data: Dictionary data to write to JSON file
        filepath: Output file path for the JSON file
        indent: Indentation level for JSON formatting (default: 2)
    
    Returns:
        True if write operation was successful
    
    Raises:
        TypeError: If data contains non-serializable objects
        PermissionError: If write to protected directory is denied
        OSError: For other file system errors
    
    Example:
        >>> data = {
        ...     "experiment": "grooming_analysis",
        ...     "parameters": {"window_size": 300},
        ...     "results": [1.2, 3.4, 5.6]
        ... }
        >>> success = write_json_output(data, "output/results.json")
        >>> print(success)
        True
    """
    # Convert filepath to Path object for easier manipulation
    file_path = Path(filepath)
    
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Attempt to serialize data first to catch TypeError early
        try:
            json_string = json.dumps(data, indent=indent)
        except TypeError as e:
            # Re-raise with more informative message
            raise TypeError(f"Data contains non-serializable objects: {e}")
        
        # Write JSON to file using context manager
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        
        return True
    
    except PermissionError as e:
        # Re-raise permission errors with context
        raise PermissionError(f"Permission denied writing to {filepath}: {e}")
    
    except OSError as e:
        # Re-raise other OS errors (disk full, invalid path, etc.)
        raise OSError(f"File system error writing to {filepath}: {e}")


# =============================================================================
# Unit 9: Calculate All Behavioral Metrics
# =============================================================================

def calculate_all_behavioral_metrics(
    events: List[Tuple[int, int]],
    total_frames: int,
    fps: float = 30.0
) -> Dict[str, float]:
    """
    Calculate complete set of behavioral metrics in one call.
    
    This function provides a comprehensive analysis of grooming behavior by
    calculating all five key behavioral metrics from a single events list.
    
    Args:
        events: List of (start_frame, end_frame) tuples representing grooming events
        total_frames: Total number of frames in video
        fps: Frames per second (default: 30.0)
    
    Returns:
        Dictionary containing all behavioral metrics:
        - 'event_frequency': Events per minute
        - 'mean_bout_duration': Mean bout duration in seconds
        - 'grooming_percentage': Percentage of time spent grooming (0-100)
        - 'fragmentation_index': Coefficient of variation of bout durations
        - 'bout_durations': List of individual bout durations in seconds
    
    Edge cases:
        - Empty events list: All numeric metrics return 0.0, bout_durations = []
        - Single event: Valid metrics with fragmentation_index = 0.0
    
    Example:
        >>> events = [(0, 29), (100, 159), (200, 289)]
        >>> metrics = calculate_all_behavioral_metrics(events, 9000, fps=30.0)
        >>> print(metrics['event_frequency'])
        0.6
        >>> print(metrics['mean_bout_duration'])
        2.0
    """
    # Calculate individual bout durations in seconds
    bout_durations = [(end - start + 1) / fps for start, end in events]
    
    # Calculate all behavioral metrics using existing functions
    event_frequency = calculate_event_frequency(events, total_frames, fps)
    mean_bout_duration = calculate_mean_bout_duration(events, fps)
    grooming_percentage = calculate_grooming_percentage(events, total_frames)
    fragmentation_index = calculate_fragmentation_index(events)
    
    # Return comprehensive metrics dictionary
    return {
        'event_frequency': event_frequency,
        'mean_bout_duration': mean_bout_duration,
        'grooming_percentage': grooming_percentage,
        'fragmentation_index': fragmentation_index,
        'bout_durations': bout_durations
    }


# =============================================================================
# Unit 10: Detect Video Frame Rate
# =============================================================================

def detect_video_fps(video_path: str, default_fps: float = 30.0) -> float:
    """
    Safely detect video frame rate with bounds checking.
    
    Attempts to detect the frame rate from video metadata. If detection fails
    or the detected fps is outside reasonable bounds (10-120 fps), returns the
    default value with a warning.
    
    Args:
        video_path: Path to video file (supports common video formats and TIF/TIFF)
        default_fps: Default frame rate to return if detection fails (default: 30.0)
    
    Returns:
        Detected fps if valid, otherwise default_fps
    
    Reasonable FPS Bounds:
        - Minimum: 10 fps (below this likely indicates incorrect metadata)
        - Maximum: 120 fps (above this likely indicates incorrect metadata)
    
    Edge Cases:
        - Non-existent file → return default with warning
        - Non-video file → return default with warning
        - Corrupted file → return default with warning
        - No exceptions propagate (graceful degradation)
    
    Example:
        >>> fps = detect_video_fps('experiment_video.mp4')
        >>> print(fps)
        30.0
        
        >>> fps = detect_video_fps('microscopy_stack.tif', default_fps=25.0)
        >>> print(fps)
        25.0
    """
    MIN_FPS = 10.0
    MAX_FPS = 120.0
    
    # Check if file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Warning: File does not exist: {video_path}. Using default fps: {default_fps}")
        return default_fps
    
    detected_fps = None
    
    # Try imageio for video reading (works for many formats including TIF)
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        metadata = reader.get_meta_data()
        if 'fps' in metadata and metadata['fps'] is not None:
            detected_fps = float(metadata['fps'])
        reader.close()
    except ImportError:
        pass  # imageio not available
    except Exception:
        pass  # Any error with imageio (corrupted file, non-video, etc.)
    
    # Try opencv as fallback
    if detected_fps is None:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps_value = cap.get(cv2.CAP_PROP_FPS)
                if fps_value > 0:
                    detected_fps = float(fps_value)
                cap.release()
        except ImportError:
            pass  # opencv not available
        except Exception:
            pass  # Any error with opencv
    
    # Try tifffile for TIF/TIFF files
    if detected_fps is None and video_file.suffix.lower() in ['.tif', '.tiff']:
        try:
            import tifffile
            # tifffile doesn't typically provide fps, but we try anyway
            with tifffile.TiffFile(video_path) as tif:
                # Most microscopy TIFFs don't have fps metadata
                # This is here for completeness but will likely not find fps
                pass
        except ImportError:
            pass  # tifffile not available
        except Exception:
            pass  # Any error with tifffile
    
    # Check if detection was successful and within bounds
    if detected_fps is not None:
        if MIN_FPS <= detected_fps <= MAX_FPS:
            return detected_fps
        else:
            print(f"Warning: Detected fps {detected_fps} is outside reasonable bounds "
                  f"({MIN_FPS}-{MAX_FPS}). Using default fps: {default_fps}")
            return default_fps
    
    # Detection failed
    print(f"Warning: Could not detect fps from {video_path}. Using default fps: {default_fps}")
    return default_fps
