Here's an enhanced README.md with a comprehensive tutorial section:

```markdown
# grooming-pipeline-analysis (~/Nextcloud/coding projects/ppe)

Pipeline for optimizing manual scoring of Drosophila grooming behavior through intelligent sampling strategies.

## Overview

This pipeline solves the problem of time-intensive manual annotation of Drosophila grooming videos (1 hour per 9000-frame video) by determining the minimal sampling strategy needed to maintain statistical power while dramatically reducing analysis time.

The pipeline consists of three scripts that work with an external annotation tool ([fly_behavior_analysis](https://github.com/example/fly_behavior_analysis)):

1. **Pilot Optimizer** - Analyzes fully-scored pilot data to find optimal sampling parameters
2. **Video Sampler** - Creates sampled videos based on optimal parameters
3. **Statistical Analyzer** - Analyzes scored sampled data and performs statistical comparisons

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate grooming-pipeline-analysis-3e3f790d

# Or using pip
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/                         # Main pipeline scripts
│   ├── common/                  # Shared library functions
│   ├── pilot_grooming_optimizer.py
│   ├── video_sampler.py
│   └── grooming_statistical_analyzer.py
├── tests/                       # Test files
├── data/                        # Data directories
│   ├── pilot_data/             # Pilot CSV data (input for Step 1)
│   │   ├── wild_type/          # One subdirectory per genotype
│   │   │   ├── fly_001.csv    # Fully annotated CSVs
│   │   │   ├── fly_002.csv
│   │   │   └── ...
│   │   └── mutant_X/
│   │       ├── fly_001.csv
│   │       └── ...
│   ├── videos/                 # Raw video files (input for Step 2)
│   │   ├── wild_type/
│   │   │   ├── fly_001.tif
│   │   │   └── ...
│   │   └── mutant_X/
│   │       └── ...
│   └── sampled/                # Output from sampling (Step 2 → Step 3)
│       ├── wild_type/
│       └── mutant_X/
├── notebooks/                   # Jupyter notebooks
├── reports/                     # Generated reports
└── docs/                        # Documentation
```

## Pipeline Workflow

### Complete Pipeline Flow

```
PREREQUISITE: Fully annotate pilot videos using fly_behavior_analysis
     ↓
[Step 1] pilot_grooming_optimizer.py
     Analyzes pilot CSVs → Finds optimal parameters
     ↓
[Step 2] video_sampler.py
     Samples full videos → Creates sampled_video.tif
     ↓
MANUAL STEP: Annotate sampled_video.tif using fly_behavior_analysis
              (Save as 'scored_sampled.csv')
     ↓
[Step 3] grooming_statistical_analyzer.py
     Analyzes scored data → Generates statistical results
```

## Data Formats

### Input: Event-Based CSV Format

All grooming annotations use this format (output from fly_behavior_analysis):

```csv
Frame
234     # Start of first grooming event
567     # End of first grooming event
1200    # Start of second grooming event
1456    # End of second grooming event
```

- Single column labeled 'Frame'
- Alternating start/stop frame numbers
- Even number of rows required
- Frame numbers must be increasing

### Directory Structure Convention

Folder names indicate genetic background:

```
experiment/
├── wild_type/       # Genetic background 1 (folder name = genotype label)
│   ├── fly_001.tif
│   └── fly_002.tif
└── mutant_X/        # Genetic background 2
    ├── fly_001.tif
    └── fly_002.tif
```

## Detailed Usage

### Step 1: Pilot Data Optimization

**Purpose:** Analyze fully-annotated pilot data to determine optimal sampling parameters.

**Prerequisites:**
- Pilot videos must be fully annotated using fly_behavior_analysis
- At least 2 genotypes required
- Recommended 10+ flies per genotype (warnings if fewer)

**Input Location:**
```
data/pilot_data/
├── wild_type/
│   ├── fly_001.csv    # From fly_behavior_analysis
│   ├── fly_002.csv
│   └── ... (10+ recommended)
└── mutant_X/
    ├── fly_001.csv
    └── ... (10+ recommended)
```

**Command:**
```bash
python src/pilot_grooming_optimizer.py \
    --data-dir data/pilot_data/ \
    --output optimization_results.json \
    --expected-frames 9000 \
    --alpha 0.05 \
    --power 0.8
```

**Parameters:**
- `--data-dir`: Directory containing genotype subdirectories with CSV files
- `--output`: Output path for optimization results JSON
- `--expected-frames`: Expected frame count per video (default: 9000)
- `--alpha`: Significance level (default: 0.05)
- `--power`: Target statistical power (default: 0.8)

**Outputs Generated:**

1. **optimization_results.json** (required for Step 2)
   ```json
   {
     "window_size": 300,
     "sampling_rate": 0.15,
     "sampling_strategy": "stratified",
     "recommended_edge_duration_min": 20
   }
   ```

2. **optimization_log.json** (detailed analysis log)
   - Complete pilot data summary
   - All tested parameter combinations
   - Performance metrics for each combination
   - Warnings and validation results

3. **optimization_report.pdf** (comprehensive report)
   - Executive summary
   - Methodology (including composite score formula: 25/25/20/15/15)
   - All visualizations
   - Validation metrics
   - Recommendations

4. **plots/** directory (visualizations)
   - `heatmap_*.pdf` - Composite scores across parameters
   - `power_curves.pdf` - Statistical power vs sampling rate
   - `pareto_frontier.pdf` - Efficiency vs power tradeoff
   - `bias_assessment.pdf` - Bias distribution by strategy
   - `strategy_comparison.pdf` - Performance comparison

**Typical Runtime:** 2-4 hours for 25 pilot videos

---

### Step 2: Video Sampling

**Purpose:** Create sampled videos containing only selected time windows based on optimal parameters.

**Prerequisites:**
- Completed Step 1 (optimization_results.json exists)
- Raw TIF videos available

**Input Location:**
```
data/videos/wild_type/
├── fly_001.tif
├── fly_002.tif
└── ... (full videos, ~900MB each)
```

**Command:**
```bash
python src/video_sampler.py \
    --video-dir data/videos/wild_type/ \
    --from-optimization optimization_results.json \
    --seed 42
```

**Parameters:**
- `--video-dir`: Directory containing TIF videos (folder name = genotype)
- `--from-optimization`: JSON file from Step 1 (required)
- `--seed`: Random seed for reproducibility

**Outputs Generated in Same Directory as Input:**

```
data/videos/wild_type/
├── fly_001.tif              # Original (unchanged)
├── fly_002.tif              # Original (unchanged)
├── sampled_video.tif        # NEW: For annotation
├── sampling_metadata.csv    # NEW: Frame mapping
└── group_info.json          # NEW: Sampling parameters
```

1. **sampled_video.tif**
   - Contains selected time windows from all videos
   - Black separator frames between segments (4 seconds each)
   - Ready for annotation with fly_behavior_analysis
   - **⚠️ CRITICAL: Must be annotated and saved as 'scored_sampled.csv'**

2. **sampling_metadata.csv**
   ```csv
   source_video,window_index,start_frame_original,end_frame_original,start_frame_sampled,end_frame_sampled,actual_total_frames,frame_type
   SEPARATOR,NA,NA,NA,0,119,NA,separator
   fly_001.tif,2,600,899,120,419,9000,content
   fly_001.tif,5,1500,1799,420,719,9000,content
   SEPARATOR,NA,NA,NA,720,839,NA,separator
   fly_002.tif,1,300,599,840,1139,8950,content
   ```

3. **group_info.json**
   ```json
   {
     "genetic_background": "wild_type",
     "n_videos": 10,
     "sampling_parameters": {
       "window_size": 300,
       "sampling_rate": 0.15,
       "sampling_strategy": "stratified",
       "recommended_edge_duration_min": 20,
       "random_seed": 42
     }
   }
   ```

**Typical Runtime:** 20-30 minutes for 10 videos

---

### MANUAL STEP: Annotate Sampled Video

**⚠️ CRITICAL: This step must be completed before Step 3**

1. Open `sampled_video.tif` in fly_behavior_analysis
2. Annotate all grooming events (ignore separator frames)
3. **Save the output as exactly `scored_sampled.csv`** in the same directory
4. Repeat for each genotype group

**Expected Result:**
```
data/videos/wild_type/
├── ...
├── sampled_video.tif
├── scored_sampled.csv       # NEW: From fly_behavior_analysis
├── sampling_metadata.csv
└── group_info.json
```

**File Format:** Same event-based CSV format as pilot data (alternating start/stop frames)

---

### Step 3: Statistical Analysis

**Purpose:** Analyze manually-scored sampled data and perform statistical comparisons between genotypes.

**Prerequisites:**
- Completed Step 2 for all genotypes
- Manual annotation completed (scored_sampled.csv exists)

**Input Location:**
```
data/videos/wild_type/
├── scored_sampled.csv       # From fly_behavior_analysis (REQUIRED)
├── sampling_metadata.csv    # From Step 2
└── group_info.json          # From Step 2

data/videos/mutant_X/
├── scored_sampled.csv       # From fly_behavior_analysis (REQUIRED)
├── sampling_metadata.csv    # From Step 2
└── group_info.json          # From Step 2
```

**Command (Two-Group Comparison):**
```bash
python src/grooming_statistical_analyzer.py \
    --group-1-dir data/videos/wild_type/ \
    --group-2-dir data/videos/mutant_X/ \
    --output-dir reports/ \
    --generate-plots
```

**Command (Single-Group Analysis):**
```bash
python src/grooming_statistical_analyzer.py \
    --group-1-dir data/videos/wild_type/ \
    --output-dir reports/ \
    --generate-plots
```

**Parameters:**
- `--group-1-dir`: Directory with scored_sampled.csv for group 1
- `--group-2-dir`: Directory with scored_sampled.csv for group 2 (optional)
- `--output-dir`: Results directory (default: ./results)
- `--generate-plots`: Generate visualization plots (optional)

**Outputs Generated:**

```
reports/
├── summary_statistics.csv
├── individual_fly_metrics.csv
├── pairwise_comparisons.csv      # Only in two-group mode
├── edge_event_sensitivity.csv
├── analysis_report.html
└── plots/                         # If --generate-plots specified
    ├── raster_plot.pdf
    ├── cumulative_curves.pdf
    ├── bout_distributions.pdf
    ├── effect_sizes.pdf           # Only in two-group mode
    ├── edge_sensitivity.pdf
    └── reliability_indicators.pdf
```

1. **summary_statistics.csv** - Group-level descriptive statistics
   ```csv
   metric,group,n,mean,std,sem,median,iqr,min,max,fallback_used
   num_events,wild_type,10,23,5,1.6,22,7,15,32,No
   grooming_frequency,wild_type,10,2.3,0.5,0.16,2.2,0.7,1.5,3.2,No
   mean_bout_duration,wild_type,10,1.8,0.3,0.09,1.7,0.4,1.2,2.5,Yes*
   ...
   ```
   *Asterisk indicates edge event fallback was used

2. **individual_fly_metrics.csv** - Per-fly detailed metrics
   ```csv
   video,group,num_events,grooming_frequency,mean_bout_duration,n_complete_events,n_edge_events,actual_frames,duration_reliability_flag
   fly_001.tif,wild_type,21,2.1,1.7,12,3,9000,High
   fly_002.tif,wild_type,25,2.5,1.9,3,4,8950,Fallback
   ...
   ```

3. **pairwise_comparisons.csv** - Statistical test results (two-group mode)
   ```csv
   metric,test_used,statistic,p_value,p_value_corrected,effect_size,effect_size_type,significant
   grooming_frequency,t-test,3.45,0.003,0.015,0.98,Cohen_d,True
   mean_bout_duration,Mann-Whitney,28,0.041,0.103,0.42,Cliff_delta,False
   ...
   ```
   - FDR correction using Benjamini-Hochberg method
   - Automatic test selection (t-test for normal, Mann-Whitney for non-normal)

4. **edge_event_sensitivity.csv** - Impact of edge event handling
   ```csv
   metric,strategy,mean_value,std_value,bias_vs_complete,edge_threshold_used
   grooming_frequency,all_events,2.3,0.5,0.0,NA
   mean_bout_duration,complete_only,1.8,0.3,0.0,NA
   mean_bout_duration,with_edge_min20,1.7,0.35,-0.06,20
   ```

5. **analysis_report.html** - Comprehensive HTML report
   - ⚠️ Prominent warning if edge event fallback was used
   - Summary statistics with reliability flags
   - All visualizations (if generated)
   - Statistical test results with FDR correction
   - Methods description and limitations
   - Footnotes explaining asterisks and flags

**Typical Runtime:** < 5 minutes for 100 videos

---

## Understanding the Outputs

### Behavioral Metrics Calculated

The pipeline calculates five standardized metrics:

1. **Event Frequency** (events per minute) - How often grooming bouts are initiated
2. **Mean Bout Duration** (seconds) - Average length of grooming episodes
3. **Grooming Percentage** (% time) - Proportion of time spent grooming
4. **Bout Duration Distribution** - Full distribution of grooming bout lengths
5. **Fragmentation Index** - Coefficient of variation of bout durations

All rate-based metrics are normalized to handle frame count variations.

### Edge Event Handling

**What are edge events?**
- Grooming bouts that are truncated at window boundaries
- Affects duration measurements but not frequency counts

**Handling strategy:**
- Frequency metrics: Use ALL events
- Duration metrics: Use complete events only (preferred)
- Fallback: If < 5 complete events, include edge events ≥ threshold
- Threshold comes from Step 1 optimization (typically 15-25 frames)

**Reliability flags:**
- `High`: Duration metrics from complete events only
- `Fallback`: Duration metrics include edge events (may underestimate)
- HTML report shows prominent warning when fallback is used

### Statistical Testing

**Automatic test selection:**
- Normality tested with Shapiro-Wilk
- Normal data → Welch's t-test + Cohen's d
- Non-normal data → Mann-Whitney U + Cliff's delta

**Multiple testing correction:**
- Benjamini-Hochberg FDR correction applied across all metrics
- Both raw and corrected p-values reported
- Significance determined from corrected p-values (α = 0.05)

---

## Troubleshooting

### Common Issues

**"At least 2 genotypes required for optimization"**
- Ensure pilot_data/ has at least 2 subdirectories
- Each subdirectory should contain CSV files

**"Expected 'scored_sampled.csv' from fly_behavior_analysis annotation"**
- After annotating sampled_video.tif, save the output as exactly `scored_sampled.csv`
- Filename is case-sensitive and must match exactly

**"Missing required fields in optimization JSON"**
- Re-run Step 1 to regenerate optimization_results.json
- Required fields: window_size, sampling_rate, sampling_strategy, recommended_edge_duration_min

**Warning: "Fewer than 10 flies recommended"**
- Script continues but results may be less reliable
- Try to collect more pilot data for better optimization

**Warning: "Edge event fallback used"**
- Some flies had < 5 complete grooming events
- Duration metrics may be slightly underestimated
- Consider increasing sampling rate if this is widespread

---

## Citation

If you use this pipeline in your research, please cite:

[Add citation information here]

---

## License

[Add license information here]

---

## Contact

For questions or issues, please contact:
- Carlo Fusco (carlo.fusco@unil.ch)
- [GitHub Issues](https://github.com/your-repo/grooming-pipeline-analysis/issues)
```
