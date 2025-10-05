# grooming-pipeline-analysis (~/Nextcloud/coding projects/ppe)

Pipeline for optimizing manual scoring of Drosophila grooming behavior through intelligent sampling strategies.

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
│   ├── pilot_data/             # Pilot CSV data
│   ├── videos/                 # Raw video files
│   └── sampled/                # Output from sampling
├── notebooks/                   # Jupyter notebooks
├── reports/                     # Generated reports
└── docs/                        # Documentation
```

## Usage

```bash
# Step 1: Optimize sampling parameters
python src/pilot_grooming_optimizer.py --data-dir data/pilot_data --output optimization_results.json

# Step 2: Sample videos
python src/video_sampler.py --video-dir data/videos/wild_type --from-optimization optimization_results.json

# Step 3: Statistical analysis
python src/grooming_statistical_analyzer.py --group-1-dir data/sampled/wild_type --output-dir reports
```
