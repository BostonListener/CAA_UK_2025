# Archaeological Site Detection Dataset Pipeline

A complete pipeline for generating multi-channel remote sensing datasets for archaeological site detection using Sentinel-2 imagery, FABDEM elevation data, and spectral indices.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)
5. [Dataset Schema](#dataset-schema)
6. [Data Augmentation Strategy](#data-augmentation-strategy)
7. [Train/Val/Test Split Guidelines](#trainvaltest-split-guidelines)
8. [Important Notes](#important-notes)

---

## Overview

This pipeline generates a balanced, augmented dataset for training archaeological site detection models. Starting from known site locations, it creates:

- **Positives**: Multi-angle views of known archaeological sites
- **Integrated Negatives**: Surrounding landscape context (same geographic area, different sampling)
- **Landcover Negatives**: Diverse negative examples (urban, water, cropland)
- **Unlabeled Data**: Background samples for semi-supervised learning

Each sample is a 1×1 km grid cell at 10m resolution (100×100 pixels) with 11 channels including spectral bands, vegetation/water/soil indices, and terrain data.

---

## File Structure
```
project_root/
├── config/
│   └── settings.yaml                       # Configuration parameters
│
├── inputs/
│   └── known_sites.csv                     # Input: known archaeological sites
│                                           # Required columns: site_id, latitude, longitude
│
├── outputs/
│   └── dataset/
│       ├── grid_metadata.parquet           # Master metadata table
│       └── grid_images/                    # Individual grid cells
│           ├── grid_000001_rot000/         # Positive (0° rotation)
│           │   ├── channels/
│           │   │   ├── B2.npy              # Sentinel-2 Blue (100×100)
│           │   │   ├── B3.npy              # Sentinel-2 Green
│           │   │   ├── B4.npy              # Sentinel-2 Red
│           │   │   ├── B8.npy              # Sentinel-2 NIR
│           │   │   ├── B11.npy             # Sentinel-2 SWIR1
│           │   │   ├── B12.npy             # Sentinel-2 SWIR2
│           │   │   ├── NDVI.npy            # Vegetation index
│           │   │   ├── NDWI.npy            # Water index
│           │   │   ├── BSI.npy             # Bare soil index
│           │   │   ├── DEM.npy             # Elevation (FABDEM)
│           │   │   └── Slope.npy           # Terrain slope
│           │   ├── labels/
│           │   │   ├── binary_label.npy    # [1] for positive
│           │   │   ├── pos_type.txt        # Site type (e.g., "geoglyph")
│           │   │   └── neg_type.txt        # "null" for positives
│           │   └── info.json               # Metadata (lat, lon, rotation, etc.)
│           │
│           ├── grid_000001_rot120/         # Positive (120° rotation)
│           ├── grid_000001_rot240/         # Positive (240° rotation)
│           ├── grid_000001_rot000_aug1/    # Augmented (bright variant)
│           ├── grid_000001_rot000_aug2/    # Augmented (dark variant)
│           ├── grid_000001_rot000_aug3/    # Augmented (noise variant)
│           │
│           ├── ineg_000001_rot000/         # Integrated negative (0°)
│           ├── ineg_000001_rot120/         # Integrated negative (120°)
│           ├── ineg_000001_rot000_aug1/    # Augmented integrated negative
│           │
│           ├── lneg_000001/                # Landcover negative (urban/water/cropland)
│           └── unla_000001/                # Unlabeled background sample
│
├── scripts/
│   ├── generate_positives.py
│   ├── generate_integrated_negatives.py
│   ├── generate_landcover_negatives.py
│   ├── generate_unlabeled.py
│   ├── generate_radiometric_augmentation.py
│   └── validate_data.py
│
├── src/
│   ├── file_manager.py
│   ├── gee_extractor.py
│   ├── index_calculator.py
│   ├── metadata_builder.py
│   └── negative_sampler.py
│
├── run_pipeline.py                         # Master script to run full pipeline
└── README.md
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
pip install earthengine-api numpy pandas scipy pyyaml pyarrow matplotlib
```

### 2. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

### 3. Configure Project

Edit `config/settings.yaml`:
```yaml
gee:
  project: 'your-gee-project-id'  # Replace with your GEE project

imagery:
  sentinel2:
    date_start: '2023-01-01'      # Adjust date range as needed
    date_end: '2024-01-01'

augmentation:
  rotation_generation:
    enabled: true
    rotation_step: 120            # Rotation angle step (e.g., 120° → 3 rotations)
                                  # Options: 120 (3×), 90 (4×), 60 (6×), 30 (12×)

site_region:
  min_lat: -17.0                  # Define your study area
  max_lat: -7.0
  min_lon: -70.0
  max_lon: -60.0
```

### 4. Prepare Input Data

Create `inputs/known_sites.csv` with columns:
- `site_id`: Unique identifier (string)
- `latitude`: Site latitude (float, WGS84)
- `longitude`: Site longitude (float, WGS84)
- `site_type`: Optional, e.g., "geoglyph", "mound" (string)

---

## Usage

### Option 1: Run Complete Pipeline (Recommended)
```bash
python run_pipeline.py
```

This runs all generation scripts in the correct order:
1. Generate positives (with rotation)
2. Generate integrated negatives (with rotation)
3. Generate landcover negatives
4. Generate unlabeled data
5. Generate radiometric augmentation

**Note**: Pipeline may take several hours depending on dataset size.

### Option 2: Run Individual Scripts
```bash
# Step 1: Generate positives
python scripts/generate_positives.py

# Step 2: Generate integrated negatives
python scripts/generate_integrated_negatives.py

# Step 3: Generate landcover negatives
python scripts/generate_landcover_negatives.py

# Step 4: Generate unlabeled data
python scripts/generate_unlabeled.py

# Step 5: Apply radiometric augmentation (requires steps 1 & 2)
python scripts/generate_radiometric_augmentation.py
```

### Data Validation
```bash
# Validate all grids
python scripts/validate_data.py --dataset-dir outputs/dataset

# Generate visualizations
python scripts/validate_data.py --visualize

# Show detailed statistics
python scripts/validate_data.py --stats

# Visualize specific grid
python scripts/validate_data.py --sample-grid grid_000001_rot000 --visualize
```

---

## Dataset Schema

### Metadata Table (`grid_metadata.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `grid_id` | string | Unique grid identifier (e.g., "grid_000001_rot000") |
| `centroid_lon` | float | Grid center longitude (WGS84) |
| `centroid_lat` | float | Grid center latitude (WGS84) |
| `label` | int | 1 = site, 0 = non-site, -1 = unlabeled |
| `label_source` | string | Data source (e.g., "literature", "integrated_context") |
| `image_path` | string | Path to grid directory |

### Channel Schema (11 channels per grid)

| Index | Channel | Source | Resolution | Description |
|-------|---------|--------|------------|-------------|
| 0 | B2 | Sentinel-2 | 10m | Blue (490nm) - atmosphere/water contrast |
| 1 | B3 | Sentinel-2 | 10m | Green (560nm) - used in NDWI |
| 2 | B4 | Sentinel-2 | 10m | Red (665nm) - used in NDVI, BSI |
| 3 | B8 | Sentinel-2 | 10m | NIR (842nm) - vegetation signal |
| 4 | B11 | Sentinel-2 | 20m→10m | SWIR1 (1610nm) - soil/moisture |
| 5 | B12 | Sentinel-2 | 20m→10m | SWIR2 (2190nm) - dryness/bare soil |
| 6 | NDVI | Calculated | 10m | (B8 - B4) / (B8 + B4) - vegetation vigor |
| 7 | NDWI | Calculated | 10m | (B3 - B8) / (B3 + B8) - water content |
| 8 | BSI | Calculated | 10m | Bare soil index - soil brightness |
| 9 | DEM | FABDEM | 30m→10m | Elevation (meters) |
| 10 | Slope | Derived | 10m | Terrain slope (degrees) |

**Storage**: Each channel saved as separate `.npy` file with shape `(100, 100)` and dtype `float32`.

---

## Data Augmentation Strategy

### 1. Rotation Generation (Geometric)

**Applied to**: Positives and integrated negatives

- **Extraction**: 1.5× size (1.5×1.5 km → 150×150 pixels)
- **Rotations**: Configurable via `rotation_step` parameter
  - `rotation_step: 120` → 3 rotations (0°, 120°, 240°)
  - `rotation_step: 90` → 4 rotations (0°, 90°, 180°, 270°)
  - `rotation_step: 60` → 6 rotations
  - `rotation_step: 30` → 12 rotations
- **Crop**: Center crop to 1×1 km (100×100 pixels)
- **Result**: R× data from each site, where **R = 360 / rotation_step**

**Grid ID Format**: 
- `rotation_step: 120` → `grid_000001_rot000`, `grid_000001_rot120`, `grid_000001_rot240`
- `rotation_step: 30` → `grid_000001_rot000`, `grid_000001_rot030`, ..., `grid_000001_rot330`

### 2. Radiometric Augmentation

**Applied to**: All rotated positives and integrated negatives

Three variants per rotated sample:
- **aug1** (Bright): +8% brightness, +5% contrast, noise σ=0.015
- **aug2** (Dark): -8% brightness, -5% contrast, noise σ=0.015  
- **aug3** (Noise): No brightness/contrast change, noise σ=0.025

**Grid ID Format**: `grid_000001_rot000_aug1`, `grid_000001_rot120_aug2`, etc.

### Final Dataset Composition

Given **N** known sites and **R** rotations (where **R = 360 / rotation_step**):

| Data Type | Count | Calculation | Label |
|-----------|-------|-------------|-------|
| **Positives (base)** | R×N | N sites × R rotations | 1 |
| **Positives (augmented)** | 3×R×N | R×N base × 3 variants | 1 |
| **Total Positives** | **4×R×N** | R×N + 3×R×N | 1 |
| **Integrated Negatives (base)** | R×N | N sites × R rotations | 0 |
| **Integrated Negatives (aug)** | 3×R×N | R×N base × 3 variants | 0 |
| **Total Integrated** | **4×R×N** | R×N + 3×R×N | 0 |
| **Landcover Negatives** | R×N | 4×R×N × 0.25 ratio | 0 |
| **Unlabeled** | ~0.5×R×N | 4×R×N × 0.13 ratio | -1 |
| **TOTAL DATASET** | **~9.5×R×N** | | |

### Examples by Rotation Configuration

| Config | R | N=100 Sites | Total Grids | Notes |
|--------|---|-------------|-------------|-------|
| `rotation_step: 120` | 3 | 100 | ~2,850 | Default, balanced |
| `rotation_step: 90` | 4 | 100 | ~3,800 | More variety |
| `rotation_step: 60` | 6 | 100 | ~5,700 | High variety |
| `rotation_step: 30` | 12 | 100 | ~11,400 | Maximum augmentation |
| `rotation_step: 360` | 1 | 100 | ~950 | No rotation (minimal) |

---

## Train/Val/Test Split Guidelines

### ⚠️ CRITICAL: Group by Site to Prevent Data Leakage

**NEVER split randomly** - this causes data leakage because rotations/augmentations of the same site would appear in multiple splits.

### Recommended Approach

**1. Group by Original Site**
```python
import pandas as pd
import numpy as np

# Load metadata
df = pd.read_parquet('outputs/dataset/grid_metadata.parquet')

# Extract base site index from grid_id
# grid_000001_rot000_aug1 -> site 000001
df['site_index'] = df['grid_id'].str.extract(r'(grid|ineg)_(\d+)')[1]

# Get unique sites
unique_sites = df[df['grid_id'].str.startswith('grid_')]['site_index'].unique()

# Shuffle and split sites (not samples!)
np.random.shuffle(unique_sites)
n_train = int(0.7 * len(unique_sites))
n_val = int(0.15 * len(unique_sites))

train_sites = unique_sites[:n_train]
val_sites = unique_sites[n_train:n_train+n_val]
test_sites = unique_sites[n_train+n_val:]

# Assign splits based on site membership
df['split'] = 'test'
df.loc[df['site_index'].isin(train_sites), 'split'] = 'train'
df.loc[df['site_index'].isin(val_sites), 'split'] = 'val'

# For landcover negatives and unlabeled, distribute randomly
mask = df['grid_id'].str.startswith(('lneg_', 'unla_'))
df.loc[mask, 'split'] = np.random.choice(
    ['train', 'val', 'test'],
    size=mask.sum(),
    p=[0.7, 0.15, 0.15]
)
```

**2. Recommended Split Ratios**

- **Train**: 70% of sites (all rotations/augmentations included)
- **Val**: 15% of sites (for hyperparameter tuning)
- **Test**: 15% of sites (held out for final evaluation)

**3. Stratification Considerations**

For small datasets or imbalanced site types:
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Load site types
sites_df = pd.read_csv('inputs/known_sites.csv')
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_idx, test_idx in splitter.split(sites_df, sites_df['site_type']):
    train_sites = sites_df.iloc[train_idx]['site_id']
    test_sites = sites_df.iloc[test_idx]['site_id']
```

### Sampling During Training

**For Balanced Training Batches**:
```python
# Within each batch, sample:
# - 50% positives (including augmentations)
# - 40% negatives (integrated + landcover)
# - 10% unlabeled (optional, for semi-supervised)

# Example using PyTorch
from torch.utils.data import WeightedRandomSampler
import torch

# Calculate sampling weights
train_df = df[df['split'] == 'train']
weights = torch.zeros(len(train_df))
weights[train_df['label'] == 1] = 0.50 / (train_df['label'] == 1).sum()
weights[train_df['label'] == 0] = 0.40 / (train_df['label'] == 0).sum()
weights[train_df['label'] == -1] = 0.10 / (train_df['label'] == -1).sum()

sampler = WeightedRandomSampler(weights, len(train_df), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## Important Notes

### Data Characteristics

1. **Integrated Negatives Strategy**
   - Extracted from the **same geographic area** as positives
   - Represents surrounding landscape context
   - 4 corners sampled after rotation (avoids center where site is located)
   - Sharp boundaries between corners (no blending)

2. **Landcover Negatives**
   - Based on known coordinates with random variations
   - Categories: Urban (40%), Water (30%), Cropland (30%)
   - Ensures model learns to reject obvious non-sites

3. **Unlabeled Data**
   - Label = -1 (NOT label = 0)
   - Random background samples with exclusion buffer around known sites
   - May contain undiscovered archaeological sites
   - Use for semi-supervised learning or RL exploration

4. **Cloud Cover**
   - Max cloud cover: 20% (configurable in `settings.yaml`)
   - Best available image selected automatically
   - Check `info.json` for actual cloud cover per grid