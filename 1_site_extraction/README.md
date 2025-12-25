# Archaeological Site Extraction Tool

A web-based tool for extracting archaeological site information from academic papers (PDFs) using Large Language Models, and downloading satellite imagery and terrain data from Google Earth Engine for identified sites.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Output Formats](#output-formats)
- [Coordinate Handling Logic](#coordinate-handling-logic)
- [Satellite Data Details](#satellite-data-details)

---

## Features

### PDF Extraction & Analysis
- **PDF Text Extraction**: Automatically extracts text from academic papers
- **LLM-Powered Analysis**: Uses GPT-4o to intelligently extract archaeological data
- **Multiple Site Handling**: Processes papers with multiple archaeological sites
- **Coordinate Recognition**: Identifies and preserves coordinate information in various formats
- **Precision Tracking**: Notes coordinate precision and confidence levels

### Satellite Data Download
- **Google Earth Engine Integration**: Download satellite imagery for identified sites
- **Multi-Spectral Imagery**: Sentinel-2 bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Terrain Data**: SRTM Digital Elevation Model (DEM) and calculated slope
- **Spectral Indices**: Auto-calculated NDVI, NDWI, and BSI
- **Visualization**: Automatic generation of RGB composites and index visualizations
- **Flexible Coordinates**: Supports multiple coordinate formats (DMS, decimal degrees)

### User Interface
- **Simple Web Interface**: Upload PDFs and download data through browser
- **Site Selection**: Review extracted sites and select which to download
- **Progress Tracking**: Real-time feedback during processing
- **JSON & ZIP Export**: Download structured metadata and satellite data packages

---

## Requirements

### Software
- Python 3.7+
- Google Chrome/Firefox (modern browser)

### API Keys & Accounts
- **OpenAI API key** (for PDF extraction)
- **Google Earth Engine account** (for satellite data download)
  - Sign up at: https://earthengine.google.com/signup/
  - Create a Cloud Project (free)

### Python Dependencies

```bash
pip install flask pypdf openai python-dotenv earthengine-api google-auth numpy scipy matplotlib
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone Repository & Install Dependencies

```bash
# Clone or download this repository
git clone https://github.com/BostonListener/CAA_UK_2025
cd CAA_UK_2025/1_site_extraction

# Install Python packages
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI key
# OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

### 3. Set Up Google Earth Engine

#### Create GEE Service Account

1. Go to **Google Cloud Console**: https://console.cloud.google.com/
2. Select or create a project
3. Enable **Earth Engine API**:
   - Go to "APIs & Services" → "Enable APIs and Services"
   - Search for "Earth Engine API"
   - Click "Enable"

4. Create **Service Account**:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Name: `gee-archaeological-extractor`
   - Click "Create and Continue"
   - Skip optional steps, click "Done"

5. Create **Service Account Key**:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose "JSON" format
   - Download the key file

6. **Register Service Account with Earth Engine**:
   - Go to https://code.earthengine.google.com/
   - Click "Register a noncommercial or commercial Cloud project"
   - Select your project
   - Complete registration

7. **Place Key File**:
   ```bash
   # Rename downloaded key to gee_service_account.json
   # Place in project root directory
   mv ~/Downloads/your-project-xxxxx.json ./gee_service_account.json
   ```

8. **Update `.env` with Project ID**:
   ```bash
   GEE_PROJECT_ID=your-project-id
   ```

---

## File Structure

```
1_site_extraction/
│
├── app.py                      # Flask web server with GEE integration
│
├── templates/
│   └── index.html              # Web interface (frontend)
│
├── static/
│   └── style.css               # Styling
│
├── gee_service_account.json    # GEE credentials (DO NOT COMMIT)
├── .env                        # API keys (DO NOT COMMIT)
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Important Files

| File | Purpose | Security |
|------|---------|----------|
| `app.py` | Flask server with PDF extraction and GEE download | Public |
| `templates/index.html` | User interface | Public |
| `gee_service_account.json` | **GEE authentication** | 🔒 **NEVER COMMIT** |
| `.env` | **API keys** | 🔒 **NEVER COMMIT** |
| `.env.example` | Template showing required variables | Public |

---

## Configuration

### Environment Variables

Create `.env` file with:

```bash
# OpenAI API Key (for PDF extraction)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# Google Earth Engine Project ID
GEE_PROJECT_ID=your-gee-project-id
```

### GEE Data Configuration

Default settings in `app.py` (modify if needed):

```python
GEE_CONFIG = {
    'project': os.getenv('GEE_PROJECT_ID'),
    'cell_size_km': 1.0,                # 1km × 1km grid cells
    'pixels_per_km': 100,               # 100 × 100 pixel output (10m resolution)
    'sentinel2': {
        'date_start': '2020-01-01',
        'date_end': '2024-12-31',
        'cloud_cover_max': 20,          # Maximum 20% cloud cover
    },
}
```

---

## Usage

### Starting the Server

```bash
python app.py
```

Expected output:
```
✓ Google Earth Engine initialized successfully

Starting Archaeological Site Extraction Web Interface...
Open http://localhost:5000 in your browser
```

### Workflow

#### Step 1: Extract Sites from PDF

1. **Open browser** to `http://localhost:5000`

2. **Upload PDF**:
   - Click "Choose PDF file"
   - Select archaeological paper
   - Click "Extract Sites"

3. **Wait for extraction** (30 seconds)
   - LLM analyzes paper
   - Identifies sites and coordinates

4. **Review Results**:
   - See site cards with metadata
   - Check which sites have coordinates
   - Download JSON if needed

#### Step 2: Download Satellite Data for Sites

For each site with coordinates:

1. **Click "Download GEE Data" button** on site card

2. **Wait for processing** (30-60 seconds per site)
   - Server queries Google Earth Engine
   - Downloads Sentinel-2 imagery
   - Extracts DEM and slope
   - Calculates spectral indices
   - Generates visualizations

3. **Automatic Download**:
   - Browser downloads ZIP file: `SiteName_lat_lon.zip`

4. **ZIP Contents**:
   ```
   SiteName_-12.3456_-67.8900/
   ├── channels/
   │   ├── B2.npy        # Blue band (490nm)
   │   ├── B3.npy        # Green band (560nm)
   │   ├── B4.npy        # Red band (665nm)
   │   ├── B8.npy        # NIR band (842nm)
   │   ├── B11.npy       # SWIR1 band (1610nm)
   │   ├── B12.npy       # SWIR2 band (2190nm)
   │   ├── DEM.npy       # Elevation (meters)
   │   ├── Slope.npy     # Terrain slope (degrees)
   │   ├── NDVI.npy      # Normalized Difference Vegetation Index
   │   ├── NDWI.npy      # Normalized Difference Water Index
   │   └── BSI.npy       # Bare Soil Index
   ├── visualizations/
   │   └── overview.png  # 6-panel visualization
   └── metadata.json     # Acquisition metadata
   ```

---

## How It Works

### PDF Extraction Pipeline

1. **PDF Upload** → Text extraction with `pypdf`
2. **LLM Analysis** → GPT-4o identifies sites and coordinates
3. **Structured Output** → JSON with site metadata
4. **Web Display** → Interactive site cards

### GEE Download Pipeline

1. **Coordinate Parsing**:
   - Supports formats: "5.23°S, 60.12°W", "5°13'48"S", "-5.23, -60.12"
   - Converts to decimal degrees

2. **GEE Query**:
   - Creates 1km × 1km bounding box around site
   - Queries Sentinel-2 collection (2020-2024)
   - Selects least cloudy image
   - Queries SRTM DEM (30m resolution)

3. **Data Extraction**:
   - Extracts bands at native resolution:
     - B2, B3, B4, B8: 10m → ~100 pixels
     - B11, B12: 20m → ~50 pixels
     - DEM: 30m → ~34 pixels
   - Resamples all to 100×100 pixels (10m effective resolution)

4. **Index Calculation**:
   ```python
   NDVI = (NIR - Red) / (NIR + Red)
   NDWI = (Green - NIR) / (Green + NIR)
   BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
   ```

5. **Visualization**:
   - RGB composite (B4-B3-B2)
   - NDVI (vegetation)
   - NDWI (water)
   - BSI (bare soil)
   - DEM (elevation)
   - Slope (terrain)

6. **Packaging**:
   - All arrays saved as `.npy` (NumPy format)
   - Visualization as PNG
   - Metadata as JSON
   - Compressed to ZIP

---

## Output Formats

### PDF Extraction Output (JSON)

```json
{
  "paper_metadata": {
    "title": "Archaeological Survey of the Upper Amazon",
    "authors": ["Smith, J.", "Jones, A."],
    "year": "2023",
    "doi": "10.1234/example"
  },
  "extraction_summary": {
    "total_sites_found": 5,
    "sites_with_explicit_coordinates": 3,
    "sites_with_descriptions_only": 2
  },
  "sites": [
    {
      "site_name": "Fazenda Colorada",
      "site_code": "FC-01",
      "coordinates": {
        "has_explicit_coordinates": true,
        "raw_text": "9.8765°S, 67.5346°W",
        "latitude": -9.8765,
        "longitude": -67.5346,
        "format": "decimal_degrees",
        "precision_level": "exact"
      },
      "location_description": "Near confluence of rivers",
      "temporal": {
        "dating": "1200-1400 CE",
        "cultural_period": "Late Pre-Columbian"
      },
      "characteristics": {
        "site_type": "Settlement mound",
        "features": ["circular plaza", "defensive ditch"],
        "size": "15 hectares"
      }
    }
  ]
}
```

### GEE Data Output (ZIP)

**Directory Structure**:
```
SiteName_-12.3456_-67.8900.zip
└── SiteName_-12.3456_-67.8900/
    ├── channels/           # 11 .npy files (100×100 each)
    ├── visualizations/     # overview.png
    └── metadata.json       # Acquisition details
```

**Metadata JSON**:
```json
{
  "site_name": "Fazenda Colorada",
  "latitude": -9.8765,
  "longitude": -67.5346,
  "cell_size_km": 1.0,
  "image_id": "COPERNICUS/S2_SR_HARMONIZED/20231115T...",
  "cloud_cover": 5.2,
  "acquisition_date": "2023-11-15T14:23:45Z"
}
```

**Channel Files** (NumPy `.npy` format):
- All arrays are `float32` dtype
- All arrays are `(100, 100)` shape
- Can be loaded with: `numpy.load('B2.npy')`

---

## Coordinate Handling Logic

### Supported Coordinate Formats

The tool automatically recognizes and converts:

| Format | Example | Output |
|--------|---------|--------|
| **Decimal Degrees** | `5.23°S, 60.12°W` | `-5.23, -60.12` |
| **Decimal with Direction** | `5.23S, 60.12W` | `-5.23, -60.12` |
| **Degrees Minutes Seconds** | `5°13'48"S, 60°7'12"W` | `-5.23, -60.12` |
| **Plain Decimal** | `-5.23, -60.12` | `-5.23, -60.12` |

### Multiple Sites vs. Multiple Points

**Scenario 1: Multiple Independent Sites**
- Each coordinate pair = separate site
- Each gets its own site card
- Each can be downloaded individually

**Scenario 2: Single Site with Multiple Points**
- One site with `coordinates` and `extra_coordinates[]`
- Main coordinate used for GEE download
- Additional points noted in metadata

**Scenario 3: No Coordinates**
- Site listed but not mappable
- No "Download GEE Data" button
- Useful for textual analysis only

**Scenario 4: Withheld Coordinates**
- `location_withheld: true`
- No download available
- Respects authors' protection of sensitive sites

---

## Satellite Data Details

### Sentinel-2 Bands

| Band | Wavelength | Resolution | Description |
|------|------------|------------|-------------|
| **B2** | 490nm (Blue) | 10m | Water penetration, soil/vegetation discrimination |
| **B3** | 560nm (Green) | 10m | Vegetation vigor peak |
| **B4** | 665nm (Red) | 10m | Chlorophyll absorption |
| **B8** | 842nm (NIR) | 10m | Biomass content, water bodies |
| **B11** | 1610nm (SWIR1) | 20m→10m | Moisture content, soil/geology |
| **B12** | 2190nm (SWIR2) | 20m→10m | Moisture stress, geology |

### Spectral Indices

**NDVI (Normalized Difference Vegetation Index)**
- Range: -1 to +1
- High values (0.6-0.9): Dense vegetation
- Low values (<0.2): Bare soil, water
- Use: Identify cleared areas, agricultural land

**NDWI (Normalized Difference Water Index)**
- Range: -1 to +1
- High values (>0.3): Water bodies
- Low values (<0): Dry land, vegetation
- Use: Detect water features, moisture

**BSI (Bare Soil Index)**
- Range: -1 to +1
- High values: Exposed soil
- Low values: Vegetation, water
- Use: Identify disturbed areas, construction

### Terrain Data

**DEM (Digital Elevation Model)**
- Source: SRTM (Shuttle Radar Topography Mission)
- Resolution: 30m → resampled to 10m
- Units: Meters above sea level
- Use: Topographic context, site placement

**Slope**
- Calculated from DEM
- Units: Degrees (0-90°)
- Use: Terrain analysis, accessibility