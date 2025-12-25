# Archaeological Site Detection: From Papers to Datasets

A complete pipeline for extracting archaeological site information from academic papers and generating multi-channel remote sensing datasets for machine learning-based site detection.

---

## Overview

This repository combines two complementary workflows designed to support reinforcement learning and deep learning approaches to archaeological site discovery:

1. **Site Extraction Pipeline** - Extract archaeological site coordinates from academic PDFs using LLMs
2. **Dataset Generation Pipeline** - Generate multi-channel satellite imagery datasets from known site locations

```
Academic Papers (PDFs)
         ↓
    [Step 1: LLM Extraction]
         ↓
    Site Coordinates (JSON/CSV)
         ↓
    [Step 2: Dataset Generation]
         ↓
    Multi-Channel Remote Sensing Dataset
         ↓
    [Your RL/ML Model]
```

**Project Origins:** This work was developed as part of our participation in the [OpenAI to Z Challenge on Kaggle](https://www.kaggle.com/competitions/openai-to-z-challenge/overview), where we explored AI-powered archaeological discovery in the Amazon. Our competition writeup detailing the approach is available [here](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/bostonlistener_digitalarchaeology). The pipeline has since evolved into a comprehensive framework for digitizing archaeological knowledge and preparing it for machine learning applications. Our approach was presented at [CAA UK 2025](https://uk.caa-international.org/caa-uk-2025/) (Computer Applications and Quantitative Methods in Archaeology) held at the University of Cambridge, December 9-10, 2025.

**Conference Materials:**
- [Abstract](https://drive.google.com/file/d/1zQ3-LrlsDmZiI1D3QkJo1lQ43e6MwYq1/view?usp=sharing)
- [Presentation Slides](https://drive.google.com/file/d/1n79eqJ7XM3h3ftmJakhP5JOrqjONAnRE/view?usp=sharing)

---

## Dataset

A pre-generated dataset created using this pipeline is available on Hugging Face:

**🤗 [Archaeological Sites Dataset (CAA UK 2025)](https://huggingface.co/datasets/lldbrett/archaeological-sites-caa2025)**

The dataset provides multi-channel remote sensing data (Sentinel-2 + FABDEM + spectral indices) with balanced positive/negative samples and augmentations for training archaeological site detection models.

### Dataset Samples

Below are example visualizations from each category in the dataset. Each sample is a 100×100 pixel multi-channel image with 11 bands (RGB composite shown for visualization).

| Positives | Integrated Negatives | Landcover Negatives | Unlabeled |
|:---------:|:--------------------:|:-------------------:|:---------:|
| ![](grid_samples/positives/grid_000018_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000018_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_000221_visualization.png) | ![](grid_samples/unlabeled/unla_000871_visualization.png) |
| ![](grid_samples/positives/grid_000145_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000145_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_000729_visualization.png) | ![](grid_samples/unlabeled/unla_001089_visualization.png) |
| ![](grid_samples/positives/grid_000206_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000206_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_000812_visualization.png) | ![](grid_samples/unlabeled/unla_001424_visualization.png) |
| ![](grid_samples/positives/grid_000317_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000317_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_000985_visualization.png) | ![](grid_samples/unlabeled/unla_001483_visualization.png) |
| ![](grid_samples/positives/grid_000601_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000601_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_001354_visualization.png) | ![](grid_samples/unlabeled/unla_001624_visualization.png) |
| ![](grid_samples/positives/grid_000685_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000685_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_001507_visualization.png) | ![](grid_samples/unlabeled/unla_001653_visualization.png) |
| ![](grid_samples/positives/grid_000832_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_000832_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_001750_visualization.png) | ![](grid_samples/unlabeled/unla_001682_visualization.png) |
| ![](grid_samples/positives/grid_001287_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_001287_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_002172_visualization.png) | ![](grid_samples/unlabeled/unla_002019_visualization.png) |
| ![](grid_samples/positives/grid_001375_rot000_visualization.png) | ![](grid_samples/integrated_negatives/ineg_001375_rot000_visualization.png) | ![](grid_samples/landcover_negatives/lneg_002241_visualization.png) | ![](grid_samples/unlabeled/unla_002099_visualization.png) |

**Categories:**
- **Positives**: Known archaeological sites (geoglyphs, mounds, settlements)
- **Integrated Negatives**: Areas surrounding positive sites, spatially close but archaeologically empty
- **Landcover Negatives**: Diverse landscapes (urban, water, cropland) to improve model robustness
- **Unlabeled**: Background samples for semi-supervised learning approaches

---

## Pipeline Independence

**Important:** Each step works as a standalone tool - you can use either one independently or combine them for the full workflow.

### Use Step 1 Alone
- Digitize archaeological site data from legacy publications
- Extract coordinates and metadata from PDFs
- Download satellite imagery for specific sites
- Export site databases for GIS or other applications
- **No need to install Step 2 dependencies**

### Use Step 2 Alone
- Generate training datasets from your own fieldwork coordinates
- Process site data from existing databases or catalogs
- Create balanced ML datasets from any site coordinate source
- **No need to install Step 1 dependencies or OpenAI API**

### Use Both Together
- Complete end-to-end pipeline from literature to ML-ready datasets
- Seamless data handoff between extraction and generation
- Ideal for comprehensive archaeological ML projects

---

## Project Structure

```
CAA_UK_2025/
│
├── 1_site_extraction/              # Step 1: Extract sites from academic papers
│   ├── app.py                      # Flask web server with LLM + GEE integration
│   ├── templates/
│   ├── static/
│   ├── .gitignore                  # Step 1 specific ignores
│   ├── requirements.txt            # Step 1 dependencies
│   └── README.md                   # Detailed documentation for Step 1
│
├── 2_dataset_generation/           # Step 2: Generate training datasets
│   ├── config/
│   ├── scripts/
│   ├── src/
│   ├── .gitignore                  # Step 2 specific ignores
│   ├── requirements.txt            # Step 2 dependencies
│   ├── README.md                   # Detailed documentation for Step 2
│   └── run_pipeline.py
│
├── README.md                       # This file
└── LICENSE                         # MIT License
```

**Note:** Each step maintains its own `.gitignore` and `requirements.txt` for independence and modularity.

---

## Key Features

### Step 1: Site Extraction
- 📄 **PDF text extraction** from archaeological publications
- 🤖 **LLM-powered analysis** (GPT-4o) to identify sites and coordinates
- 🗺️ **Multiple coordinate format support** (DMS, decimal degrees, etc.)
- 🛰️ **Satellite data download** for extracted sites (Sentinel-2 + terrain)
- 🌐 **Web interface** for easy interaction

### Step 2: Dataset Generation
- 🎯 **Balanced dataset creation** with positives, negatives, and unlabeled samples
- 🔄 **Geometric augmentation** via rotation (configurable: 3x, 4x, 6x, 12x)
- 🌈 **Radiometric augmentation** for lighting/contrast variation
- 📊 **11-channel data** (6 spectral bands + 3 indices + elevation + slope)
- 📦 **Production-ready format** with metadata and validation tools

---

## Quick Start

### Prerequisites

**For Step 1 Only:**
- Python 3.7+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Google Earth Engine account ([sign up here](https://earthengine.google.com/signup/))
- Modern web browser

**For Step 2 Only:**
- Python 3.7+
- Google Earth Engine account ([sign up here](https://earthengine.google.com/signup/))

**For Both Steps:**
- All of the above

**Python Dependencies:**
```bash
# Install for Step 1 only
cd 1_site_extraction
pip install -r requirements.txt

# Install for Step 2 only
cd 2_dataset_generation
pip install -r requirements.txt

# Or install for both if using full pipeline
```

### End-to-End Workflow

#### Step 1: Extract Sites from Papers

```bash
cd 1_site_extraction

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key and GEE project ID

# Place GEE service account JSON
cp ~/Downloads/your-gee-key.json ./gee_service_account.json

# Start web interface
python app.py
# Open http://localhost:5000
```

**What you'll do:**
1. Upload archaeological paper PDFs
2. Let GPT-4o extract site information
3. Review extracted sites with coordinates
4. Download satellite data for each site (optional)
5. Export site list as JSON

**See detailed instructions:** [1_site_extraction/README.md](1_site_extraction/README.md)

---

#### Transition: Convert JSON to CSV

Step 1 outputs JSON format, but Step 2 requires CSV input. Create `known_sites.csv`:

```csv
site_id,latitude,longitude,site_type
site_001,-9.8765,-67.5346,geoglyph
site_002,-10.1234,-68.4567,mound
site_003,-11.2345,-69.5678,settlement
```

Extract from your Step 1 JSON output:
- `site_id`: Unique identifier
- `latitude`: Decimal degrees (negative for S)
- `longitude`: Decimal degrees (negative for W)
- `site_type`: Optional classification

---

#### Step 2: Generate Training Dataset

```bash
cd 2_dataset_generation

# Prepare input
mkdir -p inputs
cp /path/to/known_sites.csv inputs/

# Configure pipeline
cp config/settings.yaml.example config/settings.yaml
# Edit settings.yaml with your GEE project ID and parameters

# Authenticate GEE
earthengine authenticate

# Run full pipeline
python run_pipeline.py
```

**What this generates:**
- Multi-angle views of each site (rotation augmentation)
- Integrated negatives from surrounding landscape
- Diverse landcover negatives (urban, water, cropland)
- Unlabeled background samples
- Radiometric augmentations (brightness/contrast/noise)

**Output:** `outputs/dataset/` with:
- `grid_metadata.parquet` - Master metadata table
- `grid_images/` - Individual 100×100×11 samples as NumPy arrays
- Ready for PyTorch/TensorFlow training

**See detailed instructions:** [2_dataset_generation/README.md](2_dataset_generation/README.md)

---

## Citation

If you use this dataset or pipeline in your research, please cite either conference or software:

### Conference Citation

```bibtex
@inproceedings{li2025fusing,
  title={{Fusing Text and Terrain}: {An LLM}-Powered Pipeline for Preparing Archaeological Datasets from Literature and Remote Sensing Imagery},
  author={Li, Linduo and Wu, Yifan and Wang, Zifeng},
  booktitle={{CAA UK 2025}: Computer Applications and Quantitative Methods in Archaeology},
  year={2025},
  month={December},
  address={University of Cambridge, UK},
  organization={CAA UK},
  note={Conference held 9--10 December 2025}
}
```

### Software Citation

```bibtex
@software{archaeological_site_detection,
  title={Archaeological Site Detection: From Papers to Datasets},
  author={Li, Linduo and Wu, Yifan and Wang, Zifeng},
  year={2025},
  url={https://github.com/BostonListener/CAA_UK_2025}
}
```

---

## Acknowledgments

This work builds upon data and resources from multiple sources:

**Satellite and Terrain Data:**
- **Sentinel-2 satellite imagery** provided by the European Space Agency (ESA) through the Copernicus Programme
- **FABDEM elevation data** provided by the University of Bristol
- **Google Earth Engine** served as the primary platform for geospatial data processing and analysis

**Archaeological Data Sources:**
We are deeply grateful to **James Q. Jacobs** for his invaluable contribution to archaeological research through his publicly accessible compilation of geoglyph locations. His meticulous curation of archaeological data has been instrumental in enabling this work:
- Jacobs, J.Q. (2025). *JQ Jacobs Archaeology*. Last modified July 31, 2025. https://jqjacobs.net/archaeology/geoglyph.html

**Technical Infrastructure:**
- **OpenAI GPT-4o** for LLM-powered text extraction and analysis
- The **Kaggle OpenAI to Z Challenge** for providing the initial impetus and platform for this research

We acknowledge that this pipeline stands on the shoulders of both cutting-edge technology and dedicated scholarly work in the archaeological community.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact & Support

- **Documentation**: See README files in each subfolder for detailed guides
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share your results and ask questions in GitHub Discussions
- **Email**: linduo.li@ip-paris.fr