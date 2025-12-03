#!/usr/bin/env python3
"""
Flask web interface for Archaeological Site Extraction with GEE Integration
"""

from flask import Flask, request, render_template, jsonify, send_file
import os
import sys
import json
import re
import tempfile
import zipfile
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import for PDF and LLM
from openai import OpenAI

# Import for Google Earth Engine
import ee
from google.oauth2 import service_account

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")

# GEE Configuration
GEE_CONFIG = {
    'project': os.getenv('GEE_PROJECT_ID'),
    'cell_size_km': 1.0,
    'pixels_per_km': 100,
    'sentinel2': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'date_start': '2020-01-01',
        'date_end': '2024-12-31',
        'cloud_cover_max': 20,
        'scale': 10
    },
    'dem': {
        'collection': 'USGS/SRTMGL1_003'  # SRTM 30m resolution DEM (single Image)
    }
}

# Initialize GEE with service account
def initialize_gee():
    """Initialize Google Earth Engine with service account"""
    try:
        service_account_path = Path(__file__).parent / 'gee_service_account.json'
        
        if not service_account_path.exists():
            print(f"ERROR: Service account file not found: {service_account_path}")
            return False
        
        credentials = service_account.Credentials.from_service_account_file(
            str(service_account_path),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        
        ee.Initialize(credentials=credentials, project=GEE_CONFIG['project'])
        print("✓ Google Earth Engine initialized successfully")
        return True
        
    except Exception as e:
        print(f"ERROR initializing GEE: {str(e)}")
        return False

# Initialize GEE on startup
GEE_INITIALIZED = initialize_gee()


# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def parse_coordinate_string(coord_str):
    """
    Parse coordinate string in various formats to decimal degrees.
    
    Handles formats like:
    - "5.23°S, 60.12°W"
    - "5°13'48"S, 60°7'12"W"
    - "-5.23, -60.12"
    - "5.23, -60.12" (assumes lat, lon order)
    
    Returns: (lat, lon) as floats, or (None, None) if parsing fails
    """
    if not coord_str:
        return None, None
    
    coord_str = coord_str.strip()
    
    # Try to split by comma
    parts = [p.strip() for p in coord_str.split(',')]
    if len(parts) != 2:
        return None, None
    
    try:
        lat = parse_single_coordinate(parts[0])
        lon = parse_single_coordinate(parts[1])
        
        if lat is not None and lon is not None:
            return lat, lon
    except:
        pass
    
    return None, None


def parse_single_coordinate(coord_str):
    """
    Parse a single coordinate value.
    
    Handles:
    - Decimal: "5.23" or "-5.23"
    - DMS with regular symbols: "5°13'48"" or "5°13'48"S"
    - DMS with prime symbols: "5°13′48″S" (Unicode prime and double prime)
    - With direction: "5.23S" or "5.23°S"
    
    Returns: float or None
    """
    coord_str = coord_str.strip()
    
    # Check for direction suffix (N/S/E/W)
    direction = 1
    if coord_str[-1].upper() in ['S', 'W']:
        direction = -1
        coord_str = coord_str[:-1].strip()
    elif coord_str[-1].upper() in ['N', 'E']:
        direction = 1
        coord_str = coord_str[:-1].strip()
    
    # Remove degree symbol if present at the end
    coord_str = coord_str.rstrip('°')
    
    # Try DMS format: degrees°minutes'seconds"
    # Handles both regular apostrophes (' ") and prime symbols (′ ″)
    # Pattern: 12°34'56" or 12°34′56″ or 12°34'56.78"
    dms_pattern = r"(-?\d+)[°\s]+(\d+(?:\.\d+)?)[′'\s]+(\d+(?:\.\d+)?)[″\"]?"
    match = re.match(dms_pattern, coord_str)
    
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        
        decimal = abs(degrees) + minutes/60 + seconds/3600
        if degrees < 0:
            decimal = -decimal
        
        return decimal * direction
    
    # Try decimal format
    try:
        decimal = float(coord_str)
        return decimal * direction
    except ValueError:
        return None


# ============================================================================
# PDF EXTRACTION AND LLM
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")
    
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        num_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n--- Page {i+1} ---\n"
            text += page_text
    
    return text


def create_extraction_prompt(paper_text):
    """Create the extraction prompt with the paper text"""
    
    prompt = """# Archaeological Site Information Extraction

        ## Task
        Extract information about archaeological sites from the provided academic paper. Your role is to act as a precise data extractor, NOT an interpreter or estimator.

        ## CRITICAL RULES - READ CAREFULLY

        1. **ONLY extract information explicitly stated in the paper**
        - If coordinates are not given, leave the coordinates field empty
        - If a site name is not mentioned, do not invent one
        - If information is ambiguous or unclear, mark it as such

        2. **NEVER:**
        - Guess or estimate coordinates from place names
        - Invent precision that doesn't exist in the source
        - Convert between coordinate systems unless explicitly shown in the paper
        - Fill in missing information based on general knowledge
        - Assume information from context alone

        3. **ALWAYS:**
        - Preserve the exact format of coordinates as written in the paper
        - Note when information is approximate, estimated, or uncertain
        - Include the specific page or section where information was found
        - Flag when coordinates are withheld or stated as "not disclosed"

        ## Information to Extract

        For each archaeological site mentioned in the paper, extract the following:

        ### Site Identification
        - **site_name**: The exact name(s) used in the paper
        - **site_code**: Any alphanumeric codes or identifiers (e.g., "PA-KU-29")
        - **alternative_names**: Other names mentioned for the same site

        ### Location Information
        - **coordinates_explicit**: ONLY if coordinates are explicitly provided
        - Extract the exact text as written (e.g., "5.23°S, 60.12°W")
        - Note the format: decimal_degrees, DMS, UTM, etc.
        - Note the datum if specified (WGS84, SAD69, etc.)
        - Mark precision level: "exact", "approximate", "rounded"
        
        - **location_description**: Textual descriptions of location
        - Examples: "near the confluence of Xingu and Amazon rivers"
        - "15 km north of modern-day Santarém"
        - "in the Upper Tapajós basin"
        
        - **administrative_location**: Modern political boundaries mentioned
        - Country, state/province, municipality, etc.

        - **location_withheld**: Boolean - true if paper explicitly states location is not disclosed

        ### Temporal Information
        - **dating**: Dates or date ranges mentioned (e.g., "1200-1400 CE", "Late Pre-Columbian")
        - **cultural_period**: Cultural phases or periods named
        - **dating_method**: If specified (radiocarbon, thermoluminescence, etc.)
        - **dating_uncertainty**: Any caveats about dating

        ### Site Characteristics
        - **site_type**: Settlement, mound, ceremonial center, earthwork, etc.
        - **site_features**: Specific features mentioned (e.g., "circular plaza", "defensive ditch")
        - **site_size**: Dimensions if provided (area, diameter, length)
        - **site_condition**: Preservation state if mentioned

        ### Context
        - **study_type**: Is this a site being actively studied or just mentioned for comparison?
        - **source_location**: Page number(s) or section where this information appears
        - **confidence_level**: Your assessment - "high" (explicit coordinates + clear description), "medium" (clear description but no coordinates), "low" (vague or passing mention)
        - **extraction_notes**: Any important caveats, ambiguities, or clarifications

        ## Output Format

        Return a JSON object with this structure:

        {
        "paper_metadata": {
            "title": "extracted from paper",
            "authors": ["list", "of", "authors"],
            "year": "publication year",
            "doi": "if available"
        },
        "extraction_summary": {
            "total_sites_found": 0,
            "sites_with_explicit_coordinates": 0,
            "sites_with_descriptions_only": 0,
            "extraction_date": "YYYY-MM-DD"
        },
        "sites": [
            {
            "site_name": "string or null",
            "site_code": "string or null",
            "alternative_names": [],
            "coordinates": {
                "has_explicit_coordinates": false,
                "raw_text": null,
                "format": null,
                "latitude": null,
                "longitude": null,
                "datum": null,
                "precision_level": null
            },
            "location_description": null,
            "administrative_location": {
                "country": null,
                "state_province": null,
                "other": null
            },
            "location_withheld": false,
            "temporal": {
                "dating": null,
                "cultural_period": null,
                "dating_method": null,
                "uncertainty": null
            },
            "characteristics": {
                "site_type": null,
                "features": [],
                "size": null,
                "condition": null
            },
            "metadata": {
                "study_type": null,
                "source_location": null,
                "confidence_level": null,
                "extraction_notes": null
            }
            }
        ]
        }

        ## Your Task

        Extract all archaeological site information from the following paper:

        """
    
    return prompt + "\n" + paper_text + "\n\nReturn ONLY the JSON output, no additional commentary."


def extract_sites_with_llm(paper_text):
    """Use ChatGPT to extract site information"""
    
    client = OpenAI()
    
    prompt = create_extraction_prompt(paper_text)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=16000
    )
    
    response_text = response.choices[0].message.content
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(response_text)
        return extracted_data
    except json.JSONDecodeError as e:
        return {"raw_response": response_text, "parse_error": str(e)}


# ============================================================================
# GOOGLE EARTH ENGINE EXTRACTION
# ============================================================================

def create_grid_bbox(lat, lon, cell_size_km):
    """Create bounding box for extraction"""
    half_size_deg = (cell_size_km / 2) / 111.32
    
    min_lon = lon - half_size_deg
    max_lon = lon + half_size_deg
    min_lat = lat - half_size_deg
    max_lat = lat + half_size_deg
    
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def get_sentinel2_image(roi):
    """Get least cloudy Sentinel-2 image for region"""
    config = GEE_CONFIG['sentinel2']
    
    collection = (
        ee.ImageCollection(config['collection'])
        .filterBounds(roi)
        .filterDate(config['date_start'], config['date_end'])
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', config['cloud_cover_max']))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )
    
    return ee.Image(collection.first())


def get_dem_data(roi):
    """Get SRTM DEM elevation data (30m resolution, globally available)"""
    # SRTM is a single global Image (simpler than ImageCollection)
    # 30m resolution, will be resampled to match grid
    dem = ee.Image('USGS/SRTMGL1_003')
    
    return dem


def calculate_slope(dem_image):
    """Calculate slope from DEM"""
    return ee.Terrain.slope(dem_image)


def extract_band_array(image, band, roi, pixels):
    """Extract a single band as numpy array - EXACTLY like working gee_extractor.py"""
    band_image = image.select(band)
    
    array = band_image.sampleRectangle(region=roi, defaultValue=0)
    data = array.get(band).getInfo()
    arr = np.array(data, dtype=np.float32)
    
    # Diagnostic info
    print(f"  Band {band}: shape={arr.shape}, min={np.min(arr):.2f}, max={np.max(arr):.2f}, mean={np.mean(arr):.2f}")
    
    # Check for issues
    if np.all(arr == 0):
        print(f"    WARNING: Band {band} is all zeros!")
    if np.isnan(arr).any():
        print(f"    WARNING: Band {band} contains NaN values!")
    
    # Resize if needed
    if arr.shape != (pixels, pixels):
        from scipy.ndimage import zoom
        original_shape = arr.shape  # Store original shape before resizing
        zoom_factors = (pixels / arr.shape[0], pixels / arr.shape[1])
        arr = zoom(arr, zoom_factors, order=1)
        print(f"    Resized from {original_shape} to ({pixels}, {pixels})")
    
    return arr


def calculate_ndvi(b8, b4):
    """Calculate NDVI"""
    numerator = b8 - b4
    denominator = b8 + b4
    ndvi = np.divide(numerator, denominator, 
                    out=np.zeros_like(numerator), 
                    where=denominator!=0)
    return ndvi.astype(np.float32)


def calculate_ndwi(b3, b8):
    """Calculate NDWI"""
    numerator = b3 - b8
    denominator = b3 + b8
    ndwi = np.divide(numerator, denominator,
                    out=np.zeros_like(numerator),
                    where=denominator!=0)
    return ndwi.astype(np.float32)


def calculate_bsi(b11, b4, b8, b2):
    """Calculate BSI"""
    numerator = (b11 + b4) - (b8 + b2)
    denominator = (b11 + b4) + (b8 + b2)
    bsi = np.divide(numerator, denominator,
                   out=np.zeros_like(numerator),
                   where=denominator!=0)
    return bsi.astype(np.float32)


def extract_gee_data(lat, lon, site_name="site"):
    """
    Extract all GEE data for a location - EXACTLY like working gee_extractor.py
    
    Returns: dict with channels and metadata
    """
    if not GEE_INITIALIZED:
        raise Exception("Google Earth Engine not initialized. Check service account configuration.")
    
    cell_size_km = GEE_CONFIG['cell_size_km']
    pixels = GEE_CONFIG['pixels_per_km']
    
    # Create region of interest
    roi = create_grid_bbox(lat, lon, cell_size_km)
    
    # Get imagery
    s2_image = get_sentinel2_image(roi)
    dem_image = get_dem_data(roi)
    slope_image = calculate_slope(dem_image)
    
    # SRTM uses 'elevation' as the band name
    dem_band = 'elevation'
    print(f"Using SRTM DEM with band: {dem_band}")
    
    # Extract all bands
    channels = {}
    
    print("Extracting Sentinel-2 bands...")
    for band in GEE_CONFIG['sentinel2']['bands']:
        channels[band] = extract_band_array(s2_image, band, roi, pixels)
    
    print("Extracting DEM...")
    channels['DEM'] = extract_band_array(dem_image, dem_band, roi, pixels)
    
    print("Extracting Slope...")
    channels['Slope'] = extract_band_array(slope_image, 'slope', roi, pixels)
    
    # Calculate indices
    print("Calculating spectral indices...")
    channels['NDVI'] = calculate_ndvi(channels['B8'], channels['B4'])
    channels['NDWI'] = calculate_ndwi(channels['B3'], channels['B8'])
    channels['BSI'] = calculate_bsi(channels['B11'], channels['B4'], 
                                     channels['B8'], channels['B2'])
    
    print(f"  NDVI: min={np.min(channels['NDVI']):.2f}, max={np.max(channels['NDVI']):.2f}")
    print(f"  NDWI: min={np.min(channels['NDWI']):.2f}, max={np.max(channels['NDWI']):.2f}")
    print(f"  BSI: min={np.min(channels['BSI']):.2f}, max={np.max(channels['BSI']):.2f}")
    
    # Get metadata
    image_info = s2_image.getInfo()
    metadata = {
        'site_name': site_name,
        'latitude': lat,
        'longitude': lon,
        'cell_size_km': cell_size_km,
        'image_id': image_info['id'] if image_info else None,
        'cloud_cover': image_info['properties'].get('CLOUDY_PIXEL_PERCENTAGE') if image_info else None,
        'acquisition_date': image_info['properties'].get('GENERATION_TIME') if image_info else None
    }
    
    return {'channels': channels, 'metadata': metadata}


def create_overview_visualization(channels, site_name, output_path):
    """
    Create 2x3 grid visualization of all channels.
    
    Layout:
    [RGB Composite] [NDVI] [NDWI]
    [BSI]           [DEM]  [Slope]
    
    Args:
        channels: Dict of channel arrays (B2, B3, B4, B8, B11, B12, DEM, Slope, NDVI, NDWI, BSI)
        site_name: Name for the plot title
        output_path: Where to save the PNG
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Site: {site_name}', fontsize=16)
    
    # RGB Composite (B4=Red, B3=Green, B2=Blue)
    r = channels['B4']
    g = channels['B3']
    b = channels['B2']
    rgb = np.stack([r, g, b], axis=-1)
    
    # Normalize RGB to 0-1 range using percentile stretch
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    
    axes[0, 0].imshow(rgb_norm)
    axes[0, 0].set_title('RGB Composite (B4-B3-B2)')
    axes[0, 0].axis('off')
    
    # NDVI
    ndvi = channels['NDVI']
    im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
    axes[0, 1].set_title('NDVI')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # NDWI
    ndwi = channels['NDWI']
    im2 = axes[0, 2].imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('NDWI')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # BSI
    bsi = channels['BSI']
    im3 = axes[1, 0].imshow(bsi, cmap='YlOrBr', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('BSI')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # DEM
    dem = channels['DEM']
    im4 = axes[1, 1].imshow(dem, cmap='terrain')
    axes[1, 1].set_title('DEM (Elevation)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Slope
    slope = channels['Slope']
    im5 = axes[1, 2].imshow(slope, cmap='plasma')
    axes[1, 2].set_title('Slope')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def package_data_as_zip(data, site_name, output_path):
    """
    Package extracted data as a ZIP file.
    
    Structure:
    site_name/
        channels/
            B2.npy, B3.npy, ..., NDVI.npy, etc.
        visualizations/
            overview.png
        metadata.json
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save each channel as .npy
        for channel_name, array in data['channels'].items():
            # Save to temporary file
            temp_npy = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            np.save(temp_npy.name, array)
            temp_npy.close()
            
            # Add to zip
            zipf.write(temp_npy.name, 
                      f"{site_name}/channels/{channel_name}.npy")
            
            # Clean up temp file
            os.unlink(temp_npy.name)
        
        # Generate and save visualization
        temp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_png.close()
        
        try:
            create_overview_visualization(
                data['channels'], 
                site_name, 
                temp_png.name
            )
            
            # Add visualization to zip
            zipf.write(temp_png.name, 
                      f"{site_name}/visualizations/overview.png")
            
            # Clean up temp file
            os.unlink(temp_png.name)
        except Exception as e:
            print(f"Warning: Failed to create visualization - {str(e)}")
            # Continue without visualization if it fails
            try:
                os.unlink(temp_png.name)
            except:
                pass
        
        # Save metadata as JSON
        metadata_str = json.dumps(data['metadata'], indent=2)
        zipf.writestr(f"{site_name}/metadata.json", metadata_str)


# ============================================================================
# FLASK ROUTES
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    """Handle PDF upload and extraction"""
    
    # Check if file was uploaded
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Extract text from PDF
        paper_text = extract_text_from_pdf(temp_path)
        
        # Extract sites using LLM
        extracted_data = extract_sites_with_llm(paper_text)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'data': extracted_data
        })
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'error': f'Processing error: {str(e)}'
        }), 500


@app.route('/download_gee', methods=['POST'])
def download_gee():
    """
    Handle GEE data extraction and download.
    
    Expects JSON:
    {
        "site_name": "Site Name",
        "coordinates_raw": "5.23°S, 60.12°W",
        "latitude": 5.23,  (optional, will be parsed from coordinates_raw)
        "longitude": -60.12  (optional)
    }
    """
    
    if not GEE_INITIALIZED:
        return jsonify({
            'error': 'Google Earth Engine not initialized. Please configure service account.'
        }), 500
    
    try:
        data = request.get_json()
        
        site_name = data.get('site_name', 'unknown_site')
        coordinates_raw = data.get('coordinates_raw', '')
        
        # Try to get coordinates from request or parse from raw text
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if lat is None or lon is None:
            # Parse from raw coordinates
            lat, lon = parse_coordinate_string(coordinates_raw)
        
        if lat is None or lon is None:
            return jsonify({
                'error': f'Could not parse coordinates from: {coordinates_raw}'
            }), 400
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                'error': f'Invalid coordinates: lat={lat}, lon={lon}'
            }), 400
        
        # Extract data from GEE
        print(f"\nExtracting data for {site_name} at ({lat:.4f}, {lon:.4f})")
        gee_data = extract_gee_data(lat, lon, site_name)
        
        # Create safe filename
        safe_site_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' 
                                for c in site_name)
        zip_filename = f"{safe_site_name}_{lat:.4f}_{lon:.4f}.zip"
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        # Package as ZIP
        print("Packaging data into ZIP file...")
        package_data_as_zip(gee_data, safe_site_name, temp_zip.name)
        print(f"✓ Package created: {zip_filename}")
        
        # Send file to browser and cleanup after
        response = send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
        # Schedule cleanup of temp file after response is sent
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(temp_zip.name)
            except:
                pass
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'GEE extraction error: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.")
        print("Create a .env file with: OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    if not GEE_INITIALIZED:
        print("\nWARNING: Google Earth Engine not initialized!")
        print("GEE download features will not work.")
        print("Please configure gee_service_account.json")
    
    print("\nStarting Archaeological Site Extraction Web Interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)