#!/usr/bin/env python3

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.gee_extractor import GEEExtractor
from src.index_calculator import IndexCalculator
from src.file_manager import FileManager
from src.metadata_builder import MetadataBuilder


def load_config(config_path='config/settings.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_expected_final_positives(sites_df, config):
    """Calculate expected final positive count from parameters."""
    num_sites = len(sites_df)
    
    # Get rotation configuration
    rotation_config = config.get('augmentation', {}).get('rotation_generation', {})
    rotation_enabled = rotation_config.get('enabled', False)
    rotation_step = rotation_config.get('rotation_step', 360)
    
    if rotation_enabled:
        num_rotations = int(360 / rotation_step)
    else:
        num_rotations = 1
    
    # Get radiometric augmentation configuration
    radiometric_config = config.get('augmentation', {}).get('radiometric', {})
    radiometric_enabled = radiometric_config.get('enabled', False)
    
    if radiometric_enabled:
        num_variants = len(radiometric_config.get('variants', ['aug1', 'aug2', 'aug3']))
        # Total = original + variants
        multiplier = 1 + num_variants
    else:
        multiplier = 1
    
    # Calculate final expected count
    final_positives = num_sites * num_rotations * multiplier
    
    return final_positives, num_sites, num_rotations, multiplier


def get_base_coordinates():
    """
    Base coordinates of known locations in Peru/Bolivia.
    These will be used as centers to generate variations around.
    """
    
    # 15 MAJOR CITIES
    urban_coords = [
        (-12.0464, -77.0428, "Lima, Peru"),
        (-16.4090, -71.5375, "Arequipa, Peru"),
        (-13.5319, -71.9675, "Cusco, Peru"),
        (-11.0187, -76.8686, "Huancayo, Peru"),
        (-8.3791, -74.5539, "Pucallpa, Peru"),
        (-12.5897, -69.1890, "Puerto Maldonado, Peru"),
        (-16.5000, -68.1500, "La Paz, Bolivia"),
        (-17.7833, -63.1821, "Santa Cruz, Bolivia"),
        (-19.0333, -65.2627, "Sucre, Bolivia"),
        (-17.3895, -66.1568, "Cochabamba, Bolivia"),
        (-11.0049, -77.6050, "Huacho, Peru"),
        (-14.0650, -75.7350, "Ica, Peru"),
        (-8.1116, -79.0288, "Trujillo, Peru"),
        (-6.7014, -79.9061, "Chiclayo, Peru"),
        (-5.1973, -80.6326, "Piura, Peru"),
    ]
    
    # 15 WATER BODIES
    water_coords = [
        (-15.8422, -69.9406, "Lake Titicaca (south)"),
        (-15.7333, -69.8667, "Lake Titicaca (central)"),
        (-16.0000, -69.0500, "Lake Titicaca (east)"),
        (-15.5500, -70.0500, "Lake Titicaca (north)"),
        (-12.0000, -77.1000, "Pacific Ocean - Lima coast"),
        (-16.4000, -72.5000, "Pacific Ocean - Arequipa coast"),
        (-8.0000, -79.0000, "Pacific Ocean - Trujillo coast"),
        (-13.2000, -76.2000, "Pacific Ocean - Pisco coast"),
        (-11.8000, -75.2000, "Rio Mantaro area"),
        (-12.5000, -69.0000, "Rio Madre de Dios"),
        (-8.5000, -74.0000, "Rio Ucayali"),
        (-4.5000, -73.2000, "Amazon River - Peru"),
        (-17.9667, -67.1167, "Lake Poopó, Bolivia"),
        (-14.0000, -76.0000, "Reserva Nacional de Paracas"),
        (-13.7000, -73.7000, "Rio Apurímac"),
    ]
    
    # 15 CROPLAND LOCATIONS
    cropland_coords = [
        (-12.2000, -76.9000, "Cañete Valley - Lima"),
        (-13.4000, -76.1000, "Pisco Valley"),
        (-14.8000, -75.9000, "Palpa Valley"),
        (-11.2000, -75.9000, "Junín agricultural area"),
        (-8.6000, -78.5000, "Chicama Valley - Trujillo"),
        (-6.9000, -79.7000, "Lambayeque Valley"),
        (-5.5000, -80.4000, "Piura agricultural region"),
        (-16.3000, -71.6000, "Arequipa agricultural valley"),
        (-17.8000, -63.0000, "Santa Cruz agricultural plain, Bolivia"),
        (-12.7000, -76.2000, "Mala Valley"),
        (-13.7000, -76.5000, "Chincha Valley"),
        (-15.5000, -71.5000, "Majes Valley"),
        (-17.0000, -66.0000, "Cochabamba Valley, Bolivia"),
        (-11.5000, -77.5000, "Huaral Valley"),
        (-9.5000, -77.5000, "Casma Valley"),
    ]
    
    return {
        'urban': urban_coords,
        'water': water_coords,
        'cropland': cropland_coords
    }


def generate_variations(base_coords, count, max_radius_km=5.0, min_spacing_km=1.5):
    """
    Generate variations around base coordinates.
    
    Args:
        base_coords: List of (lat, lon, name) tuples
        count: Total number of samples needed
        max_radius_km: Maximum distance from base coordinate (km)
        min_spacing_km: Minimum spacing between samples (km)
    
    Returns:
        List of (lat, lon, description) tuples
    """
    variations = []
    samples_per_base = int(np.ceil(count / len(base_coords)))
    
    print(f"  Generating {count} samples from {len(base_coords)} base locations")
    print(f"  ~{samples_per_base} variations per location")
    print(f"  Max radius: {max_radius_km} km, Min spacing: {min_spacing_km} km")
    
    for base_lat, base_lon, base_name in base_coords:
        if len(variations) >= count:
            break
        
        # Generate samples in a radius around this base location
        base_variations = []
        attempts = 0
        max_attempts = samples_per_base * 10
        
        while len(base_variations) < samples_per_base and attempts < max_attempts:
            attempts += 1
            
            # Random offset within max_radius_km
            # Convert km to degrees (approximate)
            lat_deg_per_km = 1.0 / 111.32
            lon_deg_per_km = 1.0 / (111.32 * np.cos(np.radians(base_lat)))
            
            # Random distance and angle
            distance_km = np.random.uniform(0, max_radius_km)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate offset
            lat_offset = distance_km * np.cos(angle) * lat_deg_per_km
            lon_offset = distance_km * np.sin(angle) * lon_deg_per_km
            
            new_lat = base_lat + lat_offset
            new_lon = base_lon + lon_offset
            
            # Check minimum spacing with existing variations
            too_close = False
            min_spacing_deg = min_spacing_km * lat_deg_per_km
            
            for existing_lat, existing_lon, _ in base_variations:
                dist = np.sqrt((new_lat - existing_lat)**2 + (new_lon - existing_lon)**2)
                if dist < min_spacing_deg:
                    too_close = True
                    break
            
            if not too_close:
                description = f"{base_name} +{distance_km:.1f}km"
                base_variations.append((new_lat, new_lon, description))
        
        variations.extend(base_variations[:samples_per_base])
    
    # Return exactly count samples
    return variations[:count]


def generate_grid_id(index, prefix='lneg'):
    """Generate grid ID with custom prefix for landcover negatives."""
    return f"{prefix}_{index:06d}"


def main():
    print("Archaeological Site Data Generator - Landcover Negatives (Parameter-Based)")
    print("=" * 70)
    print("Uses base coordinates + generates variations")
    print("Target count calculated from config parameters (no dependencies)")
    print("=" * 70)
    
    config = load_config()
    sites_df = pd.read_csv(config['paths']['input_sites'])
    
    # Calculate expected final positive count from parameters
    final_positives, num_sites, num_rotations, aug_multiplier = \
        calculate_expected_final_positives(sites_df, config)
    
    print(f"\nDataset Parameters:")
    print(f"  Number of sites: {num_sites}")
    print(f"  Rotations per site: {num_rotations}")
    print(f"  Augmentation multiplier: {aug_multiplier}x (original + variants)")
    print(f"  Expected final positives: {final_positives}")
    print(f"    Calculation: {num_sites} × {num_rotations} × {aug_multiplier} = {final_positives}")
    
    # Calculate target landcover count based on ratio
    landcover_ratio = config['dataset_ratios']['landcover_to_positive_ratio']
    target_landcover_total = int(final_positives * landcover_ratio)
    
    print(f"\nLandcover Negative Strategy:")
    print(f"  Ratio to positives: {landcover_ratio} ({landcover_ratio*100:.0f}%)")
    print(f"  Target landcover samples: {target_landcover_total}")
    print(f"    Calculation: {final_positives} × {landcover_ratio} = {target_landcover_total}")
    
    # Get landcover composition
    landcover_config = config['label_0_composition']['landcover']
    urban_ratio = landcover_config['urban_ratio']
    water_ratio = landcover_config['water_ratio']
    cropland_ratio = landcover_config['cropland_ratio']
    
    # Calculate counts for each type
    urban_count = int(target_landcover_total * urban_ratio)
    water_count = int(target_landcover_total * water_ratio)
    cropland_count = target_landcover_total - urban_count - water_count
    
    print(f"\n  Breakdown:")
    print(f"    → Urban: {urban_count} ({urban_ratio*100:.0f}%)")
    print(f"    → Water: {water_count} ({water_ratio*100:.0f}%)")
    print(f"    → Cropland: {cropland_count} ({cropland_ratio*100:.0f}%)")
    print(f"  Prefix: 'lneg_'")
    print()
    
    # Initialize components
    file_manager = FileManager(config['paths']['output_dir'])
    metadata_builder = MetadataBuilder(config['paths']['output_dir'])
    
    # Load existing metadata to determine starting index
    existing_metadata = metadata_builder.load_existing_metadata()
    if existing_metadata is not None:
        existing_landcover = existing_metadata[existing_metadata['grid_id'].str.startswith('lneg_')]
        num_existing_landcover = len(existing_landcover)
        start_index = num_existing_landcover + 1
        print(f"Existing landcover negatives: {num_existing_landcover}")
        print(f"Starting from index: {start_index}")
        
        # Calculate remaining to generate
        num_to_generate = target_landcover_total - num_existing_landcover
        if num_to_generate <= 0:
            print(f"\n✓ Already have enough landcover negatives ({num_existing_landcover} >= {target_landcover_total})")
            print("No new landcover data needed. Exiting.")
            return
        
        print(f"Will generate: {num_to_generate} new samples")
        
        # Recalculate breakdown for remaining samples
        urban_count = int(num_to_generate * urban_ratio)
        water_count = int(num_to_generate * water_ratio)
        cropland_count = num_to_generate - urban_count - water_count
        print(f"\nAdjusted breakdown for new samples:")
        print(f"  → Urban: {urban_count}")
        print(f"  → Water: {water_count}")
        print(f"  → Cropland: {cropland_count}")
    else:
        start_index = 1
        print("No existing metadata, starting from index 1")
    
    print()
    
    # Get base coordinates
    base_coords = get_base_coordinates()
    
    print(f"Base coordinates available:")
    print(f"  Urban: {len(base_coords['urban'])} cities")
    print(f"  Water: {len(base_coords['water'])} water bodies")
    print(f"  Cropland: {len(base_coords['cropland'])} agricultural areas")
    print()
    
    # Get variation parameters from config
    sampling_config = config.get('landcover_sampling', {})
    max_radius = sampling_config.get('max_radius_km', 6.0)
    min_spacing = sampling_config.get('min_spacing_km', 1.5)
    
    # Generate variations
    print("Generating coordinate variations...")
    print()
    
    all_samples = {}
    
    print("Urban samples:")
    all_samples['urban'] = generate_variations(
        base_coords['urban'], 
        urban_count,
        max_radius_km=8.0,  # Larger cities
        min_spacing_km=min_spacing
    )
    print(f"  ✓ Generated {len(all_samples['urban'])} urban samples")
    print()
    
    print("Water samples:")
    all_samples['water'] = generate_variations(
        base_coords['water'], 
        water_count,
        max_radius_km=10.0,  # Water bodies can be large
        min_spacing_km=min_spacing * 1.3
    )
    print(f"  ✓ Generated {len(all_samples['water'])} water samples")
    print()
    
    print("Cropland samples:")
    all_samples['cropland'] = generate_variations(
        base_coords['cropland'], 
        cropland_count,
        max_radius_km=max_radius,
        min_spacing_km=min_spacing
    )
    print(f"  ✓ Generated {len(all_samples['cropland'])} cropland samples")
    print()
    
    # Show examples
    print("Sample coordinates (first 3 per category):")
    for category, samples in all_samples.items():
        print(f"\n  {category.upper()}:")
        for lat, lon, desc in samples[:3]:
            print(f"    {desc}: ({lat:.4f}°, {lon:.4f}°)")
        if len(samples) > 3:
            print(f"    ... and {len(samples) - 3} more")
    print()
    
    extractor = GEEExtractor(config)
    
    print(f"\nStarting data extraction at 10m resolution...")
    print(f"Output directory: {config['paths']['output_dir']}")
    print()
    
    success_count = 0
    failure_count = 0
    
    # Process each category
    for category, samples in all_samples.items():
        print(f"\nProcessing {category} samples ({len(samples)} total)...")
        
        for idx, (lat, lon, location_desc) in enumerate(samples):
            grid_id = generate_grid_id(start_index + success_count)
            
            # Show progress every 50 samples
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(samples)}] {location_desc}")
                print(f"    Location: {lat:.4f}°, {lon:.4f}° (Grid: {grid_id})")
            
            try:
                # Extract channels at full 10m resolution
                channels, gee_metadata = extractor.extract_all_channels(lat, lon)
                
                # Calculate indices
                indices = IndexCalculator.calculate_all_indices(channels)
                
                # Save data
                grid_dir, channels_dir, labels_dir = file_manager.create_grid_structure(grid_id)
                file_manager.save_all_channels(channels_dir, channels, indices)
                
                # Save label
                label_path = labels_dir / 'binary_label.npy'
                np.save(label_path, np.array([0], dtype=np.int32))
                
                with open(labels_dir / 'neg_type.txt', 'w') as f:
                    f.write(category)
                
                with open(labels_dir / 'pos_type.txt', 'w') as f:
                    f.write('null')
                
                # Save info
                info = {
                    'grid_id': grid_id,
                    'centroid_lat': lat,
                    'centroid_lon': lon,
                    'label': 0,
                    'neg_type': category,
                    'location_description': location_desc,
                    'source': 'manual_coordinates_with_variations',
                    'sampling_method': 'base_location_plus_random_offset',
                    'gee_image_id': gee_metadata.get('image_id'),
                    'cloud_cover': gee_metadata.get('cloud_cover'),
                    'acquisition_date': gee_metadata.get('acquisition_date')
                }
                file_manager.save_info(grid_dir, info)
                
                # Add to metadata
                image_path = file_manager.get_image_path(grid_id)
                metadata_builder.add_record(
                    grid_id=grid_id,
                    lat=lat,
                    lon=lon,
                    label=0,
                    label_source=category,
                    image_path=image_path
                )
                
                if (idx + 1) % 50 == 0 or idx == 0:
                    print(f"    ✓ SUCCESS")
                
                success_count += 1
                
            except Exception as e:
                print(f"    ✗ FAILED - {str(e)}")
                failure_count += 1
        
        print(f"  Completed {category}")
    
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print(f"  Target:  {urban_count + water_count + cropland_count}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failure_count}")
    if (urban_count + water_count + cropland_count) > 0:
        print(f"  Success rate: {success_count / (urban_count + water_count + cropland_count) * 100:.1f}%")
    
    if success_count > 0:
        parquet_path = metadata_builder.save_metadata()
        print(f"\nMetadata saved to: {parquet_path}")
        
        # Show final statistics
        final_metadata = metadata_builder.load_existing_metadata()
        final_landcover = len(final_metadata[final_metadata['grid_id'].str.startswith('lneg_')])
        
        print(f"\nFinal Dataset Statistics:")
        print(f"  Total grids: {len(final_metadata)}")
        print(f"  Landcover negatives: {final_landcover}")
        print(f"  Target landcover: {target_landcover_total}")
        print(f"  Progress: {final_landcover / target_landcover_total * 100:.1f}%")
    
    print()


if __name__ == "__main__":
    main()