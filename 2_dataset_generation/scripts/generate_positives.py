#!/usr/bin/env python3

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate

sys.path.append(str(Path(__file__).parent.parent))

from src.gee_extractor import GEEExtractor
from src.index_calculator import IndexCalculator
from src.file_manager import FileManager
from src.metadata_builder import MetadataBuilder


def load_config(config_path='config/settings.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sites(sites_path):
    df = pd.read_csv(sites_path)
    required_cols = ['site_id', 'latitude', 'longitude']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def generate_grid_id(index, rotation_angle=None):
    """Generate grid ID with optional rotation suffix."""
    if rotation_angle is not None:
        return f"grid_{index:06d}_rot{int(rotation_angle):03d}"
    else:
        return f"grid_{index:06d}"


def rotate_and_crop_channels(channels_dict, rotation_angle, final_size):
    """
    Rotate all channels and crop to final size.
    
    Args:
        channels_dict: Dict of channel arrays (large size, e.g., 140x140)
        rotation_angle: Rotation angle in degrees
        final_size: Final cropped size (e.g., 100)
    
    Returns:
        Dict of rotated and cropped channel arrays
    """
    rotated_channels = {}
    
    for channel_name, data in channels_dict.items():
        # Rotate (reshape=False keeps original size)
        if rotation_angle != 0:
            rotated = rotate(data, rotation_angle, reshape=False, order=1, mode='constant', cval=0)
        else:
            rotated = data
        
        # Crop center to final size
        h, w = rotated.shape
        center_h, center_w = h // 2, w // 2
        half_size = final_size // 2
        
        cropped = rotated[
            center_h - half_size : center_h + half_size,
            center_w - half_size : center_w + half_size
        ]
        
        rotated_channels[channel_name] = cropped.astype(np.float32)
    
    return rotated_channels


def main():
    print("Archaeological Site Data Generator - Positives WITH ROTATION GENERATION")
    print("=" * 70)
    
    config = load_config()
    sites_df = load_sites(config['paths']['input_sites'])
    
    # Get rotation buffer factor
    rotation_buffer = config.get('augmentation', {}).get('rotation_buffer_factor', 1.0)
    base_cell_size = config['grid']['cell_size_km']
    base_pixels_per_km = config['grid']['pixels_per_km']
    
    # Calculate extraction parameters
    extraction_cell_size = base_cell_size * rotation_buffer
    extraction_pixels_per_side = int(base_pixels_per_km * extraction_cell_size)
    
    # Get rotation generation parameters
    rotation_config = config.get('augmentation', {}).get('rotation_generation', {})
    rotation_enabled = rotation_config.get('enabled', False)
    rotation_step = rotation_config.get('rotation_step', 90)
    
    # Calculate rotation angles
    if rotation_enabled:
        num_rotations = int(360 / rotation_step)
        rotation_angles = [i * rotation_step for i in range(num_rotations)]
    else:
        num_rotations = 1
        rotation_angles = [0]
    
    print(f"\nExtraction Parameters:")
    print(f"  Base cell size: {base_cell_size} km × {base_cell_size} km")
    print(f"  Rotation buffer factor: {rotation_buffer}x")
    print(f"  Extraction cell size: {extraction_cell_size} km × {extraction_cell_size} km")
    print(f"  Extraction pixels: {extraction_pixels_per_side} × {extraction_pixels_per_side}")
    print()
    
    print(f"Rotation Generation:")
    print(f"  Enabled: {rotation_enabled}")
    print(f"  Rotation step: {rotation_step}°")
    print(f"  Number of rotations: {num_rotations}")
    print(f"  Rotation angles: {rotation_angles}")
    print(f"  Final output size: {base_pixels_per_km} × {base_pixels_per_km} pixels")
    print()
    
    total_grids_expected = len(sites_df) * num_rotations
    print(f"Dataset Summary:")
    print(f"  Sites to process: {len(sites_df)}")
    print(f"  Rotations per site: {num_rotations}")
    print(f"  Total grids to generate: {total_grids_expected}")
    print()
    
    print(f"Output directory: {config['paths']['output_dir']}")
    print()
    
    # Temporarily modify config for extraction
    original_cell_size = config['grid']['cell_size_km']
    original_pixels_per_km = config['grid']['pixels_per_km']
    
    config['grid']['cell_size_km'] = extraction_cell_size
    config['grid']['pixels_per_km'] = extraction_pixels_per_side
    
    extractor = GEEExtractor(config)
    file_manager = FileManager(config['paths']['output_dir'])
    metadata_builder = MetadataBuilder(config['paths']['output_dir'])
    
    # Load existing metadata to determine starting index
    existing_metadata = metadata_builder.load_existing_metadata()
    if existing_metadata is not None:
        print(f"Found existing metadata with {len(existing_metadata)} records")
    
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for idx, row in sites_df.iterrows():
        site_id = row['site_id']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"[{idx+1}/{len(sites_df)}] Processing {site_id}")
        print(f"  Location: {lat:.4f}°, {lon:.4f}°")
        print(f"  Extracting: {extraction_cell_size} km × {extraction_cell_size} km")
        
        try:
            # Extract large tile ONCE from GEE
            channels, gee_metadata = extractor.extract_all_channels(lat, lon)
            
            # Verify extracted size
            expected_shape = (extraction_pixels_per_side, extraction_pixels_per_side)
            actual_shape = channels['B2'].shape
            
            if actual_shape != expected_shape:
                print(f"  ⚠️  WARNING: Expected {expected_shape}, got {actual_shape}")
            
            print(f"  → Extracted {len(channels)} bands at {actual_shape[0]}×{actual_shape[1]} pixels")
            if gee_metadata.get('cloud_cover'):
                print(f"     Cloud cover: {gee_metadata['cloud_cover']:.1f}%")
            
            # Calculate indices on large tile
            indices = IndexCalculator.calculate_all_indices(channels)
            print(f"  → Calculated {len(indices)} spectral indices")
            
            # Combine all channels
            all_channels = {**channels, **indices}
            
            # Generate all rotations
            print(f"  → Generating {num_rotations} rotations...")
            
            for rot_idx, rotation_angle in enumerate(rotation_angles):
                grid_id = generate_grid_id(idx + 1, rotation_angle)
                
                if file_manager.grid_exists(grid_id):
                    print(f"     [{rot_idx+1}/{num_rotations}] {rotation_angle:3d}° - SKIPPED (exists)")
                    skipped_count += 1
                    continue
                
                # Rotate and crop
                rotated_cropped = rotate_and_crop_channels(
                    all_channels, 
                    rotation_angle, 
                    base_pixels_per_km
                )
                
                # Save this rotation
                grid_dir, channels_dir, labels_dir = file_manager.create_grid_structure(grid_id)
                
                # Save all channels
                for name, data in rotated_cropped.items():
                    filepath = channels_dir / f"{name}.npy"
                    np.save(filepath, data.astype(np.float32))
                
                # Save label
                pos_type = row.get('site_type', 'unknown')
                label_path = labels_dir / 'binary_label.npy'
                np.save(label_path, np.array([1], dtype=np.int32))
                
                if pos_type:
                    with open(labels_dir / 'pos_type.txt', 'w') as f:
                        f.write(pos_type)
                
                with open(labels_dir / 'neg_type.txt', 'w') as f:
                    f.write('null')
                
                # Save info
                info = {
                    'grid_id': grid_id,
                    'site_id': site_id,
                    'centroid_lat': lat,
                    'centroid_lon': lon,
                    'site_type': pos_type,
                    'source': row.get('source', 'unknown'),
                    'rotation_angle': rotation_angle,
                    'rotation_step': rotation_step,
                    'extraction_size_km': extraction_cell_size,
                    'extraction_pixels': extraction_pixels_per_side,
                    'rotation_buffer_factor': rotation_buffer,
                    'base_cell_size_km': base_cell_size,
                    'final_size_pixels': base_pixels_per_km,
                    'note': f'Extracted {extraction_pixels_per_side}x{extraction_pixels_per_side}, rotated {rotation_angle}°, cropped to {base_pixels_per_km}x{base_pixels_per_km}',
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
                    label=config['metadata']['label'],
                    label_source=config['metadata']['label_source'],
                    image_path=image_path
                )
                
                print(f"     [{rot_idx+1}/{num_rotations}] {rotation_angle:3d}° - SUCCESS ({grid_id})")
                success_count += 1
            
            print(f"  ✓ Completed all rotations for {site_id}")
            
        except Exception as e:
            print(f"  ✗ FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
            failure_count += 1
            continue
        
        print()
    
    # Restore original config
    config['grid']['cell_size_km'] = original_cell_size
    config['grid']['pixels_per_km'] = original_pixels_per_km
    
    print("=" * 70)
    print("Processing Complete!")
    print(f"  Total grids generated: {success_count}")
    print(f"  Failed:  {failure_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Expected: {total_grids_expected}")
    
    if success_count > 0:
        parquet_path = metadata_builder.save_metadata()
        print(f"\nMetadata saved to: {parquet_path}")
        
        # Show rotation distribution
        if existing_metadata is not None:
            full_metadata = metadata_builder.load_existing_metadata()
            if 'grid_id' in full_metadata.columns:
                # Count rotations by parsing grid_id
                rotation_counts = {}
                for grid_id in full_metadata['grid_id']:
                    if '_rot' in grid_id:
                        angle = int(grid_id.split('_rot')[1])
                        rotation_counts[angle] = rotation_counts.get(angle, 0) + 1
                
                if rotation_counts:
                    print(f"\nRotation Distribution:")
                    for angle in sorted(rotation_counts.keys()):
                        print(f"  {angle:3d}°: {rotation_counts[angle]} grids")
    
    print()


if __name__ == "__main__":
    main()