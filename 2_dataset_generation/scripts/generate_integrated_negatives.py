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


def calculate_site_radius(row):
    """Calculate site radius from a_width and b_width."""
    a_width = row.get('a_width')
    b_width = row.get('b_width')
    
    # Convert to numeric, treating non-numeric as None
    try:
        a_width = float(a_width) if pd.notna(a_width) else None
    except (ValueError, TypeError):
        a_width = None
    
    try:
        b_width = float(b_width) if pd.notna(b_width) else None
    except (ValueError, TypeError):
        b_width = None
    
    # Determine radius
    if a_width is None:
        # Both unknown, use default 500m diameter
        return 250.0
    elif b_width is None:
        # Only a_width known, assume it's diameter
        return a_width / 2.0
    else:
        # Both known, use max as diameter
        return max(a_width, b_width) / 2.0


def rotate_entire_landscape(channels_dict, rotation_angle):
    """
    Rotate entire landscape as one continuous area.
    
    Args:
        channels_dict: Dict of channel arrays (large tile)
        rotation_angle: Rotation angle in degrees
    
    Returns:
        Dict of rotated channel arrays (same size as input)
    """
    rotated_channels = {}
    
    for channel_name, data in channels_dict.items():
        if rotation_angle != 0:
            rotated = rotate(data, rotation_angle, reshape=False, order=1, mode='constant', cval=0)
        else:
            rotated = data
        
        rotated_channels[channel_name] = rotated.astype(np.float32)
    
    return rotated_channels


def extract_4_corners_from_rotated(rotated_channels, sampling_distance_px, slice_size_px):
    """
    Extract 4 corner slices from rotated landscape, avoiding center site.
    
    In the rotated coordinate system:
    - Center is the (rotated) site
    - Sample 4 corners at sampling_distance from center
    - NW, NE, SW, SE are relative to the rotated image axes
    
    Args:
        rotated_channels: Dict of rotated channel arrays
        sampling_distance_px: Distance in pixels from center to slice centers
        slice_size_px: Size of each slice (e.g., 50 for 0.5km)
    
    Returns:
        List of 4 dicts [NW, NE, SW, SE] containing channel arrays
    """
    # Get dimensions
    sample_channel = list(rotated_channels.values())[0]
    h, w = sample_channel.shape
    center_y, center_w = h // 2, w // 2
    
    half_slice = slice_size_px // 2
    
    # Calculate slice center positions in the rotated image
    # NW: top-left (-y, -x)
    nw_center_y = center_y - sampling_distance_px
    nw_center_x = center_w - sampling_distance_px
    
    # NE: top-right (-y, +x)
    ne_center_y = center_y - sampling_distance_px
    ne_center_x = center_w + sampling_distance_px
    
    # SW: bottom-left (+y, -x)
    sw_center_y = center_y + sampling_distance_px
    sw_center_x = center_w - sampling_distance_px
    
    # SE: bottom-right (+y, +x)
    se_center_y = center_y + sampling_distance_px
    se_center_x = center_w + sampling_distance_px
    
    corners = []
    corner_positions = [
        ('NW', nw_center_y, nw_center_x),
        ('NE', ne_center_y, ne_center_x),
        ('SW', sw_center_y, sw_center_x),
        ('SE', se_center_y, se_center_x)
    ]
    
    for corner_name, cy, cx in corner_positions:
        corner_slice = {}
        
        # Extract slice centered at (cy, cx)
        y_start = int(cy - half_slice)
        y_end = int(cy + half_slice)
        x_start = int(cx - half_slice)
        x_end = int(cx + half_slice)
        
        # Bounds check
        if y_start < 0 or y_end > h or x_start < 0 or x_end > w:
            raise ValueError(f"Corner {corner_name} slice out of bounds: "
                           f"y[{y_start}:{y_end}], x[{x_start}:{x_end}], "
                           f"image size: {h}x{w}")
        
        for channel_name, data in rotated_channels.items():
            corner_slice[channel_name] = data[y_start:y_end, x_start:x_end].astype(np.float32)
        
        corners.append(corner_slice)
    
    return corners


def concatenate_corners(corners):
    """
    Concatenate 4 corner slices into 2x2 grid with SHARP boundaries.
    
    Args:
        corners: List of 4 dicts [NW, NE, SW, SE]
    
    Returns:
        Dict of concatenated channel arrays
    """
    nw, ne, sw, se = corners
    
    concatenated = {}
    channel_names = list(nw.keys())
    
    for channel in channel_names:
        # Simple concatenation - NO smoothing
        # Top row: NW | NE
        # Bottom row: SW | SE
        top_row = np.hstack([nw[channel], ne[channel]])
        bottom_row = np.hstack([sw[channel], se[channel]])
        combined = np.vstack([top_row, bottom_row])
        
        concatenated[channel] = combined.astype(np.float32)
    
    return concatenated


def generate_grid_id(index, rotation_angle=None, prefix='ineg'):
    """Generate grid ID with optional rotation suffix and custom prefix."""
    if rotation_angle is not None:
        return f"{prefix}_{index:06d}_rot{int(rotation_angle):03d}"
    else:
        return f"{prefix}_{index:06d}"


def main():
    print("Archaeological Site Data Generator - Integrated Context Negatives WITH ROTATION")
    print("=" * 70)
    print("CORRECT rotation logic:")
    print("  1. Extract SAME geographic area as positive (centered on site)")
    print("  2. Rotate ENTIRE area by angle θ")
    print("  3. In rotated coordinate system:")
    print("     - Center is rotated site")
    print("     - Sample 4 corners (NW/NE/SW/SE in rotated coords)")
    print("     - Avoid center site area")
    print("  4. Concatenate 4 corners with sharp '+' boundary")
    print("  → Same geographic area as positive, just corners without site!")
    print("=" * 70)
    
    config = load_config()
    sites_df = load_sites(config['paths']['input_sites'])
    
    # Get rotation buffer factor
    rotation_buffer = config.get('augmentation', {}).get('rotation_buffer_factor', 1.0)
    
    # Extract same size as positives
    base_output_size = 1.0  # km (same as positives)
    
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
    
    num_positives = len(sites_df)
    total_grids_expected = num_positives * num_rotations
    
    print(f"\nExtraction Parameters (SAME AS POSITIVES):")
    print(f"  Base output size: {base_output_size} km × {base_output_size} km")
    print(f"  Rotation buffer factor: {rotation_buffer}x")
    print(f"  Extraction size: {base_output_size * rotation_buffer} km × {base_output_size * rotation_buffer} km")
    print(f"  Extraction pixels: {int(100 * rotation_buffer)} × {int(100 * rotation_buffer)}")
    print()
    
    print(f"Rotation Generation:")
    print(f"  Enabled: {rotation_enabled}")
    print(f"  Rotation step: {rotation_step}°")
    print(f"  Number of rotations: {num_rotations}")
    print(f"  Rotation angles: {rotation_angles}")
    print()
    
    print(f"Dataset Strategy:")
    print(f"  Total known sites: {num_positives}")
    print(f"  Integrated negatives: 1:1 ratio with rotated positives")
    print(f"  Rotations per site: {num_rotations}")
    print(f"  Expected output: {total_grids_expected} integrated negatives")
    print(f"  Prefix: 'ineg_' (to avoid conflict with positives)")
    print()
    
    file_manager = FileManager(config['paths']['output_dir'])
    metadata_builder = MetadataBuilder(config['paths']['output_dir'])
    
    # Load existing metadata
    existing_metadata = metadata_builder.load_existing_metadata()
    if existing_metadata is not None:
        print(f"Existing metadata: {len(existing_metadata)} records")
    else:
        print("No existing metadata, starting fresh")
    
    print(f"Output directory: {config['paths']['output_dir']}")
    print()
    
    # Initialize extractor
    extractor = GEEExtractor(config)
    
    buffer_margin = config['integrated_negatives']['buffer_margin']
    
    success_count = 0
    skipped_count = 0
    failure_count = 0
    
    # Process ALL sites
    for idx, row in sites_df.iterrows():
        site_id = row['site_id']
        site_lat = row['latitude']
        site_lon = row['longitude']
        
        print(f"[{idx+1}/{len(sites_df)}] Processing {site_id}")
        print(f"  Location: {site_lat:.4f}°, {site_lon:.4f}°")
        
        # Calculate site radius
        site_radius = calculate_site_radius(row)
        print(f"  Site radius: {site_radius:.1f}m")
        
        # Calculate sampling distance (radius + 250m for half-slice + buffer)
        sampling_distance_m = site_radius + 250 + buffer_margin
        print(f"  Sampling distance: {sampling_distance_m:.1f}m")
        
        # Calculate required extraction size (SAME AS POSITIVES)
        # Need to fit: center site + sampling_distance + half slice (250m) on each side
        # Plus rotation buffer
        required_radius_m = sampling_distance_m + 250  # Add half-slice size
        extraction_size_km = (required_radius_m * 2 / 1000.0) * rotation_buffer
        extraction_pixels = int(extraction_size_km * 100)  # 100 pixels per km
        
        print(f"  Extraction size: {extraction_size_km:.2f} km ({extraction_pixels}×{extraction_pixels} pixels)")
        
        # Temporarily modify config for extraction
        original_cell_size = config['grid']['cell_size_km']
        original_pixels_per_km = config['grid']['pixels_per_km']
        
        config['grid']['cell_size_km'] = extraction_size_km
        config['grid']['pixels_per_km'] = extraction_pixels
        
        try:
            # Extract LARGE tile centered on site (SAME LOCATION AS POSITIVE)
            channels, gee_metadata = extractor.extract_all_channels(site_lat, site_lon)
            indices = IndexCalculator.calculate_all_indices(channels)
            all_channels = {**channels, **indices}
            
            print(f"  → Extracted area centered on site: {extraction_pixels}×{extraction_pixels} pixels")
            if gee_metadata.get('cloud_cover'):
                print(f"     Cloud cover: {gee_metadata['cloud_cover']:.1f}%")
            
            # Verify extraction size
            sample_channel = list(all_channels.values())[0]
            actual_size = sample_channel.shape[0]
            if actual_size < extraction_pixels * 0.9:  # Allow 10% tolerance
                print(f"  ⚠ Warning: Extracted size {sample_channel.shape} smaller than expected {extraction_pixels}")
            
            # Calculate sampling distance in pixels
            pixels_per_meter = extraction_pixels / (extraction_size_km * 1000)
            sampling_distance_px = int(sampling_distance_m * pixels_per_meter)
            slice_size_px = 50  # Each slice is 50×50 pixels (0.5km at 100px/km)
            
            print(f"  → Sampling distance: {sampling_distance_px} pixels from center")
            print(f"  → Generating {num_rotations} rotations...")
            
            # Generate all rotations
            for rot_idx, rotation_angle in enumerate(rotation_angles):
                grid_id = generate_grid_id(idx + 1, rotation_angle)
                
                if file_manager.grid_exists(grid_id):
                    print(f"     [{rot_idx+1}/{num_rotations}] {rotation_angle:3d}° - SKIPPED (exists)")
                    skipped_count += 1
                    continue
                
                # Rotate ENTIRE landscape (same as positive rotation)
                rotated_landscape = rotate_entire_landscape(all_channels, rotation_angle)
                
                # Extract 4 corners from rotated landscape (in rotated coordinate system)
                corners = extract_4_corners_from_rotated(
                    rotated_landscape,
                    sampling_distance_px,
                    slice_size_px
                )
                
                # Concatenate corners (sharp boundaries, '+' shape)
                concatenated = concatenate_corners(corners)
                
                # Verify final size
                for channel_name, data in concatenated.items():
                    if data.shape != (100, 100):
                        raise ValueError(f"Final {channel_name} has shape {data.shape}, expected (100, 100)")
                
                # Save the integrated negative
                grid_dir, channels_dir, labels_dir = file_manager.create_grid_structure(grid_id)
                
                # Save all channels
                for channel_name, data in concatenated.items():
                    file_manager.save_channel(channels_dir, channel_name, data)
                
                # Save label
                label_path = labels_dir / 'binary_label.npy'
                np.save(label_path, np.array([0], dtype=np.int32))
                
                with open(labels_dir / 'neg_type.txt', 'w') as f:
                    f.write('integrated_context')
                
                with open(labels_dir / 'pos_type.txt', 'w') as f:
                    f.write('null')
                
                # Save info
                info = {
                    'grid_id': grid_id,
                    'source_site_id': site_id,
                    'source_site_lat': site_lat,
                    'source_site_lon': site_lon,
                    'site_radius_m': site_radius,
                    'sampling_distance_m': sampling_distance_m,
                    'sampling_distance_px': sampling_distance_px,
                    'rotation_angle': rotation_angle,
                    'rotation_step': rotation_step,
                    'extraction_size_km': extraction_size_km,
                    'extraction_pixels': extraction_pixels,
                    'rotation_buffer_factor': rotation_buffer,
                    'final_size_pixels': 100,
                    'label': 0,
                    'neg_type': 'integrated_context',
                    'concatenation': 'sharp_boundaries',
                    'generation_method': 'same_area_as_positive_rotate_sample_corners',
                    'note': 'Extracted same area as positive, rotated, sampled 4 corners in rotated coords'
                }
                file_manager.save_info(grid_dir, info)
                
                # Add to metadata
                image_path = file_manager.get_image_path(grid_id)
                metadata_builder.add_record(
                    grid_id=grid_id,
                    lat=site_lat,
                    lon=site_lon,
                    label=0,
                    label_source='integrated_context',
                    image_path=image_path
                )
                
                print(f"     [{rot_idx+1}/{num_rotations}] {rotation_angle:3d}° - SUCCESS ({grid_id})")
                success_count += 1
            
            print(f"  ✓ Completed all rotations for {site_id}")
            
        except Exception as e:
            print(f"  ✗ FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
            failure_count += num_rotations
        
        finally:
            # Restore original config
            config['grid']['cell_size_km'] = original_cell_size
            config['grid']['pixels_per_km'] = original_pixels_per_km
        
        print()
    
    print("=" * 70)
    print("Processing Complete!")
    print(f"  Success: {success_count} (target: {total_grids_expected})")
    print(f"  Failed:  {failure_count}")
    print(f"  Skipped: {skipped_count}")
    if total_grids_expected > 0:
        print(f"  Success rate: {success_count / total_grids_expected * 100:.1f}%")
    
    if success_count > 0:
        parquet_path = metadata_builder.save_metadata()
        print(f"\nMetadata saved to: {parquet_path}")
        
        # Show rotation distribution
        if existing_metadata is not None or success_count > 0:
            full_metadata = metadata_builder.load_existing_metadata()
            if full_metadata is not None and 'grid_id' in full_metadata.columns:
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