#!/usr/bin/env python3

import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.file_manager import FileManager
from src.metadata_builder import MetadataBuilder
from src.index_calculator import IndexCalculator


def load_config(config_path='config/settings.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_brightness_contrast(band_data, brightness_factor, contrast_factor):
    """
    Apply brightness and contrast adjustment to a band.
    
    Args:
        band_data: numpy array
        brightness_factor: additive factor (e.g., 0.08 for +8%)
        contrast_factor: multiplicative factor (e.g., 1.05 for +5%)
    
    Returns:
        Adjusted band data
    """
    # Apply contrast first (multiplicative)
    adjusted = band_data * contrast_factor
    
    # Then brightness (additive)
    adjusted = adjusted + brightness_factor
    
    return adjusted.astype(np.float32)


def apply_gaussian_noise(band_data, sigma):
    """Add Gaussian noise to band data."""
    noise = np.random.normal(0, sigma, band_data.shape)
    noisy = band_data + noise
    return noisy.astype(np.float32)


def load_channels_from_grid(grid_dir):
    """Load all channel arrays from a grid directory."""
    channels_dir = grid_dir / 'channels'
    
    channels = {}
    spectral_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    terrain_bands = ['DEM', 'Slope']
    indices = ['NDVI', 'NDWI', 'BSI']
    
    for band in spectral_bands + terrain_bands + indices:
        band_path = channels_dir / f'{band}.npy'
        if band_path.exists():
            channels[band] = np.load(band_path)
    
    return channels


def create_augmentation_variant(channels, variant_type):
    """
    Create a radiometric augmentation variant.
    
    Args:
        channels: Dict of channel arrays
        variant_type: 'aug1' (bright), 'aug2' (dark), or 'aug3' (noise)
    
    Returns:
        Dict of augmented channels
    """
    augmented = {}
    
    # Spectral bands to augment
    spectral_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    
    if variant_type == 'aug1':
        # Variant 1: Bright
        brightness = 0.08  # +8%
        contrast = 1.05    # +5%
        noise_sigma = 0.015
        
    elif variant_type == 'aug2':
        # Variant 2: Dark
        brightness = -0.08  # -8%
        contrast = 0.95     # -5%
        noise_sigma = 0.015
        
    elif variant_type == 'aug3':
        # Variant 3: Noise only
        brightness = 0.0
        contrast = 1.0
        noise_sigma = 0.025
        
    else:
        raise ValueError(f"Unknown variant type: {variant_type}")
    
    # Apply transformations to spectral bands
    for band in spectral_bands:
        if band in channels:
            # Apply brightness and contrast
            adjusted = apply_brightness_contrast(
                channels[band], 
                brightness, 
                contrast
            )
            
            # Apply noise
            noisy = apply_gaussian_noise(adjusted, noise_sigma)
            
            augmented[band] = noisy
    
    # Keep terrain unchanged
    augmented['DEM'] = channels['DEM'].copy()
    augmented['Slope'] = channels['Slope'].copy()
    
    # Recalculate indices from augmented spectral bands
    augmented['NDVI'] = IndexCalculator.calculate_ndvi(
        augmented['B8'], augmented['B4']
    )
    augmented['NDWI'] = IndexCalculator.calculate_ndwi(
        augmented['B3'], augmented['B8']
    )
    augmented['BSI'] = IndexCalculator.calculate_bsi(
        augmented['B11'], augmented['B4'], 
        augmented['B8'], augmented['B2']
    )
    
    return augmented


def parse_grid_id(grid_id):
    """Extract base info from grid_id."""
    parts = grid_id.split('_')
    
    if grid_id.startswith('grid_'):
        # Positive: grid_000001 or grid_000001_rot030
        prefix = 'grid'
        index = int(parts[1])
        
        # Check for rotation suffix
        if len(parts) > 2 and parts[2].startswith('rot'):
            rotation = int(parts[2][3:])
        else:
            rotation = 0
            
    elif grid_id.startswith('ineg_'):
        # Integrated negative: ineg_000001 or ineg_000001_rot030
        prefix = 'ineg'
        index = int(parts[1])
        
        if len(parts) > 2 and parts[2].startswith('rot'):
            rotation = int(parts[2][3:])
        else:
            rotation = 0
    else:
        raise ValueError(f"Unknown grid_id format: {grid_id}")
    
    return prefix, index, rotation


def generate_augmented_grid_id(original_grid_id, variant_type):
    """Generate new grid_id for augmented variant."""
    # variant_type is 'aug1', 'aug2', or 'aug3'
    return f"{original_grid_id}_{variant_type}"


def copy_label_info(src_labels_dir, dst_labels_dir):
    """Copy label information from source to destination."""
    import shutil
    
    # Copy label files
    for label_file in ['binary_label.npy', 'pos_type.txt', 'neg_type.txt']:
        src_file = src_labels_dir / label_file
        dst_file = dst_labels_dir / label_file
        
        if src_file.exists():
            shutil.copy(src_file, dst_file)
    
    # Also copy landscape_type.txt if exists (for unlabeled)
    landscape_file = src_labels_dir / 'landscape_type.txt'
    if landscape_file.exists():
        shutil.copy(landscape_file, dst_labels_dir / 'landscape_type.txt')


def load_original_info(grid_dir):
    """Load info.json from original grid."""
    import json
    
    info_path = grid_dir / 'info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def get_target_grids(images_dir, target_prefixes=['grid_', 'ineg_']):
    """Get all grid directories with target prefixes that should be augmented."""
    all_grids = []
    
    for prefix in target_prefixes:
        grids = list(images_dir.glob(f'{prefix}*'))
        
        # Filter out already augmented grids
        grids = [g for g in grids if '_aug' not in g.name]
        
        all_grids.extend(grids)
    
    return sorted(all_grids)


def main():
    print("Archaeological Site Data - Radiometric Augmentation")
    print("=" * 70)
    print("Applies radiometric augmentation to positives and integrated negatives")
    print("Creates 3 variants per rotated sample:")
    print("  aug1: Bright (+8% brightness, +5% contrast, noise σ=0.015)")
    print("  aug2: Dark (-8% brightness, -5% contrast, noise σ=0.015)")
    print("  aug3: Noise only (noise σ=0.025)")
    print("=" * 70)
    print()
    
    config = load_config()
    
    file_manager = FileManager(config['paths']['output_dir'])
    metadata_builder = MetadataBuilder(config['paths']['output_dir'])
    
    # Load existing metadata
    existing_metadata = metadata_builder.load_existing_metadata()
    if existing_metadata is None:
        print("ERROR: No existing metadata found!")
        print("Please run generate_positives.py and generate_integrated_negatives.py first")
        return
    
    print(f"Existing metadata: {len(existing_metadata)} records")
    
    # Count existing data types
    num_positives = len(existing_metadata[existing_metadata['grid_id'].str.startswith('grid_')])
    num_integrated = len(existing_metadata[existing_metadata['grid_id'].str.startswith('ineg_')])
    num_landcover = len(existing_metadata[existing_metadata['grid_id'].str.startswith('lneg_')])
    num_unlabeled = len(existing_metadata[existing_metadata['grid_id'].str.startswith('unla_')])
    
    print(f"  Positives (grid_*): {num_positives}")
    print(f"  Integrated negatives (ineg_*): {num_integrated}")
    print(f"  Landcover negatives (lneg_*): {num_landcover}")
    print(f"  Unlabeled (unla_*): {num_unlabeled}")
    print()
    
    # Get augmentation configuration
    aug_config = config.get('augmentation', {}).get('radiometric', {})
    enabled = aug_config.get('enabled', True)
    variants = aug_config.get('variants', ['aug1', 'aug2', 'aug3'])
    
    if not enabled:
        print("Radiometric augmentation is disabled in config")
        return
    
    print(f"Augmentation configuration:")
    print(f"  Enabled: {enabled}")
    print(f"  Variants: {variants}")
    print()
    
    # Find all grids to augment (positives and integrated negatives)
    images_dir = file_manager.images_dir
    target_grids = get_target_grids(images_dir, target_prefixes=['grid_', 'ineg_'])
    
    if not target_grids:
        print("ERROR: No target grids found!")
        print("Please run generate_positives.py and generate_integrated_negatives.py first")
        return
    
    # Filter out already augmented grids
    base_grids = [g for g in target_grids if '_aug' not in g.name]
    
    print(f"Found {len(base_grids)} base grids to augment")
    
    # Count by type
    positives_to_aug = [g for g in base_grids if g.name.startswith('grid_')]
    integrated_to_aug = [g for g in base_grids if g.name.startswith('ineg_')]
    
    print(f"  Positives: {len(positives_to_aug)}")
    print(f"  Integrated negatives: {len(integrated_to_aug)}")
    print(f"  Variants per grid: {len(variants)}")
    print(f"  Total new grids to create: {len(base_grids) * len(variants)}")
    print()
    
    # Process each grid
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for idx, grid_dir in enumerate(base_grids):
        grid_id = grid_dir.name
        prefix, index, rotation = parse_grid_id(grid_id)
        
        grid_type = "Positive" if prefix == 'grid' else "Integrated negative"
        
        print(f"[{idx+1}/{len(base_grids)}] Processing {grid_id} ({grid_type})")
        
        try:
            # Load original channels
            channels = load_channels_from_grid(grid_dir)
            
            if len(channels) == 0:
                print(f"  ✗ FAILED - No channels found")
                failure_count += len(variants)
                continue
            
            print(f"  → Loaded {len(channels)} channels")
            
            # Load original info
            original_info = load_original_info(grid_dir)
            
            # Create all variants
            for variant_type in variants:
                aug_grid_id = generate_augmented_grid_id(grid_id, variant_type)
                
                # Check if already exists
                if file_manager.grid_exists(aug_grid_id):
                    print(f"     {variant_type}: SKIPPED (exists)")
                    skipped_count += 1
                    continue
                
                # Create augmented channels
                augmented_channels = create_augmentation_variant(channels, variant_type)
                
                # Create directory structure
                aug_grid_dir, aug_channels_dir, aug_labels_dir = \
                    file_manager.create_grid_structure(aug_grid_id)
                
                # Save all channels
                for channel_name, data in augmented_channels.items():
                    file_manager.save_channel(aug_channels_dir, channel_name, data)
                
                # Copy label information
                copy_label_info(grid_dir / 'labels', aug_labels_dir)
                
                # Create augmented info.json
                aug_info = original_info.copy()
                aug_info['grid_id'] = aug_grid_id
                aug_info['original_grid_id'] = grid_id
                aug_info['augmentation_type'] = variant_type
                aug_info['augmentation_method'] = 'radiometric'
                
                if variant_type == 'aug1':
                    aug_info['augmentation_params'] = {
                        'brightness': '+8%',
                        'contrast': '+5%',
                        'noise_sigma': 0.015
                    }
                elif variant_type == 'aug2':
                    aug_info['augmentation_params'] = {
                        'brightness': '-8%',
                        'contrast': '-5%',
                        'noise_sigma': 0.015
                    }
                elif variant_type == 'aug3':
                    aug_info['augmentation_params'] = {
                        'brightness': '0%',
                        'contrast': '0%',
                        'noise_sigma': 0.025
                    }
                
                file_manager.save_info(aug_grid_dir, aug_info)
                
                # Add to metadata
                original_record = existing_metadata[
                    existing_metadata['grid_id'] == grid_id
                ].iloc[0] if len(existing_metadata[existing_metadata['grid_id'] == grid_id]) > 0 else None
                
                if original_record is not None:
                    image_path = file_manager.get_image_path(aug_grid_id)
                    metadata_builder.add_record(
                        grid_id=aug_grid_id,
                        lat=original_record['centroid_lat'],
                        lon=original_record['centroid_lon'],
                        label=original_record['label'],
                        label_source=f"{original_record['label_source']}_augmented",
                        image_path=image_path
                    )
                
                print(f"     {variant_type}: SUCCESS")
                success_count += 1
            
            print(f"  ✓ Completed all variants for {grid_id}")
            
        except Exception as e:
            print(f"  ✗ FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
            failure_count += len(variants)
        
        print()
    
    print("=" * 70)
    print("Augmentation Complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failure_count}")
    print(f"  Skipped: {skipped_count}")
    print()
    
    if success_count > 0:
        parquet_path = metadata_builder.save_metadata()
        print(f"Metadata saved to: {parquet_path}")
        
        # Show final statistics
        final_metadata = metadata_builder.load_existing_metadata()
        
        print(f"\nFinal Dataset Statistics:")
        print(f"  Total grids: {len(final_metadata)}")
        
        # Count by prefix
        final_positives = len(final_metadata[final_metadata['grid_id'].str.startswith('grid_')])
        final_integrated = len(final_metadata[final_metadata['grid_id'].str.startswith('ineg_')])
        final_landcover = len(final_metadata[final_metadata['grid_id'].str.startswith('lneg_')])
        final_unlabeled = len(final_metadata[final_metadata['grid_id'].str.startswith('unla_')])
        
        print(f"  Positives (with augmentation): {final_positives}")
        print(f"  Integrated negatives (with augmentation): {final_integrated}")
        print(f"  Landcover negatives: {final_landcover}")
        print(f"  Unlabeled: {final_unlabeled}")
        
        # Show augmentation breakdown
        aug_count = len(final_metadata[final_metadata['grid_id'].str.contains('_aug')])
        print(f"\n  Augmented grids: {aug_count}")
        print(f"  Original grids: {len(final_metadata) - aug_count}")
    
    print()


if __name__ == "__main__":
    main()