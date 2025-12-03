#!/usr/bin/env python3

import sys
import yaml
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.gee_extractor import GEEExtractor
from src.index_calculator import IndexCalculator
from src.file_manager import FileManager
from src.metadata_builder import MetadataBuilder
from src.negative_sampler import NegativeSampler


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


def generate_grid_id(index, prefix='unla'):
    """Generate grid ID with custom prefix for unlabeled data."""
    return f"{prefix}_{index:06d}"


def main():
    print("Archaeological Site Data Generator - Unlabeled Background (Parameter-Based)")
    print("=" * 70)
    print("NOTE: This generates UNLABELED data, not confirmed negatives!")
    print("      Random sampling cannot guarantee absence of archaeological sites.")
    print("      Target count calculated from config parameters (no dependencies)")
    print("=" * 70)
    
    config = load_config()
    
    if not config['unlabeled_sampling']['enabled']:
        print("ERROR: Unlabeled sampling is disabled in config")
        print("Set unlabeled_sampling.enabled = true in config/settings.yaml")
        return
    
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
    
    # Calculate target unlabeled count based on ratio
    unlabeled_ratio = config['dataset_ratios']['unlabeled_to_positive_ratio']
    target_unlabeled_total = int(final_positives * unlabeled_ratio)
    
    print(f"\nUnlabeled Background Sampling Strategy:")
    print(f"  Ratio to positives: {unlabeled_ratio} ({unlabeled_ratio*100:.0f}%)")
    print(f"  Target unlabeled samples: {target_unlabeled_total}")
    print(f"    Calculation: {final_positives} × {unlabeled_ratio} = {target_unlabeled_total}")
    print(f"  Prefix: 'unla_'")
    print(f"\nREMINDER: These are UNLABELED (label=-1), not confirmed negatives!")
    print(f"          They may contain undiscovered archaeological sites.")
    print()
    
    file_manager = FileManager(config['paths']['output_dir'])
    metadata_builder = MetadataBuilder(config['paths']['output_dir'])
    
    # Load existing metadata to determine starting index
    existing_metadata = metadata_builder.load_existing_metadata()
    if existing_metadata is not None:
        existing_unlabeled = existing_metadata[existing_metadata['grid_id'].str.startswith('unla_')]
        num_existing_unlabeled = len(existing_unlabeled)
        
        print(f"Existing unlabeled samples: {num_existing_unlabeled}")
        
        # Calculate remaining to generate
        num_to_generate = target_unlabeled_total - num_existing_unlabeled
        
        if num_to_generate <= 0:
            print(f"\n✓ Already have enough unlabeled samples ({num_existing_unlabeled} >= {target_unlabeled_total})")
            print("No new unlabeled data needed. Exiting.")
            return
        
        print(f"Will generate: {num_to_generate} new samples")
        start_index = num_existing_unlabeled + 1
        print(f"Starting from index: {start_index}")
    else:
        num_to_generate = target_unlabeled_total
        start_index = 1
        print("No existing metadata, starting from index 1")
        print(f"Will generate: {num_to_generate} samples")
    
    print()
    
    # NegativeSampler still uses old 'negative_sampling' config structure
    # Create adapter config to maintain compatibility
    adapted_config = config.copy()
    adapted_config['negative_sampling'] = {
        'exclusion_buffer_km': config['unlabeled_sampling']['exclusion_buffer_km'],
        'study_region': config['unlabeled_sampling']['study_region'],
        'max_attempts': config['unlabeled_sampling']['max_attempts'],
        'default_difficulty': config['unlabeled_sampling']['default_difficulty'],
        'default_neg_type': config['unlabeled_sampling']['default_landscape_type']
    }
    
    # NegativeSampler is still useful for exclusion buffer logic
    sampler = NegativeSampler(adapted_config, config['paths']['input_sites'])
    sampler.get_exclusion_stats()
    
    # Sample random background locations (avoiding known sites)
    background_locations = sampler.sample_negatives(num_to_generate)
    
    if not background_locations:
        print("\nERROR: No background locations sampled")
        return
    
    print(f"\nStarting data extraction for {len(background_locations)} unlabeled samples...")
    print(f"Output directory: {config['paths']['output_dir']}")
    print()
    
    extractor = GEEExtractor(config)
    
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for idx, location_info in enumerate(background_locations):
        grid_id = generate_grid_id(start_index + idx)
        location_id = location_info['negative_id']  # Keep ID structure from sampler
        lat = location_info['latitude']
        lon = location_info['longitude']
        # NegativeSampler returns 'neg_type' field - we interpret it as landscape type
        landscape_type = location_info.get('neg_type', config['unlabeled_sampling']['default_landscape_type'])
        
        print(f"[{idx+1}/{len(background_locations)}] Processing {location_id} (Grid: {grid_id})")
        print(f"  Location: {lat:.4f}°, {lon:.4f}°")
        print(f"  Landscape: {landscape_type}")
        print(f"  Label: -1 (UNLABELED - unknown if site present)")
        
        if file_manager.grid_exists(grid_id):
            print(f"  → SKIPPED (already exists)")
            skipped_count += 1
            continue
        
        try:
            channels, gee_metadata = extractor.extract_all_channels(lat, lon)
            
            print(f"  → Extracted {len(channels)} Sentinel-2 bands + DEM/Slope")
            if gee_metadata['cloud_cover']:
                print(f"     Cloud cover: {gee_metadata['cloud_cover']:.1f}%")
            
            indices = IndexCalculator.calculate_all_indices(channels)
            print(f"  → Calculated {len(indices)} spectral indices")
            
            grid_dir, channels_dir, labels_dir = file_manager.create_grid_structure(grid_id)
            
            file_manager.save_all_channels(channels_dir, channels, indices)
            
            # Save label as -1 (UNLABELED)
            label_path = labels_dir / 'binary_label.npy'
            np.save(label_path, np.array([-1], dtype=np.int32))
            
            # Save landscape type (not "negative type")
            with open(labels_dir / 'landscape_type.txt', 'w') as f:
                f.write(landscape_type)
            
            # No pos_type for unlabeled data
            with open(labels_dir / 'pos_type.txt', 'w') as f:
                f.write('null')
            
            # No neg_type either (it's unlabeled, not a confirmed negative)
            with open(labels_dir / 'neg_type.txt', 'w') as f:
                f.write('null')
            
            info = {
                'grid_id': grid_id,
                'location_id': location_id,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'label': -1,  # UNLABELED
                'landscape_type': landscape_type,
                'sampling_method': 'random_background_with_exclusion',
                'notes': 'Unlabeled - may or may not contain archaeological sites',
                'gee_image_id': gee_metadata['image_id'],
                'cloud_cover': gee_metadata['cloud_cover'],
                'acquisition_date': gee_metadata['acquisition_date']
            }
            file_manager.save_info(grid_dir, info)
            
            image_path = file_manager.get_image_path(grid_id)
            metadata_builder.add_record(
                grid_id=grid_id,
                lat=lat,
                lon=lon,
                label=-1,
                label_source='random_background',
                image_path=image_path
            )
            
            print(f"  ✓ SUCCESS - Saved to {grid_id}/")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ FAILED - {str(e)}")
            failure_count += 1
            continue
        
        print()
    
    print("=" * 70)
    print("Processing Complete!")
    print(f"  Target:  {num_to_generate}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failure_count}")
    print(f"  Skipped: {skipped_count}")
    
    if success_count > 0:
        parquet_path = metadata_builder.save_metadata()
        print(f"\nMetadata saved to: {parquet_path}")
        
        # Show final statistics
        final_metadata = metadata_builder.load_existing_metadata()
        final_unlabeled = len(final_metadata[final_metadata['grid_id'].str.startswith('unla_')])
        
        print(f"\nFinal Dataset Statistics:")
        print(f"  Total grids: {len(final_metadata)}")
        print(f"  Unlabeled: {final_unlabeled}")
        print(f"  Target unlabeled: {target_unlabeled_total}")
        print(f"  Progress: {final_unlabeled / target_unlabeled_total * 100:.1f}%")
        
        print(f"\nIMPORTANT NOTES:")
        print(f"  - These samples are labeled -1 (UNLABELED)")
        print(f"  - They are NOT confirmed negatives (label=0)")
        print(f"  - They may contain undiscovered archaeological sites")
        print(f"  - Use for semi-supervised learning or RL exploration")
    
    print()


if __name__ == "__main__":
    main()