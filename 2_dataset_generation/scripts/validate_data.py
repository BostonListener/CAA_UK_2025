#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def validate_channel_shape(channel_path, expected_shape=(100, 100)):
    data = np.load(channel_path)
    if data.shape != expected_shape:
        return False, f"Shape {data.shape} != {expected_shape}"
    if not np.isfinite(data).all():
        return False, "Contains NaN or Inf"
    return True, "OK"


def validate_grid(grid_dir):
    grid_dir = Path(grid_dir)
    results = {
        'grid_id': grid_dir.name,
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    channels_dir = grid_dir / 'channels'
    labels_dir = grid_dir / 'labels'
    
    if not channels_dir.exists():
        results['valid'] = False
        results['errors'].append("Missing channels/ directory")
        return results
    
    if not labels_dir.exists():
        results['valid'] = False
        results['errors'].append("Missing labels/ directory")
        return results
    
    expected_channels = [
        'B2.npy', 'B3.npy', 'B4.npy', 'B8.npy', 'B11.npy', 'B12.npy',
        'NDVI.npy', 'NDWI.npy', 'BSI.npy', 'DEM.npy', 'Slope.npy'
    ]
    
    for channel in expected_channels:
        channel_path = channels_dir / channel
        if not channel_path.exists():
            results['valid'] = False
            results['errors'].append(f"Missing channel: {channel}")
        else:
            is_valid, msg = validate_channel_shape(channel_path)
            if not is_valid:
                results['valid'] = False
                results['errors'].append(f"{channel}: {msg}")
    
    label_file = labels_dir / 'binary_label.npy'
    if not label_file.exists():
        results['valid'] = False
        results['errors'].append("Missing binary_label.npy")
    else:
        label = np.load(label_file)
        if label.shape != (1,):
            results['errors'].append(f"Label shape {label.shape} != (1,)")
        if label[0] not in [0, 1, -1]:
            results['errors'].append(f"Invalid label value: {label[0]}")
    
    info_file = grid_dir / 'info.json'
    if not info_file.exists():
        results['warnings'].append("Missing info.json")
    
    return results


def load_rgb_composite(grid_dir):
    channels_dir = Path(grid_dir) / 'channels'
    
    r = np.load(channels_dir / 'B4.npy')
    g = np.load(channels_dir / 'B3.npy')
    b = np.load(channels_dir / 'B2.npy')
    
    rgb = np.stack([r, g, b], axis=-1)
    
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    
    return rgb_norm


def visualize_grid(grid_dir, output_path=None):
    grid_dir = Path(grid_dir)
    channels_dir = grid_dir / 'channels'
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Grid: {grid_dir.name}', fontsize=16)
    
    rgb = load_rgb_composite(grid_dir)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Composite (B4-B3-B2)')
    axes[0, 0].axis('off')
    
    ndvi = np.load(channels_dir / 'NDVI.npy')
    im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
    axes[0, 1].set_title('NDVI')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    ndwi = np.load(channels_dir / 'NDWI.npy')
    im2 = axes[0, 2].imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('NDWI')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    bsi = np.load(channels_dir / 'BSI.npy')
    im3 = axes[1, 0].imshow(bsi, cmap='YlOrBr', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('BSI')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    dem = np.load(channels_dir / 'DEM.npy')
    im4 = axes[1, 1].imshow(dem, cmap='terrain')
    axes[1, 1].set_title('DEM (FABDEM)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    slope = np.load(channels_dir / 'Slope.npy')
    im5 = axes[1, 2].imshow(slope, cmap='plasma')
    axes[1, 2].set_title('Slope')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_channel_stats(grid_dir):
    channels_dir = Path(grid_dir) / 'channels'
    
    stats = {}
    channel_files = sorted(channels_dir.glob('*.npy'))
    
    for channel_file in channel_files:
        data = np.load(channel_file)
        stats[channel_file.stem] = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any())
        }
    
    return stats


def get_all_grid_dirs(images_dir):
    """Find all grid directories with any prefix (grid_, ineg_, lneg_, unla_)."""
    images_dir = Path(images_dir)
    
    # Search for all known prefixes
    prefixes = ['grid_', 'ineg_', 'lneg_', 'unla_']
    all_grids = []
    
    for prefix in prefixes:
        grids = list(images_dir.glob(f'{prefix}*'))
        all_grids.extend(grids)
    
    return sorted(all_grids)


def categorize_grids_by_prefix(grid_dirs):
    """Categorize grid directories by their prefix."""
    categories = {
        'grid_': [],  # Positives
        'ineg_': [],  # Integrated negatives
        'lneg_': [],  # Landcover negatives
        'unla_': []   # Unlabeled
    }
    
    for grid_dir in grid_dirs:
        grid_name = grid_dir.name
        for prefix in categories.keys():
            if grid_name.startswith(prefix):
                categories[prefix].append(grid_dir)
                break
    
    return categories


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate generated archaeological site data')
    parser.add_argument('--dataset-dir', default='outputs/dataset', help='Dataset directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--sample-grid', type=str, help='Specific grid to visualize (e.g., grid_000001)')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / 'grid_images'
    
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist")
        return
    
    print("Archaeological Site Data Validation")
    print("=" * 70)
    
    # Check metadata
    metadata_path = dataset_dir / 'grid_metadata.parquet'
    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
        print(f"\n✓ Metadata file found: {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Show breakdown by prefix
        prefix_counts = {}
        for prefix in ['grid_', 'ineg_', 'lneg_', 'unla_']:
            count = len(df[df['grid_id'].str.startswith(prefix)])
            if count > 0:
                prefix_counts[prefix] = count
        
        if prefix_counts:
            print(f"  Prefix breakdown:")
            prefix_labels = {
                'grid_': 'Positives',
                'ineg_': 'Integrated negatives',
                'lneg_': 'Landcover negatives',
                'unla_': 'Unlabeled'
            }
            for prefix, count in prefix_counts.items():
                label = prefix_labels.get(prefix, prefix)
                print(f"    {prefix:<6} ({label:<22}): {count:>4} records")
    else:
        print(f"\n✗ Metadata file not found: {metadata_path}")
        return
    
    # Find ALL grid directories (not just grid_*)
    grid_dirs = get_all_grid_dirs(images_dir)
    
    if not grid_dirs:
        print(f"\n✗ No grid directories found in {images_dir}")
        return
    
    print(f"\n✓ Found {len(grid_dirs)} grid directories")
    
    # Categorize and show breakdown
    categories = categorize_grids_by_prefix(grid_dirs)
    print(f"  Directory breakdown:")
    category_labels = {
        'grid_': 'Positives',
        'ineg_': 'Integrated negatives',
        'lneg_': 'Landcover negatives',
        'unla_': 'Unlabeled'
    }
    for prefix, dirs in categories.items():
        if dirs:
            label = category_labels.get(prefix, prefix)
            print(f"    {prefix:<6} ({label:<22}): {len(dirs):>4} directories")
    
    # Validation
    print("\nValidating grid cells...")
    valid_count = 0
    invalid_grids = []
    
    for grid_dir in grid_dirs:
        result = validate_grid(grid_dir)
        
        if result['valid']:
            valid_count += 1
            print(f"  ✓ {result['grid_id']}")
        else:
            invalid_grids.append(result)
            print(f"  ✗ {result['grid_id']}")
            for error in result['errors']:
                print(f"      ERROR: {error}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"      WARNING: {warning}")
    
    print(f"\nValidation Summary:")
    print(f"  Valid grids:   {valid_count}/{len(grid_dirs)}")
    print(f"  Invalid grids: {len(invalid_grids)}/{len(grid_dirs)}")
    
    # Check metadata vs filesystem consistency
    metadata_ids = set(df['grid_id'].values)
    filesystem_ids = set(g.name for g in grid_dirs)
    
    missing_in_filesystem = metadata_ids - filesystem_ids
    missing_in_metadata = filesystem_ids - metadata_ids
    
    if missing_in_filesystem or missing_in_metadata:
        print(f"\n⚠ Consistency Issues:")
        if missing_in_filesystem:
            print(f"  {len(missing_in_filesystem)} records in metadata but missing directories:")
            for grid_id in sorted(list(missing_in_filesystem)[:5]):
                print(f"    - {grid_id}")
            if len(missing_in_filesystem) > 5:
                print(f"    ... and {len(missing_in_filesystem) - 5} more")
        
        if missing_in_metadata:
            print(f"  {len(missing_in_metadata)} directories but missing from metadata:")
            for grid_id in sorted(list(missing_in_metadata)[:5]):
                print(f"    - {grid_id}")
            if len(missing_in_metadata) > 5:
                print(f"    ... and {len(missing_in_metadata) - 5} more")
    else:
        print(f"\n✓ Metadata and filesystem are consistent")
    
    if args.stats and grid_dirs:
        print(f"\n{'='*70}")
        print("Channel Statistics (first grid)")
        sample_grid = grid_dirs[0]
        stats = get_channel_stats(sample_grid)
        
        print(f"\nGrid: {sample_grid.name}")
        print(f"{'Channel':<10} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Issues'}")
        print("-" * 70)
        
        for channel, s in stats.items():
            issues = []
            if s['has_nan']:
                issues.append('NaN')
            if s['has_inf']:
                issues.append('Inf')
            issue_str = ','.join(issues) if issues else 'OK'
            
            print(f"{channel:<10} {s['min']:>12.4f} {s['max']:>12.4f} {s['mean']:>12.4f} {s['std']:>12.4f} {issue_str}")
    
    if args.visualize:
        viz_dir = dataset_dir / 'validation_plots'
        viz_dir.mkdir(exist_ok=True)
        
        if args.sample_grid:
            sample_grid = images_dir / args.sample_grid
            if sample_grid.exists():
                output_path = viz_dir / f'{args.sample_grid}_visualization.png'
                visualize_grid(sample_grid, output_path)
            else:
                print(f"\nERROR: Grid {args.sample_grid} not found")
        else:
            print(f"\nGenerating visualizations for all {len(grid_dirs)} grids...")
            for i, grid_dir in enumerate(grid_dirs):
                output_path = viz_dir / f'{grid_dir.name}_visualization.png'
                visualize_grid(grid_dir, output_path)
            print(f"Visualizations saved to {viz_dir}/")
    
    print()


if __name__ == "__main__":
    main()