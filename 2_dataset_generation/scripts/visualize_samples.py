#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random


def load_rgb_composite(grid_dir):
    """Load and normalize RGB composite from B4, B3, B2 bands."""
    channels_dir = Path(grid_dir) / 'channels'
    
    r = np.load(channels_dir / 'B4.npy')
    g = np.load(channels_dir / 'B3.npy')
    b = np.load(channels_dir / 'B2.npy')
    
    rgb = np.stack([r, g, b], axis=-1)
    
    # Normalize using 2nd and 98th percentile
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    
    return rgb_norm


def load_grid_info(grid_dir):
    """Load info.json from grid directory."""
    info_path = Path(grid_dir) / 'info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def load_label_info(grid_dir):
    """Load label information from labels directory."""
    labels_dir = Path(grid_dir) / 'labels'
    
    info = {}
    
    # Load binary label
    label_file = labels_dir / 'binary_label.npy'
    if label_file.exists():
        info['label'] = int(np.load(label_file)[0])
    
    # Load pos_type
    pos_type_file = labels_dir / 'pos_type.txt'
    if pos_type_file.exists():
        info['pos_type'] = pos_type_file.read_text().strip()
    
    # Load neg_type
    neg_type_file = labels_dir / 'neg_type.txt'
    if neg_type_file.exists():
        info['neg_type'] = neg_type_file.read_text().strip()
    
    # Load landscape_type
    landscape_file = labels_dir / 'landscape_type.txt'
    if landscape_file.exists():
        info['landscape_type'] = landscape_file.read_text().strip()
    
    return info


def visualize_grid_detailed(grid_dir, output_path, title_prefix=""):
    """
    Create detailed 2x3 visualization of a grid with all channels.
    
    Args:
        grid_dir: Path to grid directory
        output_path: Path to save the visualization
        title_prefix: Additional text for the main title
    """
    grid_dir = Path(grid_dir)
    channels_dir = grid_dir / 'channels'
    
    # Load metadata
    info = load_grid_info(grid_dir)
    label_info = load_label_info(grid_dir)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Build title
    grid_id = grid_dir.name
    label = label_info.get('label', 'unknown')
    
    title_parts = [f'{title_prefix}' if title_prefix else '', f'Grid: {grid_id}', f'Label: {label}']
    
    # Add type information
    if label == 1:
        pos_type = label_info.get('pos_type', 'unknown')
        if pos_type and pos_type != 'null':
            title_parts.append(f'Type: {pos_type}')
    elif label == 0:
        neg_type = label_info.get('neg_type', 'unknown')
        if neg_type and neg_type != 'null':
            title_parts.append(f'Type: {neg_type}')
    elif label == -1:
        landscape_type = label_info.get('landscape_type', 'unknown')
        if landscape_type and landscape_type != 'null':
            title_parts.append(f'Landscape: {landscape_type}')
    
    # Add rotation info if present
    if '_rot' in grid_id:
        rotation = int(grid_id.split('_rot')[1].split('_')[0])
        title_parts.append(f'Rotation: {rotation}°')
    
    # Add coordinates
    if 'centroid_lat' in info and 'centroid_lon' in info:
        title_parts.append(f"({info['centroid_lat']:.4f}°, {info['centroid_lon']:.4f}°)")
    
    # Add cloud cover if present
    if 'cloud_cover' in info and info['cloud_cover'] is not None:
        title_parts.append(f"Cloud: {info['cloud_cover']:.1f}%")
    
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    # RGB Composite
    rgb = load_rgb_composite(grid_dir)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Composite (B4-B3-B2)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # NDVI
    ndvi = np.load(channels_dir / 'NDVI.npy')
    im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
    axes[0, 1].set_title('NDVI (Vegetation)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # NDWI
    ndwi = np.load(channels_dir / 'NDWI.npy')
    im2 = axes[0, 2].imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('NDWI (Water)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # BSI
    bsi = np.load(channels_dir / 'BSI.npy')
    im3 = axes[1, 0].imshow(bsi, cmap='YlOrBr', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('BSI (Bare Soil)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # DEM
    dem = np.load(channels_dir / 'DEM.npy')
    im4 = axes[1, 1].imshow(dem, cmap='terrain')
    axes[1, 1].set_title('DEM (Elevation)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Slope
    slope = np.load(channels_dir / 'Slope.npy')
    im5 = axes[1, 2].imshow(slope, cmap='plasma')
    axes[1, 2].set_title('Slope', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def get_rotation_angles_for_site(images_dir, base_index):
    """Find all rotation angles for a given base site index."""
    pattern = f"grid_{base_index:06d}_rot*"
    grids = list(images_dir.glob(pattern))
    
    angles = []
    for grid in grids:
        if '_rot' in grid.name:
            angle = int(grid.name.split('_rot')[1].split('_')[0])
            angles.append((angle, grid))
    
    # Also check for non-rotated version (rotation 0)
    base_grid = images_dir / f"grid_{base_index:06d}"
    if base_grid.exists():
        angles.append((0, base_grid))
    
    return sorted(angles, key=lambda x: x[0])


def select_positive_samples(images_dir, n_samples=3, preferred_rotation=0):
    """
    Select positive samples from different sites at the same rotation angle.
    Returns list of (grid_dir, description) tuples.
    
    Args:
        images_dir: Path to grid_images directory
        n_samples: Number of different sites to select
        preferred_rotation: Preferred rotation angle (default: 0)
    """
    # Find all positive grids (non-augmented only)
    all_positives = list(images_dir.glob('grid_*'))
    non_augmented = [g for g in all_positives if '_aug' not in g.name]
    
    if not non_augmented:
        print("  ⚠ No non-augmented positive samples found")
        return []
    
    # Group by base site index
    sites_dict = {}
    for grid_dir in non_augmented:
        parts = grid_dir.name.split('_')
        base_index = int(parts[1])
        
        # Extract rotation angle
        if '_rot' in grid_dir.name:
            rot_part = grid_dir.name.split('_rot')[1]
            rotation = int(rot_part)
        else:
            rotation = 0
        
        if base_index not in sites_dict:
            sites_dict[base_index] = {}
        sites_dict[base_index][rotation] = grid_dir
    
    # Select n_samples different sites, all at the same rotation angle
    selected_samples = []
    available_sites = list(sites_dict.keys())
    random.shuffle(available_sites)
    
    for site_index in available_sites:
        if len(selected_samples) >= n_samples:
            break
        
        # Try to get the preferred rotation, otherwise get any available
        if preferred_rotation in sites_dict[site_index]:
            grid_dir = sites_dict[site_index][preferred_rotation]
            desc = f"Positive site {site_index} (rotation {preferred_rotation}°)"
            selected_samples.append((grid_dir, desc))
        elif sites_dict[site_index]:  # Fallback to any rotation
            rotation = list(sites_dict[site_index].keys())[0]
            grid_dir = sites_dict[site_index][rotation]
            desc = f"Positive site {site_index} (rotation {rotation}°)"
            selected_samples.append((grid_dir, desc))
    
    return selected_samples


def select_integrated_samples(images_dir, positive_samples, n_samples=3):
    """
    Select integrated negatives that match positive samples.
    Ensures proper 1:1 correspondence with positives.
    """
    samples = []
    
    for pos_grid_dir, _ in positive_samples[:n_samples]:
        # Extract base index and rotation from positive
        pos_name = pos_grid_dir.name
        parts = pos_name.split('_')
        
        # Handle grid_XXXXXX or grid_XXXXXX_rotXXX format
        if len(parts) >= 2:
            base_index = int(parts[1])
        else:
            print(f"  ⚠ Cannot parse positive name: {pos_name}")
            continue
        
        # Check for rotation
        if '_rot' in pos_name:
            # Extract rotation angle (handle both grid_000176_rot120 and grid_000176_rot120_aug1)
            rot_part = pos_name.split('_rot')[1]
            rotation = int(rot_part.split('_')[0])  # Get just the angle, ignore any _aug suffix
            ineg_name = f"ineg_{base_index:06d}_rot{rotation:03d}"
        else:
            ineg_name = f"ineg_{base_index:06d}"
        
        ineg_dir = images_dir / ineg_name
        
        if ineg_dir.exists():
            desc = f"Integrated negative (source: {pos_grid_dir.name})"
            samples.append((ineg_dir, desc))
            print(f"  ✓ Matched: {pos_grid_dir.name} → {ineg_name}")
        else:
            print(f"  ⚠ Integrated negative not found: {ineg_name}")
    
    return samples


def select_landcover_samples(images_dir, n_samples=9):
    """
    Select landcover negatives with GUARANTEED stratified sampling.
    For n_samples=9: selects 3 urban + 3 water + 3 cropland.
    For any n_samples: distributes evenly across the 3 types.
    """
    all_landcover = list(images_dir.glob('lneg_*'))
    
    if not all_landcover:
        print("  ⚠ No landcover negatives found")
        return []
    
    # Categorize by neg_type
    urban = []
    water = []
    cropland = []
    other = []
    neg_type_counts = {}  # Track what types we're seeing
    
    for grid_dir in all_landcover:
        label_info = load_label_info(grid_dir)
        neg_type = label_info.get('neg_type', 'unknown')
        neg_type_lower = neg_type.lower()
        
        # Count types for diagnostic
        neg_type_counts[neg_type] = neg_type_counts.get(neg_type, 0) + 1
        
        if 'urban' in neg_type_lower:
            urban.append((grid_dir, neg_type))
        elif 'water' in neg_type_lower:
            water.append((grid_dir, neg_type))
        elif 'crop' in neg_type_lower:
            cropland.append((grid_dir, neg_type))
        else:
            other.append((grid_dir, neg_type))
    
    # Show diagnostic information
    print(f"  Landcover types found in dataset:")
    for neg_type, count in sorted(neg_type_counts.items()):
        print(f"    '{neg_type}': {count} samples")
    print(f"  Categorized as: {len(urban)} urban, {len(water)} water, {len(cropland)} cropland, {len(other)} other")
    
    # Calculate stratified sampling: divide n_samples evenly across 3 types
    samples_per_type = n_samples // 3
    remainder = n_samples % 3
    
    print(f"  Stratified sampling: {samples_per_type} per type" + 
          (f" + {remainder} extra" if remainder > 0 else ""))
    
    samples = []
    
    # Select from urban
    if urban:
        n_urban = min(samples_per_type + (1 if remainder > 0 else 0), len(urban))
        selected_urban = random.sample(urban, n_urban)
        for selected_dir, selected_type in selected_urban:
            samples.append((selected_dir, f"Landcover negative (urban)"))
        print(f"  ✓ Selected {n_urban} urban samples")
        if remainder > 0:
            remainder -= 1
    else:
        print(f"  ⚠ No urban samples available")
    
    # Select from water
    if water:
        n_water = min(samples_per_type + (1 if remainder > 0 else 0), len(water))
        selected_water = random.sample(water, n_water)
        for selected_dir, selected_type in selected_water:
            samples.append((selected_dir, f"Landcover negative (water)"))
        print(f"  ✓ Selected {n_water} water samples")
        if remainder > 0:
            remainder -= 1
    else:
        print(f"  ⚠ No water samples available")
    
    # Select from cropland
    if cropland:
        n_cropland = min(samples_per_type + (1 if remainder > 0 else 0), len(cropland))
        selected_cropland = random.sample(cropland, n_cropland)
        for selected_dir, selected_type in selected_cropland:
            samples.append((selected_dir, f"Landcover negative (cropland)"))
        print(f"  ✓ Selected {n_cropland} cropland samples")
    else:
        print(f"  ⚠ No cropland samples available")
    
    # If we still don't have enough (some categories were empty), fill from available types
    if len(samples) < n_samples:
        print(f"  ⚠ Only {len(samples)}/{n_samples} samples selected (some categories had insufficient data)")
        all_available = urban + water + cropland
        remaining = [item for item in all_available if item[0] not in [s[0] for s in samples]]
        
        while len(samples) < n_samples and remaining:
            selected_dir, selected_type = random.choice(remaining)
            remaining.remove((selected_dir, selected_type))
            samples.append((selected_dir, f"Landcover negative ({selected_type})"))
            print(f"  ✓ Filled with additional ({selected_type}): {selected_dir.name}")
    
    return samples[:n_samples]


def select_unlabeled_samples(images_dir, n_samples=3):
    """
    Select random unlabeled samples.
    """
    all_unlabeled = list(images_dir.glob('unla_*'))
    
    if not all_unlabeled:
        print("  ⚠ No unlabeled samples found")
        return []
    
    # Random selection
    selected = random.sample(all_unlabeled, min(n_samples, len(all_unlabeled)))
    
    samples = []
    for grid_dir in selected:
        label_info = load_label_info(grid_dir)
        landscape_type = label_info.get('landscape_type', 'unknown')
        desc = f"Unlabeled ({landscape_type})"
        samples.append((grid_dir, desc))
    
    return samples


def create_comparison_grid(samples_by_category, output_path):
    """
    Create a mega-figure showing RGB composites from all categories.
    
    Args:
        samples_by_category: Dict with keys 'positives', 'integrated', 'landcover', 'unlabeled'
                            Each value is list of (grid_dir, description) tuples
        output_path: Path to save the comparison figure
    """
    categories = ['positives', 'integrated', 'landcover', 'unlabeled']
    category_labels = {
        'positives': 'Positives (Archaeological Sites)',
        'integrated': 'Integrated Negatives (Surrounding Landscape)',
        'landcover': 'Landcover Negatives (Urban/Water/Cropland)',
        'unlabeled': 'Unlabeled (Random Background)'
    }
    
    # Determine grid size
    n_rows = len(categories)
    n_cols = max(len(samples_by_category.get(cat, [])) for cat in categories)
    
    if n_cols == 0:
        print("  ⚠ No samples to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Dataset Sample Comparison - RGB Composites', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for row_idx, category in enumerate(categories):
        samples = samples_by_category.get(category, [])
        
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(samples):
                grid_dir, description = samples[col_idx]
                
                # Load RGB
                rgb = load_rgb_composite(grid_dir)
                ax.imshow(rgb)
                
                # Set title
                grid_id = grid_dir.name
                ax.set_title(f'{grid_id}\n{description}', fontsize=10)
                
            else:
                # Empty cell
                ax.axis('off')
            
            # Add row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(category_labels[category], 
                             fontsize=12, fontweight='bold', rotation=90, labelpad=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison grid: {output_path.name}")
    
    plt.close()


def main():
    print("Dataset Sample Visualization")
    print("=" * 70)
    
    # Paths
    dataset_dir = Path('outputs/dataset')
    images_dir = dataset_dir / 'grid_images'
    output_dir = dataset_dir / 'grid_samples'
    
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist")
        return
    
    print(f"Input directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load metadata
    metadata_path = dataset_dir / 'grid_metadata.parquet'
    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
        print(f"Metadata: {len(df)} total records")
        
        # Show breakdown
        for prefix, label in [('grid_', 'Positives'), 
                               ('ineg_', 'Integrated negatives'),
                               ('lneg_', 'Landcover negatives'), 
                               ('unla_', 'Unlabeled')]:
            count = len(df[df['grid_id'].str.startswith(prefix)])
            if count > 0:
                print(f"  {label}: {count}")
        print()
    
    # Number of samples per category
    n_samples = 9
    
    print(f"Selecting {n_samples} samples from each category...")
    print(f"  (All positive/integrated samples at rotation 0° for consistency)")
    print(f"  (Landcover: {n_samples//3} urban + {n_samples//3} water + {n_samples//3} cropland)")
    print()
    
    # Select samples
    print("1. Selecting positive samples (9 different sites)...")
    positive_samples = select_positive_samples(images_dir, n_samples, preferred_rotation=0)
    print(f"   Selected {len(positive_samples)} positive samples")
    
    print("\n2. Selecting integrated negatives (matching the selected positives)...")
    integrated_samples = select_integrated_samples(images_dir, positive_samples, n_samples)
    print(f"   Selected {len(integrated_samples)} integrated negative samples")
    
    print("\n3. Selecting landcover negatives (stratified: 3 urban + 3 water + 3 cropland)...")
    landcover_samples = select_landcover_samples(images_dir, n_samples)
    print(f"   Selected {len(landcover_samples)} landcover negative samples")
    
    print("\n4. Selecting unlabeled samples (9 random background)...")
    unlabeled_samples = select_unlabeled_samples(images_dir, n_samples)
    print(f"   Selected {len(unlabeled_samples)} unlabeled samples")
    
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print()
    
    # Visualize positives
    if positive_samples:
        print("Visualizing positives...")
        positives_dir = output_dir / 'positives'
        for grid_dir, description in positive_samples:
            output_path = positives_dir / f'{grid_dir.name}_visualization.png'
            visualize_grid_detailed(grid_dir, output_path, title_prefix="POSITIVE")
    
    # Visualize integrated negatives
    if integrated_samples:
        print("\nVisualizing integrated negatives...")
        integrated_dir = output_dir / 'integrated_negatives'
        for grid_dir, description in integrated_samples:
            output_path = integrated_dir / f'{grid_dir.name}_visualization.png'
            visualize_grid_detailed(grid_dir, output_path, title_prefix="INTEGRATED NEGATIVE")
    
    # Visualize landcover negatives
    if landcover_samples:
        print("\nVisualizing landcover negatives...")
        landcover_dir = output_dir / 'landcover_negatives'
        for grid_dir, description in landcover_samples:
            output_path = landcover_dir / f'{grid_dir.name}_visualization.png'
            visualize_grid_detailed(grid_dir, output_path, title_prefix="LANDCOVER NEGATIVE")
    
    # Visualize unlabeled
    if unlabeled_samples:
        print("\nVisualizing unlabeled samples...")
        unlabeled_dir = output_dir / 'unlabeled'
        for grid_dir, description in unlabeled_samples:
            output_path = unlabeled_dir / f'{grid_dir.name}_visualization.png'
            visualize_grid_detailed(grid_dir, output_path, title_prefix="UNLABELED")
    
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  positives/              - {len(positive_samples)} detailed visualizations")
    print(f"  integrated_negatives/   - {len(integrated_samples)} detailed visualizations")
    print(f"  landcover_negatives/    - {len(landcover_samples)} detailed visualizations")
    print(f"  unlabeled/              - {len(unlabeled_samples)} detailed visualizations")
    
    # Show correspondence between positives and integrated negatives
    if positive_samples and integrated_samples:
        print(f"\nPositive ↔ Integrated Negative Correspondence:")
        for i, ((pos_dir, _), (ineg_dir, _)) in enumerate(zip(positive_samples, integrated_samples), 1):
            print(f"  {i}. {pos_dir.name} ↔ {ineg_dir.name}")
    
    print()


if __name__ == "__main__":
    main()