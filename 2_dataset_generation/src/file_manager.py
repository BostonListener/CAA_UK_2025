import numpy as np
import json
from pathlib import Path

class FileManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / 'grid_images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def create_grid_structure(self, grid_id):
        grid_dir = self.images_dir / grid_id
        channels_dir = grid_dir / 'channels'
        labels_dir = grid_dir / 'labels'
        
        channels_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        return grid_dir, channels_dir, labels_dir
    
    def save_channel(self, channels_dir, channel_name, data):
        if data.shape != (100, 100):
            raise ValueError(f"Channel {channel_name} has shape {data.shape}, expected (100, 100)")
        
        filepath = channels_dir / f"{channel_name}.npy"
        np.save(filepath, data.astype(np.float32))
    
    def save_all_channels(self, channels_dir, channels, indices):
        for name, data in channels.items():
            self.save_channel(channels_dir, name, data)
        
        for name, data in indices.items():
            self.save_channel(channels_dir, name, data)
    
    def save_labels(self, labels_dir, label, pos_type=None):
        label_array = np.array([label], dtype=np.int32)
        np.save(labels_dir / 'binary_label.npy', label_array)
        
        if pos_type:
            with open(labels_dir / 'pos_type.txt', 'w') as f:
                f.write(pos_type)
        
        with open(labels_dir / 'neg_type.txt', 'w') as f:
            f.write('null')
    
    def save_info(self, grid_dir, info_dict):
        info_path = grid_dir / 'info.json'
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
    
    def grid_exists(self, grid_id):
        grid_dir = self.images_dir / grid_id
        return grid_dir.exists()
    
    def get_image_path(self, grid_id):
        return f"grid_images/{grid_id}/"