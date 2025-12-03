import pandas as pd
from pathlib import Path

class MetadataBuilder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.records = []
    
    def add_record(self, grid_id, lat, lon, label, label_source, image_path):
        record = {
            'grid_id': grid_id,
            'centroid_lon': lon,
            'centroid_lat': lat,
            'label': label,
            'label_source': label_source,
            'image_path': image_path
        }
        self.records.append(record)
    
    def save_metadata(self):
        df = pd.DataFrame(self.records)
        
        parquet_path = self.output_dir / 'grid_metadata.parquet'
        df.to_parquet(parquet_path, index=False)
        
        return parquet_path
    
    def load_existing_metadata(self):
        parquet_path = self.output_dir / 'grid_metadata.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self.records = df.to_dict('records')
            return df
        return None