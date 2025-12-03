import numpy as np
import pandas as pd

class NegativeSampler:
    def __init__(self, config, known_sites_path):
        self.config = config
        self.neg_config = config['negative_sampling']
        
        self.known_sites = pd.read_csv(known_sites_path)
        
        self.exclusion_buffer_km = self.neg_config['exclusion_buffer_km']
        self.study_region = self.neg_config['study_region']
        self.max_attempts = self.neg_config['max_attempts']
        
        self.exclusion_buffer_deg = self.exclusion_buffer_km / 111.32
        
        self.exclusion_zones = self._create_exclusion_zones()
        
    def _create_exclusion_zones(self):
        """Create exclusion buffer zones around known sites."""
        exclusion_zones = []
        
        for _, site in self.known_sites.iterrows():
            zone = {
                'site_id': site['site_id'],
                'lat': site['latitude'],
                'lon': site['longitude'],
                'buffer_deg': self.exclusion_buffer_deg
            }
            exclusion_zones.append(zone)
        
        return exclusion_zones
    
    def _is_in_exclusion_zone(self, lat, lon):
        """Check if a point is within any exclusion buffer."""
        for zone in self.exclusion_zones:
            distance = np.sqrt(
                (lat - zone['lat'])**2 + (lon - zone['lon'])**2
            )
            
            if distance < zone['buffer_deg']:
                return True
        
        return False
    
    def _is_in_study_region(self, lat, lon):
        """Check if a point is within the study region."""
        if self.study_region['type'] == 'bbox':
            return (
                self.study_region['min_lat'] <= lat <= self.study_region['max_lat'] and
                self.study_region['min_lon'] <= lon <= self.study_region['max_lon']
            )
        else:
            raise NotImplementedError(f"Study region type {self.study_region['type']} not implemented")
    
    def _generate_random_point(self):
        """Generate a random point within the study region."""
        if self.study_region['type'] == 'bbox':
            lat = np.random.uniform(
                self.study_region['min_lat'],
                self.study_region['max_lat']
            )
            lon = np.random.uniform(
                self.study_region['min_lon'],
                self.study_region['max_lon']
            )
            return lat, lon
        else:
            raise NotImplementedError(f"Study region type {self.study_region['type']} not implemented")
    
    def sample_negatives(self, num_negatives):
        """Sample negative locations with exclusion buffers."""
        sampled_negatives = []
        attempts = 0
        
        print(f"Sampling {num_negatives} negative locations...")
        print(f"Study region: {self.study_region['type']}")
        print(f"  Latitude: {self.study_region['min_lat']:.2f} to {self.study_region['max_lat']:.2f}")
        print(f"  Longitude: {self.study_region['min_lon']:.2f} to {self.study_region['max_lon']:.2f}")
        print(f"Exclusion buffer: {self.exclusion_buffer_km} km around {len(self.known_sites)} known sites")
        print()
        
        while len(sampled_negatives) < num_negatives and attempts < self.max_attempts:
            attempts += 1
            
            lat, lon = self._generate_random_point()
            
            if self._is_in_exclusion_zone(lat, lon):
                continue
            
            if not self._is_in_study_region(lat, lon):
                continue
            
            negative_info = {
                'negative_id': f'neg_{len(sampled_negatives) + 1:03d}',
                'latitude': lat,
                'longitude': lon,
                'difficulty': self.neg_config['default_difficulty'],
                'neg_type': self.neg_config['default_neg_type']
            }
            
            sampled_negatives.append(negative_info)
            
            if len(sampled_negatives) % 5 == 0:
                print(f"  Sampled {len(sampled_negatives)}/{num_negatives} negatives (attempts: {attempts})")
        
        if len(sampled_negatives) < num_negatives:
            print(f"\nWarning: Only sampled {len(sampled_negatives)}/{num_negatives} negatives after {attempts} attempts")
            print(f"  Consider expanding study region or reducing exclusion buffer")
        else:
            print(f"\n✓ Successfully sampled {len(sampled_negatives)} negatives in {attempts} attempts")
        
        return sampled_negatives
    
    def get_exclusion_stats(self):
        """Print statistics about exclusion zones."""
        print("\nExclusion Zone Statistics:")
        print(f"  Number of known sites: {len(self.known_sites)}")
        print(f"  Exclusion buffer: {self.exclusion_buffer_km} km ({self.exclusion_buffer_deg:.4f}°)")
        
        study_area = (
            (self.study_region['max_lat'] - self.study_region['min_lat']) *
            (self.study_region['max_lon'] - self.study_region['min_lon'])
        )
        
        exclusion_area = len(self.exclusion_zones) * np.pi * (self.exclusion_buffer_deg ** 2)
        
        exclusion_percent = (exclusion_area / study_area) * 100
        
        print(f"  Study region area: ~{study_area:.2f} deg²")
        print(f"  Total exclusion area: ~{exclusion_area:.2f} deg² ({exclusion_percent:.1f}% of study region)")
        print()