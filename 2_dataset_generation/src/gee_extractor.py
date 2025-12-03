import ee
import numpy as np

class GEEExtractor:
    def __init__(self, config):
        self.config = config
        self.s2_config = config['imagery']['sentinel2']
        self.fabdem_config = config['imagery']['fabdem']
        self.grid_config = config['grid']
        
        try:
            ee.Initialize(project=config['gee']['project'])
        except Exception as e:
            ee.Authenticate()
            ee.Initialize(project=config['gee']['project'])
    
    def create_grid_bbox(self, lat, lon):
        cell_size_km = self.grid_config['cell_size_km']
        half_size_deg = (cell_size_km / 2) / 111.32
        
        min_lon = lon - half_size_deg
        max_lon = lon + half_size_deg
        min_lat = lat - half_size_deg
        max_lat = lat + half_size_deg
        
        return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    def get_sentinel2_image(self, roi):
        collection = (
            ee.ImageCollection(self.s2_config['collection'])
            .filterBounds(roi)
            .filterDate(self.s2_config['date_start'], self.s2_config['date_end'])
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.s2_config['cloud_cover_max']))
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        )
        
        image = ee.Image(collection.first())
        return image
    
    def get_fabdem_data(self, roi):
        fabdem = (ee.ImageCollection(self.fabdem_config['collection'])
                  .filterBounds(roi)
                  .first())  # Use first() - mosaic() breaks sampleRectangle
        return fabdem
    
    def calculate_slope(self, dem_image):
        slope = ee.Terrain.slope(dem_image)
        return slope
    
    def extract_band_array(self, image, band, roi):
        pixels = self.grid_config['pixels_per_km']
        scale = self.s2_config['scale']
        
        band_image = image.select(band)
        
        array = band_image.sampleRectangle(region=roi, defaultValue=0)
        data = array.get(band).getInfo()
        
        arr = np.array(data, dtype=np.float32)
        
        if arr.shape != (pixels, pixels):
            from scipy.ndimage import zoom
            zoom_factors = (pixels / arr.shape[0], pixels / arr.shape[1])
            arr = zoom(arr, zoom_factors, order=1)
        
        return arr
    
    def extract_all_channels(self, lat, lon):
        roi = self.create_grid_bbox(lat, lon)
        
        s2_image = self.get_sentinel2_image(roi)
        fabdem_image = self.get_fabdem_data(roi)
        slope_image = self.calculate_slope(fabdem_image)
        
        channels = {}
        
        for band in self.s2_config['bands']:
            channels[band] = self.extract_band_array(s2_image, band, roi)
        
        channels['DEM'] = self.extract_band_array(fabdem_image, 'b1', roi)
        channels['Slope'] = self.extract_band_array(slope_image, 'slope', roi)
        
        image_info = s2_image.getInfo()
        metadata = {
            'image_id': image_info['id'] if image_info else None,
            'cloud_cover': image_info['properties'].get('CLOUDY_PIXEL_PERCENTAGE') if image_info else None,
            'acquisition_date': image_info['properties'].get('GENERATION_TIME') if image_info else None
        }
        
        return channels, metadata