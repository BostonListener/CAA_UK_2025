import numpy as np

class IndexCalculator:
    @staticmethod
    def calculate_ndvi(b8, b4):
        numerator = b8 - b4
        denominator = b8 + b4
        
        ndvi = np.divide(numerator, denominator, 
                        out=np.zeros_like(numerator), 
                        where=denominator!=0)
        
        return ndvi.astype(np.float32)
    
    @staticmethod
    def calculate_ndwi(b3, b8):
        numerator = b3 - b8
        denominator = b3 + b8
        
        ndwi = np.divide(numerator, denominator,
                        out=np.zeros_like(numerator),
                        where=denominator!=0)
        
        return ndwi.astype(np.float32)
    
    @staticmethod
    def calculate_bsi(b11, b4, b8, b2):
        numerator = (b11 + b4) - (b8 + b2)
        denominator = (b11 + b4) + (b8 + b2)
        
        bsi = np.divide(numerator, denominator,
                       out=np.zeros_like(numerator),
                       where=denominator!=0)
        
        return bsi.astype(np.float32)
    
    @staticmethod
    def calculate_all_indices(channels):
        indices = {}
        
        indices['NDVI'] = IndexCalculator.calculate_ndvi(
            channels['B8'], channels['B4']
        )
        
        indices['NDWI'] = IndexCalculator.calculate_ndwi(
            channels['B3'], channels['B8']
        )
        
        indices['BSI'] = IndexCalculator.calculate_bsi(
            channels['B11'], channels['B4'], channels['B8'], channels['B2']
        )
        
        return indices