import os
import numpy as np
from osgeo import gdal


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds

def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels

def write_geotiff(fname, data, cols, rows, geo_transform, projection):
    """Create a 1-band GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('GTiff')
    data = data.reshape((rows, cols))
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file
    
def prepare_problem(raster_data_path, training_data_path=None):
    """Prepare a datamining problem."""
    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    projection = raster_dataset.GetProjectionRef()
    
    # Get raster data in table format
    bands_data = []
    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    
    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    
    n_pixels = rows*cols
    flat_pixels = bands_data.reshape((n_pixels, n_bands))
    
    # Supervised 
    if training_data_path is not None:
        files = [f for f in os.listdir(training_data_path) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(training_data_path, f) for f in files if f.endswith('.shp')]
        labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, projection)
        is_train = np.nonzero(labeled_pixels)
        training_labels = labeled_pixels[is_train]
        training_samples = bands_data[is_train]
        out = {"flat_pixels": flat_pixels, "rows": rows, "cols": cols,
               "training_samples": training_samples, "training_labels": training_labels, 
               "geo_transform": geo_transform, "projection": projection, "classes": classes}
    else:
        # Unsupervised
        out = {"flat_pixels": flat_pixels, "rows": rows, "cols": cols,
               "geo_transform": geo_transform, "projection": projection}
    return out
    
    
    
    
    
    
    
    
    
    
