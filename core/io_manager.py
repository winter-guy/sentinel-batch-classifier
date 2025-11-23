# core/io_manager.py
import rasterio
from rasterio.enums import Resampling
from rasterio import Affine as AffineClass
import numpy as np
import os

def _vsizip_path(zip_path, inner_path):
    """Helper for VSI path creation."""
    return f"/vsizip/{os.path.abspath(zip_path)}/{inner_path}"


class RasterIOManager:
    """Manages reading and resampling raster bands."""

    @staticmethod
    def read_raster(path_or_vsip, resample_to=None):
        """Reads a single band, optionally resampling it to a target shape."""
        with rasterio.open(path_or_vsip) as src:
            if resample_to is None:
                arr = src.read(1).astype('float32')
                profile = src.profile
            else:
                out_shape = (resample_to[0], resample_to[1])
                arr = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
                scale_x = src.width / out_shape[1]
                scale_y = src.height / out_shape[0]
                transform = src.transform * AffineClass.scale(scale_x, scale_y)
                profile = dict(src.profile)
                profile.update({'height': out_shape[0], 'width': out_shape[1], 'transform': transform})
        return arr, profile

    def read_bands(self, band_sources, logger):
        """Reads all required bands, handling single/multi-band TIFs and resampling."""
        b02, b03, b04, b08, b11, profile = None, None, None, None, None, None
        shape = None

        if 'multiband_tif' in band_sources:
            path = band_sources['multiband_tif']
            with rasterio.open(path) as src:
                b02 = src.read(1).astype('float32')
                b03 = src.read(2).astype('float32')
                b04 = src.read(3).astype('float32')
                b08 = src.read(4).astype('float32')
                profile = src.profile
                shape = b08.shape
                b11 = src.read(5).astype('float32') if src.count >= 5 else None
        else:
            if not all(b in band_sources for b in ['B02', 'B03', 'B04', 'B08']):
                logger.error("Missing required bands (B02,B03,B04,B08).")
                raise ValueError("Missing core Sentinel-2 bands.")

            b08, profile = self.read_raster(band_sources['B08'])
            shape = b08.shape

            b02, _ = self.read_raster(band_sources['B02'], resample_to=shape)
            b03, _ = self.read_raster(band_sources['B03'], resample_to=shape)
            b04, _ = self.read_raster(band_sources['B04'], resample_to=shape)

            if 'B11' in band_sources:
                try:
                    b11, _ = self.read_raster(band_sources['B11'], resample_to=shape)
                except Exception:
                    b11 = None

        if shape is None:
            raise ValueError("Could not determine shape from bands.")

        return b02, b03, b04, b08, b11, profile, shape