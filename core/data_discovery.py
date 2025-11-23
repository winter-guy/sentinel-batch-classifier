# core/data_discovery.py
import os
import zipfile
from .io_manager import RasterIOManager, _vsizip_path

class DataDiscovery:
    """Handles finding and preparing band paths from SAFE folders/zips/tifs."""

    @staticmethod
    def _find_band_paths_in_safe_folder(safe_path):
        """Only look for band files under the SAFE/<...>/IMG_DATA/ directory."""
        bands = {}
        for root, dirs, files in os.walk(safe_path):
            parts = [p.upper() for p in root.split(os.sep) if p]
            if 'IMG_DATA' in parts:
                for f in files:
                    fn = f.upper()
                    if fn.endswith('B02.JP2'): bands['B02'] = os.path.join(root, f)
                    if fn.endswith('B03.JP2'): bands['B03'] = os.path.join(root, f)
                    if fn.endswith('B04.JP2'): bands['B04'] = os.path.join(root, f)
                    if fn.endswith('B08.JP2'): bands['B08'] = os.path.join(root, f)
                    if fn.endswith('B11.JP2'): bands['B11'] = os.path.join(root, f)
        return bands

    @staticmethod
    def _find_band_paths_in_zip(zip_path):
        """Only consider files inside a path segment containing 'IMG_DATA' within the zip."""
        bands = {}
        with zipfile.ZipFile(zip_path, 'r') as z:
            for n in z.namelist():
                nn = n.replace('\\', '/').upper()
                if '/IMG_DATA/' not in nn:
                    continue
                if nn.endswith('B02.JP2'): bands['B02'] = n
                if nn.endswith('B03.JP2'): bands['B03'] = n
                if nn.endswith('B04.JP2'): bands['B04'] = n
                if nn.endswith('B08.JP2'): bands['B08'] = n
                if nn.endswith('B11.JP2'): bands['B11'] = n
        return bands

    def discover_and_prepare(self, scene_dir, logger):
        """Discovers scenes and prepares band sources for processing."""
        logger.info(f"Discovering scenes in '{scene_dir}'")
        scenes = []
        for entry in sorted(os.listdir(scene_dir)):
            full = os.path.join(scene_dir, entry)
            if entry.endswith('.zip') and '.SAFE' in entry:
                bands = self._find_band_paths_in_zip(full)
                if 'B02' in bands and 'B08' in bands:
                    vsibands = {k: _vsizip_path(full, v) for k, v in bands.items()}
                    scenes.append((os.path.splitext(entry)[0], vsibands))
            elif os.path.isdir(full) and entry.endswith('.SAFE'):
                bands = self._find_band_paths_in_safe_folder(full)
                if 'B02' in bands and 'B08' in bands:
                    scenes.append((entry, bands))
            elif entry.lower().endswith(('.tif', '.tiff')):
                try:
                    with RasterIOManager.read_raster(full) as src:
                        if src.count >= 4:
                            scenes.append((os.path.splitext(entry)[0], {'multiband_tif': full}))
                        else:
                            name = entry.upper()
                            for b in ['B02', 'B03', 'B04', 'B08', 'B11']:
                                if b in name:
                                    scenes.append((os.path.splitext(entry)[0], {b: full}))
                                    break
                except Exception:
                    pass
        # merge single-band entries
        merged = {}
        for sid, bs in scenes:
            if 'multiband_tif' in bs:
                merged[sid] = bs; continue
            base = sid.split('_B')[0] if '_B' in sid else sid
            if base not in merged: merged[base] = {}
            merged[base].update(bs)
        final = [(k, v) for k, v in merged.items() if 'B02' in v and 'B08' in v]
        logger.info(f"Found {len(final)} scene(s) to process.")
        return final