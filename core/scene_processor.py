# core/scene_processor.py
import time
import numpy as np
import os

from utils.logger import Logger
from core.io_manager import RasterIOManager
from core.feature_engineer import FeatureEngineer
from core.clustering_engine import ClusteringEngine
from core.classifier import Classifier

class SceneProcessor:
    """Orchestrates the entire processing chain for a single Sentinel-2 scene."""

    def __init__(self, band_sources, outdir, logger, kmeans_cfg, pca_cfg):
        self.band_sources = band_sources
        self.outdir = outdir
        self.logger = logger
        self.kmeans_cfg = kmeans_cfg
        self.pca_cfg = pca_cfg
        self.io_manager = RasterIOManager()
        self.feature_engineer = FeatureEngineer()
        self.clustering_engine = ClusteringEngine()
        self.classifier = Classifier()

    def _log_band_stats(self, stack, b11_present):
        """Helper to log per-band statistics for diagnostics."""
        band_names = ['B02', 'B03', 'B04', 'B08'] + (['B11'] if b11_present else [])
        for i, name in enumerate(band_names):
            band = stack[i]
            n_total = band.size
            n_zero = int((band == 0).sum())
            n_nan = int(np.isnan(band).sum())
            n_finite = int(np.isfinite(band).sum())
            try:
                bmin = float(np.nanmin(band)) if n_finite > 0 else float('nan')
                bmax = float(np.nanmax(band)) if n_finite > 0 else float('nan')
            except Exception:
                bmin, bmax = float('nan'), float('nan')
            self.logger.info(f"Band {name}: shape={band.shape}, min={bmin}, max={bmax}, zeros={n_zero}, nans={n_nan}, finite={n_finite}/{n_total}")

    def process(self, scene_id):
        self.logger.info(f"Scene '{scene_id}': starting")
        start_scene = time.time()
        
        scene_outdir = os.path.join(self.outdir, scene_id)
        os.makedirs(scene_outdir, exist_ok=True)
        self.logger.info(f"Writing outputs into: {scene_outdir}")

        # 1. Read Bands
        try:
            self.logger.info("Reading bands (B02,B03,B04,B08[,B11])")
            b02, b03, b04, b08, b11, profile, shape = self.io_manager.read_bands(self.band_sources, self.logger)
        except Exception as e:
            self.logger.error(f"Failed to read bands: {e}")
            return None
        self.logger.info(f"Read bands complete (shape: {shape})")
        
        stack_for_log = np.stack([b02, b03, b04, b08] + ([b11] if b11 is not None else []), axis=0)
        self._log_band_stats(stack_for_log, b11 is not None)

        # 2. Feature Engineering
        self.logger.info("Preprocessing: building stack, valid mask, indices, PCA, DFT")
        try:
            X, ndvi, water_img, stack, valid_mask, valid_flat, water_index = self.feature_engineer.build_features(
                b02, b03, b04, b08, b11, shape, self.logger
            )
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return None

        N = (shape[0] * shape[1])
        self.logger.info(f"Built features: total_pixels={N}, valid_pixels={X.shape[0]}")
        if X.shape[0] == 0:
            self.logger.error("No valid pixels found for this scene (X is empty). Skipping scene.")
            return None

        # 3. Normalize
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-9
        Xn = (X - mu) / sigma

        # 4. PCA + Whitening
        self.logger.info(f"PCA mode: {self.pca_cfg.get('mode')}")
        Xw, project_whiten = self.clustering_engine.run_pca_whiten(Xn, self.pca_cfg, self.kmeans_cfg, self.logger)
        self.logger.info(f"Projected to reduced space (shape={Xw.shape})")

        # 5. K-Means Clustering
        self.logger.info(f"KMeans mode: {self.kmeans_cfg.get('mode')}")
        labels_valid, centers = self.clustering_engine.run_kmeans(Xw, self.kmeans_cfg)

        # 6. Cluster Mapping and Reconstruction
        rows, cols = shape
        labels_full = -1 * np.ones(rows * cols, dtype=int)
        labels_full[valid_flat] = labels_valid
        labels_img = labels_full.reshape(rows, cols)
        class_img, cmap, cluster_stats = self.classifier.map_clusters(labels_img, ndvi, water_img, stack)

        # 7. Save Outputs
        result = self.classifier.save_outputs(
            scene_id, scene_outdir, class_img, profile, cluster_stats, 
            water_index, ndvi, water_img, valid_flat, labels_valid, self.logger
        )

        elapsed = time.time() - start_scene
        self.logger.info(f"Scene '{scene_id}' finished in {elapsed:.2f}s")
        return result