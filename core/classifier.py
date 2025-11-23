# core/classifier.py
import numpy as np
import os
import csv
from collections import Counter
import rasterio
import matplotlib.pyplot as plt

class Classifier:
    """Handles cluster mapping and output generation."""

    @staticmethod
    def map_clusters(labels_img, ndvi_img, water_img, stack):
        """Maps kmeans labels to 3 classes (0:water, 1:veg, 2:other) based on feature heuristics."""
        h, w = labels_img.shape
        labels = labels_img.flatten()
        valid_mask = labels >= 0
        if not valid_mask.any():
            return np.full((h, w), 2, dtype=np.uint8), {}, {}

        unique = np.unique(labels[valid_mask])

        ndvi_flat = ndvi_img.flatten()
        water_flat = water_img.flatten()
        nir_flat = stack[3].flatten()  # B08 is at index 3 in the stack

        cluster_features = {}
        for u in unique:
            mask = (labels == u)
            cluster_features[u] = {
                'ndvi': float(np.nanmean(ndvi_flat[mask])),
                'water': float(np.nanmean(water_flat[mask])),
                'nir': float(np.nanmean(nir_flat[mask])),
                'size': int(mask.sum())
            }

        # Prepare arrays for z-scoring
        ndvi_vals = np.array([v['ndvi'] for v in cluster_features.values()])
        water_vals = np.array([v['water'] for v in cluster_features.values()])
        nir_vals = np.array([v['nir'] for v in cluster_features.values()])

        def z(x):
            sd = x.std()
            return (x - x.mean()) / (sd + 1e-9)

        ndvi_z = z(ndvi_vals)
        water_z = z(water_vals)
        nir_z = z(nir_vals)

        cmap = {}
        mapping_info = {}
        keys = list(cluster_features.keys())
        for i, u in enumerate(keys):
            # scoring heuristics
            veg_score = ndvi_z[i] + 0.6 * nir_z[i]
            water_score = water_z[i] - 0.6 * ndvi_z[i] - 0.3 * nir_z[i]
            other_score = -0.1 * abs(ndvi_z[i])
            
            scores = {'veg': float(veg_score), 'water': float(water_score), 'other': float(other_score)}
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            assigned = 1 if best == 'veg' else (0 if best == 'water' else 2)
            cmap[u] = assigned
            mapping_info[u] = {'assigned': assigned, 'scores': scores, **cluster_features[u]}

        class_flat = np.full_like(labels, 2, dtype=np.uint8)
        for u, assigned in cmap.items():
            class_flat[labels == u] = assigned
        class_img = class_flat.reshape(h, w)
        return class_img, cmap, mapping_info

    def save_outputs(self, scene_id, scene_outdir, class_img, profile, cluster_stats, water_index, ndvi, water_img, valid_flat, labels_valid, logger):
        """Saves the classified raster, visualization, area CSV, and cluster stats."""
        
        rows, cols = class_img.shape
        counts = Counter(class_img.flatten()[class_img.flatten() >= 0])
        pixel_area_m2 = abs(profile['transform'][0]) * abs(profile['transform'][4])
        area_m2 = {cls: counts.get(cls, 0) * pixel_area_m2 for cls in [0, 1, 2]}

        # 1. Classified TIF
        out_tif = os.path.join(scene_outdir, f"{scene_id}_classified.tif")
        p = dict(profile); p.update({'count': 1, 'dtype': 'uint8'})
        with rasterio.open(out_tif, 'w', **p) as dst:
            dst.write(class_img.astype('uint8'), 1)

        # 2. Visualization PNG
        vis = np.zeros((rows, cols, 3), dtype='uint8')
        cmap_rgb = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
        for k, v in cmap_rgb.items(): vis[class_img == k] = v
        vis_path = os.path.join(scene_outdir, f"{scene_id}_classified_vis.png")
        plt.imsave(vis_path, vis)
        
        # 3. Diagnostic Scatter Plot
        try:
            sN = labels_valid.shape[0]
            samp_n = min(20000, sN)
            rng = np.random.default_rng(42)
            sidx = rng.choice(sN, size=samp_n, replace=False) if sN > 0 else np.array([], dtype=int)
            sample_ndvi = ndvi.flatten()[valid_flat][sidx] if sidx.size else np.array([])
            sample_water = water_img.flatten()[valid_flat][sidx] if sidx.size else np.array([])
            sample_labels = labels_valid[sidx] if sidx.size else np.array([])
            if sidx.size:
                plt.figure(figsize=(6,6))
                plt.scatter(sample_water, sample_ndvi, c=sample_labels, s=1) 
                plt.xlabel(f"{water_index.upper()} (water measure)")
                plt.ylabel("NDVI")
                plt.title(f"{scene_id} - NDVI vs {water_index.upper()}")
                scatter_path = os.path.join(scene_outdir, f"{scene_id}_ndvi_vs_{water_index}.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
        except Exception as e:
            logger.info(f"Diagnostic scatter failed: {e}")

        # 4. Per-cluster CSV
        try:
            cluster_csv = os.path.join(scene_outdir, f"{scene_id}_cluster_stats.csv")
            with open(cluster_csv, 'w', newline='') as cf:
                w = csv.writer(cf)
                w.writerow(['cluster','assigned_class','mean_ndvi','mean_water','mean_nir','size','score_veg','score_water','score_other'])
                for u, info in cluster_stats.items():
                    assigned = info.get('assigned', None)
                    mean_ndvi = info.get('ndvi', np.nan)
                    mean_water = info.get('water', np.nan)
                    mean_nir = info.get('nir', np.nan)
                    size = info.get('size', 0)
                    sc = info.get('scores', {})
                    w.writerow([u, assigned, mean_ndvi, mean_water, mean_nir, size, sc.get('veg',''), sc.get('water',''), sc.get('other','')])
        except Exception as e:
            logger.info(f"Failed to write cluster CSV: {e}")

        # 5. Area CSV
        csv_path = os.path.join(scene_outdir, f"{scene_id}_areas.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class_code', 'class_name', 'pixel_count', 'area_m2'])
            name_map = {0: 'water', 1: 'vegetation', 2: 'other'}
            for cls in [0, 1, 2]:
                writer.writerow([cls, name_map[cls], counts.get(cls, 0), area_m2[cls]])
        
        logger.info(f"Saved outputs: {out_tif}, {vis_path}, {csv_path}")
        return {'scene_id': scene_id, 'area_m2': area_m2, 'counts': counts, 'outdir': scene_outdir}