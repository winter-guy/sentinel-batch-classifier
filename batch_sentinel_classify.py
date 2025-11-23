# batch_sentinel_classify.py
import argparse
import os
import csv
import sys
# Add necessary paths for local imports
# This ensures Python can find classes in 'utils' and 'core'
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from utils.logger import Logger
from core.data_discovery import DataDiscovery
from core.scene_processor import SceneProcessor


# The original code's dummy path for reference
EXAMPLE_SCREENSHOT = '/mnt/data/46c3117a-13a7-4548-927e-61a5a5955374.png'


class BatchSentinelClassifier:
    """Handles CLI arguments, scene discovery, and batch processing."""

    def __init__(self):
        self.logger = Logger()
        self.discovery = DataDiscovery()

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Full patched Batch Sentinel-2 3-class classifier.")
        parser.add_argument('--input_dir', required=True, help='Directory containing Sentinel-2 SAFE folders, ZIPs, or TIFs.')
        parser.add_argument('--outdir', default='outputs', help='Output directory for classified images and statistics.')
        
        # PCA arguments
        parser.add_argument('--pca_mode', default='sample', choices=['sample', 'incremental', 'randomized'], help='Mode for PCA/whitening.')
        parser.add_argument('--pca_sample', default=100000, type=int, help='Sample size for PCA sample mode.')
        parser.add_argument('--pca_var_thresh', default=0.95, type=float, help='Variance threshold for dimension reduction.')
        parser.add_argument('--randomized_components', default=8, type=int, help='Number of components for randomized PCA.')
        parser.add_argument('--randomized_iters', default=3, type=int, help='Iterations for randomized PCA power method.')
        parser.add_argument('--pca_chunk', default=200000, type=int, help='Chunk size for chunked PCA operations.')

        # K-Means arguments
        parser.add_argument('--kmeans_mode', default='minibatch', choices=['full', 'chunked', 'minibatch'], help='K-Means implementation mode.')
        parser.add_argument('--kmeans_chunk', default=200000, type=int, help='Chunk size for chunked K-Means label assignment (if used).')
        parser.add_argument('--kmeans_batch', default=15000, type=int, help='Minibatch size for minibatch K-Means.')
        parser.add_argument('--kmeans_iters', default=2000, type=int, help='Max iterations for K-Means (Minibatch default: 2000, others: 300).')
        parser.add_argument('--kmeans_init_iters', default=5, type=int, help='Initialization batches for minibatch K-Means.')
        parser.add_argument('--kmeans_assign_chunk', default=200000, type=int, help='Chunk size for final label assignment in minibatch mode.')
        
        return parser.parse_args()

    def run(self):
        args = self._parse_args()
        scenes = self.discovery.discover_and_prepare(args.input_dir, self.logger)
        
        if len(scenes) == 0:
            self.logger.error("No scenes discovered. Place SAFE/.zip/.tif files in the input_dir.")
            self.logger.info(f"Example screenshot path (not a scene): {EXAMPLE_SCREENSHOT}")
            return

        os.makedirs(args.outdir, exist_ok=True)
        all_results = []
        
        # Prepare configs from CLI arguments
        kmeans_cfg = {
            'mode': args.kmeans_mode,
            'k': 3,
            'max_iter': args.kmeans_iters if args.kmeans_mode == 'minibatch' else 300,
            'tol': 1e-5,
            'seed': 42,
            'chunk_size': args.kmeans_chunk,
            'batch_size': args.kmeans_batch,
            'init_iters': args.kmeans_init_iters,
            'learning_rate': 0.5,
            'assign_chunk': args.kmeans_assign_chunk
        }
        
        pca_cfg = {
            'mode': args.pca_mode,
            'sample_size': args.pca_sample,
            'var_thresh': args.pca_var_thresh,
            'randomized_components': args.randomized_components,
            'randomized_iters': args.randomized_iters,
            'eps': 1e-9,
            'chunk_size': args.pca_chunk
        }

        # Process scenes
        for scene_id, band_sources in scenes:
            processor = SceneProcessor(
                band_sources, 
                args.outdir, 
                Logger(prefix=f"{scene_id} | "),
                kmeans_cfg, 
                pca_cfg
            )
            result = processor.process(scene_id)
            if result:
                all_results.append(result)

        # Write Summary CSV
        summary_csv = os.path.join(args.outdir, "summary_all_scenes.csv")
        with open(summary_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scene_id', 'water_m2', 'veg_m2', 'other_m2', 'outdir'])
            for r in all_results:
                a = r.get('area_m2', {})
                outd = r.get('outdir', '')
                w.writerow([r['scene_id'], a.get(0, 0), a.get(1, 0), a.get(2, 0), outd])
        self.logger.info("All done. Outputs in: " + os.path.abspath(args.outdir))


if __name__ == '__main__':
    # Initialize and run the main class
    BatchSentinelClassifier().run()