# Batch Sentinel-2 Classifier

A clear, descriptive **README.md** for `batch_sentinel_classify_full_fixed.py`.
This document explains what the script does, how it discovers Sentinel SAFE/ZIP/TIF inputs, the processing pipeline, all CLI options, internal configuration mapping, outputs produced, recommended usage patterns for different dataset sizes, and troubleshooting tips.

---

## Overview

`batch_sentinel_classify_full_fixed.py` is a self-contained Batch Sentinel-2 3-class classifier (water / vegetation / other) implemented in pure Python + NumPy + rasterio.
It is designed to safely handle large scenes by using chunked processing, randomized/incremental PCA, and a choice of KMeans variants (full, chunked, minibatch). The script focuses on robustness: defensive handling of missing bands, fallback water indices, chunk-safety for memory, and diagnostic outputs for debugging.

Key features

* Reads Sentinel SAFE folders or SAFE zipfiles (only scans `IMG_DATA` subfolders) and single multiband or single-band TIF/TIFF.
* Builds per-pixel features (spectral bands, NDVI, water index, PCA components, DFT magnitude).
* Supports PCA modes: `sample`, `incremental`, `randomized`.
* Supports clustering modes: `full`, `chunked`, `minibatch`.
* Outputs classified GeoTIFF, visualization PNG, cluster stats CSV, per-scene area CSV, and a summary CSV for all scenes.

---

## Quick start


1. (Optional) Start a tmux session for long runs:

```bash
brew install tmux
tmux new -s sentinel
```

2. Create & activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run (example — default/`minibatch` recommended for large scenes):

```bash
python3 batch_sentinel_classify_full_fixed.py \
  --input_dir input_data \
  --outdir outputs \
  --pca_mode randomized \
  --randomized_components 10 \
  --randomized_iters 3 \
  --kmeans_mode minibatch
```

---

## Input discovery rules

The script looks for scenes in `--input_dir` using the following logic:

* **SAFE zip files**: any `.zip` whose filename contains `.SAFE` is opened and only files whose internal path contains `/IMG_DATA/` are considered. JP2 band filenames matched: `B02.JP2, B03.JP2, B04.JP2, B08.JP2, B11.JP2`. Paths are exposed to rasterio using `/vsizip/<abs_zip>/<inner_path>`.

* **SAFE folders**: directories ending with `.SAFE` are searched but only `IMG_DATA` subtrees are scanned for band JP2 files. This prevents picking up ancillary non-image files in the SAFE tree.

* **TIF/TIFF files**:

  * If `rasterio.open(file).count >= 4`, it's treated as a `multiband_tif` (expecting bands in order).
  * Otherwise, the filename is inspected for band tokens (`B02`, `B03`, `B04`, `B08`, `B11`) and single-band entries are grouped together into scenes.

* After discovery, single-band files are merged by common basename into scene groups so the rest of the pipeline can read a full band set or a multiband TIF.

**Required bands:** B02 (green), B03 (blue), B04 (red), B08 (NIR). B11 (SWIR1) is optional: when missing, the script falls back from MNDWI to NDWI (McFeeters).

---

## Processing pipeline (high-level)

1. **Read bands** — read B02, B03, B04, B08 and optionally B11. Resampling happens to B08 shape when bands are in separate files.

2. **Diagnostics** — prints per-band stats (shape, min/max, zeros, NaNs) to help with data validity checks.

3. **Masking** — constructs a `valid_mask` using finite & non-zero tests, falling back to finite-only if everything was masked.

4. **Feature computation**

   * NDVI: `(NIR - RED) / (NIR + RED)`
   * Water: `MNDWI` using (GREEN, SWIR1) if B11 available, otherwise `NDWI` (GREEN, NIR) fallback.
   * PCA features: `compute_pca_features` (simple PCA on stacked bands) used for additional descriptors.
   * DFT magnitude: FFT-based magnitude of first PCA component (normalized).
   * Feature vector per pixel includes spectral bands, NDVI, water index, first two PCA components, and DFT magnitude.

5. **Preprocessing** — flatten, filter by valid mask, standardize features (z-score).

6. **Dimensionality reduction** — options:

   * `sample` — PCA on a random sample (`--pca_sample`) then whiten.
   * `incremental` — chunked covariance accumulation (exact but slower).
   * `randomized` — randomized power-method PCA for approximate components.

7. **Clustering** — KMeans variants:

   * `full` — vectorized full KMeans (memory heavy).
   * `chunked` — KMeans with chunked label assignment to reduce memory footprint.
   * `minibatch` — Mini-batch KMeans (default) with minibatches for scaling.

8. **Postprocessing**

   * Build full-label image (labels for masked pixels set to -1).
   * Map clusters to classes (water=0, vegetation=1, other=2) using z-scored cluster-level heuristics on NDVI, water, and NIR means.
   * Produce diagnostics (NDVI vs water scatter), cluster stats CSV, areas (pixels → m²), GeoTIFF and visualization PNG.

---

## CLI options (detailed)

```
--input_dir            (required)    Path with SAFE folders/zips or TIF files.
--outdir               (str)         Output directory. Default: outputs

# PCA options
--pca_mode             (sample|incremental|randomized)  Default: sample
--pca_sample           (int)         Sample size for sample-mode PCA. Default: 100000
--pca_var_thresh       (float)       Variance retention threshold for PCA. Default: 0.95
--randomized_components (int)        #components for randomized PCA. Default: 8
--randomized_iters     (int)         Power iterations for randomized PCA. Default: 3
--pca_chunk            (int)         Chunk size used during PCA incremental/randomized ops. Default: 200000

# KMeans / clustering
--kmeans_mode          (full|chunked|minibatch)  Default: minibatch
--kmeans_chunk         (int)         Chunk size for chunked-mode label assignment (defaults to 200000 internally if not provided)
--kmeans_batch         (int)         Minibatch size for minibatch kmeans. Default: 15000
--kmeans_iters         (int)         Max iterations for minibatch kmeans. Default: 2000
--kmeans_init_iters    (int)         Initialization minibatch iterations (minibatch mode). Default: 5
--kmeans_assign_chunk  (int)         Chunk used when assigning final labels in minibatch mode. Default: 200000
```

> Note: The script normalizes some `None` CLI values to safe defaults (e.g., chunk sizes default to `200000` inside `main()`), so you may omit them if using standard settings.

---

## Internal config mapping

* `kcfg` — used for `chunked` and `full` paths. Contains: `mode, k, max_iter, tol, seed, chunk_size, batch_size, max_iter, init_iters, learning_rate, assign_chunk`.
* `kparams_for_minibatch` — used for `minibatch` function call. Contains: `k, batch_size, max_iter, init_iters, learning_rate, seed, assign_chunk`.
* `pcfg` — PCA config: `mode, sample_size, var_thresh, randomized_components, randomized_iters, eps, chunk_size`.

The script currently hardcodes `k = 3` cluster centers (water / veg / other). If you need a variable `k` or variable `seed`, add CLI flags and pass them to the `kcfg` / `kparams_for_minibatch` dicts.

---

## Outputs

For each detected scene `<scene_id>` the script writes to `<outdir>/<scene_id>/`:

* `<scene_id>_classified.tif` — single-band uint8 GeoTIFF with class codes (0=water, 1=vegetation, 2=other).
* `<scene_id>_classified_vis.png` — RGB visualization (blue=water, green=veg, red=other).
* `<scene_id>_ndvi_vs_<water_index>.png` — diagnostic scatter of NDVI vs water-index (sampled points).
* `<scene_id>_cluster_stats.csv` — per-cluster statistics and heuristic scores used for mapping to classes.
* `<scene_id>_areas.csv` — pixel counts and computed area (m²) per class.
* `outputs/summary_all_scenes.csv` — aggregated table of scenes and their areas.

---

## Example commands

**Default / recommended for big scenes (minibatch):**

```bash
python3 batch_sentinel_classify_full_fixed.py \
  --input_dir input_data \
  --outdir outputs \
  --pca_mode randomized \
  --randomized_components 10 \
  --randomized_iters 3 \
  --kmeans_mode minibatch
```

**Chunked KMeans (lower memory than full):**

```bash
python3 batch_sentinel_classify_full_fixed.py \
  --input_dir input_data \
  --outdir outputs \
  --pca_mode sample \
  --pca_sample 100000 \
  --kmeans_mode chunked \
  --kmeans_chunk 200000
```

**Full KMeans (only when dataset is small enough to fit in memory):**

```bash
python3 batch_sentinel_classify_full_fixed.py \
  --input_dir input_data \
  --outdir outputs \
  --pca_mode sample \
  --kmeans_mode full
```

**Incremental PCA + chunked KMeans (exact PCA for very large data):**

```bash
python3 batch_sentinel_classify_full_fixed.py \
  --input_dir input_data \
  --outdir outputs \
  --pca_mode incremental \
  --pca_chunk 200000 \
  --kmeans_mode chunked \
  --kmeans_chunk 200000
```

---

## Performance & tuning tips

* **Memory tuning:** increase `--kmeans_chunk`, `--kmeans_assign_chunk`, and `--pca_chunk` to reduce I/O passes but at the cost of RAM; decrease them to lower memory footprint.
* **Minibatch sizing:** increase `--kmeans_batch` for better center stability per iteration if you have RAM; decrease for lower memory usage.
* **PCA strategy:** use `randomized` for speed with large D/N and approximate components; use `incremental` for exact PCA when accuracy matters and you can accept the I/O cost; `sample` is a practical default.
* **Reproducibility:** the script uses a fixed RNG seed (`42`) internally. Add `--seed` into CLI and pass it into the `kcfg` and minibatch params if you need variable seeds or reproducible runs with a different seed.
* **Missing bands:** If B11 is missing the script will use `NDWI (GREEN,NIR)` as a fallback water index. Required bands are B02,B03,B04,B08.

---

## Troubleshooting

* **"No scenes discovered"** — ensure `--input_dir` contains `.SAFE` folders, SAFE-like `.zip` files, or TIF/TIFFs. Check that zip names include `.SAFE` if using SAFE zips.
* **"No valid pixels"** — check per-band diagnostics printed to logs. The script treats `0` and `NaN` as invalid by default. If your sensor uses fill values not equal to `0` (or if `0` is a valid value), adjust the masking logic.
* **Long runtimes / OOM errors** — reduce chunk sizes and minibatch sizes, or use `tmux` to keep sessions alive and monitor memory.
* **Unexpected class mapping** — check `*_cluster_stats.csv` to see per-cluster NDVI/water/NIR means and heuristic scores; tweak mapping heuristics in `map_clusters()` if needed.

---

## Extending the script

Common modifications you may want:

* Add CLI flags for `--k` (cluster count) and `--seed`.
* Add logging verbosity level to toggle debug prints.
* Add optional multi-threaded I/O or use rasterio windowed reads for micro-optimizations.
* Add a simple progress bar for per-chunk operations.
