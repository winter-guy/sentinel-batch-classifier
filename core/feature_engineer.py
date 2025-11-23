# core/feature_engineer.py
import numpy as np

class FeatureEngineer:
    """Computes various spectral indices, PCA, and DFT features."""

    @staticmethod
    def compute_ndvi(nir, red):
        denom = (nir + red)
        nd = np.zeros_like(nir, dtype='float32')
        mask = denom != 0
        nd[mask] = (nir[mask] - red[mask]) / denom[mask]
        return nd

    @staticmethod
    def compute_mndwi(green, swir1):
        denom = (green + swir1)
        m = np.zeros_like(green, dtype='float32')
        mask = denom != 0
        m[mask] = (green[mask] - swir1[mask]) / denom[mask]
        return m

    @staticmethod
    def compute_ndwi_mcfeeters(green, nir):
        """NDWI (McFeeters): (GREEN - NIR) / (GREEN + NIR) — fallback when SWIR1 not available."""
        denom = (green + nir)
        nd = np.zeros_like(green, dtype='float32')
        mask = denom != 0
        nd[mask] = (green[mask] - nir[mask]) / denom[mask]
        return nd

    @staticmethod
    def compute_pca_features(stack, n_components=3):
        bands, rows, cols = stack.shape
        X = stack.reshape(bands, -1).T
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        proj = Xc.dot(eigvecs[:, :n_components])
        pcs = proj.T.reshape(n_components, rows, cols)
        return pcs

    @staticmethod
    def compute_dft_magnitude(image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
        return mag

    def build_features(self, b02, b03, b04, b08, b11, shape, logger):
        """Computes all features and combines them into an (Npix, D) array."""
        stack = np.stack([b02, b03, b04, b08] + ([b11] if b11 is not None else []), axis=0)

        # Robust valid mask
        finite_mask = np.all(np.isfinite(stack), axis=0)
        nonzero_mask = np.any(stack != 0, axis=0)
        valid_mask = finite_mask & nonzero_mask

        if not valid_mask.any():
            logger.info("No valid pixels after finite+nonzero test — falling back to finite-only mask")
            valid_mask = finite_mask

        ndvi = self.compute_ndvi(b08, b04)
        if b11 is not None:
            water_index = 'mndwi'
            water_img = self.compute_mndwi(b03, b11)
        else:
            logger.info("B11 missing — using NDWI (green,NIR) as fallback")
            water_index = 'ndwi'
            water_img = self.compute_ndwi_mcfeeters(b03, b08)

        pcs = self.compute_pca_features(stack, n_components=3)
        dft_feat = self.compute_dft_magnitude(pcs[0])

        # build features array
        feature_list = [b02.flatten(), b03.flatten(), b04.flatten(), b08.flatten()]
        if b11 is not None: feature_list.append(b11.flatten())
        feature_list.extend([ndvi.flatten(), water_img.flatten(), pcs[0].flatten(), pcs[1].flatten(), dft_feat.flatten()])
        features = np.stack(feature_list, axis=1)  # (Npix, D)
        
        valid_flat = valid_mask.flatten()
        X = features[valid_flat].astype('float32')

        return X, ndvi, water_img, stack, valid_mask, valid_flat, water_index