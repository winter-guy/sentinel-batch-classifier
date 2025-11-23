# core/clustering_engine.py
import numpy as np

class ClusteringEngine:
    """
    Implements K-Means variations (vectorized, chunked, minibatch)
    and the PCA/Whitening logic.
    """

    @staticmethod
    def _assign_labels_in_chunks(X, centers, chunk_size=500000):
        # Defensive handling for chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = 200000
        
        N = X.shape[0]
        labels = np.empty(N, dtype=np.int32)
        # Precompute squared norms for X and centers (only needed once for X)
        x2 = (X * X).sum(axis=1)[:, None]
        c2 = (centers * centers).sum(axis=1)[None, :]
        
        for i in range(0, N, chunk_size):
            chunk = X[i:i+chunk_size]
            xc = chunk.dot(centers.T)
            # Distance: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x.c
            dists_sq = x2[i:i+chunk.shape[0]] + c2 - 2.0 * xc
            labels[i:i+chunk.shape[0]] = np.argmin(dists_sq, axis=1)
        return labels

    def kmeans_vectorized(self, X, k=3, max_iter=200, tol=1e-4, seed=42, verbose=False):
        rng = np.random.default_rng(seed)
        N, D = X.shape
        if N == 0: raise ValueError("kmeans_vectorized called with empty dataset (N==0).")
        init_idx = rng.choice(N, size=k, replace=False)
        centers = X[init_idx].astype(np.float64)
        for it in range(max_iter):
            x2 = (X * X).sum(axis=1)[:, None]
            c2 = (centers * centers).sum(axis=1)[None, :]
            xc = X.dot(centers.T)
            dists_sq = x2 + c2 - 2.0 * xc
            labels = np.argmin(dists_sq, axis=1)
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = (labels == j)
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(0, N)]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if verbose and (it % 10 == 0): print(f"[kmeans] iter={it}, shift={shift:.6f}")
            if shift < tol: break
        return labels, centers

    def kmeans_chunked(self, X, k=3, max_iter=200, tol=1e-4, seed=42, chunk_size=200000, verbose=False):
        rng = np.random.default_rng(seed)
        N, D = X.shape
        if N == 0: raise ValueError("kmeans_chunked called with empty dataset (N==0).")
        
        chunk_size = chunk_size if chunk_size is not None else 200000
        try:
            chunk_size = int(chunk_size)
            if chunk_size <= 0: chunk_size = 200000
        except Exception: chunk_size = 200000

        init_idx = rng.choice(N, size=k, replace=False)
        centers = X[init_idx].astype(np.float64)
        for it in range(max_iter):
            labels = self._assign_labels_in_chunks(X, centers, chunk_size=chunk_size)
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = (labels == j)
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(0, N)]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if verbose and (it % 10 == 0): print(f"[kmeans_chunked] iter={it}, shift={shift:.6f}")
            if shift < tol: break
        return labels, centers

    def minibatch_kmeans(self, X, k=3, batch_size=10000, max_iter=2000, init_iters=5,
                         learning_rate=0.5, seed=42, verbose=False, assign_chunk=200000):
        rng = np.random.default_rng(seed)
        N, D = X.shape
        if N == 0: raise ValueError("minibatch_kmeans called with empty dataset (N==0).")
        init_idx = rng.choice(N, size=k, replace=False)
        centers = X[init_idx].astype('float64')

        # light initialization using a few minibatches
        for _ in range(init_iters):
            idx = rng.choice(N, size=min(batch_size, N), replace=False)
            batch = X[idx]
            x2 = (batch * batch).sum(axis=1)[:, None]
            c2 = (centers * centers).sum(axis=1)[None, :]
            xc = batch.dot(centers.T)
            dists = x2 + c2 - 2.0 * xc
            labels_b = np.argmin(dists, axis=1)
            for j in range(k):
                mask = (labels_b == j)
                if mask.any():
                    centers[j] = batch[mask].mean(axis=0)

        counts = np.zeros(k, dtype=np.int64)
        it = 0
        while it < max_iter:
            it += 1
            idx = rng.choice(N, size=min(batch_size, N), replace=False)
            batch = X[idx]
            x2 = (batch * batch).sum(axis=1)[:, None]
            c2 = (centers * centers).sum(axis=1)[None, :]
            xc = batch.dot(centers.T)
            dists = x2 + c2 - 2.0 * xc
            labels_b = np.argmin(dists, axis=1)

            for j in range(k):
                mask = (labels_b == j)
                if not mask.any(): continue
                counts[j] += mask.sum()
                eta = learning_rate / (1.0 + counts[j]*1e-6)
                batch_mean = batch[mask].mean(axis=0)
                centers[j] = (1 - eta) * centers[j] + eta * batch_mean

            if verbose and (it % 200 == 0): print(f"[minibatch] iter={it}")

        labels_full = self._assign_labels_in_chunks(X, centers, chunk_size=assign_chunk)
        return labels_full, centers

    @staticmethod
    def _compute_pca_whiten_safe(X, sample_size=100000, var_thresh=0.95, eps=1e-9, chunk_size=200000):
        N, D = X.shape
        # Compute mean
        sum_x = np.zeros((D,), dtype='float64')
        for i in range(0, N, chunk_size):
            sum_x += X[i:i+chunk_size].sum(axis=0)
        mu = (sum_x / N).astype('float32')

        # Compute covariance
        if sample_size is not None and N > sample_size:
            idx = np.random.choice(N, size=sample_size, replace=False)
            Xs = (X[idx] - mu).astype('float32')
            cov = np.cov(Xs, rowvar=False).astype('float64')
        else:
            sum_xx = np.zeros((D, D), dtype='float64')
            for i in range(0, N, chunk_size):
                chunk = X[i:i+chunk_size].astype('float64')
                sum_xx += chunk.T.dot(chunk)
            cov = (sum_xx / N) - np.outer(mu.astype('float64'), mu.astype('float64'))
            cov = (cov + cov.T) / 2.0
            
        eigvals, eigvecs = np.linalg.eigh(cov)
        idxs = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idxs].real
        eigvecs = eigvecs[:, idxs].real
        explained = eigvals / eigvals.sum()
        cum = explained.cumsum()
        m = int(np.searchsorted(cum, var_thresh) + 1)
        W = eigvecs[:, :m].astype('float32')
        Lambda = eigvals[:m].astype('float32')
        
        def project_whiten(Xfull, chunk_size_proj=None):
            Nfull = Xfull.shape[0]
            if chunk_size_proj is None: chunk_size_proj = 200000
            
            out = np.empty((Nfull, m), dtype='float32')
            for i in range(0, Nfull, chunk_size_proj):
                c = (Xfull[i:i+chunk_size_proj] - mu).astype('float32')
                xp = c.dot(W)
                out[i:i+xp.shape[0]] = (xp / np.sqrt(Lambda + eps)).astype('float32')
            return out
        return W, Lambda, mu, project_whiten

    @staticmethod
    def _randomized_pca_power(X, n_components=8, n_iter=3, seed=42, chunk_size=200000):
        rng = np.random.default_rng(seed)
        N, D = X.shape
        m = min(n_components, D)
        Q = rng.normal(size=(D, m)).astype('float32')
        
        for _ in range(n_iter):
            Z = np.zeros((D, m), dtype='float64')
            for i in range(0, N, chunk_size):
                chunk = X[i:i+chunk_size].astype('float64')
                Y = chunk.dot(Q)
                Z += chunk.T.dot(Y)
            Q, _ = np.linalg.qr(Z)
            
        B = np.zeros((m, m), dtype='float64')
        for i in range(0, N, chunk_size):
            chunk = X[i:i+chunk_size].astype('float64')
            Y = chunk.dot(Q)
            B += Y.T.dot(Y)
            
        eigvals_small, eigvecs_small = np.linalg.eigh(B)
        idx = np.argsort(eigvals_small)[::-1]
        eigvals_small = eigvals_small[idx]
        eigvecs_small = eigvecs_small[:, idx]
        
        eigvecs = Q.dot(eigvecs_small).astype('float32')
        eigvals = eigvals_small.astype('float32')
        return eigvecs, eigvals

    def run_pca_whiten(self, Xn, pca_cfg, kmeans_cfg, logger):
        """Applies PCA and whitening based on configuration."""
        pca_mode = pca_cfg.get('mode', 'sample')
        chunk_size_pca = pca_cfg.get('chunk_size', 200000)
        
        if pca_mode == 'randomized':
            ncomp = pca_cfg.get('randomized_components', min(8, Xn.shape[1]))
            niters = pca_cfg.get('randomized_iters', 3)
            eigvecs, eigvals = self._randomized_pca_power(
                Xn, n_components=ncomp, n_iter=niters, chunk_size=chunk_size_pca
            )
            total_var = eigvals.sum() + 1e-12
            explained = eigvals / total_var
            cum = np.cumsum(explained)
            m = int(np.searchsorted(cum, pca_cfg.get('var_thresh', 0.95)) + 1)
            W = eigvecs[:, :m]
            Lambda = eigvals[:m]
            eps = pca_cfg.get('eps', 1e-9)
            
            def project_whiten(Xfull, chunk_size_proj=None):
                Nfull = Xfull.shape[0]
                chunk_size_proj = chunk_size_proj if chunk_size_proj is not None else 200000
                out = np.empty((Nfull, m), dtype='float32')
                for i in range(0, Nfull, chunk_size_proj):
                    c = Xfull[i:i+chunk_size_proj]
                    xp = c.dot(W)
                    out[i:i+xp.shape[0]] = (xp / np.sqrt(Lambda + eps)).astype('float32')
                return out
        else:
            sample_size = pca_cfg.get('sample_size', 100000)
            if pca_mode == 'incremental':
                sample_size = None
            
            W, Lambda, mu_pca, project_whiten = self._compute_pca_whiten_safe(
                Xn,
                sample_size=sample_size,
                var_thresh=pca_cfg.get('var_thresh', 0.95),
                eps=pca_cfg.get('eps', 1e-9),
                chunk_size=chunk_size_pca
            )

        Xw = project_whiten(Xn, chunk_size_proj=kmeans_cfg.get('chunk_size', None))
        return Xw, project_whiten


    def run_kmeans(self, Xw, kmeans_cfg):
        """Runs the configured K-Means algorithm."""
        kmeans_mode = kmeans_cfg.get('mode', 'minibatch')
        
        if kmeans_mode == 'full':
            labels_valid, centers = self.kmeans_vectorized(
                Xw,
                k=kmeans_cfg.get('k', 3),
                max_iter=kmeans_cfg.get('max_iter', 300),
                tol=kmeans_cfg.get('tol', 1e-5),
                seed=kmeans_cfg.get('seed', 42),
                verbose=kmeans_cfg.get('verbose', False)
            )
        elif kmeans_mode == 'chunked':
            labels_valid, centers = self.kmeans_chunked(
                Xw,
                k=kmeans_cfg.get('k', 3),
                max_iter=kmeans_cfg.get('max_iter', 300),
                tol=kmeans_cfg.get('tol', 1e-5),
                seed=kmeans_cfg.get('seed', 42),
                chunk_size=kmeans_cfg.get('chunk_size', 200000),
                verbose=kmeans_cfg.get('verbose', False)
            )
        else:  # minibatch
            labels_valid, centers = self.minibatch_kmeans(
                Xw,
                k=kmeans_cfg.get('k', 3),
                batch_size=kmeans_cfg.get('batch_size', 15000),
                max_iter=kmeans_cfg.get('max_iter', 2000),
                init_iters=kmeans_cfg.get('init_iters', 5),
                learning_rate=kmeans_cfg.get('learning_rate', 0.5),
                seed=kmeans_cfg.get('seed', 42),
                verbose=kmeans_cfg.get('verbose', False),
                assign_chunk=kmeans_cfg.get('assign_chunk', 200000)
            )
        return labels_valid, centers