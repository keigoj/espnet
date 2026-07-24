# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging
import os
import random
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from espnet.utils.cli_readers import file_reader_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def _centers_path(km_path: str) -> str:
    if km_path.endswith(".mdl"):
        return km_path[: -len(".mdl")] + ".npy"
    return km_path + ".npy"


def _save_centers(km_path: str, centers: np.ndarray) -> None:
    centers_path = _centers_path(km_path)
    np.save(centers_path, np.asarray(centers))
    logger.info("saved cluster centers to %s", centers_path)


def _assign_batched(X: np.ndarray, centers: np.ndarray, batch_size: Optional[int]) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float32, order="C")
    centers = np.asarray(centers, dtype=np.float32, order="C")
    n = X.shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = n
    assignments = np.empty((n,), dtype=np.int64)
    inertia = 0.0
    centers_t = centers.T  # (D, K)
    center_norm = (centers * centers).sum(axis=1)  # (K,)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        x_norm = (xb * xb).sum(axis=1, keepdims=True)  # (B, 1)
        dist2 = x_norm + center_norm[None, :] - 2.0 * (xb @ centers_t)
        assign = np.argmin(dist2, axis=1)   # (B,)
        assignments[start:end] = assign     # (B,)
        inertia += dist2[np.arange(end - start), assign].sum()

    return assignments, float(inertia) # (K,)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)

    parser.add_argument(
        "--RVQ_layers",
        type=int,
        default=1,
        help="Number of RVQ layers.",
    )
    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier",
        type=str,
        nargs="+",
        help="Read specifier for feats. e.g. ark:some.ark",
    )
    parser.add_argument(
        "--label_rspecifier",
        type=str,
        default=None,
        help="Optional phoneme/frame labels in 'utt_id val1 val2 ...' format.",
    )
    parser.add_argument(
        "--label_filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "text_int", "text"],
        help=(
            "Label format. 'text_int' expects space separated integers per utt. "
            "'text' expects space separated symbols per utt."
        ),
    )
    parser.add_argument(
        "--kmeans_method",
        type=str,
        default="base",
        choices=["base", "phone-based", "anchore-based"],
        help="KMeans variant.",
    )
    parser.add_argument(
        "--anchor_center_path",
        type=str,
        default=None,
        help="Path to anchor centers for anchore-based kmeans (.npy or .npz).",
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=1.0,
        help="Anchor regularization weight for anchore-based kmeans.",
    )
    parser.add_argument(
        "--anchor_mode",
        type=str,
        default="cluster",
        choices=["cluster", "frame"],
        help="Anchor mode for anchore-based kmeans.",
    )
    parser.add_argument(
        "--one_to_one",
        action="store_true",
        help="Whether to use one-to-one mapping for anchors.",
    )
    return parser


class PhoneBasedKMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 50,
        tol: float = 1e-5,
        seed: int = 0,
        lambda_phone: float = 1.0,
        default_phone: int = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.lambda_phone = lambda_phone
        self.default_phone = default_phone
        self.verbose = verbose

        # fitted attrs
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, feat: np.ndarray):
        X, phone = self._split_feat_and_phone(feat)

        rng = np.random.default_rng(self.seed)
        self.cluster_centers_ = self._init_kmeanspp(X, rng)  # or random init

        prev_inertia = None
        for it in tqdm(range(1, self.max_iter + 1)):
            assign, inertia = self._assign(X, self.cluster_centers_)
            centers_new = self._update_centers(X, phone, assign)

            # judge convergence (inertia or center shift)
            if prev_inertia is not None and abs(prev_inertia - inertia) <= self.tol:
                self.n_iter_ = it
                self.inertia_ = inertia
                self.cluster_centers_ = centers_new
                break

            prev_inertia = inertia
            self.cluster_centers_ = centers_new
            self.inertia_ = inertia
            self.n_iter_ = it

            if self.verbose:
                print(f"[iter {it}] inertia={inertia:.4e}")

        return self

    def predict(self, feat: np.ndarray) -> np.ndarray:
        X, _ = self._split_feat_and_phone(feat)
        assign, _ = self._assign(X, self.cluster_centers_)
        return assign

    def score(self, feat: np.ndarray) -> float:
        X, _ = self._split_feat_and_phone(feat)
        _, inertia = self._assign(X, self.cluster_centers_)
        return -float(inertia)

    def _split_feat_and_phone(self, feat: np.ndarray):
        if feat.ndim != 2 or feat.shape[1] < 2:
            raise ValueError("feat must be 2D with at least 2 columns (X + phone_id).")
        X = feat[:, :-1].astype(np.float32, copy=False)
        phone = feat[:, -1].astype(np.int64, copy=False)
        return X, phone

    def _assign(self, X: np.ndarray, centers: np.ndarray):
        diff = X[:, None, :] - centers[None, :, :]              # X: (N, D), centers: (K, D) -> diff: (N, K, D)
        dist2 = np.sum(diff * diff, axis=-1)                    # (N, K)
        assign = np.argmin(dist2, axis=1)                       # (N,)
        inertia = np.sum(dist2[np.arange(X.shape[0]), assign])  # scalar
        return assign, float(inertia)

    def _update_centers(self, X: np.ndarray, phone: np.ndarray, assign: np.ndarray) -> np.ndarray:
        K, D = self.n_clusters, X.shape[1]
        centers = np.zeros((K, D), dtype=np.float32)
        
        for k in range(K):
            idx = np.where(assign == k)[0]
            
            if len(idx) == 0:
                continue

            Xk = X[idx]
            pk = phone[idx]

            mu = Xk.mean(axis=0)

            # select most frequent phone excluding default_phone
            # if default_phone is given, ignore it in mode calculation
            if self.default_phone is not None:
                valid = pk != self.default_phone
            else:
                valid = np.ones_like(pk, dtype=bool)
            
            if np.any(valid):
                pk2 = pk[valid]
                Xk2 = Xk[valid]
                
                # compute mode
                vals, cnts = np.unique(pk2, return_counts=True)
                phone_mode = vals[np.argmax(cnts)]
                X_pure = Xk2[pk2 == phone_mode]
                p_mu = X_pure.mean(axis=0)
            else:
                p_mu = mu

            lam = float(self.lambda_phone)
            centers[k] = (Xk.sum(axis=0) + lam * p_mu) / (len(idx) + lam)

        return centers
    
    def _init_kmeanspp(self, X, rng):
        N, D = X.shape
        K = self.n_clusters
        centers = np.empty((K, D), dtype=np.float32)
        
        # select first center randomly
        i0 = rng.integers(0, N)
        centers[0] = X[i0]

        # select remaining centers
        dist2 = np.sum((X - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            prob = dist2 / (dist2.sum() + 1e-12)
            ik = rng.choice(N, p=prob)
            centers[k] = X[ik]
            dist2 = np.minimum(dist2, np.sum((X - centers[k]) ** 2, axis=1))

        return centers


class BatchedPhoneBasedKMeans(PhoneBasedKMeans):
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 50,
        tol: float = 0.0,
        seed: int = 0,
        lambda_phone: float = 1.0,
        default_phone: int = None,
        verbose: bool = False,
        assign_batch_size: Optional[int] = None,
    ):
        super().__init__(
            n_clusters,
            max_iter,
            tol,
            seed,
            lambda_phone,
            default_phone,
            verbose,
        )
        self.assign_batch_size = assign_batch_size

    def _assign(self, X: np.ndarray, centers: np.ndarray):
        return _assign_batched(X, centers, self.assign_batch_size) # (K,)


class AnchoredKMeans(PhoneBasedKMeans):
    def __init__(
            self,
            n_clusters: int,
            max_iter: int = 50,
            tol: float = 0.0,
            seed: int = 0,
            lambda_phone: float = 1.0,
            default_phone: int = None,
            verbose: bool = False,
            anchor_centers: np.ndarray | None = None,
            anchor_mode: str = "cluster",
            one_to_one: bool = True,
        ):
        super().__init__(
            n_clusters, 
            max_iter, 
            tol, 
            seed, 
            lambda_phone, 
            default_phone, 
            verbose
        )
        if anchor_centers is None:
            self.anchor_centers = None
        else:
            self.anchor_centers = np.asarray(anchor_centers, dtype=np.float32)
        
        if anchor_mode not in ("cluster", "frame"):
            raise ValueError("anchor_mode must be 'cluster' or 'frame'")
        self.anchor_mode = anchor_mode
        self.one_to_one = bool(one_to_one)

    def _split_feat_and_phone(self, feat: np.ndarray):
        if feat.ndim != 2:
            raise ValueError("feat must be 2D feature matrix.")
        X = feat.astype(np.float32, copy=False)
        return X, None

    def _validate_anchor_centers(self, X: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        if anchor.ndim != 2:
            raise ValueError("anchor_centers must be a 2D array.")
        if anchor.shape[0] != self.n_clusters:
            raise ValueError(
                f"anchor_centers must have {self.n_clusters} rows, got {anchor.shape[0]}"
            )
        if anchor.shape[1] != X.shape[1]:
            raise ValueError(
                f"anchor_centers dim {anchor.shape[1]} does not match X dim {X.shape[1]}"
            )
        anchor = anchor.astype(np.float32, copy=False)
        self.anchor_centers = anchor
        return anchor
    
    @staticmethod
    def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # returns squared Euclidean distance matrix: (A_rows, B_rows)
        return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)

    def _cluster_to_anchor_mapping(self, mu_bar: np.ndarray, counts: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """
        For mode='cluster': decide a(k) for each cluster k based on distance between mu_bar[k] and anchor[j].
        - Non-empty clusters are mapped.
        - Empty clusters keep a(k)=k by default.
        """
        K = self.n_clusters
        a = np.arange(K, dtype=np.int64)

        nonempty = np.where(counts > 0)[0]
        if len(nonempty) == 0:
            return a

        dist = self._sqdist(mu_bar[nonempty], anchor)  # (M, K)

        if self.one_to_one:
            # Hungarian if available and M<=K; else greedy unique
            use_hungarian = False
            try:
                from scipy.optimize import linear_sum_assignment
                use_hungarian = (dist.shape[0] <= dist.shape[1])
            except Exception:
                use_hungarian = False

            if use_hungarian:
                row_ind, col_ind = linear_sum_assignment(dist)
                for r, c in zip(row_ind, col_ind):
                    a[nonempty[r]] = int(c)
                return a

            # Greedy unique matching fallback
            taken = set()
            order = np.argsort(dist.min(axis=1))  # easy-first
            for r in order:
                k = int(nonempty[r])
                cand = np.argsort(dist[r])
                chosen = None
                for c in cand:
                    if int(c) not in taken:
                        chosen = int(c)
                        break
                if chosen is None:
                    chosen = int(cand[0])  # allow collision
                a[k] = chosen
                taken.add(chosen)
            return a

        # Not one-to-one: simple nearest
        nearest = dist.argmin(axis=1)
        for i, k in enumerate(nonempty):
            a[int(k)] = int(nearest[i])
        return a

    def _frame_to_anchor_index(self, X: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """
        For mode='frame': for each frame t, choose nearest anchor j*(t).
        Returns (T,) int64 indices.
        """
        dist = self._sqdist(X, anchor)  # (T, K)
        return dist.argmin(axis=1).astype(np.int64)

    def _update_centers(self, X: np.ndarray, phone: np.ndarray, assign: np.ndarray) -> np.ndarray:
        K, D = self.n_clusters, X.shape[1]
        if self.anchor_centers is None:
            raise ValueError("anchor_centers must be provided for AnchoredKMeans.")
        anchor = self._validate_anchor_centers(X, np.asarray(self.anchor_centers))
        lam = float(self.lambda_phone)

        centers = np.zeros((K, D), dtype=np.float32)

        # --- compute cluster means and counts
        mu_bar = np.zeros((K, D), dtype=np.float32)
        counts = np.zeros((K,), dtype=np.int64)
        cluster_indices = [None] * K
        for k in range(K):
            idx = np.where(assign == k)[0]
            cluster_indices[k] = idx
            counts[k] = len(idx)
            if counts[k] > 0:
                mu_bar[k] = X[idx].mean(axis=0)

        if self.anchor_mode == "cluster":
            # (1) cluster-level anchoring: one anchor per cluster
            a = self._cluster_to_anchor_mapping(mu_bar, counts, anchor)  # (K,)
            for k in range(K):
                idx = cluster_indices[k]
                ak = int(a[k])
                if len(idx) == 0:
                    centers[k] = anchor[ak]
                else:
                    Xk = X[idx]
                    centers[k] = (Xk.sum(axis=0) + lam * anchor[ak]) / (len(idx) + lam)
            return centers
        elif self.anchor_mode == "frame":
            # (2) frame-level anchoring: each frame picks its nearest anchor, then cluster center is pulled to mean(anchor_of_frames)
            j_star = self._frame_to_anchor_index(X, anchor)  # (T,)
            for k in range(K):
                idx = cluster_indices[k]
                if len(idx) == 0:
                    centers[k] = anchor[k]  # deterministic fallback
                    continue
                Xk = X[idx]                               # (n, D)
                Ak = anchor[j_star[idx]]                  # (n, D) frame-chosen anchors
                # treat each frame's anchor as lam pseudo-sample (so total pseudo count = lam*n)
                n = len(idx)
                centers[k] = (Xk.sum(axis=0) + lam * Ak.sum(axis=0)) / (n + lam * n)
            return centers
        else:
            raise ValueError(f"Unknown anchor_mode: {self.anchor_mode}")


class BatchedAnchoredKMeans(AnchoredKMeans):
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 50,
        tol: float = 0.0,
        seed: int = 0,
        lambda_phone: float = 1.0,
        default_phone: int = None,
        verbose: bool = False,
        anchor_centers: np.ndarray | None = None,
        anchor_mode: str = "cluster",
        one_to_one: bool = True,
        assign_batch_size: Optional[int] = None,
    ):
        super().__init__(
            n_clusters,
            max_iter,
            tol,
            seed,
            lambda_phone,
            default_phone,
            verbose,
            anchor_centers=anchor_centers,
            anchor_mode=anchor_mode,
            one_to_one=one_to_one,
        )
        self.assign_batch_size = assign_batch_size

    def _assign(self, X: np.ndarray, centers: np.ndarray):
        return _assign_batched(X, centers, self.assign_batch_size)


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    seed,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=seed,
    )


def _text_int_reader(label_rspecifier: str) -> Iterable[Tuple[str, np.ndarray]]:
    """Simple reader for text labels: each line 'utt_id 1 2 3'."""
    path = (
        label_rspecifier.replace("scp:", "", 1)
        if label_rspecifier.startswith("scp:")
        else label_rspecifier
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            utt, *vals = parts
            if len(vals) == 0:
                raise ValueError(f"No label values found for utt {utt} in {path}")
            yield utt, np.asarray([int(v) for v in vals], dtype=np.int64)


def _text_symbol_reader(label_rspecifier: str) -> Iterable[Tuple[str, List[str]]]:
    """Simple reader for text labels: each line 'utt_id sym1 sym2 sym3'."""
    path = (
        label_rspecifier.replace("scp:", "", 1)
        if label_rspecifier.startswith("scp:")
        else label_rspecifier
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            utt, *vals = parts
            if len(vals) == 0:
                raise ValueError(f"No label values found for utt {utt} in {path}")
            yield utt, vals


def _build_symbol_dict(label_rspecifier: str) -> Dict[str, int]:
    counts: Counter = Counter()
    for _, vals in _text_symbol_reader(label_rspecifier):
        counts.update(vals)
    if not counts:
        raise ValueError(f"No label symbols found in {label_rspecifier}")
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    token_to_id = {tok: i for i, (tok, _) in enumerate(items)}
    logger.info("built %d label tokens from %s", len(token_to_id), label_rspecifier)
    return token_to_id


def convert_label_id(label_rspecifier: str) -> Dict[str, np.ndarray]:
    token_to_id = _build_symbol_dict(label_rspecifier)
    label_dict: Dict[str, np.ndarray] = {}
    for utt, vals in _text_symbol_reader(label_rspecifier):
        label_dict[utt] = np.asarray([token_to_id[v] for v in vals], dtype=np.int64)
    logger.info("loaded %d label entries from %s", len(label_dict), label_rspecifier)
    return label_dict


def load_label_dict(label_rspecifier: Optional[str], label_filetype: str) -> Dict[str, np.ndarray]:
    """Load label dictionary from rspecifier."""
    if label_rspecifier is None:
        return {}
    if label_filetype == "text_int":
        reader = _text_int_reader(label_rspecifier)
    elif label_filetype == "text":
        return convert_label_id(label_rspecifier)
    else:
        reader = file_reader_helper(label_rspecifier, label_filetype)

    label_dict: Dict[str, np.ndarray] = {}
    for utt, lab in reader:
        label_dict[utt] = lab
    logger.info("loaded %d label entries from %s", len(label_dict), label_rspecifier)
    return label_dict


def load_anchor_centers(anchor_center_path: str) -> np.ndarray:
    data = np.load(anchor_center_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "centers" in data:
            centers = data["centers"]
        elif len(data.files) == 1:
            centers = data[data.files[0]]
        else:
            raise ValueError(
                "anchor_center_path .npz must contain 'centers' or a single array"
            )
    else:
        centers = data
    if centers.ndim != 2:
        raise ValueError("anchor centers must be a 2D array")
    return centers.astype(np.float32, copy=False)


def load_feature_shard(
    rspecifier, 
    in_filetype, 
    percent, 
    label_dict: Optional[Dict[str, np.ndarray]] = None, 
):
    """Load feature shard from rspecifier."""
    feats = []
    cropped = 0
    cropped_examples = []
    for utt, feat in file_reader_helper(rspecifier, in_filetype):
        if label_dict:
            if utt not in label_dict:
                raise ValueError(f"Label for {utt} is missing in {len(label_dict)} provided labels")
            lab = label_dict[utt]
            if lab.ndim == 1:
                lab = lab[:, None]
            if lab.shape[0] != feat.shape[0]:
                if abs(feat.shape[0] - lab.shape[0]) <= 1:
                    cropped += 1
                    if len(cropped_examples) < 5:
                        cropped_examples.append((utt, feat.shape[0], lab.shape[0]))
                    min_len = min(feat.shape[0], lab.shape[0])
                    feat = feat[:min_len]
                    lab = lab[:min_len]
                else:
                    raise ValueError(
                        f"Length mismatch for {utt}: feat {feat.shape[0]} vs label {lab.shape[0]}"
                    )
            feat = np.concatenate([feat, lab], axis=1) # feat: (T, D), lab: (T, 1)
        feats.append(feat)

    if cropped > 0:
        logger.warning(
            "cropped %d feature/label pairs with <=1 frame mismatch in %s; examples=%s",
            cropped,
            rspecifier,
            cropped_examples,
        )
    
    if percent < 0:
        return np.concatenate(feats, axis=0)
    else:
        nsample = int(np.ceil(len(feats) * percent))
        sampled_feat = random.sample(feats, nsample)
        sampled_feat = np.concatenate(
            sampled_feat,
            axis=0,
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from rspecifier {rspecifier}"
            )
        )
        return sampled_feat # (N, D)


def load_feature(rspecifiers, in_filetype, percent, label_dict: Optional[Dict[str, np.ndarray]] = None):
    """Load features from one or more rspecifiers."""
    assert percent <= 1.0
    if not isinstance(rspecifiers, list):
        rspecifiers = [rspecifiers]
    feat = np.concatenate(
        [
            load_feature_shard(rspecifier, in_filetype, percent, label_dict=label_dict)
            for rspecifier in rspecifiers
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    rspecifier,
    in_filetype,
    km_path,
    n_clusters,
    RVQ_layers,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
    label_rspecifier=None,
    label_filetype="mat",
    kmeans_method="base",
    anchor_center_path=None,
    lambda_anchor=1.0,
    anchor_mode="cluster",
    one_to_one=True,
):
    np.random.seed(seed)


    if kmeans_method == "phone-based":
        if label_rspecifier is None:
            raise ValueError("label_rspecifier is required for phone-based kmeans.")
        label_dict = load_label_dict(label_rspecifier, label_filetype)
    else:
        if label_rspecifier is not None:
            raise ValueError("label_rspecifier is only supported for phone-based kmeans.")
        label_dict = None
    feat = load_feature(rspecifier, in_filetype, percent, label_dict=label_dict)
    
    if kmeans_method == "base":
        for i in range(RVQ_layers):
            km_model = get_km_model(
                n_clusters,
                init,
                max_iter,
                batch_size,
                tol,
                max_no_improvement,
                n_init,
                reassignment_ratio,
                seed,
            )
            km_model.fit(feat)
            km_path_ = (
                km_path.replace(".mdl", f"_RVQ_{i}.mdl") if RVQ_layers > 1 else km_path
            )
            joblib.dump(km_model, km_path_)
            _save_centers(km_path_, km_model.cluster_centers_)

            inertia = -km_model.score(feat) / len(feat)
            logger.info(
                "{}total intertia: %.5f".format(f"RVQ_{i} " if RVQ_layers > 1 else ""),
                inertia,
            )
            c = km_model.predict(feat)
            r = km_model.cluster_centers_[c]
            feat = feat - km_model.cluster_centers_[km_model.predict(feat)]
    elif kmeans_method == "phone-based":
        for i in range(RVQ_layers):
            km_model = BatchedPhoneBasedKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                lambda_phone=lambda_anchor,
                default_phone=None,
                verbose=True,
                assign_batch_size=batch_size,
            )
            km_model.fit(feat)
            km_path_ = (
                km_path.replace(".mdl", f"_RVQ_{i}.mdl") if RVQ_layers > 1 else km_path
            )
            joblib.dump(km_model, km_path_)
            _save_centers(km_path_, km_model.cluster_centers_)

            inertia = -km_model.score(feat) / len(feat)
            logger.info("total intertia: %.5f", inertia)
    elif kmeans_method == "anchore-based":
        if anchor_center_path is None:
            raise ValueError("anchor_center_path is required for anchore-based kmeans.")
        anchor_centers = load_anchor_centers(anchor_center_path)
        for i in range(RVQ_layers):
            km_model = BatchedAnchoredKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                lambda_phone=lambda_anchor,
                default_phone=None,
                verbose=True,
                anchor_centers=anchor_centers,
                anchor_mode=anchor_mode,
                one_to_one=one_to_one,
                assign_batch_size=batch_size,
            )
            km_model.fit(feat)
            km_path_ = (
                km_path.replace(".mdl", f"_RVQ_{i}.mdl") if RVQ_layers > 1 else km_path
            )
            joblib.dump(km_model, km_path_)
            _save_centers(km_path_, km_model.cluster_centers_)

            inertia = -km_model.score(feat) / len(feat)
            logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
