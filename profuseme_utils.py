"""
profuseme_utils.py
------------------
Backend module for the PROFUSEme educational notebook.
All S3 access, feature loading, fusion, and prediction logic lives here
so the notebook stays readable and focused on explanations.
"""

import io
import json
import logging
import warnings
import numpy as np
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
log = logging.getLogger("profuseme")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────

BUCKET = "chimera-challenge"
PATIENTS = [
    "1003", "1010", "1011", "1021", "1025", "1026", "1028",
    "1030", "1031", "1035", "1036", "1037", "1039", "1041", "1048",
]

S3_CLINICAL  = "v2/task1/clinical_data"
S3_PATH_FEAT = "v2/task1/pathology/features/features"   # .pt tensors (if present)
S3_PATH_COORD= "v2/task1/pathology/features/coordinates"  # .npy coordinates
S3_RAD_FEAT  = "v2/task1/radiology/features"             # .pt tensors
S3_WSI       = "v2/task1/pathology/images"               # raw .tif (if present)

# Clinical fields — keys are updated from verify_s3_paths.py output if needed
CLINICAL_FIELD_MAP = {
    # JSON key in CHIMERA          → internal name
    "age":                           "age",
    "isup_grade":                    "isup_grade",
    "pt_stage":                      "pt_stage",
    "lymph_node_involvement":        "lymph_nodes",
    "capsular_penetration":          "capsular_penetration",
    "surgical_margins":              "surgical_margins",
    "seminal_vesicle_invasion":      "seminal_vesicle_invasion",
    "lymphovascular_invasion":       "lymphovascular_invasion",
}
# Possible names for the BCR outcome label in the JSON
BCR_FIELD_CANDIDATES = ["BCR", "bcr", "biochemical_recurrence", "recurrence", "outcome"]
TIME_FIELD_CANDIDATES = ["time_to_bcr", "time_bcr", "time_to_recurrence", "survival_time"]


# ── S3 Client ─────────────────────────────────────────────────────────────────

class S3Client:
    """Thin wrapper around boto3 with anonymous access."""

    def __init__(self):
        self._client = None
        for region in ["us-west-2", "us-east-1", "eu-west-1"]:
            try:
                c = boto3.client(
                    "s3", region_name=region,
                    config=Config(signature_version=UNSIGNED)
                )
                c.head_bucket(Bucket=BUCKET)
                self._client = c
                log.info(f"Connected to S3 via {region}")
                break
            except Exception:
                continue
        if self._client is None:
            raise RuntimeError("Cannot connect to S3 bucket 'chimera-challenge'")

    def download_bytes(self, key: str) -> bytes:
        buf = io.BytesIO()
        self._client.download_fileobj(BUCKET, key, buf)
        buf.seek(0)
        return buf.read()

    def list_prefix(self, prefix: str) -> List[str]:
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def key_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=BUCKET, Key=key)
            return True
        except Exception:
            return False


# ── Clinical Loader ───────────────────────────────────────────────────────────

class ClinicalLoader:
    """Loads and encodes clinical JSON records from S3."""

    def load_patient(self, patient_id: str, s3: S3Client) -> dict:
        key = f"{S3_CLINICAL}/{patient_id}.json"
        data = s3.download_bytes(key)
        return json.loads(data)

    def extract_label(self, raw: dict) -> Optional[int]:
        """Return BCR label (0/1) from a clinical record, or None if not found."""
        for field in BCR_FIELD_CANDIDATES:
            if field in raw:
                val = raw[field]
                if isinstance(val, bool):
                    return int(val)
                if isinstance(val, (int, float)):
                    return int(bool(val))
                if isinstance(val, str):
                    try:
                        return int(bool(float(val)))
                    except ValueError:
                        return 1 if val.lower() in ("yes", "true", "positive") else 0
        # Fall back: if there is a time-to-event field, BCR=1 if time < 60 months
        for field in TIME_FIELD_CANDIDATES:
            if field in raw:
                t = raw[field]
                if isinstance(t, (int, float)) and t > 0:
                    for ef in BCR_FIELD_CANDIDATES:
                        if ef in raw:
                            event = int(bool(raw[ef]))
                            return event
        return None

    def encode_features(self, raw: dict) -> np.ndarray:
        """
        Encode one patient's clinical record into an 8-dimensional feature vector.
        Missing values are filled with 0 (binary) or -1 (continuous) and logged.
        """
        vec = np.zeros(8, dtype=np.float32)

        # 0: age — normalize: (age - 60) / 15
        age_val = raw.get("age_at_prostatectomy", raw.get("age", raw.get("Age", None)))
        if age_val is not None:
            vec[0] = (float(age_val) - 60.0) / 15.0
        else:
            log.warning("Missing 'age' — imputing 0")

        # 1: ISUP/Gleason grade (ordinal 1–5)
        for key in ("ISUP", "isup_grade", "isup", "gleason_grade", "ISUP_grade"):
            if key in raw:
                vec[1] = float(raw[key]) / 5.0
                break
        else:
            log.warning("Missing ISUP grade — imputing 0")

        # 2: pT stage (pT2=0, pT3a=0.33, pT3b=0.67, pT4=1.0)
        stage_map = {"pt2": 0.0, "pt3a": 0.33, "pt3b": 0.67, "pt4": 1.0}
        for key in ("pT_stage", "pt_stage", "pathological_stage", "T_stage"):
            if key in raw:
                s = str(raw[key]).lower().replace(" ", "").replace("_", "")
                if not s.startswith("pt"):
                    s = "pt" + s
                vec[2] = stage_map.get(s, 0.0)
                break

        # 3–7: binary features
        binary_fields = [
            ("lymph_nodes",           ["positive_lymph_nodes", "lymph_node_involvement", "lymph_nodes", "N_stage"]),
            ("capsular_penetration",  ["capsular_penetration", "extracapsular_extension"]),
            ("surgical_margins",      ["positive_surgical_margins", "surgical_margins", "positive_margins"]),
            ("seminal_vesicle",       ["invasion_seminal_vesicles", "seminal_vesicle_invasion", "SVI"]),
            ("lymphovascular",        ["lymphovascular_invasion", "LVI"]),
        ]
        for i, (name, candidates) in enumerate(binary_fields):
            for key in candidates:
                if key in raw:
                    v = raw[key]
                    if isinstance(v, bool):
                        vec[3 + i] = float(v)
                    elif isinstance(v, (int, float)):
                        vec[3 + i] = float(bool(v))
                    elif isinstance(v, str):
                        vec[3 + i] = 1.0 if v.lower() in ("yes", "true", "1", "positive") else 0.0
                    break

        return vec

    def load_all(self, patients: List[str], s3: S3Client):
        """
        Returns:
            X: np.ndarray of shape (N, 8) — clinical feature matrix
            y: np.ndarray of shape (N,)   — BCR labels (0/1)
            patient_ids: list of patient IDs with valid labels
        """
        X_list, y_list, valid_ids = [], [], []
        for pid in patients:
            try:
                raw = self.load_patient(pid, s3)
                label = self.extract_label(raw)
                if label is None:
                    log.warning(f"Patient {pid}: no BCR label found — skipping")
                    continue
                feat = self.encode_features(raw)
                X_list.append(feat)
                y_list.append(label)
                valid_ids.append(pid)
            except Exception as e:
                log.warning(f"Patient {pid}: failed to load clinical data ({e})")
        X = np.stack(X_list)
        y = np.array(y_list, dtype=int)
        return X, y, valid_ids

    def get_feature_names(self) -> List[str]:
        return [
            "Age (normalized)",
            "ISUP Grade (normalized)",
            "pT Stage",
            "Lymph Node Involvement",
            "Capsular Penetration",
            "Surgical Margins",
            "Seminal Vesicle Invasion",
            "Lymphovascular Invasion",
        ]


# ── Pathology Loader ──────────────────────────────────────────────────────────

class PathologyLoader:
    """
    Loads pre-computed pathology features from S3.
    For each patient, loads the patch-level feature tensor and mean-pools
    across all patches to get one patient-level vector.
    """

    def __init__(self, s3: S3Client):
        self.s3 = s3
        self._feature_keys = None

    def _discover_feature_keys(self) -> List[str]:
        if self._feature_keys is None:
            self._feature_keys = self.s3.list_prefix(S3_PATH_FEAT + "/")
            if not self._feature_keys:
                # fallback: check if features live in a different subdir
                all_path = self.s3.list_prefix("v2/task1/pathology/features/")
                self._feature_keys = [
                    k for k in all_path
                    if k.endswith(".pt") and "coordinates" not in k
                ]
        return self._feature_keys

    def _load_tensor(self, key: str) -> np.ndarray:
        import torch
        data = self.s3.download_bytes(key)
        tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(tensor, dict):
            # some .pt files are dicts with a 'features' key
            for k in ("features", "embeddings", "x"):
                if k in tensor:
                    tensor = tensor[k]
                    break
        tensor = tensor.float()
        if tensor.dim() == 1:
            return tensor.numpy()
        return tensor.mean(dim=0).numpy()   # mean-pool over patches → (D,)

    def load_patient(self, patient_id: str) -> Optional[np.ndarray]:
        feat_keys = self._discover_feature_keys()
        patient_keys = [k for k in feat_keys if f"/{patient_id}_" in k or k.endswith(f"/{patient_id}.pt")]

        if not patient_keys:
            log.warning(f"Patient {patient_id}: no pathology feature file found")
            return None

        slide_vecs = []
        for key in sorted(patient_keys):
            try:
                vec = self._load_tensor(key)
                slide_vecs.append(vec)
            except Exception as e:
                log.warning(f"  Failed to load {key}: {e}")

        if not slide_vecs:
            return None
        return np.mean(slide_vecs, axis=0)  # average over slides

    def load_coordinates(self, patient_id: str, slide_num: int = 1) -> Optional[np.ndarray]:
        """Load patch coordinate .npy for visualization."""
        key = f"{S3_PATH_COORD}/{patient_id}_{slide_num}.npy"
        try:
            data = self.s3.download_bytes(key)
            return np.load(io.BytesIO(data))
        except Exception as e:
            log.warning(f"Could not load coordinates for {patient_id}: {e}")
            return None

    def load_all(self, patients: List[str]) -> Tuple[Optional[np.ndarray], List[str]]:
        vecs, valid_ids = [], []
        for pid in patients:
            v = self.load_patient(pid)
            if v is not None:
                vecs.append(v)
                valid_ids.append(pid)
        if not vecs:
            log.warning("No pathology feature .pt files found on S3 — pathology modality will be skipped.")
            return None, []
        return np.stack(vecs), valid_ids


# ── Radiology Loader ──────────────────────────────────────────────────────────

class RadiologyLoader:
    """Loads pre-computed radiology feature tensors from S3."""

    def __init__(self, s3: S3Client):
        self.s3 = s3

    def _load_tensor(self, key: str) -> np.ndarray:
        import torch
        data = self.s3.download_bytes(key)
        tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(tensor, dict):
            for k in ("features", "embeddings", "x"):
                if k in tensor:
                    tensor = tensor[k]
                    break
        tensor = tensor.float()
        if tensor.dim() == 1:
            return tensor.numpy()
        return tensor.mean(dim=0).numpy()   # mean-pool over slices/sequences

    def load_patient(self, patient_id: str) -> Optional[np.ndarray]:
        # Try _0001, _0002, ... up to _0005
        vecs = []
        for seq in range(1, 6):
            key = f"{S3_RAD_FEAT}/{patient_id}_{seq:04d}_features.pt"
            if self.s3.key_exists(key):
                try:
                    vecs.append(self._load_tensor(key))
                except Exception as e:
                    log.warning(f"  {key}: {e}")

        if not vecs:
            log.warning(f"Patient {patient_id}: no radiology features found")
            return None
        return np.mean(vecs, axis=0)

    def load_all(self, patients: List[str]) -> Tuple[np.ndarray, List[str]]:
        vecs, valid_ids = [], []
        for pid in patients:
            v = self.load_patient(pid)
            if v is not None:
                vecs.append(v)
                valid_ids.append(pid)
        if not vecs:
            raise RuntimeError("No radiology features could be loaded.")
        return np.stack(vecs), valid_ids


# ── Multi-Modal Fusion ────────────────────────────────────────────────────────

class MultiModalFusion:
    """
    Fuses clinical, pathology, and radiology feature vectors.
    Uses PCA per modality to reduce dimensionality before concatenation.
    PCA is always fit inside the training fold to avoid data leakage.
    """

    def __init__(self, path_components: int = 8, rad_components: int = 8):
        self.path_components = path_components
        self.rad_components = rad_components

    def fuse(
        self,
        X_clin: np.ndarray,
        X_path: Optional[np.ndarray],
        X_rad: Optional[np.ndarray],
        train_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse modalities with PCA fit only on train_mask rows.
        Returns the full fused matrix (train + test), scaled consistently.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        parts = [X_clin]

        if X_path is not None:
            n_comp = min(self.path_components, X_path[train_mask].shape[0] - 1, X_path.shape[1])
            pca = PCA(n_components=n_comp)
            scaler = StandardScaler()
            train_p = scaler.fit_transform(X_path[train_mask])
            pca.fit(train_p)
            parts.append(pca.transform(scaler.transform(X_path)))

        if X_rad is not None:
            n_comp = min(self.rad_components, X_rad[train_mask].shape[0] - 1, X_rad.shape[1])
            pca = PCA(n_components=n_comp)
            scaler = StandardScaler()
            train_r = scaler.fit_transform(X_rad[train_mask])
            pca.fit(train_r)
            parts.append(pca.transform(scaler.transform(X_rad)))

        return np.hstack(parts)

    def fuse_for_visualization(
        self,
        X_clin: np.ndarray,
        X_path: Optional[np.ndarray],
        X_rad: Optional[np.ndarray],
    ) -> np.ndarray:
        """Fuse using ALL data (for visualization only — not for model training)."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        train_mask = np.ones(len(X_clin), dtype=bool)
        return self.fuse(X_clin, X_path, X_rad, train_mask)


# ── BCR Predictor ─────────────────────────────────────────────────────────────

class BCRPredictor:
    """
    Trains and evaluates a BCR prediction model using leave-one-out cross-validation.
    Uses logistic regression — appropriate for very small datasets (N=15).
    """

    def __init__(self, C: float = 0.01):
        self.C = C

    def evaluate_loo(
        self,
        X_clin: np.ndarray,
        y: np.ndarray,
        X_path: Optional[np.ndarray] = None,
        X_rad: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Leave-one-out cross-validation with StandardScaler inside each fold.
        Returns dict with predictions, AUC, and bootstrap CI.
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return {"error": "Only one class in labels — cannot compute AUC"}

        fusion = MultiModalFusion()
        loo = LeaveOneOut()
        all_proba = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X_clin):
            train_mask = np.zeros(len(y), dtype=bool)
            train_mask[train_idx] = True

            X_fused = fusion.fuse(X_clin, X_path, X_rad, train_mask)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_fused[train_idx])
            X_te = scaler.transform(X_fused[test_idx])

            clf = LogisticRegression(C=self.C, max_iter=2000, random_state=42, solver="lbfgs")
            clf.fit(X_tr, y[train_idx])
            all_proba[test_idx] = clf.predict_proba(X_te)[0, 1]

        auc = roc_auc_score(y, all_proba)
        ci_low, ci_high = self._bootstrap_ci(y, all_proba)

        preds_binary = (all_proba >= 0.5).astype(int)
        tp = np.sum((preds_binary == 1) & (y == 1))
        fn = np.sum((preds_binary == 0) & (y == 1))
        tn = np.sum((preds_binary == 0) & (y == 0))
        fp = np.sum((preds_binary == 1) & (y == 0))

        return {
            "auc": auc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "accuracy": (tp + tn) / len(y),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "predicted_proba": all_proba.tolist(),
            "true_labels": y.tolist(),
        }

    def _bootstrap_ci(self, y, proba, n_boot=1000, alpha=0.05) -> Tuple[float, float]:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(42)
        aucs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[idx], proba[idx]))
        if not aucs:
            return (0.0, 0.0)
        return (np.percentile(aucs, 100 * alpha / 2), np.percentile(aucs, 100 * (1 - alpha / 2)))

    def ablation_study(
        self,
        X_clin: np.ndarray,
        y: np.ndarray,
        X_path: Optional[np.ndarray] = None,
        X_rad: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Run LOO-CV for all modality combinations and return an AUC table."""
        combos = []
        if X_path is not None and X_rad is not None:
            combos = [
                ("Clinical only",           X_clin, None,   None),
                ("Pathology only",          None,   X_path, None),
                ("Radiology only",          None,   None,   X_rad),
                ("Clinical + Pathology",    X_clin, X_path, None),
                ("Clinical + Radiology",    X_clin, None,   X_rad),
                ("Pathology + Radiology",   None,   X_path, X_rad),
                ("All three (PROFUSEme)",   X_clin, X_path, X_rad),
            ]
        elif X_path is not None:
            combos = [
                ("Clinical only",         X_clin, None,   None),
                ("Pathology only",        None,   X_path, None),
                ("Clinical + Pathology",  X_clin, X_path, None),
            ]
        elif X_rad is not None:
            combos = [
                ("Clinical only",        X_clin, None,  None),
                ("Radiology only",       None,   None,  X_rad),
                ("Clinical + Radiology", X_clin, None,  X_rad),
            ]
        else:
            combos = [("Clinical only", X_clin, None, None)]

        rows = []
        for name, xc, xp, xr in combos:
            # Need at least clinical for the feature vector to be non-empty
            effective_clin = xc if xc is not None else np.zeros((len(y), 1))
            result = self.evaluate_loo(effective_clin, y, xp, xr)
            if "error" in result:
                rows.append({"Modality Combination": name, "AUC": float("nan"), "95% CI": "N/A"})
            else:
                ci = f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]"
                rows.append({
                    "Modality Combination": name,
                    "AUC": round(result["auc"], 3),
                    "Sensitivity": round(result["sensitivity"], 3),
                    "Specificity": round(result["specificity"], 3),
                    "95% CI": ci,
                })
        return pd.DataFrame(rows)



# ── Convenience helpers for notebook visualizations ───────────────────────────

def align_patients(
    clinical_ids: List[str],
    y: np.ndarray,
    X_clin: np.ndarray,
    path_ids: Optional[List[str]] = None,
    X_path: Optional[np.ndarray] = None,
    rad_ids: Optional[List[str]] = None,
    X_rad: Optional[np.ndarray] = None,
) -> dict:
    """
    Align all modalities to a common set of patients.
    Returns dict with aligned arrays and the intersection patient list.
    """
    common = set(clinical_ids)
    if path_ids:
        common &= set(path_ids)
    if rad_ids:
        common &= set(rad_ids)
    common = sorted(common)

    clin_map = {pid: i for i, pid in enumerate(clinical_ids)}
    path_map = {pid: i for i, pid in enumerate(path_ids)} if path_ids else {}
    rad_map  = {pid: i for i, pid in enumerate(rad_ids)}  if rad_ids  else {}

    idx_clin = [clin_map[p] for p in common]
    result = {
        "patients": common,
        "y": y[idx_clin],
        "X_clin": X_clin[idx_clin],
        "X_path": X_path[[path_map[p] for p in common]] if X_path is not None and path_map else None,
        "X_rad":  X_rad[[rad_map[p]  for p in common]] if X_rad  is not None and rad_map  else None,
    }
    return result
