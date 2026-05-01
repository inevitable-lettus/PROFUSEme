"""
verify_s3_paths.py
------------------
Run this BEFORE the main notebook to confirm what data is available in the
CHIMERA S3 bucket and learn the exact file paths and JSON schema.

Usage:
    python3 verify_s3_paths.py > verify_results.txt
"""

import io
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "chimera-challenge"
TEST_PATIENT = "1003"

def make_client():
    for region in ["us-west-2", "us-east-1", "eu-west-1"]:
        try:
            c = boto3.client("s3", region_name=region, config=Config(signature_version=UNSIGNED))
            c.head_bucket(Bucket=BUCKET)
            print(f"[S3] Connected via region: {region}")
            return c
        except Exception:
            continue
    raise RuntimeError("Could not connect to S3 bucket in any region")


def list_prefix(client, prefix, max_keys=50):
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= max_keys:
                return keys
    return keys


def check_key_exists(client, key):
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def download_bytes(client, key):
    buf = io.BytesIO()
    client.download_fileobj(BUCKET, key, buf)
    buf.seek(0)
    return buf.read()


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    client = make_client()

    # ── 1. Pathology features directory ──────────────────────────────
    section("PATHOLOGY FEATURES — all keys under v2/task1/pathology/features/")
    path_keys = list_prefix(client, "v2/task1/pathology/features/", max_keys=60)
    for k in path_keys:
        print(f"  {k}")
    print(f"\n  Total shown: {len(path_keys)}")

    # Group by subdirectory
    subdirs = {}
    for k in path_keys:
        parts = k.split("/")
        if len(parts) >= 6:
            subdir = parts[4]  # e.g. "coordinates" or "features"
            subdirs.setdefault(subdir, []).append(k)
    print("\n  Subdirectories found:")
    for sd, ks in subdirs.items():
        print(f"    {sd}/  ({len(ks)} files shown, e.g. {ks[0].split('/')[-1]})")

    # ── 2. Raw WSI images ─────────────────────────────────────────────
    section("RAW WSI IMAGES — checking v2/task1/pathology/images/")
    wsi_keys = list_prefix(client, "v2/task1/pathology/images/", max_keys=10)
    if wsi_keys:
        print("  RAW WSI FILES FOUND:")
        for k in wsi_keys:
            print(f"    {k}")
    else:
        print("  No raw WSI files found at v2/task1/pathology/images/")
        # Try direct path
        direct_key = f"v2/task1/pathology/images/{TEST_PATIENT}/{TEST_PATIENT}_1.tif"
        exists = check_key_exists(client, direct_key)
        print(f"  Direct check ({direct_key}): {'EXISTS' if exists else 'NOT FOUND'}")

    # ── 3. Radiology features ─────────────────────────────────────────
    section("RADIOLOGY FEATURES — listing all under v2/task1/radiology/features/")
    rad_keys = list_prefix(client, "v2/task1/radiology/features/", max_keys=50)
    print(f"  Total shown: {len(rad_keys)}")
    # Group by patient
    patients_rad = {}
    for k in rad_keys:
        fname = k.split("/")[-1]
        pid = fname.split("_")[0]
        patients_rad.setdefault(pid, []).append(fname)
    print("\n  Files per patient (sample):")
    for pid, fnames in sorted(patients_rad.items())[:5]:
        print(f"    {pid}: {fnames}")

    # ── 4. Radiology feature tensor shape ────────────────────────────
    section(f"RADIOLOGY TENSOR SHAPE — loading {TEST_PATIENT}_0001_features.pt")
    rad_key = f"v2/task1/radiology/features/{TEST_PATIENT}_0001_features.pt"
    try:
        data = download_bytes(client, rad_key)
        import torch
        tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if hasattr(tensor, "shape"):
            print(f"  Type: torch.Tensor | Shape: {tensor.shape} | Dtype: {tensor.dtype}")
        elif isinstance(tensor, dict):
            print(f"  Type: dict | Keys: {list(tensor.keys())}")
            for k, v in tensor.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape}")
        else:
            print(f"  Type: {type(tensor)}")
    except ImportError:
        print("  [!] torch not installed — run: pip install torch")
        print("  Saving raw bytes to check manually...")
        with open(f"/tmp/{TEST_PATIENT}_0001_features.pt", "wb") as f:
            f.write(data)
        print(f"  Saved to /tmp/{TEST_PATIENT}_0001_features.pt")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 5. Clinical JSON schema ───────────────────────────────────────
    section(f"CLINICAL JSON SCHEMA — {TEST_PATIENT}.json")
    clin_key = f"v2/task1/clinical_data/{TEST_PATIENT}.json"
    try:
        data = download_bytes(client, clin_key)
        record = json.loads(data)
        print(f"  All fields:")
        for k, v in record.items():
            print(f"    {k!r:40s} = {v!r}")

        # Guess BCR label field
        bcr_candidates = [k for k in record if any(
            kw in k.lower() for kw in ["bcr", "recur", "outcome", "event", "label", "psa"]
        )]
        if bcr_candidates:
            print(f"\n  Likely BCR / outcome fields: {bcr_candidates}")
        else:
            print("\n  No obvious BCR field found — review all fields above")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 6. Pathology coordinate .npy shape ───────────────────────────
    section(f"PATHOLOGY COORDINATE .npy — {TEST_PATIENT}_1.npy")
    coord_key = f"v2/task1/pathology/features/coordinates/{TEST_PATIENT}_1.npy"
    try:
        import numpy as np
        data = download_bytes(client, coord_key)
        arr = np.load(io.BytesIO(data))
        print(f"  Shape: {arr.shape} | Dtype: {arr.dtype}")
        print(f"  First 5 rows: {arr[:5]}")
        print(f"  Value range: min={arr.min()}, max={arr.max()}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 7. Summary ────────────────────────────────────────────────────
    section("SUMMARY — What to expect in profuseme_utils.py")
    print("  Review the output above to confirm:")
    print("  1. Pathology feature tensor paths (coordinates/ vs features/ subdirs)")
    print("  2. Whether raw WSI exists (determines PathologyLoader mode)")
    print("  3. Radiology tensor shape (sets D_rad for fusion)")
    print("  4. Clinical JSON field names for BCR label")
    print()
    print("  Once confirmed, run the Jupyter notebook:")
    print("  $ jupyter notebook PROFUSEme_Educational_Pipeline.ipynb")


if __name__ == "__main__":
    main()
