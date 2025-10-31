import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


try:
    import pymap3d as pm  # type: ignore
except Exception as e:
    raise ImportError(
        "gps_helpers_pymap3d requires the 'pymap3d' package. Install it via 'pip install pymap3d'."
    ) from e


@dataclass
class GpsSample:
    t_ns: int
    t_s: float
    lat: float
    lon: float
    alt: float

def read_gps_csv(csv_path: str) -> List[GpsSample]:
    rows: List[GpsSample] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t_ns = int(r.get("timestamp_ns") or 0)
            t_s = float(r.get("timestamp_s") or 0.0)
            lat = float(r["latitude"])  # raises if missing
            lon = float(r["longitude"])  # raises if missing
            alt = float(r["altitude"])  # meters
            rows.append(GpsSample(t_ns=t_ns, t_s=t_s, lat=lat, lon=lon, alt=alt))
    if not rows:
        raise ValueError(f"No rows parsed from {csv_path}")
    return rows

def extract_ts_ns(path: str) -> Optional[int]:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    m = re.match(r"^(\d{12,})$", stem) #be careful here. 
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def match_images_to_gps(
    image_paths: List[str],
    gps: List[GpsSample],
    tolerance_ns: int,
) -> Dict[str, GpsSample]:
    ts_list = np.array([g.t_ns for g in gps], dtype=np.int64)
    order = np.argsort(ts_list)
    ts_sorted = ts_list[order]
    gps_sorted = [gps[i] for i in order]

    def nearest(ts: int) -> Optional[GpsSample]:
        idx = int(np.searchsorted(ts_sorted, ts))
        candidates = []
        if 0 <= idx < len(gps_sorted):
            candidates.append(gps_sorted[idx])
        if idx - 1 >= 0:
            candidates.append(gps_sorted[idx - 1])
        if idx + 1 < len(gps_sorted):
            candidates.append(gps_sorted[idx + 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda g: abs(g.t_ns - ts))
        if abs(best.t_ns - ts) <= tolerance_ns:
            return best
        return None

    matches: Dict[str, GpsSample] = {}
    for img in image_paths:
        ts = extract_ts_ns(img)
        if ts is None:
            continue
        g = nearest(ts)
        if g is not None:
            matches[img] = g
    return matches

def attach_translation_poses_from_gps(
        views: List[Dict],
        image_paths: List[str],
        gps_csv_path: str,
        tolerance_ms: float = 100.0
) -> Tuple[int, int]:
    gps_rows = read_gps_csv(gps_csv_path)
    origin = gps_rows[0]
    tol_ns = int(tolerance_ms * 1e6)

    matches = match_images_to_gps(image_paths, gps_rows, tolerance_ns=tol_ns)

    # Find earliest matched view index and drop all views prior to it to satisfy
    # the model constraint that the first view must have a pose if any do.
    matched_view_indices = [int(v.get("idx", 0)) for v in views if image_paths[int(v.get("idx", 0))] in matches]
    if matched_view_indices:
        first_match_idx = min(matched_view_indices)
        # Drop all views with idx < first_match_idx (in-place)
        views[:] = [v for v in views if int(v.get("idx", 0)) >= first_match_idx]

    matched_count = 0
    for v in views:
        idx = int(v.get("idx", 0))
        path = image_paths[idx]
        gps = matches.get(path, None)

        if gps is None:
            # Leave unmatched views without camera_poses (RGB-only)
            continue

        # pymap3d ENU
        e, n, u = pm.geodetic2enu(gps.lat, gps.lon, gps.alt, origin.lat, origin.lon, origin.alt)
        # Map ENU -> OpenCV RDF world: [E, -U, N]
        trans_rdf = torch.tensor([float(e), float(-u), float(n)], dtype=torch.float32)
        # v["camera_poses"] = (quat.clone()[None], trans_rdf.clone()[None])
        # Provide translation-only priors; rotation should be recovered.
        v["camera_poses"] = trans_rdf.clone()[None]
        v["is_metric_scale"] = torch.tensor([True])
        matched_count += 1

    return matched_count, len(views)