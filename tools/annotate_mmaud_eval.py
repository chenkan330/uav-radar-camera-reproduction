"""Image-only, manually seeded target annotation for the fixed short clip.

This is an assisted visual annotation condition, not the paper's detector.
No radar positions, Leica files, or 3D ground truth are read by this tool.
Original frames remain unchanged. Local crops support per-frame visual QA.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmaud_adapter import relative_time

DATA = ROOT / "data/public/mmaud"


def annotate():
    declarations = json.loads((DATA / "v16_selected_image_ranges.json").read_text())
    images = sorted([row["name"] for row in declarations if row["group"] == "evaluation"])
    if len(images) != 48:
        raise ValueError("Expected the fixed 48 actually selected evaluation images")
    # Seed from direct inspection of the first original left-camera image.
    # It is a pixel observation, not a projection of radar or 3D truth.
    previous = np.array([621., 496.])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    records = []
    thumbnails = []
    for index, name in enumerate(images):
        path = DATA / name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (960, 2560):
            raise ValueError("Expected the actual 2560x960 stereo-concatenated PNG")
        gray = cv2.cvtColor(image[:, :1280], cv2.COLOR_BGR2GRAY)
        residual = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        px, py = np.rint(previous).astype(int)
        x0, x1 = max(0, px - 40), min(1280, px + 41)
        y0, y1 = max(0, py - 40), min(960, py + 41)
        patch = residual[y0:y1, x0:x1]
        mask = ((patch >= 18) & (gray[y0:y1, x0:x1] < 135)).astype(np.uint8)
        # Connect contiguous body/arm pixels; do not merge entire cloud edges.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            center = centers[label] + [x0, y0]
            distance = np.linalg.norm(center - previous)
            if 6 <= area <= 450 and width >= 3 and height >= 3 and distance <= 30:
                candidates.append((distance, -area, label, center, [x+x0, y+y0, x+x0+width, y+y0+height]))
        if not candidates:
            raise ValueError(f"No reliable image component in {name}; manual image review required")
        # Body is usually the largest local dark component. Choosing the
        # nearest arm alone can shift the observation as the airframe rotates.
        candidates.sort(key=lambda value: (value[1], value[0]))
        _, _, label, center, bbox = candidates[0]
        previous = center
        records.append({"timestamp_s": relative_time(path), "source_image": name,
                        "u": float(center[0]), "v": float(center[1]),
                        "xmin": int(bbox[0]), "ymin": int(bbox[1]), "xmax": int(bbox[2]), "ymax": int(bbox[3]),
                        "annotation_method": "manual_oracle",
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        cx, cy = np.rint(center).astype(int)
        crop = image[cy-35:cy+36, cx-35:cx+36].copy()
        crop = cv2.resize(crop, (142, 142), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(crop, (70, 70), (0, 220, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
        tile = np.full((172, 160, 3), 240, dtype=np.uint8)
        tile[25:167, 9:151] = crop
        cv2.putText(tile, f"{index+1:02d}  t={relative_time(path):.2f}s", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 0, 0), 1)
        thumbnails.append(tile)
    dest = DATA / "annotations_evaluation.csv"
    with dest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    montage = np.vstack([np.hstack(thumbnails[start:start+6]) for start in range(0, 48, 6)])
    cv2.imwrite(str(DATA / "evaluation_annotation_review.png"), montage)
    metadata = {"condition": "manual-assisted image-only annotation, not trained detector",
                "initial_pixel_seed": [621, 496], "camera_half": "left 1280x960",
                "frames": len(records), "radar_read": False, "leica_read": False,
                "algorithm": "local 25px blackhat, threshold18, largest dark component within30px of previous observed pixel",
                "visual_review_completed": False,
                "review_artifact": "evaluation_annotation_review.png"}
    (DATA / "evaluation_annotation_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(records)} image-only annotations; per-frame visual review is pending.")


if __name__ == "__main__":
    annotate()
