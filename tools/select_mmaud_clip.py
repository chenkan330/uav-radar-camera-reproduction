"""Regenerate the fixed data-only selection plan from the audited ZIP index."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmaud_adapter import CALIBRATION_INTERVALS_UNIX, EVALUATION_INTERVAL_UNIX


def select():
    data = ROOT / "data/public/mmaud"
    index = json.loads((data / "archive_index.json").read_text())
    images = [row for row in index if row["name"].startswith("Mavic3/image/") and row["name"].endswith(".png")]
    baseline = [row["name"] for row in index if row["name"].endswith(".npy") and
                row["name"].startswith(("Mavic3/ground_truth/", "Mavic3/radar_enhance_pcl/"))]
    requests = []
    for group, starts in (("calibration", [float(t) for start,end in CALIBRATION_INTERVALS_UNIX
                                         for t in range(int(start), int(end)+1)]),
                           ("evaluation", [EVALUATION_INTERVAL_UNIX[0] + i*.2 for i in range(48)])):
        for timestamp in starts:
            row = min(images, key=lambda x: abs(float(Path(x["name"]).stem)-timestamp))
            requests.append({"name": row["name"], "start": row["offset"],
                             "end": row["offset"]+row["span_bytes"]-1,
                             "bytes": row["span_bytes"], "group": group,
                             "requested_timestamp": timestamp})
    if len(requests) != 79 or len({row["name"] for row in requests}) != 79:
        raise ValueError("Fixed 31 calibration/48 evaluation image selection changed")
    plans = {"v16_selected_image_ranges.json": requests,
             "v16_selected_image_members.json": [row["name"] for row in requests],
             "v16_required_members.json": baseline+[row["name"] for row in requests]}
    for name, values in plans.items():
        (data/name).write_text(json.dumps(values, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Fixed independent calibration31/evaluation48 images; {len(baseline)} radar/truth arrays. No error metrics read.")


if __name__ == "__main__":
    select()
