"""Save public MMAUD calibration sources on D: without running their contents.

The public GitHub snapshot contains documentation and CAD, not verified camera
or radar extrinsics. An optional publicly obtained Drive download URL can save
the fisheye ZIP. Authentication/permission pages are rejected, never bypassed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


UPSTREAM_COMMIT = "0213ad16c9a6489c6a99f474ac759cebf5cb0f6f"
PUBLIC_FOLDER = "https://drive.google.com/drive/folders/1wk-c5xVX6701WNI_In1ba3_D4LSjRYv5"
RAW_BASE = f"https://raw.githubusercontent.com/ntu-aris/MMAUD/{UPSTREAM_COMMIT}/"
FILES = ("README.md", "sensor_calibration.md", "sensors_and_usage.md",
         "evaluation_tutorial.md", "drawing.rar")
CAD_SHA256 = "182ac1daffc73862f08495b6939bb1c7973837ff440b9929b8aad5cc17e1b632"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(url: str, destination: Path, maximum_bytes: int, expected_kind: str) -> dict:
    """Bounded normal HTTPS download; preserve partial bytes for inspection."""
    parsed = urlparse(url)
    allowed = {"raw.githubusercontent.com", "drive.google.com",
               "drive.usercontent.google.com"}
    if parsed.scheme != "https" or parsed.hostname not in allowed:
        raise ValueError("Only ordinary HTTPS URLs at the declared public sources are allowed")
    record = {"url": url, "path": str(destination), "status": "failed"}
    if destination.name == "drawing.rar" and destination.exists() and digest(destination) == CAD_SHA256:
        record.update(status="cached_verified", bytes=destination.stat().st_size,
                      sha256=CAD_SHA256)
        return record
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "MMAUD-public-calibration-audit/1.6"})
        with urlopen(request, timeout=25) as response, partial.open("wb") as file:
            record["resolved_url"] = response.url
            record["content_type"] = response.headers.get("Content-Type", "")
            length = response.headers.get("Content-Length")
            if length and int(length) > maximum_bytes:
                raise ValueError("Source exceeds the declared download limit")
            size = 0
            first = b""
            while True:
                chunk = response.read(min(1024 * 1024, maximum_bytes + 1 - size))
                if not chunk:
                    break
                if not first:
                    first = chunk[:256]
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("Source exceeds the declared download limit")
                file.write(chunk)
        if expected_kind == "zip" and not zipfile.is_zipfile(partial):
            raise ValueError("Expected a ZIP; a permission, login or download page is not calibration data")
        if expected_kind == "rar" and not first.startswith(b"Rar!\x1a\x07"):
            raise ValueError("Expected a RAR CAD archive")
        if expected_kind == "text":
            partial.read_text(encoding="utf-8")
        if destination.name == "drawing.rar" and digest(partial) != CAD_SHA256:
            raise ValueError("CAD hash differs from the audited official snapshot")
        partial.replace(destination)
        record.update(status="downloaded", bytes=destination.stat().st_size,
                      sha256=digest(destination))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if partial.exists():
            record["partial_path"] = str(partial)
    return record


def inspect_zip(path: Path) -> list[dict]:
    """List archive metadata only. Do not extract or execute any member."""
    with zipfile.ZipFile(path) as archive:
        return [{"name": item.filename, "bytes": item.file_size,
                 "compressed_bytes": item.compress_size, "crc32": f"{item.CRC:08x}"}
                for item in archive.infolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description="获取 MMAUD 公开标定说明与 CAD，记录来源和哈希")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/public/mmaud/calibration")
    parser.add_argument("--fisheye-url", help="已从作者公开目录确认的 ZIP 下载 URL；不接受私有权限绕过")
    parser.add_argument("--inspect-local-zip", type=Path, help="只读检查已经正常取得的 ZIP")
    parser.add_argument("--zip-limit-mib", type=int, default=512)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.drive and root.drive.upper() != "D:":
        parser.error("按本机存放偏好，原始资料必须保存到 D 盘")
    if not 1 <= args.zip_limit_mib <= 2048:
        parser.error("ZIP 上限必须在 1 至 2048 MiB 之间")
    root.mkdir(parents=True, exist_ok=True)
    records = [fetch(RAW_BASE + name, root / name, 16 * 1024 * 1024,
                     "rar" if name.endswith(".rar") else "text") for name in FILES]
    if args.fisheye_url:
        records.append(fetch(args.fisheye_url, root / "fisheye_calibration.zip",
                             args.zip_limit_mib * 1024 * 1024, "zip"))
    zip_path = args.inspect_local_zip or root / "fisheye_calibration.zip"
    listing = inspect_zip(zip_path) if zip_path.exists() else None
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "official_folder": PUBLIC_FOLDER,
        "files": records,
        "fisheye_zip_status": "saved_and_listed" if listing is not None else "not_obtained",
        "zip_listing": listing,
        "calibration_verified": False,
        "note": "Successful download does not prove camera intrinsics, sensor extrinsics, reference point or clock alignment.",
    }
    manifest_path = root / "calibration_fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(record["status"], Path(record["path"]).name,
              record.get("bytes", record.get("error", "")))
    print("fisheye ZIP:", manifest["fisheye_zip_status"])
    print("来源与校验记录：", manifest_path)
    return 0 if all(record["status"] in {"downloaded", "cached_verified"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
