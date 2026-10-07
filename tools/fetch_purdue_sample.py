"""Fetch one real UAV-to-UAV video from the official Purdue archive.

Only Clip_1.mov and its published annotations are fetched. No downloaded code
is executed. The full Videos.zip archive is never downloaded or extracted.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
from threading import Lock
import time
import urllib.parse
import urllib.request
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "purdue"
OUT = ROOT / "output" / "v2_data_check"
BASE = "https://engineering.purdue.edu/~bouman/UAV_Dataset/"
VIDEO_URL = BASE + "Videos.zip"
ANNOTATION_URL = BASE + "Video_Annotation-v2.zip"
LICENSE_URL = BASE + "pubs/LICENSE.txt"
VIDEO_MEMBER = "Videos/Clip_1.mov"
ANNOTATION_MEMBER = "Video_Annotation-v2/refined_gt/Clip_1_refined.txt"
VIDEO_ARCHIVE_BYTES = 2_035_637_654
TOTAL_DOWNLOAD_LIMIT = 12_000_000
downloaded_bytes = 0
download_lock = Lock()


def _fetch_once(url: str, limit: int, byte_range: tuple[int, int] | None = None) -> bytes:
    """Bound each body and the total transfer; reject foreign redirects."""
    global downloaded_bytes
    if url not in (VIDEO_URL, ANNOTATION_URL, LICENSE_URL):
        raise ValueError("Only the three fixed official URLs are permitted")
    if downloaded_bytes + limit > TOTAL_DOWNLOAD_LIMIT:
        raise ValueError("Requested transfer exceeds the 12 MB budget")
    headers = {"User-Agent": "UAV-reproduction-data-check/2.0"}
    if byte_range is not None:
        start, end = byte_range
        if not 0 <= start <= end < VIDEO_ARCHIVE_BYTES:
            raise ValueError("Invalid archive range")
        if end - start + 1 != limit:
            raise ValueError("Range length must equal the exact byte limit")
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    deadline = time.monotonic() + 180
    with urllib.request.urlopen(request, timeout=45) as response:
        actual = urllib.parse.urlsplit(response.url)
        if actual.scheme != "https" or actual.netloc != "engineering.purdue.edu":
            raise ValueError("Unexpected redirect outside the official host")
        if byte_range is not None:
            expected = f"bytes {start}-{end}/{VIDEO_ARCHIVE_BYTES}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server must return exact HTTP 206 Content-Range")
        elif response.status != 200:
            raise ValueError("Unexpected HTTP status")
        content_length = response.headers.get("Content-Length")
        if content_length is None or int(content_length) > limit:
            raise ValueError("Missing or excessive Content-Length")
        pieces = []
        received = 0
        while received < int(content_length):
            if time.monotonic() > deadline:
                raise TimeoutError("Download exceeded its bounded time budget")
            block = response.read(min(65_536, int(content_length) - received))
            if not block:
                raise ValueError("Truncated HTTP body")
            received += len(block)
            with download_lock:
                downloaded_bytes += len(block)
                if received > limit or downloaded_bytes > TOTAL_DOWNLOAD_LIMIT:
                    raise ValueError("Download size limit exceeded")
            pieces.append(block)
        payload = b"".join(pieces)
    if byte_range is not None and len(payload) != limit:
        raise ValueError("Range body length mismatch")
    return payload


def fetch(url: str, limit: int, byte_range: tuple[int, int] | None = None) -> bytes:
    for attempt in range(3):
        try:
            return _fetch_once(url, limit, byte_range)
        except OSError:
            if attempt == 2:
                raise
            # Bytes received during a failed body are still charged to the
            # global budget; invalid range/header responses are never retried.
            print(f"Transient network error; retry {attempt + 1}/2", flush=True)
    raise RuntimeError("Unreachable")


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")


def video_from_range() -> tuple[bytes, dict]:
    tail = fetch(VIDEO_URL, 65_536, (VIDEO_ARCHIVE_BYTES - 65_536, VIDEO_ARCHIVE_BYTES - 1))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError("Missing ZIP end-of-central-directory")
    end = struct.unpack_from("<4s4H2IH", tail, eocd)
    if end[1] != 0 or end[2] != 0 or end[3] != end[4]:
        raise ValueError("Multi-disk ZIP archives are unsupported")
    central_bytes, central_offset, comment_bytes = end[5:8]
    if eocd + 22 + comment_bytes != len(tail):
        raise ValueError("Unexpected trailing bytes in ZIP archive")
    position = central_offset - (VIDEO_ARCHIVE_BYTES - len(tail))
    if position < 0 or position + central_bytes != eocd:
        raise ValueError("Central directory does not fit the bounded ZIP tail")
    entries = []
    for _ in range(end[4]):
        if position + 46 > eocd:
            raise ValueError("Truncated ZIP central directory")
        row = struct.unpack_from("<4s6H3I5H2I", tail, position)
        if row[0] != b"PK\x01\x02":
            raise ValueError("Invalid central-directory signature")
        name_size, extra_size, note_size = row[10:13]
        name = tail[position + 46:position + 46 + name_size].decode("utf-8")
        safe_name(name)
        entries.append((name, row))
        position += 46 + name_size + extra_size + note_size
    if position != eocd:
        raise ValueError("Central-directory size mismatch")
    matches = [row for name, row in entries if name == VIDEO_MEMBER]
    if len(matches) != 1:
        raise ValueError("The fixed Clip_1 video must appear exactly once")
    row = matches[0]
    flags, method, expected_crc = row[3], row[4], row[7]
    compressed_size, original_size, offset = row[8], row[9], row[16]
    if flags & 1 or method not in (0, 8):
        raise ValueError("Encrypted or unsupported compression")
    if not 0 < compressed_size < 9_000_000 or not 0 < original_size < 9_000_000:
        raise ValueError("Video is not the expected small sample")
    local = fetch(VIDEO_URL, 30, (offset, offset + 29))
    header = struct.unpack("<4s5H3I2H", local)
    if header[0] != b"PK\x03\x04" or header[2] != flags or header[3] != method:
        raise ValueError("Local and central ZIP headers disagree")
    if not flags & 8 and header[6:9] != (expected_crc, compressed_size, original_size):
        raise ValueError("Local ZIP CRC and sizes disagree with the directory")
    name_size, extra_size = header[9:11]
    if name_size != len(VIDEO_MEMBER.encode("utf-8")) or extra_size > 4096:
        raise ValueError("Unexpected local header length")
    prefix_size = name_size + extra_size
    prefix = fetch(VIDEO_URL, prefix_size, (offset + 30, offset + 29 + prefix_size))
    if prefix[:name_size].decode("utf-8") != VIDEO_MEMBER:
        raise ValueError("Local ZIP member name differs from Clip_1.mov")
    body_offset = offset + 30 + prefix_size
    # Small independent ranges keep slow public-server transfers resumable per
    # request. Fixed slices still total exactly one compressed member.
    slices = [
        (body_offset + first, body_offset + min(first + 1_048_576, compressed_size) - 1)
        for first in range(0, compressed_size, 1_048_576)
    ]

    def fetch_slice(bounds: tuple[int, int]) -> bytes:
        result = fetch(VIDEO_URL, bounds[1] - bounds[0] + 1, bounds)
        print(f"Verified range: {len(result):,} bytes", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        compressed = b"".join(pool.map(fetch_slice, slices))
    if method == 8:
        inflater = zlib.decompressobj(-15)
        video = inflater.decompress(compressed, original_size + 1)
        if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("Incomplete or oversized DEFLATE payload")
    else:
        video = compressed
    if len(video) != original_size or zlib.crc32(video) & 0xFFFFFFFF != expected_crc:
        raise ValueError("Video size or CRC mismatch")
    return video, {
        "member": VIDEO_MEMBER,
        "compressed_bytes": compressed_size,
        "local_header_offset": offset,
        "crc32": f"{expected_crc:08x}",
        "crc_verified": True,
        "archive_bytes": VIDEO_ARCHIVE_BYTES,
        "range_response_required": 206,
    }


def read_annotation(archive: bytes) -> tuple[bytes, dict]:
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        infos = zipped.infolist()
        if len(infos) > 1000:
            raise ValueError("Unexpected annotation archive entry count")
        for info in infos:
            safe_name(info.filename)
            if info.file_size > 1_000_000:
                raise ValueError("Unexpected oversized annotation member")
        matches = [info for info in infos if info.filename == ANNOTATION_MEMBER]
        if len(matches) != 1:
            raise ValueError("The fixed Clip_1 annotation must appear exactly once")
        annotation = zipped.read(matches[0])  # zipfile verifies this member's CRC.
        expected_crc = matches[0].CRC
    rows = []
    for row in csv.reader(io.StringIO(annotation.decode("utf-8"))):
        if not row:
            continue
        if len(row) != 10:
            raise ValueError("Expected ten comma-separated annotation columns")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite annotation value")
        if any(values[i] != int(values[i]) or values[i] < 1 for i in (0, 1)):
            raise ValueError("Frames and target IDs must be positive integers")
        if values[4] <= 0 or values[5] <= 0:
            raise ValueError("Bounding-box width and height must be positive")
        rows.append(values)
    if not rows:
        raise ValueError("Empty annotation")
    frames = sorted({int(row[0]) for row in rows})
    target_ids = sorted({int(row[1]) for row in rows})
    if len({(int(row[0]), int(row[1])) for row in rows}) != len(rows):
        raise ValueError("Duplicate frame/target annotation")
    by_frame = {frame: sum(int(row[0]) == frame for row in rows) for frame in frames}
    return annotation, {
        "member": ANNOTATION_MEMBER,
        "crc32": f"{expected_crc:08x}",
        "crc_verified": True,
        "row_count": len(rows),
        "annotated_frame_count": len(frames),
        "frame_min": frames[0],
        "frame_max": frames[-1],
        "frame_numbering": "observed_positive_integer_starting_at_1" if frames[0] == 1 else "observed_positive_integer",
        "target_ids": target_ids,
        "target_count": len(target_ids),
        "max_targets_per_annotated_frame": max(by_frame.values()),
        "column_count": 10,
        "columns": ["frame", "target_id", "left_px", "top_px", "width_px", "height_px", "annotation_flag", "unused_1", "unused_2", "unused_3"],
        "annotation_flag_values": sorted({row[6] for row in rows}),
        "interpolated_row_count": sum(row[6] < 1 for row in rows),
        "direct_annotated_row_count": sum(row[6] >= 1 for row in rows),
        "unused_values": [sorted({row[i] for row in rows}) for i in (7, 8, 9)],
        "first_row": rows[0],
        "note": "Ground-truth boxes; the author renderer identifies seventh-column values below 1 as interpolated boxes. It is not detector confidence. Frame IDs are 1-based according to the bundled renderer, which is read as text and never executed.",
    }


def save(name: str, payload: bytes, url: str) -> dict:
    safe_name(name)
    if PurePosixPath(name).name != name:
        raise ValueError("Output must be a single fixed basename")
    path = DATA / name
    path.write_bytes(payload)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "source_url": url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def check_license(payload: bytes) -> None:
    # Exact title and clauses verified from the official LICENSE.txt; the
    # copyright holder is Charles A. Bouman, not an institution-name field.
    text = payload.decode('utf-8').replace('\r\n', '\n')
    if not text.startswith('BSD 3-Clause License\n') or not all(
        clause in text for clause in (
            '1. Redistributions of source code',
            '2. Redistributions in binary form',
            '3. Neither the name of the copyright holder',
            'THIS SOFTWARE IS PROVIDED',
        )
    ):
        raise ValueError('Unexpected official BSD license text')


def write_status(files: list[dict], checks: dict | None, video: dict | None = None,
                 errors: list[str] | None = None, transfer_bytes: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transferred = downloaded_bytes if transfer_bytes is None else transfer_bytes
    checked_date = date.today().isoformat()
    manifest = {
        'checked_date': checked_date,
        'dataset': 'Purdue UAV-to-UAV Detection and Tracking Dataset',
        'official_page': BASE,
        'scope': 'real_material_download_and_format_check_only',
        'data_license': 'BSD-3-Clause',
        'viewpoint': 'camera_mounted_on_a_flying_UAV',
        'video_status': 'saved_and_crc_verified' if video else 'not_saved',
        'video_source_url': VIDEO_URL,
        'video_member': VIDEO_MEMBER,
        'video_archive_bytes': VIDEO_ARCHIVE_BYTES,
        'video_range_access_verified': True,
        'video_prior_attempt_note': 'A prior bounded Clip_1 download passed CRC, but its program stopped before saving because the license-text check incorrectly required the institution name. No retained video or SHA-256 is claimed unless a video file is explicitly listed.',
        'video_zip_member': video,
        'detector_evaluation_completed': False,
        'tracking_evaluation_completed': False,
        'localization_evaluation_completed': False,
        'downloaded_bytes_this_run': transferred,
        'download_limit_bytes': TOTAL_DOWNLOAD_LIMIT,
        'annotation': checks,
        'files': files,
        'errors': errors or [],
        'raw_data_in_git': False,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    status = ('指定 Clip_1 视频已保存并通过大小和 CRC 校验；尚未解码或评价。' if video else
              '**当前未保存视频，本地视觉材料仍缺视频。** 官方归档范围访问已核验；此前一次限定视频下载曾通过 CRC，但程序因错误地要求 BSD 许可包含学校名而在落盘前退出。此判断已纠正，不代表数据集不能公开访问。')
    report = f'''# 第二版真实无人机对无人机数据核验

核验日期：{checked_date}。

来源：[Purdue 作者官方页面]({BASE})。原始材料保存于本项目 `data/public/purdue`，不上传 GitHub。官方数据许可为 BSD 3-Clause；版权人为 Charles A. Bouman。仅获取指定样本，不下载完整视频归档。

{status}

## 已保存文件

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
'''
    for entry in files:
        report += f"| {Path(entry['path']).name} | {entry['bytes']:,} | `{entry['sha256']}` |\n"
    if checks:
        report += f'''
## 标注格式和计数

- 指定标注成员 CRC-32 为 `{checks['crc32']}`，已校验。只读取标注文本与作者渲染脚本说明，未执行压缩包内的代码。
- 每行 10 个逗号分隔字段：帧号、目标编号、框左上角横坐标、框左上角纵坐标、框宽、框高、标注置信标志及三个未使用字段。坐标单位为像素，帧号从 1 开始；这是作者附带渲染器明确采用的约定。
- 共 {checks['row_count']} 条框标注，覆盖 {checks['annotated_frame_count']} 个标注帧；实际标注帧号范围 {checks['frame_min']}–{checks['frame_max']}。尚未通过视频解码核对帧数。
- 目标编号为 `{checks['target_ids']}`，共 {checks['target_count']} 个目标；单个标注帧最多 {checks['max_targets_per_annotated_frame']} 个目标。未发现重复的“帧号、目标编号”组合；数值有限，框宽高为正。
- 第七列实际取值为 `{checks['annotation_flag_values']}`。根据作者渲染器，该列小于 1 表示插值框；本样本有 {checks['interpolated_row_count']} 条插值框、{checks['direct_annotated_row_count']} 条非插值框。这些真值标志不能冒充检测器的置信度或检测输出。
'''
    if video:
        report += f"\n视频 CRC-32 为 `{video['crc32']}`，已校验。\n"
    if errors:
        report += '\n## 尚未完成的获取步骤\n\n' + '\n'.join('- ' + item for item in errors) + '\n'
    report += '''
## 验证边界与后续工作

此次仅核验真实材料、完整性与格式，**没有运行目标检测、二维跟踪或三维定位评价**，没有把真值框冒充实际检测结果。该视觉数据适合后续无人机对无人机检测与跟踪；当前样本没有同步雷达、三维位置真值或本机位姿/IMU，不能验证雷达相机融合三维误差或原论文实测结果。

下一步应保存和解码视频，核对帧号，再独立运行检测/跟踪。训练和评价按完整视频序列隔离。完整状态见 `manifest.json`。

运行 `python tools/fetch_purdue_sample.py --annotations-only` 仅取小材料；运行 `python tools/fetch_purdue_sample.py` 才尝试限定视频。获取器先保存许可，再保存标注，每个成功文件立即落盘并更新报告；后续失败保留已成功文件和状态。所有请求仅指向固定的三个作者官方网址，视频请求必须返回精确 HTTP 206，限制每段和总字节，核对 ZIP 头、固定成员名、解压大小与 CRC，拒绝路径穿越。网络错误最多重试两次，失败读取的字节仍计入 12 MB 上限。
'''
    (OUT / 'report.md').write_text(report, encoding='utf-8')


def main(only_annotations: bool = False) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    files = []
    checks = None
    write_status(files, checks)
    try:
        print('Fetching and saving the official BSD license first...', flush=True)
        license_text = fetch(LICENSE_URL, 16_384)
        check_license(license_text)
        files.append(save('LICENSE.txt', license_text, LICENSE_URL))
        write_status(files, checks)
        print('Fetching and saving the updated annotation archive...', flush=True)
        archive = fetch(ANNOTATION_URL, 1_000_000)
        annotation, checks = read_annotation(archive)
        files.append(save('Video_Annotation-v2.zip', archive, ANNOTATION_URL))
        write_status(files, checks)
        files.append(save('Clip_1_refined.txt', annotation, ANNOTATION_URL))
        write_status(files, checks)
        if not only_annotations:
            print('Fetching only Clip_1 with bounded HTTP ranges...', flush=True)
            payload, video = video_from_range()
            # Persist the verified video before any later report formatting.
            files.append(save('Clip_1.mov', payload, VIDEO_URL))
            write_status(files, checks, video)
    except (OSError, ValueError, zipfile.BadZipFile, zlib.error) as error:
        write_status(files, checks, errors=[type(error).__name__ + ': ' + str(error)])
        print('Incomplete download; successful files and an accurate report were preserved.', flush=True)
        return 1
    print(json.dumps({'annotation': checks, 'files': files}, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--annotations-only']):
        raise SystemExit('Usage: fetch_purdue_sample.py [--annotations-only]')
    raise SystemExit(main(only_annotations=bool(sys.argv[1:])))
