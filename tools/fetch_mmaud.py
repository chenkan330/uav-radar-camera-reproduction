"""Fetch public MMAUD V1 Mavic3 material using normal published download links.

No credentials, permission changes, or archive code execution. Default operation
fetches the small independent ground truth and a ZIP/ZIP64 directory. Repeat
--member with exact names from data/public/mmaud/archive_index.json to fetch data.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import http.cookiejar
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "mmaud"
OUT = ROOT / "output" / "v16_data_audit"
OFFICIAL = "https://ntu-aris.github.io/MMAUD/"
ZIP_SHARE = "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/ETe4YTM-IKdMnA11Q1Dv_PgBWGfQ38iwYoFpkFWthhfJsQ?e=CJU8JN"
GT_SHARE = "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/Eaol9EuQTZ5BmYffvBSo3kIB0DwKDcYstGvwg-Qrtnts4A?e=ZaJpsa"
ZIP_BYTES = 10_750_360_115
GT_BYTES = 158_950
HOST = "entuedu-my.sharepoint.com"
SPAN_HASHES: dict[tuple[str, int, int], str] = {}


def numpy_header(path: Path) -> dict:
    """Inspect the bounded NPY literal header without pickle or NumPy loading."""
    with path.open("rb") as stream:
        prefix = stream.read(12)
        if prefix[:6] != b"\x93NUMPY" or prefix[6] not in (1, 2, 3):
            raise ValueError("Unexpected numeric array header")
        start = 10 if prefix[6] == 1 else 12
        size = int.from_bytes(prefix[8:start], "little")
        if size > 16_384:
            raise ValueError("NPY header exceeds metadata bound")
        stream.seek(start)
        header = ast.literal_eval(stream.read(size).decode("utf-8" if prefix[6] == 3 else "latin1"))
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("Unexpected NPY header fields")
    return header


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\x00" in name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive path")


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


class PublicDownload:
    def __init__(self, budget: int = 1_000_000_000):
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.used = 0
        self.budget = budget
        self._public_cookies: list[dict] = []
        self.allow_network = True

    def get(self, url: str, limit: int, bounds: tuple[int, int, int] | None = None) -> bytes:
        if not self.allow_network:
            raise FileNotFoundError("Offline cache is missing or invalid; network is disabled")
        if self.used >= self.budget:
            raise ValueError("Transfer budget exhausted")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != HOST:
            raise ValueError("Only the published SharePoint host is allowed")
        if sys.platform == "win32":
            return self._windows_get(url, limit, bounds)
        headers = {"User-Agent": "UAV-paper-reproduction/1.6"}
        if bounds:
            start, end, total = bounds
            if not 0 <= start <= end < total or end - start + 1 != limit:
                raise ValueError("Invalid byte range")
            headers["Range"] = f"bytes={start}-{end}"
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(3):
            received = 0
            try:
                deadline = time.monotonic() + 240
                with self.opener.open(request, timeout=45) as response:
                    final = urllib.parse.urlsplit(response.url)
                    if final.netloc != HOST or "login" in final.path.lower():
                        raise PermissionError("Public link redirected away or requested login")
                    if bounds:
                        expected = f"bytes {start}-{end}/{total}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Server did not return the exact requested public range")
                    elif response.status != 200:
                        raise ValueError("Unexpected public download status")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) > limit:
                        raise ValueError("HTTP body exceeds its bound")
                    blocks = []
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Bounded request time exceeded")
                        allowance = min(65_536, limit + 1 - received, self.budget + 1 - self.used)
                        if allowance <= 0:
                            raise ValueError("Transfer limit exhausted")
                        block = response.read(allowance)
                        if not block:
                            break
                        self.used += len(block)
                        received += len(block)
                        if self.used > self.budget or received > limit:
                            raise ValueError("Transfer limit exceeded")
                        blocks.append(block)
                    result = b"".join(blocks)
                    if bounds and len(result) != limit:
                        raise ValueError("Truncated range body")
                    if length is not None and len(result) != int(length):
                        raise ValueError("Truncated public body")
                    return result
            except PermissionError:
                raise
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise PermissionError("Published public download requires access") from error
                if attempt == 2:
                    raise
            except OSError:
                if attempt == 2:
                    raise
            print(f"Transient request failed after {received} bytes; retry {attempt + 1}/2", flush=True)
        raise RuntimeError("Unreachable")

    def _windows_get(self, url: str, limit: int, bounds: tuple[int, int, int] | None) -> bytes:
        """Use Windows TLS, checking headers before reading a strictly bounded body.

        Public URL is delivered by stdin, not command line or logged files.
        Only the response body briefly occupies a random D-drive .part file.
        """
        if bounds and (not 0 <= bounds[0] <= bounds[1] < bounds[2] or bounds[1] - bounds[0] + 1 != limit):
            raise ValueError("Invalid byte range")
        worker = r'''
$ErrorActionPreference = 'Stop'
$spec = [Console]::In.ReadToEnd() | ConvertFrom-Json
$received = 0L
$response = $null
$destination = $null
$client = $null
try {
  $cookies = [Net.CookieContainer]::new()
  foreach ($item in $spec.cookies) {
    $cookie = [Net.Cookie]::new($item.name, $item.value, $item.path, $item.domain)
    $cookie.Secure = $true
    $cookies.Add($cookie)
  }
  $handler = [Net.Http.HttpClientHandler]::new()
  $handler.CookieContainer = $cookies
  $handler.UseCookies = $true
  $client = [Net.Http.HttpClient]::new($handler)
  $client.Timeout = [TimeSpan]::FromSeconds(30)
  $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $spec.url)
  $request.Headers.UserAgent.ParseAdd('Mozilla/5.0')
  if ($spec.bounds) { $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new([long]$spec.bounds[0], [long]$spec.bounds[1]) }
  $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
  $uri = $response.RequestMessage.RequestUri
  if ($uri.Host -ne $spec.host -or $uri.AbsolutePath -match 'login') {
    throw [UnauthorizedAccessException]::new('Public link requested login or changed host')
  }
  if ([int]$response.StatusCode -in @(401, 403)) { throw [UnauthorizedAccessException]::new('Public link requires access') }
  if ($spec.bounds) {
    $expected = 'bytes ' + $spec.bounds[0] + '-' + $spec.bounds[1] + '/' + $spec.bounds[2]
    if ([int]$response.StatusCode -ne 206 -or $response.Content.Headers.ContentRange.ToString() -ne $expected) {
      throw [IO.InvalidDataException]::new('Unexpected exact range response')
    }
  } elseif ([int]$response.StatusCode -ne 200) { throw [IO.InvalidDataException]::new('Unexpected status') }
  $length = $response.Content.Headers.ContentLength
  if ($length -and $length -gt $spec.limit) { throw [IO.InvalidDataException]::new('Body exceeds bound') }
  $destination = [IO.File]::Open($spec.path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
  $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $buffer = New-Object byte[] 65536
  $clock = [Diagnostics.Stopwatch]::StartNew()
  while ($true) {
    if ($clock.Elapsed.TotalSeconds -gt 120) { throw [TimeoutException]::new('Bounded transfer deadline') }
    $allowance = [int][Math]::Min(65536, [Math]::Min($spec.limit + 1 - $received, $spec.remaining + 1 - $received))
    if ($allowance -le 0) { throw [IO.InvalidDataException]::new('Transfer budget exhausted') }
    $count = $stream.Read($buffer, 0, $allowance)
    if ($count -eq 0) { break }
    $received += $count
    if ($received -gt $spec.limit -or $received -gt $spec.remaining) { throw [IO.InvalidDataException]::new('Transfer budget exceeded') }
    $destination.Write($buffer, 0, $count)
  }
  if (($spec.bounds -and $received -ne $spec.limit) -or ($length -and $received -ne $length)) {
    throw [IO.InvalidDataException]::new('Truncated response')
  }
  $session = @($cookies.GetAllCookies() | ForEach-Object {
    @{ name = $_.Name; value = $_.Value; domain = $_.Domain; path = $_.Path }
  })
  @{ ok = $true; bytes = $received; cookies = $session } | ConvertTo-Json -Compress -Depth 4
} catch {
  $cause = $_.Exception
  while ($cause.InnerException) { $cause = $cause.InnerException }
  $denied = $cause -is [UnauthorizedAccessException]
  if ($cause.Response -and [int]$cause.Response.StatusCode -in @(401, 403)) { $denied = $true }
  @{ ok = $false; bytes = $received; denied = $denied; type = $cause.GetType().Name } | ConvertTo-Json -Compress
} finally {
  if ($destination) { $destination.Dispose() }
  if ($response) { $response.Dispose() }
  if ($client) { $client.Dispose() }
}
'''
        for attempt in range(2):
            temporary = DATA / (".http-" + uuid.uuid4().hex + ".part")
            specification = {"url": url, "limit": limit, "bounds": bounds, "host": HOST,
                             "remaining": self.budget - self.used, "path": str(temporary),
                             "cookies": self._public_cookies}
            try:
                shell = shutil.which("pwsh.exe")
                if shell is None:
                    raise OSError("PowerShell 7 is required for the bounded Windows transport")
                result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", worker],
                                        input=json.dumps(specification), capture_output=True, text=True, timeout=160)
                metadata = json.loads(result.stdout.strip())
                self.used += int(metadata.get("bytes", 0))
                if not metadata.get("ok"):
                    if metadata.get("denied"):
                        raise PermissionError("Published public download requires access; no bypass attempted")
                    raise OSError("Bounded Windows download failed: " + metadata.get("type", "Unknown"))
                self._public_cookies = metadata.get("cookies", [])
                payload = temporary.read_bytes()
                if len(payload) > limit or len(payload) != metadata["bytes"] or self.used > self.budget:
                    raise ValueError("Windows response size check failed")
                return payload
            except PermissionError:
                raise
            except (OSError, subprocess.TimeoutExpired):
                if attempt == 1:
                    raise
                print("Transient Windows request failed; retry 1/1", flush=True)
            finally:
                temporary.unlink(missing_ok=True)
        raise RuntimeError("Unreachable")

    def public_metadata(self, share: str, expected_name: str, expected_bytes: int) -> dict:
        page = self.get(share, 2_000_000).decode("utf-8")
        def field(key: str) -> str:
            match = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:\\.|[^"\\])*)"', page)
            if not match:
                raise PermissionError("File metadata absent; no attempt to bypass login")
            return json.loads('"' + match.group(1) + '"')
        name, size = field("FileLeafRef"), int(field("FileSizeDisplay"))
        if name != expected_name or size != expected_bytes:
            raise ValueError("Published file identity or size changed")
        # This endpoint is disclosed by the anonymous file viewer itself.
        return {"name": name, "bytes": size, "download": field(".downloadUrlNoAuth"), "source": share}


def write_file(relative: str, payload: bytes, source: str, crc: int | None = None) -> dict:
    safe_name(relative)
    path = DATA / relative
    if not path.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Resolved data path leaves the MMAUD directory")
    if sys.platform == "win32" and path.resolve().drive.upper() != "D:":
        raise ValueError("This workstation stores downloaded project material on D drive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "source_url": source,
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "crc32": None if crc is None else f"{crc:08x}", "crc_verified": crc is not None}


def zip_index(client: PublicDownload, url: str) -> list[dict]:
    tail_size = 131_072
    tail_path = DATA / "archive_tail.bin"
    if tail_path.exists() and tail_path.stat().st_size == tail_size:
        tail = tail_path.read_bytes()
    else:
        tail = client.get(url, tail_size, (ZIP_BYTES - tail_size, ZIP_BYTES - 1, ZIP_BYTES))
        tail_path.write_bytes(tail)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise ValueError("ZIP end record missing")
    end = struct.unpack_from("<4s4H2IH", tail, position)
    if position + 22 + end[7] != len(tail):
        raise ValueError("ZIP end record does not match the published file end")
    if end[1] or end[2] or end[3] != end[4]:
        raise ValueError("Multi-disk archive unsupported")
    directory_size, directory_offset, entries = end[5], end[6], end[4]
    if max(directory_offset, directory_size) == 0xFFFFFFFF or entries == 0xFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, position)
        if locator < 0:
            raise ValueError("ZIP64 locator missing")
        _, disk, offset64, disks = struct.unpack_from("<4sIQI", tail, locator)
        # This author archive declares total disks=0 in its ZIP64 locator.
        # Tolerate that writer anomaly only after EOCD and all entries prove a
        # single contiguous disk; do not silently permit genuine split ZIPs.
        if disk or disks not in (0, 1):
            raise ValueError(f"ZIP64 locator needs audit: disk={disk}, disks={disks}, offset={offset64}")
        tail_start = ZIP_BYTES - tail_size
        if tail_start <= offset64 and offset64 + 56 <= ZIP_BYTES:
            record = tail[offset64 - tail_start:offset64 - tail_start + 56]
        else:
            record = client.get(url, 56, (offset64, offset64 + 55, ZIP_BYTES))
        row64 = struct.unpack("<4sQ2H2I4Q", record)
        if row64[0] != b"PK\x06\x06" or row64[4] or row64[5] or row64[6] != row64[7]:
            raise ValueError("Invalid ZIP64 record")
        entries, directory_size, directory_offset = row64[7:10]
        if directory_offset + directory_size != offset64 or offset64 + 12 + row64[1] != tail_start + locator:
            raise ValueError("ZIP64 single-disk layout is not contiguous")
    if directory_size > 32_000_000 or entries > 300_000:
        raise ValueError("Directory exceeds bounded metadata size")
    directory_path = DATA / "archive_central_directory.bin"
    if directory_path.exists() and directory_path.stat().st_size == directory_size:
        directory = directory_path.read_bytes()
    else:
        directory = client.get(url, directory_size,
                               (directory_offset, directory_offset + directory_size - 1, ZIP_BYTES))
        directory_path.write_bytes(directory)
    result = []
    pointer = 0
    for _ in range(entries):
        row = struct.unpack_from("<4s6H3I5H2I", directory, pointer)
        if row[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP directory signature")
        if row[13] != 0:
            raise ValueError("ZIP member is on a different disk")
        name_size, extra_size, comment_size = row[10:13]
        name = directory[pointer + 46:pointer + 46 + name_size].decode("utf-8" if row[3] & 0x800 else "cp437")
        safe_name(name)
        original, compressed, offset = row[9], row[8], row[16]
        extra = directory[pointer + 46 + name_size:pointer + 46 + name_size + extra_size]
        extptr = 0
        while extptr + 4 <= len(extra):
            kind, length = struct.unpack_from("<HH", extra, extptr)
            extptr += 4
            if kind == 1:
                valueptr = extptr
                for field in ("original", "compressed", "offset"):
                    value = {"original": original, "compressed": compressed, "offset": offset}[field]
                    if value == 0xFFFFFFFF:
                        replacement = struct.unpack_from("<Q", extra, valueptr)[0]
                        valueptr += 8
                        if field == "original": original = replacement
                        elif field == "compressed": compressed = replacement
                        else: offset = replacement
            extptr += length
        result.append({"name": name, "bytes": original, "compressed_bytes": compressed,
                       "offset": offset, "method": row[4], "flags": row[3], "crc32_int": row[7]})
        if not 0 <= offset < directory_offset or offset + 30 + name_size + compressed > directory_offset:
            raise ValueError("ZIP member lies outside the single data disk")
        pointer += 46 + name_size + extra_size + comment_size
    if pointer != len(directory):
        raise ValueError("ZIP directory length mismatch")
    ordered = sorted(result, key=lambda entry: entry["offset"])
    for first, second in zip(ordered, ordered[1:]):
        first["span_bytes"] = second["offset"] - first["offset"]
        if first["span_bytes"] < 30 + len(first["name"].encode("utf-8")) + first["compressed_bytes"]:
            raise ValueError("Overlapping ZIP members")
    ordered[-1]["span_bytes"] = directory_offset - ordered[-1]["offset"]
    return result


def fetch_member(client: PublicDownload, url: str, entry: dict) -> dict:
    name = entry["name"]
    if name.endswith("/") or Path(name).suffix.lower() not in (".pcd", ".png", ".jpg", ".jpeg", ".npy", ".json", ".yaml", ".yml", ".txt", ".csv", ".mat", ".bag"):
        raise ValueError("Only data members can be extracted")
    path = DATA / name
    if path.exists():
        cached = path.read_bytes()
        if len(cached) == entry["bytes"] and zlib.crc32(cached) & 0xFFFFFFFF == entry["crc32_int"]:
            return write_file(name, cached, ZIP_SHARE, entry["crc32_int"])
    offset, span = entry["offset"], entry["span_bytes"]
    if not 30 <= span <= 128_000_000:
        raise ValueError("Member compressed span exceeds bound")
    raw = client.get(url, span, (offset, offset + span - 1, ZIP_BYTES))
    return decode_member(entry, raw)


def decode_member(entry: dict, raw: bytes) -> dict:
    name = entry["name"]
    safe_name(name)
    if name.endswith("/") or Path(name).suffix.lower() not in (".pcd", ".png", ".jpg", ".jpeg", ".npy", ".json", ".yaml", ".yml", ".txt", ".csv", ".mat", ".bag"):
        raise ValueError("Only data members can be extracted")
    header = struct.unpack_from("<4s5H3I2H", raw)
    if header[0] != b"PK\x03\x04" or header[2] != entry["flags"] or header[3] != entry["method"]:
        raise ValueError("Local ZIP header disagrees with directory")
    if header[9] + header[10] > 65_536 or entry["bytes"] > 128_000_000:
        raise ValueError("Member exceeds bounded extraction size")
    prefix_bytes = header[9] + header[10]
    prefix = raw[30:30 + prefix_bytes]
    if prefix[:header[9]].decode("utf-8" if entry["flags"] & 0x800 else "cp437") != name:
        raise ValueError("Local ZIP member name mismatch")
    body_start = 30 + prefix_bytes
    compressed = raw[body_start:body_start + entry["compressed_bytes"]]
    if len(compressed) != entry["compressed_bytes"]:
        raise ValueError("Compressed member body is truncated")
    if entry["flags"] & 1:
        raise ValueError("Encrypted data unsupported")
    if entry["method"] == 0:
        payload = compressed
    elif entry["method"] == 8:
        inflater = zlib.decompressobj(-15)
        payload = inflater.decompress(compressed, entry["bytes"] + 1)
        if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("Invalid or excessive compressed payload")
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(payload) != entry["bytes"] or zlib.crc32(payload) & 0xFFFFFFFF != entry["crc32_int"]:
        raise ValueError("Member size or CRC mismatch")
    record = write_file(name, payload, ZIP_SHARE, entry["crc32_int"])
    print(f"Saved verified data member: {name} ({len(payload)} bytes)", flush=True)
    return record


def fetch_members(client: PublicDownload, url: str, entries: list[dict]):
    """Batch neighbouring local records, capped at 8 MB per exact HTTP range."""
    groups: list[list[dict]] = []
    for entry in sorted(entries, key=lambda item: item["offset"]):
        if entry["name"].endswith("/"):
            continue
        if groups:
            last = groups[-1][-1]
            gap = entry["offset"] - last["offset"] - last["span_bytes"]
            total = entry["offset"] + entry["span_bytes"] - groups[-1][0]["offset"]
            if 0 <= gap <= 65_536 and total <= 8_000_000:
                groups[-1].append(entry)
                continue
        groups.append([entry])
    for group in groups:
        missing = []
        for entry in group:
            path = DATA / entry["name"]
            if path.exists():
                payload = path.read_bytes()
                if len(payload) == entry["bytes"] and zlib.crc32(payload) & 0xFFFFFFFF == entry["crc32_int"]:
                    yield write_file(entry["name"], payload, ZIP_SHARE, entry["crc32_int"])
                    continue
            missing.append(entry)
        if not missing:
            continue
        start = missing[0]["offset"]
        finish = missing[-1]["offset"] + missing[-1]["span_bytes"]
        if finish - start > 8_000_000:
            raise ValueError("Requested member or batch exceeds the 8 MB range cap")
        raw = client.get(url, finish - start, (start, finish - 1, ZIP_BYTES))
        for entry in missing:
            position = entry["offset"] - start
            yield decode_member(entry, raw[position:position + entry["span_bytes"]])


def record_status(files: list[dict], index: list[dict], used: int, error: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for entry in index:
        prefix = str(PurePosixPath(entry["name"]).parent)
        counts[prefix] = counts.get(prefix, 0) + 1
    material = {}
    for entry in index:
        if entry["name"].endswith("/"):
            continue
        key = str(PurePosixPath(entry["name"]).parent)
        group = material.setdefault(key, {"archive_count": 0, "saved_count": 0,
                                         "first_filename_time": None, "last_filename_time": None})
        group["archive_count"] += 1
        try:
            timestamp = float(PurePosixPath(entry["name"]).stem)
            group["first_filename_time"] = timestamp if group["first_filename_time"] is None else min(timestamp, group["first_filename_time"])
            group["last_filename_time"] = timestamp if group["last_filename_time"] is None else max(timestamp, group["last_filename_time"])
        except ValueError:
            pass
    for entry in files:
        key = str(PurePosixPath(entry["path"].removeprefix("data/public/mmaud/")).parent)
        if key in material:
            material[key]["saved_count"] += 1
    ranges = []
    declarations = DATA / "range_requests.json"
    if declarations.exists():
        ranges.extend(json.loads(declarations.read_text(encoding="utf-8")))
    for filename in ("sample_image_ranges.json", "v16_selected_image_ranges.json"):
        image_declarations = DATA / filename
        if not image_declarations.exists():
            continue
        for entry in json.loads(image_declarations.read_text(encoding="utf-8")):
            ranges.append({"start": entry["start"], "end": entry["end"], "bytes": entry["bytes"],
                           "path": "data/public/mmaud/image_archive_spans/" + PurePosixPath(entry["name"]).stem + ".bin"})
    verified_ranges = []
    seen_spans = set()
    for entry in ranges:
        safe_name(entry["path"])
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():
            continue
        length = entry["end"] - entry["start"] + 1
        if length != entry["bytes"] or length > 8_000_000 or path.stat().st_size != length:
            continue
        if entry["path"] in seen_spans:
            continue
        seen_spans.add(entry["path"])
        metadata = path.stat()
        hash_key = (str(path), metadata.st_mtime_ns, metadata.st_size)
        if hash_key not in SPAN_HASHES:
            SPAN_HASHES[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()
        verified_ranges.append({"path": path.relative_to(ROOT).as_posix(), "start": entry["start"],
                                "end": entry["end"], "bytes": length,
                                "sha256": SPAN_HASHES[hash_key]})
    manifest = {"checked_date": datetime.now().astimezone().isoformat(), "dataset": "MMAUD V1 Mavic3",
                "official_page": OFFICIAL, "zip_source": ZIP_SHARE, "ground_truth_source": GT_SHARE,
                "archive_bytes": ZIP_BYTES, "ground_truth_bytes": GT_BYTES,
                "files": files, "zip_directory_entries": len(index), "directory_counts": counts,
                "downloaded_bytes_this_run": used, "error": error,
                "login_bypassed": False, "archive_code_executed": False,
                "ground_truth_used_as_observation": False, "raw_data_in_git": False,
                "zip64_locator_anomaly": "Author ZIP locator total_disk_count=0; accepted only after EOCD disk=0, all member disk=0, contiguous directory and ZIP64 record validation."}
    manifest["material_summary"] = material
    manifest["saved_archive_spans"] = verified_ranges
    manifest["archive_metadata"] = []
    for filename in ("archive_tail.bin", "archive_central_directory.bin"):
        path = DATA / filename
        if path.is_file():
            payload = path.read_bytes()
            manifest["archive_metadata"].append({"path": path.relative_to(ROOT).as_posix(),
                                                 "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    radar = [entry for entry in index if "/radar_enhance_pcl/" in entry["name"] and entry["name"].endswith(".npy")]
    arrays = []
    for entry in radar:
        path = DATA / entry["name"]
        if path.is_file():
            arrays.append(numpy_header(path))
    manifest["radar_numeric_format"] = {"saved_header_count": len(arrays),
        "dtype_descriptors": sorted({str(entry["descr"]) for entry in arrays}),
        "column_counts": sorted({entry["shape"][1] for entry in arrays if len(entry["shape"]) == 2}),
        "minimum_points": min((entry["shape"][0] for entry in arrays), default=None),
        "maximum_points": max((entry["shape"][0] for entry in arrays), default=None),
        "archive_unique_crc32": len({entry["crc32_int"] for entry in radar}),
        "adjacent_equal_crc32": sum(a["crc32_int"] == b["crc32_int"] for a, b in zip(radar, radar[1:])),
        "velocity_intensity_fields_present": False,
        "coordinate_axis_meaning_verified": False}
    atomic_text(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    report = "# MMAUD 真实数据获取与审计\n\n"
    report += f"日期：{datetime.now().astimezone().date()}。来源：[作者官方页面]({OFFICIAL})。仅使用作者发布的匿名文件页所提供的公开下载入口，不登录或绕过权限。原始材料位于 D 盘本项目 `data/public/mmaud`，不上传 GitHub。\n\n"
    report += f"Mavic3 归档共 {ZIP_BYTES:,} 字节；本次审计进程网络读取 {used:,} 字节（离线核验时为 0）。已核验目录中 {len(index)} 个成员，保存并校验 {len(files)} 个材料文件；另保留 {sum(entry['bytes'] for entry in verified_ranges):,} 字节归档局部跨度供再次提取。\n\n"
    report += "| ZIP 文件夹 | 归档数量 | 已保存数量 | 首/末文件名时间（秒） |\n|---|---:|---:|---|\n"
    for key, value in material.items():
        report += f"| {key} | {value['archive_count']} | {value['saved_count']} | {value['first_filename_time']}–{value['last_filename_time']} |\n"
    report += "\n以下展示各类已保存文件的一个例子；全部逐文件 SHA-256 与 ZIP CRC32 见同目录 `manifest.json`。\n\n| 已保存文件示例 | 字节 | SHA-256 |\n|---|---:|---|\n"
    seen = set()
    for entry in files:
        key = str(PurePosixPath(entry["path"]).parent)
        if key in seen:
            continue
        seen.add(key)
        report += f"| {entry['path']} | {entry['bytes']} | `{entry['sha256']}` |\n"
    report += "\n材料获取与格式核验不等于算法精度验证。本获取审计未将真值作为雷达或相机观测；未执行压缩包内代码。后续若使用独立训练真值做固定标定，须明确披露，评估段真值只在两组滤波完成后用于评价。相机和雷达字段、坐标、时间及标定必须根据真实文件核验。\n"
    if arrays:
        details = manifest["radar_numeric_format"]
        report += f"\n实际 enhanced radar 导出数组：{len(arrays)} 个 NPY，数据类型 {details['dtype_descriptors']}，每帧 {details['minimum_points']}–{details['maximum_points']} 行、{details['column_counts']} 列。文件没有速度、强度、字段名、rig 外参；列的物理轴定义未在此审计中证明。目录中 {details['adjacent_equal_crc32']} 对相邻雷达文件 CRC 相同，不能把所有文件当作独立测量。\n"
    report += "\n本 ZIP 仅包含 `radar_enhance_pcl`，未包含原始 `radar_pcl`、`radar_trk`、内外参或图像标签。文件名是导出的时间戳，不能据此宣称硬件同步。已下载相机 PNG 为双鱼眼拼接图；本材料审计不证明可直接转成原论文普通双目模型。论文所述 MATLAB 标定不能替代缺少的实际参数。\n"
    if error:
        report += "\n当前未完成环节：" + error + "\n"
    report += "\n可复用入口：`tools/fetch_mmaud.py --offline --import-spans` 会只读已有 D 盘局部归档，验证局部成员名、压缩大小、原始大小与 ZIP CRC32，立即保存有效成员；`--members-file` 可传入 JSON 文件名清单。`--offline` 禁止网络请求。正常公开获取使用原作者分享页，Windows 匿名会话 cookie 只在进程内保留，不写入结果；下载按精确 HTTP 206/Content-Range 检查，单范围最多 8 MB，总预算可设。\n"
    report += "\n数据许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，限非商业学术用途；上游软件MIT许可与数据许可分开。数据引用：Yuan et al., MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature Drone Threats, ICRA 2024, pp.2745–2751, [DOI](https://doi.org/10.1109/ICRA57147.2024.10610957)。来源与许可说明：[作者官网](https://ntu-aris.github.io/MMAUD/)。\n"
    atomic_text(OUT / "report.md", report)


def saved_members(index: list[dict]) -> list[dict]:
    """Rebuild the audit from validated disk files before any new HTTP request."""
    result = []
    for entry in index:
        path = DATA / entry["name"]
        if not entry["name"].endswith("/") and path.is_file():
            payload = path.read_bytes()
            if len(payload) == entry["bytes"] and zlib.crc32(payload) & 0xFFFFFFFF == entry["crc32_int"]:
                result.append({"path": path.relative_to(ROOT).as_posix(), "source_url": ZIP_SHARE,
                               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                               "crc32": f"{entry['crc32_int']:08x}", "crc_verified": True})
    return result


def import_spans(index: list[dict]):
    """Extract only validated data records covered by known D-drive range caches."""
    declarations = []
    group_path = DATA / "range_requests.json"
    if group_path.exists():
        declarations.extend(json.loads(group_path.read_text(encoding="utf-8")))
    for filename in ("sample_image_ranges.json", "v16_selected_image_ranges.json"):
        request_path = DATA / filename
        if request_path.exists():
            for entry in json.loads(request_path.read_text(encoding="utf-8")):
                declarations.append({"start": entry["start"], "end": entry["end"], "bytes": entry["bytes"],
                                     "path": "data/public/mmaud/image_archive_spans/" + PurePosixPath(entry["name"]).stem + ".bin"})
    seen = set()
    for request in declarations:
        safe_name(request["path"])
        path = (ROOT / request["path"]).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():
            continue
        if request["bytes"] != request["end"] - request["start"] + 1 or not 0 < request["bytes"] <= 8_000_000 or path.stat().st_size != request["bytes"]:
            raise ValueError("Cached ZIP span size or bounds are invalid")
        raw = path.read_bytes()
        for entry in index:
            if entry["name"].endswith("/") or entry["name"] in seen:
                continue
            offset = entry["offset"] - request["start"]
            if 0 <= offset and offset + entry["span_bytes"] <= len(raw):
                existing = DATA / entry["name"]
                if existing.is_file():
                    payload = existing.read_bytes()
                    if len(payload) == entry["bytes"] and zlib.crc32(payload) & 0xFFFFFFFF == entry["crc32_int"]:
                        seen.add(entry["name"])
                        continue
                record = decode_member(entry, raw[offset:offset + entry["span_bytes"]])
                seen.add(entry["name"])
                yield record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--members-file", type=Path, help="Local JSON array of exact member names")
    parser.add_argument("--offline", action="store_true", help="Audit existing verified material without network")
    parser.add_argument("--import-spans", action="store_true", help="CRC-check and extract existing named range caches")
    parser.add_argument("--budget-mb", type=int, default=1000)
    args = parser.parse_args()
    if args.members_file is not None:
        names = json.loads(args.members_file.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            parser.error("Members file must contain a JSON string array")
        args.member.extend(names)
    if not 1 <= args.budget_mb <= 2000:
        parser.error("Budget must be between 1 and 2000 MB")
    DATA.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32" and DATA.resolve().drive.upper() != "D:":
        raise ValueError("MMAUD download directory must be on D drive")
    client = PublicDownload(args.budget_mb * 1_000_000)
    client.allow_network = not args.offline
    files, index = [], []
    try:
        gt_name = "2023-08-24-11-14-40_mavic3.bag"
        gt_path = DATA / gt_name
        if gt_path.exists() and hashlib.sha256(gt_path.read_bytes()).hexdigest() == "c8e2adfb3528ce93dd2c37fe2ccf9a3853108a26bfee04845b78695f87cb0cf3":
            ground_truth = gt_path.read_bytes()
        else:
            if args.offline:
                raise FileNotFoundError("Previously verified independent GT bag is absent")
            gt_meta = client.public_metadata(GT_SHARE, gt_name, GT_BYTES)
            ground_truth = client.get(gt_meta["download"], GT_BYTES)
        if len(ground_truth) != GT_BYTES or not ground_truth.startswith(b"#ROSBAG V2.0\n"):
            raise ValueError("Unexpected independent GT bag content")
        files.append(write_file(gt_name, ground_truth, GT_SHARE))
        if (DATA / "archive_tail.bin").exists() and (DATA / "archive_central_directory.bin").exists():
            index = zip_index(client, "https://" + HOST + "/")
            files.extend(saved_members(index))
        if args.import_spans:
            for record in import_spans(index):
                files = [entry for entry in files if entry["path"] != record["path"]]
                files.append(record)
                if len(files) % 100 == 0 or record["path"].endswith(".png"):
                    record_status(files, index, client.used)
        record_status(files, index, client.used)
        print("Independent ground truth saved immediately.", flush=True)
        if args.offline:
            existing_names = {entry["path"] for entry in files}
            for name in args.member:
                safe_name(name)
                if "data/public/mmaud/" + name not in existing_names:
                    raise FileNotFoundError("Requested member has not been saved and verified")
            print(f"Offline audit: {len(files)} verified files; no network requests.", flush=True)
            return 0
        zip_meta = client.public_metadata(ZIP_SHARE, "Mavic3.zip", ZIP_BYTES)
        index_path = DATA / "archive_index.json"
        index = zip_index(client, zip_meta["download"])
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files = [entry for entry in files if entry["path"] == "data/public/mmaud/" + gt_name] + saved_members(index)
        record_status(files, index, client.used)
        print(json.dumps({"entries": len(index), "first_members": index[:20]}, ensure_ascii=True), flush=True)
        by_name = {entry["name"]: entry for entry in index}
        selected = []
        for name in dict.fromkeys(args.member):
            if name not in by_name:
                raise ValueError("Requested member is not in the published archive")
            selected.append(by_name[name])
        for record in fetch_members(client, zip_meta["download"], selected):
            files = [entry for entry in files if entry["path"] != record["path"]]
            files.append(record)
            if len(files) % 100 == 0 or record["path"].endswith(".png"):
                record_status(files, index, client.used)
        record_status(files, index, client.used)
    except (OSError, ValueError, struct.error, zlib.error) as error:
        record_status(files, index, client.used, type(error).__name__ + ": " + str(error))
        print("Acquisition stopped; successful files and an accurate audit were preserved.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
