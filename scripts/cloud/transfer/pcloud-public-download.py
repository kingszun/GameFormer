#!/usr/bin/env python3
"""
pCloud public link parallel downloader.

Usage:
    pcloud-public-download.py <CODE> <DEST_DIR> [--parallel N]

Downloads all files from a pCloud public folder share to DEST_DIR.
- No authentication required (uses pCloud public API).
- Idempotent: skips files where local size matches share metadata.
- Parallel downloads via ThreadPoolExecutor + curl (resumable).
- Retries failed downloads up to 3 times with exponential backoff.

API: https://docs.pcloud.com/methods/public_links/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

API_BASE = "https://api.pcloud.com"


def api_get(endpoint, params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_BASE}/{endpoint}?{qs}"
    with urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def list_files(code):
    data = api_get("showpublink", {"code": code})
    if data.get("result") != 0:
        sys.exit(f"showpublink failed: result={data.get('result')} error={data.get('error')}")
    md = data["metadata"]
    if not md.get("isfolder"):
        sys.exit("share root must be a folder, got file")
    files = []

    def walk(item, prefix=""):
        for c in item.get("contents", []):
            if c.get("isfolder"):
                walk(c, f"{prefix}{c['name']}/")
            else:
                files.append({
                    "fileid": c["fileid"],
                    "name": f"{prefix}{c['name']}",
                    "size": c.get("size", 0),
                })

    walk(md)
    return md.get("name"), files


def get_download_url(code, fileid):
    data = api_get("getpublinkdownload", {"code": code, "fileid": fileid})
    if data.get("result") != 0:
        raise RuntimeError(f"getpublinkdownload failed: {data}")
    return f"https://{data['hosts'][0]}{data['path']}"


def download_one(code, info, dest_dir, retries=3):
    fileid, name, expected = info["fileid"], info["name"], info["size"]
    dst = os.path.join(dest_dir, name)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    if os.path.exists(dst) and os.path.getsize(dst) == expected:
        return "SKIP", name, expected, 0.0

    for attempt in range(retries):
        try:
            t0 = time.time()
            url = get_download_url(code, fileid)
            subprocess.run(
                ["curl", "-fsSL", "--connect-timeout", "30", "-o", dst, url],
                check=True, capture_output=True, timeout=3600,
            )
            sz = os.path.getsize(dst)
            if sz != expected:
                raise RuntimeError(f"size mismatch got={sz} expected={expected}")
            return "OK", name, sz, time.time() - t0
        except Exception as e:
            if attempt == retries - 1:
                return "FAIL", name, 0, str(e)
            time.sleep(2 ** attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("code", help="pCloud public link code (e.g., kt4)")
    p.add_argument("dest", help="destination directory")
    p.add_argument("--parallel", type=int, default=8, help="parallel downloads (default 8)")
    args = p.parse_args()

    print(f"[pcloud] listing code={args.code}", flush=True)
    name, files = list_files(args.code)
    total_size = sum(f["size"] for f in files)
    print(f"[pcloud] share={name} files={len(files)} total={total_size/(1024**3):.2f}GiB", flush=True)

    os.makedirs(args.dest, exist_ok=True)
    t0 = time.time()
    completed = 0
    bytes_done = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(download_one, args.code, f, args.dest): f for f in files}
        for fut in as_completed(futures):
            status, fname, size, info = fut.result()
            completed += 1
            if status in ("OK", "SKIP"):
                bytes_done += size
            else:
                failures.append((fname, info))
            elapsed = time.time() - t0
            rate = bytes_done / elapsed / (1024**2) if elapsed > 0 else 0
            print(f"[{completed:3d}/{len(files)}] {status:4s} {fname} {size/(1024**2):.0f}MB elapsed={elapsed:.0f}s avg={rate:.1f}MB/s", flush=True)

    elapsed = time.time() - t0
    print(f"\n[pcloud] done {bytes_done/(1024**3):.2f}GiB in {elapsed:.0f}s ({bytes_done/elapsed/(1024**2):.1f}MB/s avg)", flush=True)
    if failures:
        print(f"[pcloud] FAILED {len(failures)} files:", flush=True)
        for f, err in failures:
            print(f"  {f}: {err}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
