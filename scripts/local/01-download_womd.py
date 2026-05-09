#!/usr/bin/env python3
"""WOMD scenario tfrecord download via google-cloud-storage.

cloud (ADC) + host (gcloud auth login + ADC) 둘 다 동작.
gsutil 이 user OAuth ADC 거부하는 issue 우회 — python google-cloud-storage 사용.
"""

import argparse
import os
import sys
import time

from google.cloud import storage


def main() -> int:
    p = argparse.ArgumentParser(description="WOMD shard downloader (google-cloud-storage)")
    p.add_argument("--version", default=os.environ.get("WOMD_VERSION", "1_2_1"))
    p.add_argument("--subset", default=os.environ.get("WOMD_SUBSET", "training_20s"))
    p.add_argument("--shards", default=os.environ.get("WOMD_SHARDS", "2"),
                   help="integer or 'all'")
    p.add_argument("--dest", default=os.environ.get("WOMD_DEST", "./data/raw"),
                   help="local destination root (subset directory will be created underneath)")
    p.add_argument("--project", default=os.environ.get(
        "GCP_PROJECT", "influential-bit-427111-u9"
    ), help="GCP project for quota (any project where ADC user has access)")
    args = p.parse_args()

    bucket_name = f"waymo_open_dataset_motion_v_{args.version}"
    prefix = f"uncompressed/scenario/{args.subset}/"
    target = os.path.join(args.dest, args.subset)
    os.makedirs(target, exist_ok=True)

    print(f"version={args.version} subset={args.subset} prefix={prefix}")
    print(f"target={target}")
    print(f"project={args.project}")

    client = storage.Client(project=args.project)
    bucket = client.bucket(bucket_name)

    shards = [b for b in bucket.list_blobs(prefix=prefix) if ".tfrecord" in b.name]
    print(f"available shards: {len(shards)}")

    if args.shards == "all":
        chosen = shards
    else:
        n = int(args.shards)
        chosen = shards[:n]

    for blob in chosen:
        fname = blob.name.split("/")[-1]
        dst = os.path.join(target, fname)
        if os.path.exists(dst) and os.path.getsize(dst) == blob.size:
            print(f"skip (exists): {fname}")
            continue
        t0 = time.time()
        blob.download_to_filename(dst)
        sz_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"downloaded {fname}  {sz_mb:.1f} MB  {time.time() - t0:.1f}s")

    print(f"done -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
