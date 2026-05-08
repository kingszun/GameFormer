#!/usr/bin/env bash
# container 내부 GPU + torch + waymo import smoke test
# - 기대 출력: torch 2.3.1 / cu118 / GPU device name / matmul OK / waymo proto OK
# - usage: bash scripts/04-smoke_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

docker compose exec -T gameformer python <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('device name:', torch.cuda.get_device_name(0))
print('cuda runtime:', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version())
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('matmul ok, sum:', y.sum().item())

import tensorflow as tf
print('tf:', tf.__version__)
from waymo_open_dataset.protos import scenario_pb2
print('waymo proto: OK')
PY
