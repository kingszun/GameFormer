## 00 - overview

ICCV'23 GameFormer (https://github.com/MCZhi/GameFormer) repo를 현대 환경에서 재현하기 위한 작업 기록.

### 목표

- 원본 paper의 학습 path 재현 — interaction_prediction (multi-agent joint prediction) + open_loop_planning.
- 3060 (sm_86, 12 GB) local에서 환경/코드 검증 → H200 또는 다른 cloud GPU에서 본격 학습.
- container 기반 — 동일 image를 local과 cloud 양쪽에서 그대로 재사용.

### 원본 stack vs 현 stack

| 항목 | 원본 (`requirements.txt`) | 현 stack | 변경 사유 |
| --- | --- | --- | --- |
| PyTorch | 1.12.0+cu116 | 2.3.1+cu118 | cu116 wheel은 sm_86까지만 native, cu118 wheel은 sm_86 + sm_90 native 포함 |
| Python | 3.8 | 3.10 | torch 2.3.1+cu118 wheel default |
| CUDA | 11.6 | 11.8 | sm_90 (H200) native 지원 |
| TensorFlow | 2.11 (waymo dep) | 2.11 동일 | waymo-open-dataset-tf-2-11-0가 강제 |
| Launcher | `torch.distributed.launch` | 동일 (deprecated warning만) | torchrun 전환은 차후 |

torch 2.x에서 코드 호환성 문제는 없음 — model code는 plain transformer (Linear, MHA, LayerNorm) + DDP + LSTM. PyG/torch_scatter 같은 custom CUDA extension 사용 안 함.

### 현재 진행 상황

| 단계 | 상태 | 비고 |
| --- | --- | --- |
| Docker image build (local) | done | `gameformer:cu118-py310-torch2.3.1`, 5 GB |
| Docker Hub push | done | `docker.io/kingszun/gameformer:cu118-py310-torch2.3.1` |
| WOMD subset download (host) | done | training_20s 2 shards (164 MB), training 2 shards (886 MB) |
| open_loop_planning preprocess | done | 3610 .npz, 1.8 GB |
| open_loop_planning train (1 epoch, 3060) | done | loss 91 → 48 |
| interaction_prediction preprocess | done | 8518 .npz, 2.2 GB |
| interaction_prediction train (1 epoch, 3060 single-GPU DDP) | done | loss 22000 → 109, val_loss 109 |
| Cloud (RunPod) 1×4090 smoke | pending | 다음 단계 |

### 디렉토리 구조

```
GameFormer/
├── Dockerfile, compose.yaml, .env.example, .dockerignore, .gitignore
├── data -> /mnt/e/datasets/womd          (host symlink)
│   ├── raw/{training_20s,training}/
│   └── processed/{open_loop,interaction}/{train,valid}/
├── scripts/                              (실행 자동화)
│   ├── 01-download_womd.sh ~ 99-down.sh
│   └── README.md
├── docs/                                 (이 폴더)
├── interaction_prediction/, open_loop_planning/, model/, utils/   (upstream)
└── requirements.txt                      (upstream — Docker 안 쓰면 참고용)
```

### 다음 작업 지점

`docs/05-cloud_plan.md` 참조. 결정 완료 사항:
- 학습 범위: 1×4090 smoke ($0.3, 1~2h)
- WOMD 인증: host의 user OAuth ADC json 복사 (service account는 Waymo bucket ACL 거부, 26-05-08 검증)
- 코드 전달: git push → cloud pod git clone
