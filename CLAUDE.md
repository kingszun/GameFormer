## GameFormer (project CLAUDE.md)

ICCV'23 GameFormer reproduction — torch 2.3.1+cu118 stack, container 기반 (local 3060 + cloud H200/H100/4090 호환).

### 작업 컨벤션

- 한국어 + 기술 용어는 영어 (한글 음차 금지)
- bold (`**...**`), emoji 사용 금지
- file: lowercase, segment간 `-`, 단어간 `_` (예: `02-build_image.sh`)
- 시간 표기: `yy-mm-dd-hh:mm:ss`
- 설계 후 구현 (design-first); 구현 중 설계 변경 필요 시 일단 정지 → 설계 update → review → 재개

### Rules

추가 규칙은 `.claude/rules/` 하위 파일에 정의되어 있다.

- `kck-jira-workflow.md` — jira ticket workflow

### Jira

- project key: `KAK` (https://kingszun.atlassian.net/jira/software/projects/KAK)
- name: kakao
- 모든 작업은 KAK 의 ticket 으로 등록 후 commit message 에 `[KAK-N]` 포함
- 자세한 규칙은 `.claude/rules/kck-jira-workflow.md` 참조

### 진입 시 먼저 읽을 것

- `docs/00-overview.md` — 프로젝트 목표, 진행 상황, 디렉토리 구조
- `docs/01-environment.md` — Docker/compose stack, env var, hardware compat
- `docs/02-workflow.md` — 단계별 실행 흐름 (host vs container)
- `docs/03-patches.md` — upstream code 변경 사항 (driveway, --local-rank)
- `docs/04-smoke_results.md` — 3060에서 검증된 결과 (open_loop + interaction 모두 1 epoch 통과)
- `docs/05-cloud_plan.md` — RunPod 진행 계획
- `scripts/README.md` — script 목록과 실행 순서

### 핵심 사실

- Image: `docker.io/kingszun/gameformer:cu118-py310-torch2.3.1` (5949 MB compressed, public). base `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`, uv venv at `/opt/venv/gameformer` (KAK-16). sshd / gcloud / venv 모든 dep 내장 — pod 에서 추가 install 없이 즉시 사용.
- 코드는 image에 포함 안 됨 — host repo를 mount 또는 cloud pod에서 git clone.
- `data` 는 host symlink (예: `data -> /mnt/e/datasets/womd`). compose가 host에서 resolve 후 container에 bind-mount.
- WOMD bucket: `gs://waymo_open_dataset_motion_v_1_2_1/` (us-central1).
- Waymo license + `gcloud auth login` (host) 필요. cloud pod 인증은 host의 user OAuth ADC json (`~/.config/gcloud/application_default_credentials.json`) 복사 — service account는 Waymo bucket ACL 거부 (26-05-08 검증).

### 자주 쓰는 command

container shell:
```
docker compose exec gameformer bash
```

GPU 상태:
```
docker compose exec gameformer nvidia-smi
```

전체 흐름 (smoke):
```
WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash scripts/local/01-download_womd.sh
bash scripts/local/02-build_image.sh
bash scripts/local/03-up.sh
bash scripts/local/04-smoke_test.sh
bash scripts/local/05-open_loop_preprocess.sh
BATCH_SIZE=8 EPOCHS=1 bash scripts/local/06-open_loop_train.sh
```

interaction_prediction (DDP, 직접 호출):
```
docker compose exec gameformer bash -lc "
cd interaction_prediction && python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=28596 train.py \
    --batch_size=4 --workers=2 --training_epochs=1 \
    --train_set=/workspace/GameFormer/data/processed/interaction/train \
    --valid_set=/workspace/GameFormer/data/processed/interaction/train \
    --name=smoke
"
```

### 실수 방지

- WOMD v1.1 사용 금지 — `training_20s` annotation 결손 issue 있음. 항상 v1.2.1.
- training 데이터 받을 때 shard 크기 주의 — `training` subset 1 shard ≈ 440 MB (training_20s의 5배).
- TF + torch 동거 시 `TF_FORCE_GPU_ALLOW_GROWTH=true` env 유지 (compose에 이미 포함).
- `interaction_prediction/train.sh` 의 placeholder path (`path_to_trainset` 등) 채우지 않은 채로 직접 실행 금지 — 우회는 위 직접 호출.
- `git add -A` / `git add .` 금지 — 변경 file 개별 지정.
- commit은 user 명시 지시 시에만. 작업 즉시 `git add` 까지만.

### upstream 호환성 patch

| file | line | 변경 | 사유 |
| --- | --- | --- | --- |
| `open_loop_planning/data_process.py` | 62 | `raise TypeError` → `continue` | WOMD v1.2.1 신규 map type (`driveway`) skip |
| `interaction_prediction/train.py` | 263 | `--local_rank` → `--local_rank, --local-rank` | torch 2.x DDP launcher 호환 |

### cloud (RunPod) 진행 결정

- 1차: 1×4090 smoke ($0.3, 1~2h)
- WOMD 인증: host의 user OAuth ADC json 복사 (`~/.config/gcloud/application_default_credentials.json` → pod로 scp)
- 코드 전달: git push → cloud pod git clone

세부는 `docs/05-cloud_plan.md`.
