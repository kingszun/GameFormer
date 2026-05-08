## 03 - patches

upstream GameFormer 코드 대비 변경 사항. 모두 환경 호환성 패치이며 모델/학습 logic은 건드리지 않음.

### patch 1 — open_loop_planning/data_process.py: WOMD v1.2.1 신규 map type 호환

|  |  |
| --- | --- |
| 파일 | `open_loop_planning/data_process.py:62` |
| 원본 | `else: raise TypeError` |
| 변경 | `else: continue` |

배경:
- 원본 코드는 WOMD v1.1 시점 작성. map feature type을 `lane / road_line / road_edge / stop_sign / crosswalk / speed_bump` 6종으로 가정.
- WOMD v1.2.x에서 `driveway` 등 신규 map feature type 추가 → 알 수 없는 oneof 만남 → `raise TypeError` 로 process 중단.
- `interaction_prediction/data_process.py:71` 은 이미 같은 위치를 `continue` 로 처리해둠 (upstream maintainer가 한쪽만 patch한 것으로 추정).
- 동일 패턴으로 통일.

영향:
- 알 수 없는 type을 silently skip — driveway 등은 사용 안 하던 feature이므로 모델 입력에 영향 없음.

### patch 2 — interaction_prediction/train.py: torch 2.x DDP launcher 호환

|  |  |
| --- | --- |
| 파일 | `interaction_prediction/train.py:263` |
| 원본 | `parser.add_argument("--local_rank", type=int)` |
| 변경 | `parser.add_argument("--local_rank", "--local-rank", type=int, default=0)` |

배경:
- torch 1.x의 `torch.distributed.launch` 는 child process에 `--local_rank=N` (underscore) 전달.
- torch 2.x 의 `torch.distributed.launch` 와 `torchrun` 은 `--local-rank=N` (hyphen) 전달.
- argparse는 underscore/hyphen 자동 변환 안 함 → torch 2.x launcher 사용 시 `unrecognized arguments: --local-rank=0` 에러.
- alias 추가로 양쪽 launcher 호환 + `default=0` 으로 single-process 직접 실행도 가능.

영향:
- 코드 logic 변화 없음. argparse alias만 추가.

### patch 3 — open_loop_planning/train.py: file_system sharing strategy

|  |  |
| --- | --- |
| 파일 | `open_loop_planning/train.py:13~16` |
| 변경 | import 직후 `torch.multiprocessing.set_sharing_strategy('file_system')` 추가 |
| ticket | KAK-30 |

배경:
- DataLoader 가 `num_workers=os.cpu_count()` 로 worker spawn. cloud GPU pod (4090, 128 core) 에서 worker 128개.
- default sharing strategy `file_descriptor` 는 worker 간 tensor 공유에 fd 사용 → pod 의 ulimit -n 1024 (soft) 한계 초과.
- 학습 41% 지점에서 `RuntimeError: Too many open files. Communication with the workers is no longer possible` 발생.
- 3060 host (12 core) 에서는 worker 12개라 1024 안에 들어가서 통과 — cloud 환경에서만 발견.

영향:
- file_system 전략은 worker 간 tensor 공유에 fd 대신 임시 파일 (`/dev/shm` 또는 `$TMPDIR`) 사용.
- fd 한계 무관, 추가 의존성 없음, batch 8 size 에선 I/O overhead 무시 가능.
- 3060 환경에도 동일하게 안전 (file_system 도 정상 동작).

### patch 4 — train.py log_path env 외부화

|  |  |
| --- | --- |
| 파일 | `open_loop_planning/train.py:111~117`, `interaction_prediction/train.py:1~18, 148~155` |
| 변경 | `log_path = f"./training_log/{args.name}/"` → `log_home = os.environ.get('TRAINING_LOG_HOME', './training_log')` 후 `f"{log_home}/{args.name}/"` |
| ticket | KAK-33 |

배경:
- 산출물 (training_log + checkpoint) 이 cwd 기준 상대 path 라 cloud pod container disk 에 저장 → pod destroy 시 사라짐 (KAK-9 진행 시 scp 회수).
- 두 환경 (host docker compose vs cloud pod) 의 산출물 경로를 분리해야 — host: `./training_log/`, pod: `/workspace/data/runs/` (volume).
- env 로 외부화하면 train.py 수정 없이 script wrapper 에서 분기 가능.

영향:
- env 미지정 시 default `./training_log/` 그대로 — 기존 host 동작 무변경.
- `interaction_prediction/train.py` 도 동일 patch + 동시에 KAK-30 의 `set_sharing_strategy('file_system')` 같이 적용 (cloud DDP 환경에서 동일 fd 한계 위험).

### patch 5 — interaction_prediction/train.py: file_system sharing strategy

|  |  |
| --- | --- |
| 파일 | `interaction_prediction/train.py:1~18` |
| 변경 | `torch.multiprocessing.set_sharing_strategy('file_system')` 추가 |
| ticket | KAK-33 |

KAK-30 의 open_loop 와 동일 — high-core pod 환경에서 fd 한계 초과 방지. cloud DDP 학습 시 effective.

### 미적용 후보 patch (현재 불필요)

- `torch.distributed.launch` → `torchrun` 전환:
  - torch 2.x에서 `torch.distributed.launch` 는 deprecated이지만 동작은 함.
  - 전환 시 `train.sh` script 수정 필요 (`python -m torch.distributed.launch` → `torchrun`).
  - 환경변수 `LOCAL_RANK` 자동 주입 — argparse `--local_rank` 의존 코드는 그대로 동작.
  - 차후 deprecated 완전 제거되면 그때 전환.

- `interaction_prediction/train.sh` 의 placeholder path 채우기:
  - `--train_set=path_to_trainset` 등 placeholder가 그대로 들어감 → 직접 실행 시 에러.
  - 현재는 우회 (직접 `python -m torch.distributed.launch ... train.py ...` 호출).
  - 차후 `scripts/06-interaction_train.sh` 에 wrapping 예정.

### 회귀 검증

patch 1, 2 모두 적용 후:
- open_loop_planning: 1 epoch 학습 통과 (loss 91 → 48)
- interaction_prediction: 1 epoch DDP 학습 통과 (loss 22000 → 109, val_loss 109)
