# 08. 원본 repo 의 코드 / 문서 문제점

reproduction 과정에서 발견한 원본 [MCZhi/GameFormer](https://github.com/MCZhi/GameFormer) repo 의 issue. 각 항목은 (1) 무엇이 문제인지, (2) 어떻게 해결했는지, (3) upstream 에 PR 가능 여부를 정리.

## 1. 일관성 부재 — `data_process.py` 의 unknown map type 처리

### 문제

`open_loop_planning/data_process.py:62` 와 `interaction_prediction/data_process.py:71` 가 같은 logic (WOMD map feature 의 unknown oneof 처리) 인데 다르게 구현됨:

```python
# open_loop_planning/data_process.py:62 — strict
else:
    raise TypeError

# interaction_prediction/data_process.py:71 — permissive
else:
    continue
```

### 영향

WOMD v1.2.x 부터 새 map feature type (`driveway`) 추가. open_loop preprocess 가 첫 driveway scenario 에서 즉시 abort. interaction preprocess 는 silently skip — 결과 OK.

### 해결

`open_loop_planning/data_process.py` 도 `continue` 로 통일 (이 reproduction 의 patch 1).

### upstream PR 가능성

높음. upstream maintainer 가 한쪽만 patch 한 것으로 보임 — single-line 수정으로 통일 가능. 단 upstream 이 maintained 되지 않을 가능성 있음 (last commit 이 오래됨).

## 2. torch 2.x DDP launcher 비호환

### 문제

`interaction_prediction/train.py` 의 argparse:
```python
parser.add_argument("--local_rank", type=int)  # underscore only
```

torch 2.x 의 `torch.distributed.launch` 는 child process 에 `--local-rank=N` (hyphen) 전달. argparse 는 underscore/hyphen 자동 변환 안 함 → `unrecognized arguments` 에러.

### 영향

torch 1.x 만 동작. torch 2.x + cu118 같은 최신 환경 (sm_89, sm_90 GPU) 에서 사용 불가.

### 해결

argparse 에 alias 추가:
```python
parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
```

(이 reproduction 의 patch 2)

### upstream PR 가능성

높음. backward compatible (torch 1.x 의 underscore 도 그대로 받음). 1줄 변경.

## 3. Dataloader fd limit 초과 위험

### 문제

torch 의 default `multiprocessing` 전략은 `file_descriptor` — DataLoader worker 가 fork 될 때 worker 간 tensor 공유에 fd 사용. worker 수 ↑ → fd 누적 → `ulimit -n` 한계 초과 → "OSError: [Errno 24] Too many open files".

원본 코드는 어디에도 sharing strategy 명시 안 함 → default 사용. cloud pod 의 default `ulimit -n=1024` 환경에서 worker=8 이상 사용 시 발생.

### 영향

local docker (ulimit 보통 1024+) 또는 high-spec workstation 에서는 안 보이는 issue. cloud pod 으로 옮기면 발견.

### 해결

`open_loop_planning/train.py` 와 `interaction_prediction/train.py` 의 import 직후:
```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

(이 reproduction 의 patch 3)

### upstream PR 가능성

가능. backward compatible (file_system 전략은 fd 대신 임시 파일). 단 어떤 환경 (low-fd cloud) 에서 문제 발생하는지 PR 설명에 포함 필요.

## 4. Hard-coded 학습 산출물 경로

### 문제

train.py 의 logging 코드:
```python
log_path = f"./training_log/{args.name}/"
```

cwd-relative 로 hardcode. cloud pod 에서 다른 path 에 학습 log 저장 필요한 경우 (e.g., `/workspace/logs/{hostname}/runs/`) 코드 수정 필요.

### 해결

env var 로 외부화:
```python
log_home = os.environ.get('TRAINING_LOG_HOME', './training_log')
log_path = f"{log_home}/{args.name}/"
```

(이 reproduction 의 patch 4)

### upstream PR 가능성

가능. backward compatible (env 미지정 시 원본 동작 그대로).

## 5. `interaction_prediction/train.sh` 의 placeholder path

### 문제

```bash
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    train.py \
    # specify your own args:
    --batch_size=16 \
    --train_set=path_to_trainset \         # ← placeholder
    --valid_set=path_to_valset \           # ← placeholder
    --name=exp_log_name \
    --workers=8 \                            # ← trailing backslash with no continuation
```

placeholder (`path_to_trainset`, `path_to_valset`) 를 user 가 직접 수정해야 함. 게다가 마지막 줄 `--workers=8 \` 에 trailing `\` 가 있는데 다음 line 이 비어있음 — bash 가 다음 빈 line 을 continuation 으로 해석해서 silently 동작 (혹은 error).

### 영향

documentation 부족 — README 에 "specify the processed paths for `--train_set` and `--valid_set`" 라고만 적혀있음. 처음 사용자는 placeholder 를 모르고 그대로 실행 → "FileNotFoundError: path_to_trainset/*" 에러.

### 해결

이 reproduction 은 train.sh 미사용. 대신 wrapper script (`scripts/local/06-open_loop_train.sh`) 가 env var 로 path 받음:
```bash
TRAIN_SET="${DATASET_HOME}/processed/open_loop/${TRAIN_SPLIT}"
```

cloud 학습은 `scripts/cloud/train/pod-train-from-scratch-*.sh` 가 동일 패턴.

### upstream PR 가능성

가능. train.sh 를 env var 받는 형태로 정리 + README 보완. 단 user-facing 변경이라 upstream 의 design 의도 확인 필요.

## 6. distributed preprocess 미지원

### 문제

`data_process.py` (둘 다) 가 single pod 에서 multiprocessing.Pool 로만 동작. multi-pod 분산 preprocess 위한 shard subset 처리 옵션 없음.

```python
data_files = sorted(glob.glob(load_path + '/*'))
if args.use_multiprocessing:
    with mp.Pool(args.processes) as p:
        p.map(multiprocessing, data_files)
```

### 영향

interaction full set (1000 shard, ~425 GB raw → ~870 GB processed) 을 single pod 에서 처리 시 8+ 시간 소요. multi-pod 분산하려면 코드 수정 필요.

### 해결

이 reproduction 의 추가 patch — `--shard_range start:end` 옵션 추가:
```python
parser.add_argument('--shard_range', type=str, default='', help='start:end (Python slice)')
...
if args.shard_range:
    s, e = map(int, args.shard_range.split(':'))
    data_files = data_files[s:e]
```

8 pod 에 shard 0:125, 125:250, ..., 875:1000 분배 → ~1 시간 마무리.

### upstream PR 가능성

높음. additive change (옵션 미지정 시 원본 동작 그대로).

## 7. checkpoint 명명 — open_loop vs interaction inconsistency

### 문제

두 학습의 checkpoint 명명 규칙이 다름:

| 학습 | 명명 | 예 |
| --- | --- | --- |
| open_loop_planning | `predictor_{epoch}_{val_metric}.pth` | `predictor_1_0.8345.pth` |
| interaction_prediction | `epochs_{epoch}.pth` | `epochs_0.pth` |

open_loop 은 val_metric 포함 (정보 풍부) 이지만 string formatting 이 복잡. interaction 은 simple. 일관성 없음.

### 영향

checkpoint loader (학습 재개 / inference) 가 두 학습 별로 다른 logic 필요. 통합 자동화 시 추가 작업.

### 해결

이 reproduction 은 명명 규칙 보존 (upstream 호환성). 단 wrapper script (`checkpoint-uploader.sh`) 가 glob `*.pth` 로 둘 다 처리.

### upstream PR 가능성

medium. user-facing 변경 — naming convention 통일 시 기존 사용자의 script 가 깨질 수 있음.

## 8. open-loop dataset sampling 의 비결정성

### 문제

paper Sec 4.2.2: "We randomly select 10,000 20-second scenarios". 그러나 코드 어디에도 sampling random seed 가 fix 되어 있지 않음. data_process.py 가 매 실행 시 다른 sample 선택 가능 (glob ordering + filter 결과 변동).

### 영향

reproduction 의 학습 sample 이 paper 의 실제 sample 과 일치하지 않을 수 있음. 단 통계적으로 비슷한 distribution 이라 학습 결과는 ±5% 영향 정도로 추정.

### 해결

이 reproduction 은 그대로 사용 (paper 와 같은 ambiguity 받아들임). full reproducibility 필요 시 sampling seed 추가 patch 필요.

### upstream PR 가능성

가능. data_process.py 에 `--sample_seed` 옵션 추가.

## 9. open_loop train_log.csv 의 NaN

### 문제 (잠재)

WOMD v1.1 + open_loop 의 `training_20s` 일부 scenario 에 annotation 결손 → train_loss 가 NaN 발생 후 모든 후속 step 도 NaN.

### 영향

학습 fail (loss NaN → backward NaN → optimizer step 후 weights 가 모두 NaN).

### 해결

이 reproduction 은 WOMD v1.2.1 사용으로 회피. (`docs/03-patches.md` patch 3 의 dataset 변경)

### upstream PR 가능성

낮음. WOMD version 변경은 user 의 책임 (paper 도 v1.1 사용). 단 README 에 "v1.2+ 권장" 노트 추가는 가능.

## 10. README 의 dataset 안내 부족

### 문제

원본 README:
> Download the Waymo Open Motion Dataset v1.1.

v1.1 만 명시. v1.2+ 의 driveway type 호환성 / training_20s annotation 결손 등 trade-off 안 적힘.

### 해결

이 reproduction 의 README 에는:
- WOMD v1.2.1 사용 (training_20s 결손 issue 회피)
- driveway type 호환 patch 적용
- gcloud auth 방법 (service account 거부 → user OAuth)

### upstream PR 가능성

가능. README 에 dataset version note + 권장 patch 안내 PR.

## 11. 정리 — patch 분류

| patch | 목적 | upstream PR 가능 | 본 reproduction 의 처리 |
| --- | --- | --- | --- |
| 1. driveway 호환 | 환경 호환성 | yes | applied |
| 2. --local-rank alias | torch 2.x 호환 | yes | applied |
| 3. fd sharing strategy | 안정성 | yes | applied |
| 4. TRAINING_LOG_HOME | configurability | yes | applied |
| 5. train.sh placeholder | usability | yes | wrapper script 로 우회 |
| 6. --shard_range 옵션 | 분산 preprocess | yes | applied (additive) |
| 7. checkpoint naming inconsistency | usability | medium | 보존 (upstream 호환성) |
| 8. sampling seed | reproducibility | yes | unfixed (paper 와 같은 ambiguity 수용) |
| 9. v1.1 NaN | dataset version | no (user choice) | v1.2.1 사용 으로 회피 |
| 10. README dataset note | doc | yes | 본 reproduction 에 명시 |

총 6 개 patch 가 적용됨 (1, 2, 3, 4, 6, 10), 1 개는 wrapper 로 우회 (5), 1 개는 dataset 변경으로 회피 (9), 2 개는 미수정 (7, 8).

upstream contribution 가능 patch: **1, 2, 3, 4, 6**.

## 12. 결론

원본 GameFormer repo 는 paper architecture / 학습 logic 측면에서 무결. 발견된 이슈는 모두 환경 호환성 / 운영 편의성 관련. 핵심 코드 (`model/`, train logic) 는 변경 없이 그대로 reproduction 가능.

단 cloud GPU 환경에서 본격 학습 시:
- DDP launcher 호환 (patch 2)
- DataLoader fd 한계 (patch 3)
- 분산 preprocess (patch 6)

위 3 개는 사실상 필수 — upstream 에 contribute 하면 다른 사용자의 reproduction 이 한결 쉬워짐.
