## 07d - training subset 상세 (interaction prediction train)

KAK-53 의 doc. WOMD `training` raw → `processed/interaction/train/` → `interaction_prediction/train.py` 의 train_epoch (DDP 4 GPU) 흐름.

전체 overview 는 [`07-data_pipeline.md`](07-data_pipeline.md) 참조.
**raw schema, preprocess 함수, output structure 는 validation_interactive 와 동일** — [`07c-interaction_validation.md`](07c-interaction_validation.md) 참조 (이 doc 는 차이점 + train 관련 detail 만).

---

### 1. raw 의 차이점 (vs validation_interactive)

| 항목 | validation_interactive | training |
|---|---|---|
| 위치 | `gs://.../scenario/validation_interactive/*` | `gs://.../scenario/training/*` |
| shard 수 | 150 | **1000** (가장 큼) |
| 총 size | 38 GB | **425 GB** |
| 1 shard size | 평균 ~250 MB | 평균 ~425 MB (validation_interactive 보다 1.7배) |
| 용도 | interaction **eval** | interaction **train** |
| Scene 형태 | **interactive only** (objects_of_interest marked) | **일반 + interactive 모두** (objects_of_interest 가 비어있을 수도) |

#### 1.1. training 의 특성

- **WOMD 의 main training set** — 일반 driving + interaction 모두
- 1000 shard × 평균 ~290 scenario ≈ **290,000 scenario** (가장 큼)
- 일부 scene 만 interactive (objects_of_interest 채워져 있음)
- 나머지는 random pair augmentation 으로 학습 데이터 확보

---

### 2. preprocess (validation_interactive 와 동일 script)

`interaction_prediction/data_process.py` 의 함수 모두 동일:
1. `build_map(...)` — map static + dynamic
2. `interactive_process(tracks_list, interesting_ids, tracks)` — pair 식별 (interesting + random)
3. `ego_process(sdc_ids, tracks)` — 2 agent × 11 hist
4. `neighbors_process(sdc_ids, tracks)` — top 32 neighbor
5. `map_process(traj)` — 6 lane × 300 point × 17 attr (per agent)
6. `ground_truth_process(sdc_ids, tracks)` — 2 agent × **80 future step**
7. `normalize_data(...)` — ego[0] frame 으로 변환
8. `np.savez(filename, ...)` — 9 field

처리 단위: per scenario per pair = 1 .npz file. 1 scenario 평균 5~10 pair.

차이: 단지 `--load_path` 가 `/workspace/data/raw/training` 으로 다를 뿐.

#### 2.1. 추가 challenge

- raw 425 GB → cross-region download (EU-CZ-1 ↔ US-IL-1) 필요
- container disk **1204 GB 로 resize** (KAK-41 chain v2)
- preprocess 시간: 130 worker, 1000 shard / 130 = 7.7 batch × 47분 ≈ **6시간**

---

### 3. output structure (validation_interactive 와 동일)

| field | shape | 의미 |
|---|---|---|
| ego | (2, 11, 9) | 2 agent × 11 hist × 9 |
| neighbors | (32, 11, 9) | 32 neighbor |
| map_lanes | (2, 6, 300, 17) | 2 × 6 lane × 300 point × 17 |
| map_crosswalks | (2, 4, 100, 3) | 2 × 4 crosswalk × 100 × 3 |
| object_type | (2,) | 2 agent type |
| region_6 | (6, 2) | region anchor |
| object_index | (2,) | 2 agent id |
| current_state | (4,) | ego[0] pose |
| gt_future_states | (2, 80, 5) | 2 × 80 future × 5 |

file 명 pattern: `<scenario_id>_<sdc_id1>_<sdc_id2>_{interest|r}.npz`

#### 3.1. 추정 file 수

- 1000 shard × 평균 ~290 scenario × 평균 ~6 pair ≈ **1,740,000 file**
- 실제 측정 시 update 필요

---

### 4. 학습 사용 — `interaction_prediction/train.py` 의 `train_epoch` (DDP 4 GPU)

#### 4.1. valid_epoch 와의 차이

| 항목 | valid_epoch (07c) | train_epoch |
|---|---|---|
| model mode | `model.eval()` | `model.train()` |
| dropout | inference | training |
| gradient | `torch.no_grad()` | enabled |
| optimizer | X | `zero_grad()` + `backward()` + `step()` |
| grad clip | X | clip_grad_norm_(5.0) |
| 결과 사용 | metric 측정 | model parameter update |

#### 4.2. DDP setup

```python
# torch.distributed.launch 또는 torchrun 으로 실행
# 각 process 에 LOCAL_RANK env (0, 1, 2, 3) 자동 주입

import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# DDP wrap
model = GameFormer(...).to(rank)
ddp_model = DDP(model, device_ids=[rank])

# DistributedSampler 가 4 GPU rank 별로 batch 분배
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, ...)
```

#### 4.3. KAK-30 patch — `--local-rank` arg

torch 1.x 의 launcher 는 `--local_rank` (underscore) 만 받았지만 torch 2.x 의 launcher 는 `--local-rank` (hyphen) 를 사용. KAK-30 patch 가 둘 다 받게 추가:

```python
parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
```

이걸로 torch 1.x / 2.x launcher 모두 호환.

#### 4.4. train_epoch logic

```python
def train_epoch(rank, world_size, model, train_loader, optimizer, epoch):
    model.train()
    train_sampler.set_epoch(epoch)  # shuffle reproducibility
    
    for batch in train_loader:
        inputs = {
            'ego_state': batch[0].to(rank),
            'neighbors_state': batch[1].to(rank),
            'map_lanes': batch[2].to(rank),
            'map_crosswalks': batch[3].to(rank),
            'object_type': batch[4].to(rank),
            'region_6': batch[5].to(rank),
        }
        gt = batch[8].to(rank)  # (B, 2, 80, 5)
        
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = joint_loss(outputs, gt)
        loss.backward()
        nn.utils.clip_grad_norm_(ddp_model.parameters(), 5)
        optimizer.step()
        
        # DDP 가 자동으로 gradient all-reduce across GPUs
    
    return mean_loss
```

#### 4.5. hyperparameter (KAK-34 v0.3, D option)

| param | value | 의미 |
|---|---|---|
| batch_size (per GPU) | 32 | 1 GPU 당 32 sample |
| effective batch_size | **128** (32 × 4 GPU) | DDP gradient sync 후 |
| learning_rate | **1e-4** | paper 그대로 (batch 32 per GPU 동일하니 lr scaling X) |
| training_epochs | **30** (paper) | 12h 추정 |
| optimizer | Adam | train.py default |
| scheduler | (paper milestones) | epoch 별 decay |
| grad clip | 5.0 | gradient norm clip |
| workers (DataLoader) | (CLI) | num_workers per GPU |

#### 4.6. CLI 명령 (KAK-44 의 plan)

```
cd interaction_prediction && python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=28596 train.py \
    --batch_size=32 --workers=4 --training_epochs=30 \
    --learning_rate=1e-4 \
    --train_set=/workspace/data/processed/interaction/train \
    --valid_set=/workspace/data/processed/interaction/valid \
    --name=full_d
```

`--nproc_per_node=4` → 4 GPU process spawn → 각자 LOCAL_RANK 0~3.

---

### 5. 학습 흐름 (전체)

```python
for epoch in range(training_epochs):
    # train (DDP, all 4 GPU)
    train_loss = train_epoch(rank, world_size, ddp_model, train_loader, optimizer, epoch)
    
    # valid (rank 0 만 또는 모두)
    valid_loss, valid_metrics = valid_epoch(rank, world_size, ddp_model, valid_loader)
    
    if rank == 0:  # logging 은 rank 0 만
        logging.info(f"Epoch {epoch}: train_loss={train_loss}, valid_loss={valid_loss}")
        logging.info(f"  metrics: PlanningADE={valid_metrics[0]}, Collision={valid_metrics[1]}")
        logging.info(f"           minADE={valid_metrics[2]}, minFDE={valid_metrics[3]}, miss={valid_metrics[4]}")
        
        # save checkpoint
        torch.save(ddp_model.module.state_dict(), f"{run_dir}/predictor_{epoch}.pth")
    
    scheduler.step()
    
    dist.barrier()  # 모든 rank sync
```

---

### 6. paper baseline 비교

paper Table 2 (interaction prediction, K=4):
- Planning ADE: **0.8329**
- Collision rate: **0.0198**

paper Table 3 (motion prediction):
- minADE: **0.79**
- minFDE: **1.85**
- miss rate: **0.30**

acceptance: paper 값 대비 ±10~15% margin.

ETA + 비용:
- 30 epoch × ~22 min/epoch (4×H100 추정) = **11h**
- 비용: 4×H100 SXM × $2.99/h × 11h = **$132**

---

### 7. 현재 진행 상태

| 단계 | status |
|---|---|
| raw download | RTX 3090 chain v2 prefetch 진행 중 (training 105/1000, 42 GB / 425 GB, 10%) |
| preprocess | pending (chain v2 step 7 — open_loop 전체 끝나야) |
| output sync to S3 | preprocess 끝나면 (network volume `processed/interaction/train/`) |
| 학습 train_epoch | preprocess 끝나면 4×H100 stock 재검증 + DDP train |

ETA (전체):
- chain v2 valid: ~20분 더
- chain v2 train_20s preprocess: ~60분
- chain v2 training prefetch wait + interaction preprocess: ~6h
- 총 약 **7~8h** 후 interaction 학습 trigger

---

### 8. 관련

- **KAK-53** (이 sub-task)
- KAK-49 (parent story)
- KAK-52 (validation_interactive, 같은 preprocess script)
- KAK-44 (학습 ticket — interaction multi-GPU train, 4×H100 DDP)
- code: `interaction_prediction/data_process.py`, `interaction_prediction/train.py`, `model/GameFormer.py`
- doc: [`07-data_pipeline.md`](07-data_pipeline.md), [`07c-interaction_validation.md`](07c-interaction_validation.md), [`02-workflow.md`](02-workflow.md), [`06-full_training_plan.md`](06-full_training_plan.md)
