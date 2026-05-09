## 07b - training_20s subset 상세 (open_loop planning train)

KAK-51 의 doc. WOMD `training_20s` raw → `processed/open_loop/train/` → `open_loop_planning/train.py` 의 train_epoch 흐름.

전체 overview 는 [`07-data_pipeline.md`](07-data_pipeline.md) 참조.
**raw schema, preprocess 함수, output structure 는 validation 과 동일** — [`07a-open_loop_validation.md`](07a-open_loop_validation.md) 참조 (이 doc 는 차이점만 명시).

---

### 1. raw 의 차이점 (vs validation)

| 항목 | validation | training_20s |
|---|---|---|
| 위치 | `gs://.../scenario/validation/*` | `gs://.../scenario/training_20s/*` |
| shard 수 | 150 | **344** (description 의 252 가 아니라 실측 — KAK-50 에서도 검증) |
| 총 size | 39 GB | 30 GB (1 shard 당 평균 ~87 MB, validation 의 1/3) |
| 용도 | open_loop **eval** | open_loop **train** |
| Scenario length | **9.1초** (91 timestep) | **20초** (200 timestep, 이름 그대로) ← 단, GameFormer 는 91 step 만 사용 |

#### 1.1. training_20s 의 "20s" 의 의미

이름 그대로 **scenario length 가 20초** (200 timestep × 0.1s) — 일반 1.7초/91 timestep 보다 길음. 더 긴 horizon training 가능.

**단 GameFormer script 는 hist_len=11, future_len=50, time_len 까지 사용** — 즉 scenario 의 첫 91 timestep 만 활용. 나머지 109 timestep 은 사용 X.

이유: GameFormer 의 model 이 91 step input 으로 학습되어서 — 더 긴 sequence 처리 X.

추후 long-horizon model 학습 가능성 있음 (paper 에서 future work 으로 언급).

---

### 2. preprocess (validation 과 동일 script)

`open_loop_planning/data_process.py` 의 함수 모두 동일:
1. `build_map(map_features, dynamic_map_states)`
2. `ego_process(sdc_id, timestep, tracks)` — 11 step history
3. `route_process(sdc_id, timestep, ...)` — 1000 waypoint reference
4. `neighbors_process(sdc_id, timestep, tracks)` — top 10 neighbor
5. `map_process(traj, timestep, type)` — 6 lane × 100 point × 16 attribute
6. `ground_truth_process(sdc_id, timestep, tracks)` — 50 future step
7. `normalize_data(...)` — ego frame
8. `np.savez(filename, ...)` — 6 field

처리 단위 동일: per scenario per timestep (5 step interval) → 7 file/scenario.

차이: 단지 `--load_path` 가 `/workspace/data/raw/training_20s` 로 다를 뿐.

---

### 3. output structure (validation 과 동일)

| field | shape | 의미 |
|---|---|---|
| ego | (11, 9) | SDC 11 hist × 9 attr |
| neighbors | (10, 11, 9) | top 10 neighbor |
| map_lanes | (11, 6, 100, 16) | (1+10) × 6 lane × 100 point × 16 attr |
| map_crosswalks | (11, 4, 100, 3) | (1+10) × 4 crosswalk × 100 point × 3 |
| ref_line | (1000, 5) | 1000 waypoint |
| gt_future_states | (11, 50, 5) | (1+10) × 50 future × 5 attr |

file size: 512,088 bytes uniform.

#### 추정 file 수

- 344 shard × 평균 ~270 scenario × 7 timestep ≈ **650,000 file**
- 이전 KAK-9 smoke 시 252 shard 추정 했지만 실제 344 → file 수 더 많음

---

### 4. 학습 사용 — `open_loop_planning/train.py` 의 `train_epoch`

#### 4.1. valid_epoch 와의 차이

| 항목 | valid_epoch | train_epoch |
|---|---|---|
| model mode | `model.eval()` | `model.train()` |
| dropout / batchnorm | inference mode | training mode |
| gradient | `torch.no_grad()` | enabled |
| optimizer | X | `optimizer.zero_grad()` + `loss.backward()` + `optimizer.step()` |
| grad clip | X | `nn.utils.clip_grad_norm_(model.parameters(), 5)` |
| 결과 사용 | metric 측정 + best model 선택 | model parameter update |

#### 4.2. train_epoch logic

```python
def train_epoch(data_loader, model, optimizer):
    epoch_loss = []
    epoch_metrics = []
    model.train()  # dropout, batchnorm training mode

    with tqdm(data_loader, desc="Training", unit="batch") as epoch:
        for batch in epoch:
            inputs = {
                'ego_state': batch[0].to(args.device),
                'neighbors_state': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'ref_line': batch[4].to(args.device)
            }
            ego_future = batch[5][:, 0].to(args.device)
            neighbors_future = batch[5][:, 1:].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # gradient 계산 + parameter update
            optimizer.zero_grad()
            level_k_outputs = model(inputs)
            loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
            plan = results[:, 0]
            prediction = results[:, 1:]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient explosion 방지
            optimizer.step()

            metrics = motion_metrics(plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_loss.append(loss.item())
            epoch_metrics.append(metrics)

    return mean_loss, [plannerADE, plannerFDE, predictorADE, predictorFDE]
```

#### 4.3. hyperparameter (KAK-34 v0.3, B option)

| param | value | 의미 |
|---|---|---|
| batch_size | 64 | 1 batch 당 64 sample |
| learning_rate | 2e-4 | sqrt scaling (paper 의 batch 32 + lr 1e-4 → batch 64 + lr √2 × 1e-4 ≈ 1.4e-4. v0.3 에서 2e-4 로 결정) |
| training_epochs | 20 | 1 epoch sanity → 19 epoch 추가 |
| optimizer | Adam (default) | train.py 안에서 정의 |
| scheduler | MultiStepLR (paper milestones) | 일정 epoch 마다 lr decay |
| grad clip | 5.0 | gradient norm clip |
| workers | (CLI) | DataLoader num_workers |

#### 4.4. file_descriptor sharing strategy (KAK-30 patch)

```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

worker 수가 많을수록 file_descriptor sharing 시 fd 누적 → ulimit -n 초과. file_system 전략 으로 전환하면 worker 간 tensor 공유에 fd 대신 임시 파일 사용 → fd 한계 무관.

---

### 5. 학습 흐름 (전체 epoch)

```python
for epoch in range(training_epochs):
    # train
    train_loss, train_metrics = train_epoch(train_loader, model, optimizer)
    # valid (07a 의 valid_epoch 호출)
    valid_loss, valid_metrics = valid_epoch(valid_loader, model)
    
    # log
    logging.info(f"Epoch {epoch}: train_loss={train_loss}, valid_loss={valid_loss}")
    logging.info(f"  train: plannerADE={train_metrics[0]}, plannerFDE={train_metrics[1]}")
    logging.info(f"  valid: plannerADE={valid_metrics[0]}, plannerFDE={valid_metrics[1]}")
    
    # scheduler step
    scheduler.step()
    
    # save checkpoint
    torch.save(model.state_dict(), f"{run_dir}/predictor_{epoch}.pth")
```

---

### 6. 현재 진행 상태

| 단계 | status |
|---|---|
| raw download | RTX 3090 chain v2 의 prefetch 진행 중 (training_20s 339/344, 99%) |
| preprocess | pending (chain v2 step 4 — valid 끝나야) |
| output sync to S3 | preprocess 끝나면 |
| 학습 train_epoch | preprocess 끝나면 H100 start + train.py 의 매 epoch 별 자동 |

ETA:
- chain v2 valid 끝 (~20분 더)
- train preprocess (252 → **344** shard, 130 worker) — 약 60분
- 총 약 80분 후 open_loop 학습 trigger

---

### 7. paper baseline 비교

paper Table 1 (open_loop planning):
- plannerADE / plannerFDE
- predictorADE / predictorFDE

acceptance: paper 값 대비 ±10~15% margin (KAK-34 v0.3).

---

### 8. 관련

- **KAK-51** (이 sub-task)
- KAK-49 (parent story)
- KAK-50 (validation, 같은 preprocess script)
- KAK-41 (학습 ticket)
- code: `open_loop_planning/data_process.py`, `open_loop_planning/train.py`, `model/GameFormer.py`, `utils/open_loop_train_utils.py`
- doc: [`07-data_pipeline.md`](07-data_pipeline.md), [`07a-open_loop_validation.md`](07a-open_loop_validation.md), [`02-workflow.md`](02-workflow.md), [`06-full_training_plan.md`](06-full_training_plan.md)
