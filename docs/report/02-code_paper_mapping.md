# 02. Code ↔ Paper 매핑

paper 의 architecture 와 학습 logic 이 코드 어디에 구현됐는지 line 단위로 매핑.

## 1. 모델 정의 (`model/`)

### 1.1. `model/GameFormer.py`

| paper section | 코드 |
| --- | --- |
| Sec 3.2 Encoder | `class Encoder` (line 5~72) |
| Sec 3.3 Decoder (level k reasoning) | `class Decoder` (line 75~111) |
| Sec 3.1 전체 architecture | `class GameFormer` (line 114~124) |

#### Encoder (line 5~72)

```python
def forward(self, inputs):
    # agent encoding
    encoded_ego = self.ego_encoder(ego)                                     # eq. (1)
    encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in ...]
    encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)

    # map encoding
    encoded_map_lanes = self.lane_encoder(map_lanes)                        # eq. (2)
    encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)

    # attention fusion (per-agent local map)
    for i in range(N):                                                      # eq. (3)
        ...
        fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
        encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
```

- agent encoding: paper eq.(1) — agent type + history trajectory 를 LSTM 으로 256-dim feature 로 encode
- map encoding: paper eq.(2) — lane / crosswalk polyline 을 PointNet 으로 encode
- attention fusion: paper eq.(3) — Transformer encoder 6-layer 로 모든 agent context 의 정보를 융합. **per-agent local frame** (각 agent 의 시점에서 본 map) 을 별도로 encoding

#### Decoder (line 75~111)

```python
def forward(self, encoder_inputs):
    # level 0 (initial guess)
    results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) ...]
    decoder_outputs['level_0_interactions'] = last_level
    decoder_outputs['level_0_scores'] = last_scores

    # level k reasoning
    for k in range(1, self._levels+1):
        results = [interaction_decoder(i, current_states[:, :N], last_level, last_scores, ...)]
        decoder_outputs[f'level_{k}_interactions'] = last_level
        decoder_outputs[f'level_{k}_scores'] = last_scores
```

- level 0: marginal prediction 에 해당 — `InitialDecoder` 로 각 agent 별 M개 modal trajectory + score 출력
- level k (k=1..K): `InteractionDecoder` 로 `last_level` (전 level 의 prediction) 을 받아 interaction 추론 → 새 prediction
- 모든 level 의 출력이 보존되어 deep supervision 으로 학습

### 1.2. `model/modules.py`

| paper concept | 코드 |
| --- | --- |
| Agent encoder (LSTM) | `class AgentEncoder` (line 27~39) |
| Lane encoder (PointNet + position encoding) | `class LaneEncoder` (line 42~80) |
| Crosswalk encoder | `class CrosswalkEncoder` (line 83~91) |
| Future encoder (trajectory → embedding) | `class FutureEncoder` (line 94~120) |
| GMM output (μx, μy, log_σx, log_σy + score) | `class GMMPredictor` (line 123~135) |
| Self-attention transformer block | `class SelfTransformer` (line 138~152) |
| Cross-attention transformer block | `class CrossTransformer` (line 155~169) |
| Initial decoder (level 0) | `class InitialDecoder` (line 172~198) |
| Interaction decoder (level k) | `class InteractionDecoder` (line 201~232) |

#### AgentEncoder (line 27~39)

```python
def forward(self, inputs):
    traj, _ = self.motion(inputs[:, :, :8])      # LSTM hidden over 11-step history
    output = traj[:, -1]                          # last hidden state (agent feature)
    type = self.type_emb(inputs[:, -1, 8].int()) # type embedding (vehicle/ped/cyclist)
    output = output + type                        # combined feature
```

paper Sec 3.2: 8-dim 의 history (x, y, heading, vx, vy, w, l, h) 11 step 을 LSTM 통해 256-dim agent feature 로 변환. agent type 정보를 embedding 으로 더함.

#### InteractionDecoder (line 201~232) — game-theoretic core

```python
def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
    # 1. encode prev-level futures (각 agent 의 M modal future trajectories)
    multi_futures = self.future_encoder(actors[..., :2], current_states)    # (B, N, M, dim)
    futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2) # weighted avg by mode score

    # 2. self-attention 으로 agent 간 interaction 추론
    interaction = self.interaction_encoder(futures, mask[:, :N])             # (B, N, dim)

    # 3. context (encoding) + interaction 결합
    encoding = torch.cat([interaction, encoding], dim=1)
    mask = torch.cat([mask[:, :N], mask], dim=1).clone()
    mask[:, id] = True   # mask the agent's own future from prev level

    # 4. cross-attention decoder + GMM prediction
    query = last_content + multi_futures[:, id]
    query_content = self.query_encoder(query, encoding, encoding, mask)
    trajectories, scores = self.decoder(query_content)
```

- Step 1 (paper eq.(7)): 다른 agent 들의 prev-level prediction (M modal × score) 을 weighted avg 로 압축 → 그 agent 의 "예상되는" 행동 representation
- Step 2 (paper eq.(8)): agent 들 사이의 interaction 을 self-attention 으로 encoding (level k thinking 의 핵심)
- Step 3: 자기 자신의 prev prediction 은 mask out (다른 agent 의 행동 만 본다는 의미)
- Step 4 (paper eq.(9)): 결합된 context 를 query 로 cross-attention → GMM (μ, σ) + score 출력

이 한 단위가 paper 의 "k-th level reasoning" 1 단계.

## 2. 학습 logic

### 2.1. `interaction_prediction/train.py`

| paper section | 코드 |
| --- | --- |
| Sec 3.4 Loss function (multi-level GMM NLL + classification) | `interaction_loss` in `utils/inter_pred_utils.py` |
| Sec 4.2.1 Training setup (DDP, batch 16/GPU × 4 GPU, lr 1e-4, epoch 30) | `train.py:267~284` (argparse defaults) |
| Sec 4.2.1 Evaluation (minADE, minFDE, MR, mAP) | `valid_epoch()` (line 75~110) |
| LR schedule (MultiStepLR milestones=[20,22,24,26,28] gamma=0.5) | `train.py:217~218` |

### 2.2. `open_loop_planning/train.py`

| paper section | 코드 |
| --- | --- |
| Sec 4.2.2 dataset (training_20s 9000+1000 scenarios) | `data_process.py` 의 candidate scene filtering |
| Sec 4.2.2 Training (single GPU, batch 32, lr 1e-4, epoch 20) | `train.py:177~190` (argparse defaults) |
| Validation metrics (planner ADE/FDE, prediction ADE/FDE) | `valid_epoch()` |

## 3. Data preprocessing

### 3.1. `interaction_prediction/data_process.py`

| paper concept | 코드 |
| --- | --- |
| Scene 의 ego/neighbor 추출 (interaction 후보 pair) | `class DataProcess` |
| trajectory + map feature → fixed-length tensor | `process_one_scenario()` |
| valid scene filtering (kalman score 등 quality filter) | `valid_check()` |
| modal points (target endpoint 의 K-mean clustering) | `point_dir/{vehicle,pedestrian,cyclist}_{K}.pkl` 사용 |

### 3.2. `open_loop_planning/data_process.py`

| paper concept | 코드 |
| --- | --- |
| 20s scene 에서 long-horizon trajectory 추출 | `class DataProcess.process_data` |
| reference path (route lane network) 추출 | `process_data` 의 `ref_line` 부분 |
| ego trajectory + neighbor 5 (paper Sec 4.2.2) | `num_neighbors=5` (interaction 의 32 보다 적음) |

## 4. Loss function (paper Sec 3.4)

paper 식:
$$L_{total} = \sum_{k=0}^{K} (L_{traj}^{(k)} + L_{score}^{(k)})$$

- $L_{traj}^{(k)}$: 각 level 의 trajectory 와 GT 의 GMM negative log-likelihood (winner-takes-all — 가장 가까운 mode 만 update)
- $L_{score}^{(k)}$: closest mode 를 positive 로 한 cross-entropy classification

`utils/inter_pred_utils.py` 의 loss function:
```python
# winner-takes-all: 각 GT 에 대해 가장 가까운 mode 선택
distances = torch.norm(pred[..., :2] - gt[..., None, :, :2], dim=-1).mean(-1)  # (B, N, M)
best_mode = distances.argmin(dim=-1)  # (B, N)

# trajectory loss: best mode 의 GMM NLL
nll = -gaussian_log_prob(pred[best_mode], gt)

# score loss: best_mode 를 target 으로 한 CE
score_loss = F.cross_entropy(scores, best_mode)
```

이 winner-takes-all 패턴이 multi-modal prediction 의 mode collapse 를 방지 — 모든 mode 가 동일 trajectory 로 수렴하지 않고 다양성 유지.

## 5. 학습 entry-point flow

```
python -m torch.distributed.launch --nproc_per_node=4 train.py
        │
        ▼
train.py:
  1. argparse → config
  2. seed (fix 3407)
  3. DDP init (init_process_group 'nccl')
  4. data loader (DistributedSampler)
  5. model = GameFormer(modalities=6, neighbors=1, future_len=80, levels=3)
  6. optimizer = AdamW(lr=1e-4, weight_decay=1e-4)
  7. scheduler = MultiStepLR(milestones=[20,22,24,26,28], gamma=0.5)
  8. for epoch in 0..training_epochs:
        train_epoch(): forward → loss → backward → step
        valid_epoch(): metrics on validation (no gradient)
        rank=0 save: train_log.csv + epochs_{epoch}.pth
```

위 흐름에 어떤 model 변경도 없음 — paper 의 학습 setup 을 그대로 사용. 단, 환경 호환성 patch 만 적용 ([04-stack_migration](04-stack_migration.md) 참조).
