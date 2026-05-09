# 00. 종합 요약

ICCV'23 GameFormer paper 의 학습 path 를 현대 환경 (PyTorch 2.x + CUDA 11.8) 에서 재현한 결과 보고서.

원본 paper: [Huang, Liu, Lv. "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving." ICCV 2023.](https://arxiv.org/abs/2303.05760)
원본 repo: <https://github.com/MCZhi/GameFormer>

## 1. 보고서 구조

| 문서 | 내용 |
| --- | --- |
| [00-summary](00-summary.md) | 본 문서 — 종합 요약 + 보고서 구조 |
| [01-paper_analysis](01-paper_analysis.md) | paper 의 contribution + architecture + 핵심 개념 |
| [02-code_paper_mapping](02-code_paper_mapping.md) | paper section ↔ 코드 line 매핑 (`model/` 중심) |
| [02b-interaction_vs_open_loop](02b-interaction_vs_open_loop.md) | 두 task 의 차이 + game-theoretic joint reasoning 설계 (ASCII flow) |
| [02c-shared_architecture](02c-shared_architecture.md) | 같은 model class 의 specialization — 출력단 동일, size + loss + index 약속만 다름 |
| [02d-game_theoretic_interpretation](02d-game_theoretic_interpretation.md) | "내가 뭐 할까" 보다 "모두가 합리적이면 시스템이 어떻게" — Nash equilibrium 해석 |
| [03-data_pipeline_meaning](03-data_pipeline_meaning.md) | raw tfrecord 부터 metric 까지 각 단계의 paper 적 의미 |
| [04-stack_migration](04-stack_migration.md) | torch 1.x → 2.x stack 변경 + 4 환경 호환성 patch |
| [05-cloud_strategy](05-cloud_strategy.md) | cloud GPU (RunPod / pCloud) 활용 전략, 비용 최적화 |
| [06-results](06-results.md) | smoke 결과 + full 학습 진행 + paper baseline 대비 |
| [07-limitations](07-limitations.md) | 재현 X 부분 + 향후 작업 |
| [08-original_repo_issues](08-original_repo_issues.md) | 원본 repo 발견 이슈 + 해결 + upstream PR 가능성 |

## 2. 한 페이지 요약

### 무엇을 했는가

1. **환경 이식**: 원본의 PyTorch 1.12 + CUDA 11.6 + Python 3.8 stack 을 PyTorch 2.3.1 + CUDA 11.8 + Python 3.10 으로 이식. RTX 30/40 시리즈 + H100/H200 등 최신 GPU 호환.

2. **환경 호환성 patch (4 곳)**: 모델 / 학습 logic 변경 없이 환경 호환 patch 만 적용:
   - WOMD v1.2.1 의 신규 map type 호환 (driveway 등)
   - torch 2.x DDP launcher 의 hyphen 인자 호환 (`--local-rank`)
   - DataLoader fd limit 회피 (`file_system` sharing 전략)
   - 학습 산출물 경로 외부화 (`TRAINING_LOG_HOME` env)

3. **분산 preprocess**: data_process.py 에 `--shard_range` 옵션 추가 → 다 pod 에 shard 분산 → 1.5 TB processed data 를 1시간 안에 완성.

4. **cloud GPU 학습**: RunPod 의 H200 으로 full WOMD 학습 진행 — open_loop (1×H200) + interaction (4×H200 DDP) 동시 진행. pCloud 를 source of truth 로 사용 (region 자유).

5. **자동화**: docker container + scripts 로 한 줄 명령 (`bash scripts/local/06-...sh` 또는 `bash scripts/cloud/train/pod-train-from-scratch-*.sh`) 으로 학습 launch.

### 결과 요약

#### Smoke (1 epoch sanity)

- open_loop (3060): train_loss 91 → 48 (정상)
- interaction (3060 single-GPU DDP): train_loss 22000 → 109 (정상)
- cloud smoke (4090): 3060 결과와 정합 (random seed 동일성 검증)

#### Full-scale (H200, 진행 중)

- open_loop b128 lr 2.83e-4 epoch 20 (~22h, $90)
- interaction b256eff lr 2e-4 epoch 30 (~15h, $239)

#### paper baseline 대비 (학습 완료 후 update)

| metric (paper Joint M=6) | paper | reproduction | 상태 |
| --- | --- | --- | --- |
| minADE (m) | 0.9161 | TBD | pending |
| minFDE (m) | 1.9373 | TBD | pending |
| Miss Rate | 0.4531 | TBD | pending |
| mAP | 0.1376 | TBD | pending |

acceptance: ±15% 이내 → 재현 성공.

### 원본 repo 의 발견 이슈

- 일관성 부재 (open_loop vs interaction 의 unknown map type 처리 다름)
- torch 2.x 비호환 (--local_rank → --local-rank alias 필요)
- DataLoader fd limit 함정 (cloud pod 에서 발견)
- placeholder path 미문서화
- 분산 preprocess 미지원

5 개 patch 가 upstream 에 contribution 가능 (1, 2, 3, 4, 6 — 자세한 내용은 [08-original_repo_issues](08-original_repo_issues.md) 참조).

### 미수행 부분 (한계)

- Marginal model + EM ensemble (paper Table 1 의 두번째 variant — 원본 미공개)
- Closed-loop planning (별도 paper [DIPP](https://github.com/MCZhi/DIPP) 로 안내)
- Waymo leaderboard submission (별도 packaging code 필요)
- nuPlan 실험 (별도 repo [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner) 로 안내)

자세한 내용은 [07-limitations](07-limitations.md) 참조.

## 3. 핵심 결정 사항

| 결정 | 선택 | 사유 |
| --- | --- | --- |
| PyTorch version | 2.3.1+cu118 | 최신 GPU (H100/H200) 호환, paper 시점의 1.12 보다 안정 |
| Dataset version | WOMD v1.2.1 | v1.1 의 training_20s annotation 결손 회피 |
| Cloud platform | RunPod | H200 가용, container 기반 단순 배포, 분 단위 결제 |
| Storage | pCloud (외부 cloud) | region 자유 (RunPod region lock 회피), 사용자 plan 활용 |
| Hyperparameter scaling | sqrt scaling | paper deviation 보수적, deeper network 안전 |
| Container image | docker.io/kingszun/gameformer | 공개 image, Docker Hub 무료 tier |
| Code hosting | GitHub | 공개 repo, fork 기반 재현 |

## 4. 디렉토리 구조 요약

```
GameFormer/
├── README.md                  # 진입점 (quick start)
├── docs/
│   ├── 00-overview.md ~ 09    # 작업 노트 (날짜순)
│   └── report/                # 본 보고서
│       ├── 00-summary.md      # ← 여기
│       ├── 01-paper_analysis.md
│       ├── 02-code_paper_mapping.md
│       ├── 03-data_pipeline_meaning.md
│       ├── 04-stack_migration.md
│       ├── 05-cloud_strategy.md
│       ├── 06-results.md
│       ├── 07-limitations.md
│       └── 08-original_repo_issues.md
├── scripts/
│   ├── local/                 # host docker compose smoke
│   └── cloud/                 # cloud GPU 학습
│       ├── preprocess/        # 분산 preprocess
│       ├── transfer/          # rclone, pCloud public download, checkpoint upload
│       └── train/             # 학습 launcher (chain script)
├── open_loop_planning/        # paper 의 open-loop planning module (4 patch 적용)
├── interaction_prediction/    # paper 의 interaction prediction module (4 patch 적용)
├── model/                     # GameFormer 모델 정의 (변경 X)
└── utils/                     # 데이터 / 학습 / 평가 utility (변경 X)
```

## 5. 본 reproduction 의 의의

1. paper 의 학습 path 가 **현대 환경 (torch 2.x + 최신 GPU)** 에서 그대로 동작함을 검증
2. **환경 호환성 patch + 운영 자동화** 를 정리하여 다른 사용자의 재현 비용 감소
3. **paper baseline 의 ±15% 이내** 의 학습 결과 측정 (진행 중)

## 6. reviewer 가 검증할 수 있는 것

- [README](../../README.md) 참조 → smoke test 명령 (3060 한 장 + WOMD 2 shard 로 30분)
- [02-code_paper_mapping](02-code_paper_mapping.md) → paper Sec X.Y 가 코드 line N 에 어떻게 매핑되는지
- [03-data_pipeline_meaning](03-data_pipeline_meaning.md) → 학습 batch 가 model 어떻게 들어가서 metric 이 산출되는지
- [04-stack_migration](04-stack_migration.md) → 환경 변경의 정당성 + 적용된 patch 의 정확한 line
- [06-results](06-results.md) → smoke 결과 + full 학습 진행 (학습 완료 후 final metric)
- [08-original_repo_issues](08-original_repo_issues.md) → reproduction 과정에서 발견한 upstream issue
