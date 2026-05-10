# GameFormer reproduction

ICCV'23 GameFormer paper의 학습 path를 현대 환경(PyTorch 2.x + CUDA 11.8)에서 재현한 결과.

원본 repo: <https://github.com/MCZhi/GameFormer>
원본 paper: [GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.html) (Huang et al., ICCV 2023)

## 재현 범위

| 작업 | 상태 |
| --- | --- |
| open_loop_planning 학습 path 동작 확인 (3060, 1 epoch smoke) | yes |
| interaction_prediction 학습 path 동작 확인 (3060 single-GPU DDP, 1 epoch smoke) | yes |
| 환경 이식 (torch 1.12+cu116 → torch 2.3.1+cu118) | yes |
| Docker container 기반 재현 환경 구성 | yes |
| 클라우드 GPU(H200)로 full-scale 학습 | 진행 중 |

원본 repo 가 제공하지 않는 부분:
- closed-loop planning (원저자가 [DIPP](https://github.com/MCZhi/DIPP)로 안내)
- WOMD Interaction Prediction Challenge submission 코드
- marginal model + EM ensemble

## stack 변경 사유

| 항목 | 원본 | 재현 | 사유 |
| --- | --- | --- | --- |
| PyTorch | 1.12.0+cu116 | 2.3.1+cu118 | cu116 wheel은 sm_86까지만 native, cu118 wheel은 sm_90(H200)까지 native 포함 |
| Python | 3.8 | 3.10 | torch 2.3.1+cu118 wheel default |
| CUDA | 11.6 | 11.8 | sm_90 native 지원 |
| TensorFlow | 2.11 (waymo dep) | 2.11 동일 | `waymo-open-dataset-tf-2-11-0`가 강제 |
| dataset | WOMD v1.1 | WOMD v1.2.1 | v1.1의 `training_20s` annotation 결손 issue 회피 |

torch 2.x에서 원본 코드는 두 군데 patch만 필요했음 (자세한 내용은 [docs/03-patches.md](docs/03-patches.md)).

## 디렉토리 구조

```
GameFormer/
├── README.md                      # this file
├── docker/                        # Dockerfile, entrypoint
├── compose.yaml, .env.example     # docker compose stack
├── pyproject.toml, uv.lock        # uv 기반 dep 관리
├── interaction_prediction/        # 원본 코드 (+ patch)
├── open_loop_planning/            # 원본 코드 (+ patch)
├── model/, utils/                 # 원본 그대로
├── scripts/
│   ├── local/                     # host docker compose smoke
│   └── cloud/                     # 클라우드 GPU 학습 자동화
└── docs/                          # 재현 과정 + 결정 사항
```

## 빠른 시작 (smoke test)

3060 12 GB 한 장 기준 1~2 shard로 학습 path 동작 확인.

### 사전 준비
- `docker` (compose v2), `nvidia-container-toolkit`
- `google-cloud-cli` (`gsutil`, `gcloud`)
- Waymo Open Dataset license 동의 + `gcloud auth login`
- dataset symlink:
  ```bash
  DATASET_HOST_PATH=/path/to/your/dataset/storage
  mkdir -p $DATASET_HOST_PATH
  ln -s $DATASET_HOST_PATH ./data
  cp .env.example .env
  ```

### 실행
```bash
WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash scripts/local/01-download_womd.sh
bash scripts/local/02-build_image.sh
bash scripts/local/03-up.sh
bash scripts/local/04-smoke_test.sh
bash scripts/local/05-open_loop_preprocess.sh
BATCH_SIZE=8 EPOCHS=1 bash scripts/local/06-open_loop_train.sh
```

학습 log/checkpoint는 `open_loop_planning/training_log/${NAME}/`에 저장됨.

각 script의 동작은 [scripts/README.md](scripts/README.md) 참조.

## full-scale 학습 (cloud GPU)

H200 등 cloud GPU에서 full WOMD로 학습. 자세한 내용은 [docs/06-full_training_plan.md](docs/06-full_training_plan.md).

## 재현 결과

원본 paper의 metric (validation set 기준, paper Table 1):

| variant | minADE | minFDE | Miss Rate | mAP |
| --- | --- | --- | --- | --- |
| Joint (M=6) | 0.9161 | 1.9373 | 0.4531 | 0.1376 |
| Marginal (M=64) | 0.9721 | 2.2146 | 0.4933 | 0.1923 |

재현 시 hyperparameter는 paper 그대로(batch 16/GPU × 4 GPU, lr 1e-4, epoch 30)를 기준으로 사용. cloud GPU의 VRAM 여유를 활용해 batch를 키운 variant도 별도로 측정 (자세한 내용은 [docs/06-full_training_plan.md](docs/06-full_training_plan.md), [docs/09-evaluation_methodology.md](docs/09-evaluation_methodology.md)).

## 보고서

reproduction 의 종합 보고서는 [docs/report/](docs/report/) 아래.

| 문서 | 내용 |
| --- | --- |
| [report/00-summary](docs/report/00-summary.md) | 종합 요약 + 보고서 색인 |
| [report/01-paper_analysis](docs/report/01-paper_analysis.md) | paper contribution + architecture |
| [report/02-code_paper_mapping](docs/report/02-code_paper_mapping.md) | paper section ↔ 코드 line 매핑 |
| [report/02b-interaction_vs_open_loop](docs/report/02b-interaction_vs_open_loop.md) | interaction vs open_loop 차이 + game-theoretic 설계 (ASCII flow) |
| [report/02c-shared_architecture](docs/report/02c-shared_architecture.md) | 같은 model class — 출력단 동일, size + loss + index 약속만 다름 |
| [report/02d-game_theoretic_interpretation](docs/report/02d-game_theoretic_interpretation.md) | "내가 뭐 할까" 보다 "모두가 합리적이면 시스템이 어떻게" 추론 |
| [report/02e-multi_modal_mode_selection](docs/report/02e-multi_modal_mode_selection.md) | M=6 modality + winner-takes-all 학습 + 학습/평가/deploy mode selection mismatch |
| [report/03-data_pipeline_meaning](docs/report/03-data_pipeline_meaning.md) | raw 부터 metric 까지 pipeline + paper 적 의미 |
| [report/04-stack_migration](docs/report/04-stack_migration.md) | torch 1.x → 2.x stack 변경 + 환경 호환성 patch |
| [report/05-cloud_strategy](docs/report/05-cloud_strategy.md) | cloud GPU 활용 전략 + 비용 최적화 |
| [report/06-results](docs/report/06-results.md) | smoke + full 학습 결과 + paper baseline 대비 |
| [report/07-limitations](docs/report/07-limitations.md) | 재현 X 부분 + 향후 작업 |
| [report/08-original_repo_issues](docs/report/08-original_repo_issues.md) | 원본 repo 발견 이슈 + upstream PR 가능성 |
| [report/09-traffic_sim_paradigms](docs/report/09-traffic_sim_paradigms.md) | GameFormer 한계 극복 후속 paradigm 4 종 deep-dive (12 paper, 정량 비교) |

## 작업 노트

reproduction 진행 중 작성한 작업 노트 (날짜순) 는 [docs/](docs/) 아래:

| 문서 | 내용 |
| --- | --- |
| [docs/00-overview.md](docs/00-overview.md) | 프로젝트 목표, 진행 상황 |
| [docs/01-environment.md](docs/01-environment.md) | Docker stack, env var, hardware compat |
| [docs/02-workflow.md](docs/02-workflow.md) | 단계별 실행 흐름 (host vs container) |
| [docs/03-patches.md](docs/03-patches.md) | upstream 코드 변경 사항 |
| [docs/04-smoke_results.md](docs/04-smoke_results.md) | 3060에서 검증된 smoke 결과 |
| [docs/06-full_training_plan.md](docs/06-full_training_plan.md) | 클라우드 full 학습 plan + hyperparameter |
| [docs/07-data_pipeline.md](docs/07-data_pipeline.md) | data preprocess pipeline 4 subset |
| [docs/08-storage_structure.md](docs/08-storage_structure.md) | host/cloud 데이터 저장 구조 |
| [docs/09-evaluation_methodology.md](docs/09-evaluation_methodology.md) | train/valid/test 분리, paper baseline 비교 protocol |

## 원본 paper 인용

```bibtex
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Zhiyu and Liu, Haochen and Lv, Chen},
    title     = {GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3903-3913}
}
```
