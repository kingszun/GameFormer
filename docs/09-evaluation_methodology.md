# 학습 평가 methodology + Waymo leaderboard 위치

GameFormer reproduction 의 학습/평가 setup 과 paper baseline 비교 protocol.

## Train / Valid / Test 의 역할

표준 ML methodology:

| split | gradient update | 학습 중 사용 | 평가용 |
|---|---|---|---|
| **train** | 모델 parameter update 의 source | 매 step | — |
| **valid (= dev)** | ✗ | **매 epoch 후 loss 측정** | ✓ 최종 metric |
| **test** | ✗ | ✗ | ✓ unbiased final metric |

valid 의 학습 중 역할:
- inference only (gradient 안 계산)
- overfitting 감지 (train loss ↓ but valid loss ↑ 의 신호)
- early stopping (valid loss 가 N epoch 동안 개선 안 되면 학습 중단)
- best model selection (가장 낮은 valid loss 의 epoch checkpoint 채택)
- hyperparameter tuning (lr/batch_size 변경 시 valid metric 비교)

## Waymo Open Motion Dataset (WOMD) 의 split

| set | 용도 | label 공개 |
|---|---|---|
| training (487k scenarios) | gradient update | ✓ |
| training_20s (344 sparse shards of 1000) | open_loop_planning train | ✓ |
| validation (44k scenarios) | open_loop_planning eval (val) | ✓ |
| validation_interactive (44k scenarios) | interaction_prediction eval (val) | ✓ |
| testing (44k scenarios) | leaderboard 만 | ✗ (Waymo private) |

→ test set label 비공개 → 학술 논문은 **validation set 에서 metric 보고**.

## GameFormer reproduction 의 setup

| task | train data | valid data | test data |
|---|---|---|---|
| open_loop_planning | training_20s | validation | (사용 안 함) |
| interaction_prediction | training | validation_interactive | (사용 안 함) |

→ **valid 가 monitor + 최종 보고 둘 다** 사용 (Waymo 공통 protocol)

## 정보 leakage 우려 (= "사기" 인가?)

valid 를 monitor + 보고 둘 다 사용 시:
- best checkpoint 을 valid loss 로 선택 → valid 정보가 model 선택에 반영됨
- 따라서 valid metric 는 진짜 generalization metric 보다 **약간 overestimate** (information leakage)

그래도 community 공통 protocol 인 이유:
1. **parameter 직접 학습 안 함** — gradient 는 train 만으로 (valid 는 inference only)
2. **모든 paper 같은 룰** — relative comparison 공정 (paper A 의 valid metric vs paper B 의 valid metric)
3. **Waymo test set label 비공개** → leaderboard 제출 외에는 unbiased metric 불가

## GameFormer paper 의 보고 metric

paper Table 1 (interaction prediction, WOMD validation_interactive):

| variant | minADE | minFDE | Miss Rate | mAP |
|---|---|---|---|---|
| Joint (M=6) | 0.9161 | 1.9373 | 0.4531 | 0.1376 |
| Marginal (M=64) | 0.9721 | 2.2146 | 0.4933 | 0.1923 |

→ 모두 **WOMD validation set** 에서 측정 (test/leaderboard 아님). paper 명시: "trained on the entire WOMD training dataset" + "evaluated using the official evaluation metrics" — validation/benchmark results.

## Waymo leaderboard 제출 여부

조사 결과:

1. **GameFormer GitHub repository (MCZhi/GameFormer)** 의 README:
   - "Code for packaging and submitting prediction results to the WOMD Interaction Prediction Challenge" — **공개 안 됨**
   - 별도 ecosystem (DIPP, GameFormer-Planner) 에 submission 코드 가능성 시사

2. **2023 Waymo Open Motion Challenge** (closed 2023-05-23):
   - Motion Prediction + Interaction Prediction + Sim Agents 별도 leaderboard
   - GameFormer entry **공개 검색 결과 없음**
   - 2023 challenge winner = MTR-A (별도 method)

3. **paper 의 metric** = WOMD validation set (test/leaderboard 아님)

→ **GameFormer 는 Waymo leaderboard 에 공식 제출되지 않은 것으로 보임** (paper 의 reported metric 은 validation set 에서 측정)

## reproduction 시 평가 protocol

우리 학습:
- training 데이터로 SGD
- 매 epoch 후 validation 에서 loss/metric 측정
- best valid loss 의 checkpoint 저장
- 최종 보고 metric = validation 에서 측정

paper 와 같은 protocol → paper Table 1 의 baseline metric 과 직접 비교 가능 (relative comparison fair).

## 진짜 unbiased test metric 이 필요하면 (별도 작업)

| 옵션 | 동작 | 상태 |
|---|---|---|
| Waymo Motion Challenge submit | Waymo 의 test set 에서 metric 받기 (leaderboard 점수) | 미구현 (별도 KAK ticket 필요) |
| training 의 일부 (5%) 를 hold-out → self test | 학습 외 별도 split 사용 | 미구현 |
| k-fold cross validation | 비싸지만 robust | 미구현 |

reproduction 목적 상 paper 의 reported metric 재현이 우선 → 위 옵션은 추가 작업 시 별도 ticket.

## 참고 (재현 시 주의)

- WOMD v1.2.1 의 validation_interactive 는 chain v3 에서 처리됨 — 86958 npz file (43479 unique scenes × 2 direction pair, network volume 보존 + pCloud upload 진행 중)
- 학습 시 valid loss 가 train loss 보다 너무 낮으면 (예: 1/2 이하) → data preprocessing 또는 split 오류 의심
- valid metric 가 paper baseline 보다 ±15% 이내 → reproduction 성공 기준
