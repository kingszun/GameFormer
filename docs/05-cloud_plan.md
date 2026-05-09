## 05 - cloud plan

local 검증 완료 후 RunPod에서 학습 진행하기 위한 계획. 결정 사항 + 구체 절차.

### 결정 사항

| 항목 | 선택 | 비고 |
| --- | --- | --- |
| 학습 범위 | 1×4090 smoke | $0.3, 1~2h. 본격 학습 전 cloud 환경 통과만 확인 |
| WOMD 인증 | host의 user OAuth ADC json 복사 | service account는 Waymo bucket ACL을 통과 못 함 (아래 노트). user OAuth credential 만 valid |
| 코드 전달 | git push → cloud pod git clone | Docker Hub image (5GB)는 한 번만 push. 코드 변경은 git만 |

### 인증 결정 배경 (왜 service account 가 안 되는가)

26-05-08 검증 결과:
- Waymo bucket (`gs://waymo_open_dataset_motion_v_1_2_1/`) 은 Waymo 소유. 우리 IAM 변경 권한 없음.
- Waymo는 Google account 단위로 license 동의 후 ACL을 부여 — `kingszun@gmail.com` 이 https://waymo.com/open/licensing/ 에서 동의했기 때문에 host gsutil 통과.
- service account 는 license 동의 절차가 없어 ACL 통과 못 함:
  ```
  AccessDeniedException: 403 waymo-reader@... does not have storage.objects.list access
  ```
- 우리 GCP project IAM에 어떤 role 을 줘도 무관 — bucket-level 거부.
- 따라서 cloud pod에서도 host와 동일한 user OAuth credential 을 사용해야 함.

smoke 통과 후 후속 결정:
- (b) open_loop_planning full — 1×A100 ~$30~50, ~24h
- (c) interaction_prediction single-GPU — 1×H100 ~$60~120, 1~2일
- (d) interaction_prediction 4×H100 paper 재현 — ~$100~300, 5~15h

### 사전 user 준비 사항

| # | 항목 | 상태 |
| --- | --- | --- |
| 1 | RunPod 가입 + 결제 ($20 충전 권장) | done? user 진행 |
| 2 | RunPod API key 발급 + `RUNPOD_API_KEY` env | done (`~/.zshrc` 등록 확인) |
| 3 | Docker Hub `kingszun` push | done (`kingszun/gameformer:cu118-py310-torch2.3.1`) |
| 4 | host에 user OAuth ADC json 생성 | pending — user 단계 |
| 5 | git remote (push 대상) 결정 + commit + push | done (`https://github.com/kingszun/GameFormer.git`, main, 26-05-08) |

### user OAuth ADC json 준비 절차 (user 단계)

ADC = Application Default Credentials. `~/.config/gcloud/application_default_credentials.json` 단일 파일.

1. host:
   ```
   gcloud auth application-default login
   ```
   - browser 인증 진행. Waymo license 동의한 동일 Google account (`kingszun@gmail.com`) 선택.
   - 결과: `~/.config/gcloud/application_default_credentials.json` 생성.
2. 파일 메타 확인:
   ```
   ls -la ~/.config/gcloud/application_default_credentials.json
   ```
3. host 인증 검증:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json gsutil ls gs://waymo_open_dataset_motion_v_1_2_1/ | head
   ```

cloud pod로 전달은 pod 띄운 후 (아래 절차 참조).

보안 노트:
- ADC 는 user account refresh token이고 default scope 가 `cloud-platform` → 모든 GCP API 호출 가능.
- cloud pod 외부 노출 시 즉시 host 에서 `gcloud auth application-default revoke`.
- pod 종료 후 더 이상 안 쓰면 즉시 revoke 권장.

### 코드 전달 (3a) 절차

1. local repo의 변경 사항 commit:
   - file: docker/{Dockerfile,entrypoint.sh}, pyproject.toml, uv.lock, compose.yaml, .env.example, .dockerignore, .gitignore, scripts/*, docs/*
   - 코드 수정: `interaction_prediction/train.py`, `open_loop_planning/data_process.py`
2. user 소유 git remote에 push (예: `kingszun/gameformer-fork`)
3. cloud pod에서 `git clone <remote>` 또는 `git pull`

### Cloud pod 생성 절차 (agent 자동화 가능 부분)

agent가 `runpodctl` 또는 GraphQL API로 처리:

1. pod 생성:
   - GPU type: `RTX 4090` (community cloud 가장 저렴)
   - container image: `kingszun/gameformer:cu118-py310-torch2.3.1`
   - container disk: 30 GB (image 5GB + data 일부)
   - volume disk: 50 GB (선택, dataset cache용)
   - region: us-central (WOMD bucket region 일치)
   - command: `sleep infinity`
2. pod 기동 후 SSH 진입 (RunPod 자동 expose)
3. cloud pod 내부:
   - git clone
   - host에서 ADC json upload (scp 또는 RunPod web upload):
     ```
     scp ~/.config/gcloud/application_default_credentials.json <pod>:/root/.config/gcloud/application_default_credentials.json
     ```
   - pod에서 환경변수:
     ```
     export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json
     ```
   - 검증: `gsutil ls gs://waymo_open_dataset_motion_v_1_2_1/ | head`
   - `bash scripts/local/01-download_womd.sh` (training_20s 1~2 shard)
   - `bash scripts/local/02-build_image.sh` 불필요 (image 이미 pull됨, 그러나 코드 변경 시 image 안의 코드도 update 필요 — image에 코드는 안 들어있으므로 git clone만으로 OK)
   - 학습 launch
4. 학습 완료 후 pod stop (비용 절약)

### 비용 통제

- pod stop ≠ 삭제. stop 시 storage 비용만 소량 (~$0.07/GB/월). compute 청구는 정지.
- 작업 후 즉시 stop 또는 terminate 습관화.
- 4090은 시간당 $0.3 수준 — 1~2h smoke은 $0.3~0.6.
- 학습 launch 전 GPU/region/disk 비용 확인하고 진행.

### 미해결 / 위험

- multi-GPU NCCL 정합성: 3060 1장으로는 검증 불가. cloud single-GPU에서도 우회되므로 multi-GPU 단계 진입 시 별도 검증 필요.
- WOMD 전체 (수백 GB) cloud download 속도: GCS bucket region (us-central1)과 pod region 일치 시 1~5 Gbps. 실측 후 단계 결정.
- TF + torch 동거 안정성: smoke에서 통과 확인. 장시간 학습에서 TF GPU init 시도가 OOM 유발할 가능성 — `CUDA_VISIBLE_DEVICES=""` 로 TF 격리 필요 시 별도 metric process 분리 검토.
