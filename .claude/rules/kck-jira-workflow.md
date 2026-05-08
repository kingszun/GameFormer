## Jira workflow

작업 추적은 jira 에서 수행한다. 모든 코드 변경은 ticket과 연결한다. 정확한 jira project key는 `CLAUDE.md` 참조.

### Ticket 계층

- Epic — 큰 목표 단위 (예: "GameFormer reproduction"). 보통 1개 project당 1~2개.
- Task — Epic 하위 실행 단위. 1~5일 분량. Epic link 필수. default issue type.
- Story — user-facing 결과물 단위 (UI 추가, API 신규, 외부 공개 demo 등). 사용자 가치 전달이 명확할 때만 사용. agent 임의로 Story 등록 금지.
- Sub-task — Task / Story 안의 step. 수 시간 ~ 1일 분량. parent 필수.
- Bug — 문제 발생 시 별도 등록. 발생한 작업의 Task / Story / Epic 에 link.

각 ticket 은 이전 단계 완료 후 다음 단계 진입 전 status 갱신.

### Issue type 선택 가이드

| 작업 성격 | issue type | label | 비고 |
| --- | --- | --- | --- |
| 큰 목표 (multi-step, 1주+) | Epic | — | sub-Task 다수 묶음 |
| 일반 기술 작업 (코드 변경, infra, 문서) | Task | — | default |
| 실험 / spike / 회고적 측정 | Task | `experiment` | 양식은 "실험 ticket 양식" section 참조 |
| user-facing 기능 결과물 | Story | — | 사용자 가치가 명확할 때만 |
| 결함 / 회귀 | Bug | — | 발견 시 즉시 등록 |
| 위 type 의 sub-step | Sub-task | — | parent 필수 |

판단 기준:
- "사용자 (외부) 가 이 결과물을 직접 본/쓸 수 있는가?" — Yes 면 Story 후보, No 면 Task.
- 코드 변경이 없거나 측정/조사가 주요 목적 → 실험 (Task + label).
- 의심되면 Task 로 등록 후 필요 시 type 전환.

### 작업 시작 시

- 새 작업을 시작하기 전에 해당 작업을 다루는 ticket 이 있는지 확인.
  - 없으면 신규 등록 (적절한 type + parent/Epic link).
  - 있으면 status 를 `In Progress` 로 전환.
- ticket 단위가 너무 크면 sub-task 로 분할.
- ticket 등록은 작업 시작 시 즉시 수행. 사후 정리 금지.

### Status 전환

- `To Do` → `In Progress` — 작업 시작 시
- `In Progress` → `Done` — 작업 완료 + verify 통과 시
- `In Progress` → `Blocked` — 외부 의존으로 막힘. 막힌 사유를 comment 로 기록
- `Done` 전환 규칙은 "작업 결과 / log 기록" + "Done 시 의무 sync" section 참조

### Commit / branch 매핑

- commit message: `{type}(scope): [ISSUE_ID] subject`
  - 예: `feat(cloud): [KAK-3] RunPod 1차 smoke pod create`
  - `kck-git-workflow.md` 의 Conventional Commits format 과 일치
- ticket id 가 없는 commit (history fix 등) 은 prefix 없이 작성
- branch 이름: `{type}/{ISSUE_ID}-short-desc`
  - 예: `feat/KAK-3-runpod-smoke`
  - main 직접 commit 일 경우 branch 생략

### MCP tool 활용

모든 jira 조작은 `mcp__atlassian-api__jira_*` MCP tool 로 수행.

| 작업 | tool |
| --- | --- |
| 진행 중 ticket 조회 | `jira_search` (JQL: `project = <KEY> AND status = "In Progress"`) |
| ticket 상세 조회 | `jira_get_issue` |
| 신규 ticket 등록 | `jira_create_issue` |
| status 전환 | `jira_transition_issue` |
| comment 작성 | `jira_add_comment` |
| Task → Epic link | `jira_link_to_epic` |
| 다중 ticket 일괄 등록 | `jira_batch_create_issues` |

ticket 만들거나 수정할 때 항상 user 에게 결과 보고 (key + summary + 변경 내용).

### 등록 규칙

- summary 는 한국어 또는 영어 일관 (project 단위 통일).
- description 에 다음을 포함:
  - 목표 / acceptance criteria
  - 관련 file / commit / 외부 reference
  - 외부 의존성 (cloud 비용, 데이터 다운로드 시간 등)
- assignee 는 등록 시점에 알 수 없으면 미지정 (현재 user 가 단독 작업 중이면 본인).
- estimate 는 알 수 있을 때만 (불확실하면 비워둠).

### 작업 결과 / log 기록 (모든 ticket 공통)

`In Progress` → `Done` 으로 전환 시 ticket comment 에 다음을 모두 포함한다:

- 실행한 command — 그대로 복사 가능한 형식 (code block). 요약 금지.
- 결과 metric — runtime, throughput, output size/count, GPU/CPU/mem util, loss 등. 표 형식 권장.
- 검증 결과 — `PASS` / `FAIL` + 사유.
- 산출물 path — log file, checkpoint, output directory, 관련 commit hash.
- baseline 비교 — 이전 측정값 또는 reference 가 있으면 같은 표에 병기.

raw log 보관 정책:

- host repo `logs/{TICKET-ID}_{short_desc}.log` 에 보관 (예: `logs/KAK-9_train_smoke.log`).
- `logs/` 는 `.gitignore` (git tracking 금지 — 용량 누적 + secret 가능성).
- jira attachment upload 가능하면 첨부. 안 되면 comment 에 `host file path` + 핵심 발췌 (head/tail 50 line) 포함.
- pod 등 원격에서 생성한 log 는 `scp` 로 host 에 회수 후 동일 위치 저장.

### Done 시 의무 sync (다중 위치)

ticket Done 전환과 같은 turn 안에 다음을 모두 update — SSOT 유지:

- 관련 docs (`docs/0X-*.md`) — 결과 표, baseline, diagram, fact 변경.
- `CLAUDE.md` — 핵심 fact (image digest, image tag, command, 비용 등) 변경 시.
- memory (`~/.claude/projects/.../memory/*.md`) — project state, reference, feedback.
- `git add` 변경 file — commit 은 user 지시 시에만 (`kck-git-workflow.md` 준수).

작업 중 발견한 추가 작업/위험은 즉시 신규 ticket 발행 (사후 정리 금지). 발견 ticket comment 에 신규 ticket key 를 link.

### 명령 실행 / log 보존 정책

모든 외부 system 명령 (runpodctl, gcloud, gsutil, ssh, scp, docker, curl 등) 의 실행은:

- comment 에 명령 + 결과를 함께 기록 (요약 X — raw stdout/stderr 그대로 또는 head/tail).
- script 가 host-only 또는 pod-only 패턴 (예: `docker compose exec`) 으로 묶여있어 우회한 경우, 우회한 직접 명령 + 사유를 명시. script 자체 개선은 별도 ticket 으로 발행.
- 인증 정보 (token, key, password, ADC, refresh token 등) 는 결과에서 redact (`***` 또는 `<redacted>`) 후 기록.
- 긴 stdout 은 host log file 로 redirect 한 후 (`> logs/{TICKET-ID}_{cmd}.log 2>&1`) ticket 에는 path + 발췌 첨부.

### 실험 ticket 양식

실험 / spike / 측정 작업은 Task + label `experiment` 로 등록한다. description 은 다음 section 을 모두 포함:

- 목표 — 무엇을 검증/측정하려는가.
- 환경 — GPU / OS / container / dataset / hyperparameter 등 재현에 필요한 모든 값.
- 결과 metric — 측정값 (loss, runtime, GPU memory, throughput 등). 표 형식 권장.
- 산출물 — checkpoint 경로, log 파일, 관련 docs.
- Status — 통과 / 실패 / 부분 통과 + 짧은 결론.

회고적 등록 (이미 끝난 실험) 도 동일 양식. comment 에 reproducibility 정보 (commit hash, log path) 추가.
