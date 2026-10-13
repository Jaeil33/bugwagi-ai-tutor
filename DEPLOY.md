# 배포 · 운영 가이드

## 1. 배포 구성

```
GitHub 비공개 저장소 ──(git push 하면 자동 반영)──▶ Streamlit Community Cloud ──▶ https://<앱 이름>.streamlit.app
                                                       │  Secrets: ANTHROPIC_API_KEY, (APP_PASSWORD), DAILY_BUDGET_USD
                                                       ├─ 앱 로그(JSON 한 줄/요청) → Manage app → Logs
                                                       └─ Claude API (Anthropic)
```

### 플랫폼 선택: Streamlit Community Cloud

| 기준 | Streamlit Cloud | AWS App Runner · GCP Cloud Run · Azure Container Apps |
|---|---|---|
| 비용 | 무료 | 사용량 과금 (프리티어 · 크레딧 있음), 카드 등록 필요 |
| 배포 방법 | GitHub 저장소 연결만 하면 됨 | 컨테이너 이미지 빌드 → 레지스트리 → 서비스 생성 |
| 메모리 | 최대 2.7GB (공식 문서, 2024-02 기준) → 실측 1.2GB로 충분 | 원하는 만큼 (2GB 이상 권장) |
| 확장성 | 컨테이너 1개, 12시간 접속이 없으면 잠듦 | 자동 확장(스케일링), 0대까지 축소 가능 |
| 맞는 단계 | **프로토타입 · 데모 (지금)** | 학교 전체 운영 |

지금은 멘토 · 면접관이 써 보는 데모 단계라 무료이고 설정이 가장 단순한 Streamlit Cloud를 골랐어요.
실제 학교 서비스로 운영할 때를 대비해 컨테이너 배포용 `Dockerfile`도 준비했어요(아래 B).

> 측정한 메모리: 앱 실행 151MB → 임베딩 모델 · 벡터DB 로드 후 약 1.1GB (최대 1.2GB). 대부분 임베딩 모델과 torch예요.

## 2. 배포 절차

### A. Streamlit Community Cloud (사용 중)

1. 이 폴더를 GitHub **비공개** 저장소로 올린다. (`data/RFP_교육.pdf`는 코멘토 실습 자료라 공개 저장소에 올리지 않는다)
2. https://share.streamlit.io 에 GitHub 계정으로 로그인한다.
   비공개 저장소는 Workspace settings → Linked accounts → Source control에서 **비공개 저장소 접근 권한**을 추가로 허용해야 보인다.
3. **Create app** → 저장소 · 브랜치(`main`) · Main file `app.py`를 고르고, 원하는 주소(`<이름>.streamlit.app`)를 정한다.
4. **Advanced settings**에서 Python **3.13**을 고르고, Secrets에 아래 내용을 넣는다.
   최상위 값은 환경변수로도 들어가서 코드(`os.environ`)를 바꿀 필요가 없다.
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   # 워크스페이스에 묶이지 않은 키라면
   # ANTHROPIC_WORKSPACE_ID = "wrkspc_..."
   DAILY_BUDGET_USD = "2"
   MAX_QUESTIONS_PER_SESSION = "30"
   # 입장 코드를 쓰려면
   # APP_PASSWORD = "..."
   ```
5. **Deploy**. 첫 배포는 패키지 설치와 임베딩 모델 다운로드로 몇 분 걸린다.
6. 배포된 주소에서 예시 질문 하나를 눌러, 답변 아래에 **출처 쪽수 · 응답 시간 · 비용**이 나오고 📈 운영 현황 탭에 집계되는지 확인한다.

> **패키지 설치 주의:** Streamlit Cloud는 `uv`로 설치하는데, `uv`는 보안상 **처음 찾은 인덱스에서만** 패키지를 고른다.
> PyTorch CPU 인덱스를 `--extra-index-url`로 넣으면 그 인덱스에 있는 옛 `requests`만 보고 설치가 실패한다(로컬에서 재현 · 확인).
> 그래서 `requirements.txt`는 `--find-links https://download.pytorch.org/whl/cpu/torch/`로 torch만 그곳에서 받는다.
> (`uv pip compile`로 Linux · Python 3.13 기준 설치 가능 여부를 미리 확인함)

### B. 컨테이너 서비스 (AWS App Runner / GCP Cloud Run / Azure Container Apps)

```bash
docker build -t bugwagi-ai-tutor .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... -e DAILY_BUDGET_USD=2 bugwagi-ai-tutor

# 예: GCP Cloud Run (소스에서 바로 빌드 · 배포)
gcloud run deploy bugwagi-ai-tutor --source . --region asia-northeast3 --memory 2Gi \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest --max-instances 2 --allow-unauthenticated
```

- 포트 8080(Cloud Run은 `PORT` 환경변수를 따름), 메모리 **2GB 이상**, 헬스 체크 경로 `/_stcore/health`.
- Streamlit은 WebSocket을 쓰므로 세션 고정(session affinity)을 켜는 게 좋다 (Cloud Run: `--session-affinity`).
- Dockerfile은 root가 아닌 사용자로 실행하고, 임베딩 모델을 이미지에 미리 넣어 첫 요청 지연을 없앴다.
- ⚠️ 로컬에 Docker가 없어 이 Dockerfile의 빌드는 직접 확인하지 못했다. 쓰기 전에 `docker build`로 먼저 확인할 것.

## 3. 운영

### 모니터링 · 로그 (튜토리얼 4절)

| 무엇을 | 어디서 |
|---|---|
| 질문 수, 평균 · 95% 응답 시간, 오늘 API 비용, 오류 수, 👍/👎 | 앱의 **📈 운영 현황** 탭 (서버 메모리 집계, 재시작하면 초기화) |
| 요청별 상세 기록 | **Manage app → Logs** (Cloud Run은 Cloud Logging, App Runner는 CloudWatch) |
| 실제 API 사용액 | Anthropic Console → Usage |

로그는 요청마다 JSON 한 줄이에요. **질문 원문은 개인정보가 섞일 수 있어 남기지 않고 길이만** 기록해요.

```json
{"event": "answer", "session": "953505f1", "model": "claude-sonnet-5-5", "search": "하이브리드", "k": 7,
 "followup": true, "rewritten": true, "pages": [2, 7, 9], "seconds": 2.75, "first_token": 1.71,
 "input_tokens": 1901, "output_tokens": 190, "cost": 0.0057, "status": "정상", "request_id": "req_...", "error": null}
```

- `status`가 `오류→대체`이면 `error`에 오류 종류(AuthenticationError, RateLimitError 등)가 남는다. `request_id`는 Anthropic에 문의할 때 쓴다.
- 오류 상세(스택트레이스)는 `client.showErrorDetails = "none"`으로 브라우저에 보이지 않게 했다.

### 비용 관리 (튜토리얼 1-3)

1. **앱 안의 상한:** 하루 비용이 `DAILY_BUDGET_USD`(기본 $2, 한국 시간 자정 초기화)를 넘거나 한 접속에서 `MAX_QUESTIONS_PER_SESSION`(기본 30회)을 넘으면, 앱이 멈추지 않고 **오프라인 모드(찾은 자료 원문)** 로 바뀐다.
   - 실측 질문당 비용: Sonnet 5.5 약 $0.007 → $2는 약 280문항.
   - 앱 집계는 서버가 재시작되면 초기화되므로, 아래 2번을 함께 건다.
2. **Anthropic Console의 월 사용 한도(spend limit)** 를 걸어 둔다. 앱 밖에서 거는 최종 안전장치다.
3. 데모 기간이 끝나면 Streamlit Cloud에서 앱을 **Delete** 하거나 Secrets의 API 키를 지운다(키가 없으면 오프라인 모드로만 동작).

### 보안 (튜토리얼 1-4, 3절)

- API 키는 Secrets(환경변수)로만 넣고, `.env` · `.streamlit/secrets.toml`은 `.gitignore`로 저장소에서 뺐다.
- 입력 제한: 질문 500자, PDF 20MB × 3개. 업로드 파일은 내용 해시 폴더에 따로 저장해 접속자끼리 섞이지 않는다.
- `APP_PASSWORD`를 넣으면 입장 코드를 아는 사람만 쓸 수 있다(비교는 `hmac.compare_digest`).
- API Gateway: Streamlit은 REST API가 아니라 WebSocket 화면 앱이라 API Gateway를 앞에 두는 구조가 아니다.
  지금은 앱 안에서 입장 코드 · 사용량 상한으로 대신하고, 학교 운영 단계에서는 Cloud Run + 로드밸런서/IAP(학교 계정 로그인) 또는 `st.login`(OIDC)으로 바꾼다.

### 장애 대응

| 상황 | 앱의 동작 | 운영자가 할 일 |
|---|---|---|
| API 키 오류 · 크레딧 소진 | "AI 서비스 인증 설정에 문제가 있어…" + 찾은 자료 원문으로 대체 | Secrets의 키 확인, Console 결제 확인 |
| API 과부하 · 속도 제한(429/5xx) | SDK가 2회 자동 재시도 → 실패 시 대체 응답 | 로그의 `error` 빈도 확인, 계속되면 Haiku로 기본 모델 변경 |
| 안전 분류기 거절 | 서버측 fallbacks로 다른 모델이 이어서 답변 | - |
| 메모리 초과로 앱 중단 | Streamlit Cloud가 "over resource limits" 표시 | Manage app → Reboot. 반복되면 컨테이너 서비스(B)로 이전 |
| 12시간 무접속 후 잠듦 | 첫 방문자가 깨우기 버튼을 누르면 1~2분 뒤 시작 | 시연 전에 미리 한 번 접속 |

### 업데이트

1. `main` 브랜치에 push한다.
2. Streamlit Cloud에서 **Manage app → ⋮ → Reboot app**. 코드 변경 감시(`fileWatcherType`)를 꺼 두었기 때문에 다시 시작해야 반영된다.
   - 감시를 켜 두면 transformers의 모듈 수천 개를 훑으면서 `ModuleNotFoundError: torchvision` 로그를 수백만 자 쏟아냈다(첫 배포 로그에서 확인).
   - 운영 중에는 코드가 바뀌지 않으므로 끄는 쪽이 로그도 깨끗하고 CPU도 아낀다.
3. 패키지(`requirements.txt`)를 바꾸면 다시 설치하느라 몇 분 걸린다.

### 배포하며 겪은 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| (배포 전 확인) 패키지 설치 실패 | `uv`가 `--extra-index-url`의 PyTorch 인덱스에서만 `requests`를 찾음 | `--find-links`로 torch만 PyTorch에서 받기 |
| Secrets에 키를 넣었는데 오프라인 모드로 시작 | 앱 시작 시점에 `os.environ`에 키가 없었음 | 시작할 때 `st.secrets`를 환경변수로 직접 옮기고, 읽은 개수 · 키 존재 여부만 로그에 기록 |
| 로그가 오류로 가득 참 | Streamlit 코드 변경 감시가 transformers 모듈을 전부 훑음 | `fileWatcherType = "none"` |
| Secrets 입력 시 `.env`의 다음 줄까지 따옴표 안에 붙여 넣음 → 키를 못 읽음 | TOML 문자열은 한 줄이어야 하는데, 고친 뒤 **Save 없이 Deploy**해서 깨진 내용이 저장됨 | App settings → Secrets에서 다시 고치고 **Save changes** → Reboot. `.env`와 TOML은 형식이 다름(TOML은 값에 따옴표 필요) |
| 첫 배포 직후 후속 질문 2건이 6~10초 (첫 글자 6~9초) | 특정하지 못함. 같은 질문을 다시 물었을 때는 2.9초 | 단계별 시간(검색어 재작성 · 검색)을 화면과 로그(`rewrite_s`, `search_s`)에 남기도록 추가. 재측정: 재작성 0.8~0.9초 · 검색 0.1초 |

### 배포 후 확인 (2026-10-13, Claude Sonnet 5.5)

| 질문 | 결과 | 응답 시간 |
|---|---|---|
| 사업 기간은 언제까지야? (예시 버튼) | 정답 "계약체결일 ∼ 2024년 12월 31일" (p.2) | 2.4초 (첫 글자 1.2초) |
| 그럼 예산은 얼마야? (후속) | 검색어를 "이 사업의 총 예산은 얼마인가요?"로 바꿔 정답 103,792,000원 (p.2) | 2.9초 (첫 글자 1.8초) |
| 그 금액에 부가세도 포함된 거야? (후속) | 검색어 "사업 예산 103,792,000원에 부가세가 포함되어 있나요?" → 포함 | 2.9초 (첫 글자 1.8초) |

- 질문당 비용 $0.006~0.008, 📈 운영 현황 탭에 3건 · 평균 응답 시간 · 비용이 집계되는 것까지 확인했다.
- 헬스 체크: `https://bugwagi-ai-tutor.streamlit.app/~/+/_stcore/health` → `ok`
