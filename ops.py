"""
운영 도구: 로그 · 사용량 집계 · 일일 예산 (배포 튜토리얼 1-3 비용 관리, 4 모니터링 및 로깅)

- 로그: 요청마다 JSON 한 줄을 표준출력에 남긴다. 클라우드는 표준출력을 그대로 모아 준다.
  (Streamlit Cloud: Manage app → Logs / Cloud Run: Cloud Logging / App Runner: CloudWatch)
  질문 원문에는 개인정보가 섞일 수 있어 남기지 않고 길이만 기록한다.
- 사용량: 서버 프로세스 하나가 모든 접속자의 질문 수 · 응답 시간 · 비용을 함께 집계한다.
  메모리에만 두므로 앱이 재시작되면 초기화된다.
- 예산: 오늘(한국 시간) 쓴 API 비용이 DAILY_BUDGET_USD를 넘으면 앱이 오프라인 모드로 전환한다.
"""
import json
import logging
import sys
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))

logger = logging.getLogger("bugwagi")
if not logger.handlers:  # Streamlit은 화면을 다시 그릴 때마다 app.py를 재실행하므로 핸들러 중복 방지
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event, level=logging.INFO, **fields):
    logger.log(level, json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


def now_kst():
    return datetime.now(KST)


class UsageTracker:
    """모든 접속자가 공유하는 사용량 집계 (Streamlit은 접속자마다 스레드가 달라서 lock으로 보호)"""

    def __init__(self, daily_budget_usd):
        self.daily_budget_usd = daily_budget_usd
        self.started_at = now_kst()
        self._lock = threading.Lock()
        self._day = self.started_at.date()
        self._cost_today = 0.0
        self._records = deque(maxlen=500)  # 최근 요청만 보관
        self._feedback = {"좋아요": 0, "아쉬워요": 0}

    def _roll_day(self):
        today = now_kst().date()
        if today != self._day:
            self._day, self._cost_today = today, 0.0

    def add(self, record):
        with self._lock:
            self._roll_day()
            self._cost_today += record.get("cost", 0.0)
            self._records.append({"시각": now_kst().strftime("%m-%d %H:%M:%S"), **record})

    def add_feedback(self, liked, previous=None):
        """liked: 이번 평가(True/False, 취소면 None), previous: 같은 답변에 앞서 준 평가 → 바꾸면 이전 표는 뺀다"""
        with self._lock:
            if previous is not None:
                self._feedback["좋아요" if previous else "아쉬워요"] -= 1
            if liked is not None:
                self._feedback["좋아요" if liked else "아쉬워요"] += 1

    def over_budget(self):
        with self._lock:
            self._roll_day()
            return self._cost_today >= self.daily_budget_usd

    def snapshot(self):
        with self._lock:
            self._roll_day()
            return {
                "records": list(self._records),
                "cost_today": self._cost_today,
                "feedback": dict(self._feedback),
                "started_at": self.started_at,
            }
