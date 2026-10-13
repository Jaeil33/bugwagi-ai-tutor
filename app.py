"""
부산과기대 학습지원 AI 조교 - Streamlit 화면 (Frontend)

실행:  streamlit run app.py

3주차(서비스화 · 배포) 변경
    - 기본 설정을 실험으로 고른 값으로 (chunk 200 / k 7 / 하이브리드 검색 / 개선 프롬프트). 조절 슬라이더는 '실험 설정'으로 접어 둠
    - 답변 스트리밍 + 검색 과정 표시, 이어지는 질문(대화 맥락), 답변 평가(👍/👎)
    - 운영: 업로드 파일을 내용별로 분리 저장, 접속당 질문 수 · 하루 API 비용 상한,
            API 오류 시 검색 결과로 대체 응답, 요청 로그 + 운영 현황 탭
"""
import hashlib
import hmac
import logging
import os
import tempfile
import uuid
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

import ops
from rag_module import MODELS, OFFLINE_DEFAULT, PROMPT_VERSIONS, SEARCH_MODES, USER_TYPES, RAGPipeline


@st.cache_resource
def load_secrets_to_env():
    """Streamlit Cloud의 Secrets(최상위 값)를 환경변수로 옮긴다. 서버 프로세스당 한 번만 실행.
    Streamlit Cloud에 처음 배포했을 때 Secrets를 넣었는데도 os.environ에 키가 없어서 직접 옮기게 했다.
    로컬(.env) · 컨테이너(환경변수)에는 secrets.toml이 없으므로 아무것도 하지 않는다."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # secrets.toml 없음 또는 형식 오류
        secrets = {}
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and not os.environ.get(key):
            os.environ[key] = str(value)
    # 값은 남기지 않고, 몇 개를 읽었는지와 키가 있는지만 기록
    ops.log_event("config", secrets=len(secrets), api_key_set=bool(os.environ.get("ANTHROPIC_API_KEY")))


load_secrets_to_env()

DEFAULT_PDF = "data/RFP_교육.pdf"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "bugwagi_uploads"
MAX_UPLOADS = 3
MAX_QUESTION_CHARS = 500
MAX_QUESTIONS_PER_SESSION = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "30"))
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "2"))
TARGET_SECONDS = 10  # 제안서 목표 응답 시간
API_KEY_SET = bool(os.environ.get("ANTHROPIC_API_KEY"))
SUGGESTIONS = ["이 사업 예산은 얼마야?", "사업 기간은 언제까지야?", "기술평가와 가격평가 비율은?", "하자담보책임기간은 얼마나 돼?"]

st.set_page_config(page_title="부과기 AI 조교", page_icon="🎓", layout="wide")

# 배포판 접근 제한: APP_PASSWORD 가 설정돼 있으면 입장 코드를 맞혀야 쓸 수 있다 (API 비용 보호)
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if APP_PASSWORD and not st.session_state.get("authed"):
    st.title("🎓 부과기 AI 조교")
    code = st.text_input("입장 코드", type="password")
    if code and hmac.compare_digest(code.encode(), APP_PASSWORD.encode()):
        st.session_state.authed = True
        st.rerun()
    elif code:
        st.error("입장 코드가 맞지 않습니다.")
    st.stop()


@st.cache_resource(show_spinner=False, max_entries=6)
def load_pipeline(pdf_paths, chunk_size, chunk_overlap):
    """같은 문서·같은 설정이면 벡터DB를 다시 만들지 않고 모든 접속자가 재사용.
    max_entries: 문서·설정 조합이 늘어도 서버 메모리가 끝없이 커지지 않게 최근 6개만 보관"""
    return RAGPipeline(list(pdf_paths), chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@st.cache_resource
def get_tracker():
    return ops.UsageTracker(DAILY_BUDGET_USD)


def save_upload(file):
    """업로드 파일을 '내용 해시/파일명'으로 저장한다.
    2주차에는 uploads/파일명 에 저장해서, 여러 사람이 같은 이름의 다른 파일을 올리면
    서로 덮어쓰고 이전 사람의 벡터DB 캐시가 재사용되는 문제가 있었다."""
    data = file.getvalue()
    folder = UPLOAD_DIR / hashlib.sha256(data).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(file.name).name
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def friendly_error(e):
    """사용자에게 보여줄 오류 설명 (자세한 내용은 로그에만 남김)"""
    if isinstance(e, anthropic.AuthenticationError):
        return "AI 서비스 인증 설정에 문제가 있어"
    if isinstance(e, anthropic.RateLimitError):
        return "요청이 몰려 지금은 AI 답변을 만들 수 없어"
    if isinstance(e, anthropic.APIConnectionError):
        return "AI 서버에 연결하지 못해"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "AI 서버가 일시적으로 응답하지 않아"
    return "AI 답변을 만드는 중 문제가 생겨"


def record_feedback(msg_id):
    value = st.session_state.get(f"fb_{msg_id}")
    previous = st.session_state.feedback_given.get(msg_id)
    if value == previous:
        return
    liked = None if value is None else value == 1
    get_tracker().add_feedback(liked=liked, previous=None if previous is None else previous == 1)
    st.session_state.feedback_given[msg_id] = value
    ops.log_event("feedback", session=st.session_state.session_id, message=msg_id, liked=liked)


def show_meta(msg):
    meta = msg["meta"]
    badge = "✅" if meta["seconds"] <= TARGET_SECONDS else "⚠️"
    first = f" (첫 글자 {meta['first_token']:.1f}초)" if meta.get("first_token") is not None else ""
    if meta["input_tokens"]:
        usage = f"토큰 입력 {meta['input_tokens']:,} / 출력 {meta['output_tokens']:,} · ${meta['cost']:.4f}"
    else:
        usage = f"프롬프트 {meta['prompt_chars']:,}자"
    st.caption(f"{badge} {meta['seconds']:.1f}초{first} · 목표 {TARGET_SECONDS}초 이내 · {meta['model']} · "
               f"{meta['user_type']} · {meta['search']} 검색 · 프롬프트 {meta['prompt']} · k={meta['k']} · {usage}")
    if meta["search_query"] != meta["question"]:
        st.caption(f"🔁 이전 대화를 반영한 검색어: {meta['search_query']}")
    with st.expander(f"🔎 참고한 자료 {len(meta['sources'])}개"):
        for i, s in enumerate(meta["sources"], start=1):
            st.markdown(f"**[{i}] {s['source']} · p.{s['page']}**")
            st.text(s["text"])
    st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=record_feedback, args=(msg["id"],))


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex[:8]  # 로그에서 같은 접속의 요청을 묶어 보기 위한 번호
    st.session_state.questions_asked = 0  # 대화를 지워도 줄지 않음 (상한 우회 방지)
    st.session_state.feedback_given = {}

# ──────────────────────────────────────────────
# 1. 사이드바: 사용자 유형 · 모델 · 문서 · 실험 설정
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    user_type = st.segmented_control("사용자 유형", USER_TYPES, default="학습자", required=True)
    model_names = list(MODELS)
    offline_index = model_names.index("오프라인 모드 (API 없이 검색 결과만)")
    model_label = st.selectbox("모델", model_names, index=offline_index if OFFLINE_DEFAULT or not API_KEY_SET else 0)
    if not API_KEY_SET:
        st.caption("API 키가 설정되지 않아 오프라인 모드로 시작했습니다.")

    st.subheader("📚 학내 자료")
    use_default = st.checkbox("기본 자료: RFP_교육.pdf", value=True,
                              help="학사규정 등 실제 학내 자료 대신 1주차에 분석한 RFP로 시연합니다.")
    uploaded_files = st.file_uploader(f"PDF 추가 업로드 (최대 {MAX_UPLOADS}개, 파일당 20MB)", type=["pdf"],
                                      accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > MAX_UPLOADS:
        st.warning(f"앞의 {MAX_UPLOADS}개 파일만 사용합니다.")

    with st.expander("🧪 실험 설정 (개발자용)"):
        st.caption("기본값은 3주차 실험에서 정답률과 비용이 가장 좋았던 조합입니다.")
        search_mode = st.segmented_control("검색 방식", SEARCH_MODES, default="하이브리드", required=True,
                                           help="하이브리드: 키워드 검색(BM25) + 의미 검색(임베딩)을 합침")
        prompt_version = st.segmented_control("프롬프트", PROMPT_VERSIONS, default="개선", required=True,
                                              help="기본: 예제 코드 원본 / 개선: 역할·형식·Few-shot·제약조건 적용")
        chunk_size = st.slider("chunk_size (청크 크기)", 100, 1500, 200, step=100)
        chunk_overlap = st.slider("chunk_overlap (겹침)", 0, min(500, chunk_size - 50), min(100, chunk_size - 50), step=50)
        k = st.slider("k (검색할 청크 수)", 1, 10, 7)

    remaining_slot = st.empty()  # 질문을 처리한 뒤에 채움
    if st.button("🗑️ 대화 지우기", width="stretch"):
        st.session_state.messages = []

# 사용할 문서 목록 정리
pdf_paths = [DEFAULT_PDF] if use_default else []
pdf_paths += [save_upload(f) for f in (uploaded_files or [])[:MAX_UPLOADS]]

# ──────────────────────────────────────────────
# 2. 본문
# ──────────────────────────────────────────────
st.title("🎓 부과기 AI 조교")
st.caption("부산과학기술대학교 학내 자료 기반 RAG 질의응답 서비스 · "
           "1주차 제안서(SFR-003 생성형 AI 서비스, SFR-004 사용자별 맞춤형 프롬프트) 구현")

tab_chat, tab_exp, tab_ops = st.tabs(["💬 챗봇", "📊 실험 결과", "📈 운영 현황"])

rag = None
picked = None
with tab_chat:
    if not pdf_paths:
        st.info("왼쪽 사이드바에서 기본 자료를 선택하거나 PDF를 업로드하면 대화를 시작할 수 있습니다.")
    else:
        try:
            with st.spinner("문서를 분석 중입니다... (서버가 막 켜졌다면 임베딩 모델을 불러오느라 1~2분 걸릴 수 있어요)"):
                rag = load_pipeline(tuple(pdf_paths), chunk_size, chunk_overlap)
        except Exception as e:  # 손상된 PDF, 글자가 없는 스캔 PDF 등
            ops.log_event("index_error", level=logging.ERROR, error=type(e).__name__, detail=str(e))
            st.error(f"자료를 분석하지 못했습니다. 다른 PDF로 다시 시도해 주세요. ({e})")
        else:
            st.caption(f"📚 문서 {len(pdf_paths)}개 → 청크 {len(rag.chunks)}개 · {search_mode} 검색 · "
                       f"chunk_size={chunk_size}, overlap={chunk_overlap}, k={k}")

    # 기존 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("meta", {}).get("notice"):
                st.info(msg["meta"]["notice"])
            st.markdown(msg["content"])
            if msg.get("meta"):
                show_meta(msg)

    if rag is not None and not st.session_state.messages:
        picked = st.pills("예시 질문", SUGGESTIONS, key="suggestion")

with tab_exp:
    summary_path = "results/summary.csv"
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        offline = "정답률" not in summary
        st.subheader("설정별 결과 요약 · 테스트 질문 16개")
        if offline:
            st.caption("🔌 오프라인 모드로 측정: Claude 답변 없이 검색 단계(찾아온 자료의 품질)와 프롬프트 길이만 비교했습니다.")
        st.dataframe(summary, width="stretch", hide_index=True, alt="설정별 정답률, 근거 포함률, 비용 비교표")

        quality_col = "근거포함률" if offline else "정답률"
        cost_col = "평균프롬프트글자수" if offline else "질문당 비용(USD)"
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{quality_col} (%)**")
            st.bar_chart(summary.set_index("설정")[quality_col], horizontal=True, alt=f"설정별 {quality_col}")
        with c2:
            st.markdown(f"**{cost_col}**")
            st.bar_chart(summary.set_index("설정")[cost_col], horizontal=True, alt=f"설정별 {cost_col}")

        followup_path = "results/followup.csv"
        if os.path.exists(followup_path):
            st.subheader("후속 질문: 앞 대화 반영 전후")
            st.dataframe(pd.read_csv(followup_path), width="stretch", hide_index=True,
                         alt="후속 질문을 앞 대화 없이 물었을 때와 반영했을 때의 검색어와 정답 비교")

        deploy_path = "results/deploy_check.csv"
        if os.path.exists(deploy_path):
            deploy = pd.read_csv(deploy_path)
            st.subheader("배포 설정 점검 · Claude Sonnet 5.5 + 3주차 최종 설정")
            st.caption(f"정답률 {deploy['정답'].mean():.0%} · 평균 응답 {deploy['응답시간(초)'].mean():.1f}초 "
                       f"(첫 글자 {deploy['첫 글자(초)'].mean():.1f}초) · 질문당 ${deploy['비용(USD)'].mean():.4f}")
            st.dataframe(deploy, width="stretch", hide_index=True, alt="배포 기본 설정으로 측정한 문항별 결과")

        raw_path = "results/raw_results.csv"
        if os.path.exists(raw_path):
            with st.expander("문항별 상세 결과"):
                st.dataframe(pd.read_csv(raw_path), width="stretch", hide_index=True, alt="실험 설정별 문항 결과")
    else:
        st.info("아직 실험 결과가 없습니다. 터미널에서 `python experiment.py`를 실행하세요.")

# 사용자 입력 처리 (입력창은 화면 하단에 고정)
question = st.chat_input("질문을 입력하세요", max_chars=MAX_QUESTION_CHARS, submit_mode="disable", disabled=rag is None)
question = question or picked

if rag is not None and question:
    tracker = get_tracker()
    model_id = MODELS[model_label]
    notice = None
    if model_id != "offline":
        if st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
            model_id = "offline"
            notice = f"이번 접속에서 쓸 수 있는 AI 답변 {MAX_QUESTIONS_PER_SESSION}회를 모두 써서, 찾은 자료 원문을 보여드립니다."
        elif tracker.over_budget():
            model_id = "offline"
            notice = "오늘 준비된 AI 사용량을 모두 써서, 찾은 자료 원문을 보여드립니다. 내일 다시 이용해 주세요."
        else:
            st.session_state.questions_asked += 1

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question, "id": uuid.uuid4().hex})
    error = None
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if notice:
                st.info(notice)
            try:
                with st.status(":shimmer[자료를 찾는 중]", type="compact") as status:
                    turn = rag.prepare(question, history, k=k, model_id=model_id, prompt_version=prompt_version,
                                       user_type=user_type, search_mode=search_mode)
                    if turn.search_query != question:
                        st.write(f"이전 대화를 반영한 검색어: {turn.search_query}")
                    st.write("찾은 자료: " + ", ".join(f"p.{d.metadata['page_no']}" for d in turn.docs))
                    status.update(label=f"자료 {len(turn.docs)}개를 찾았습니다", state="complete")

                answer_slot = st.empty()
                try:
                    with answer_slot.container():
                        st.write_stream(turn.stream())
                except anthropic.AnthropicError as e:
                    # Claude 호출이 실패해도 서비스는 멈추지 않는다: 찾은 자료 원문으로 대체
                    error = e
                    notice = f"{friendly_error(e)} 찾은 자료 원문을 대신 보여드립니다."
                    turn.model_id, turn.first_token_seconds = "offline", None
                    with answer_slot.container():
                        st.warning(notice)
                        st.write_stream(turn.stream())
                if turn.refused:
                    answer_slot.markdown(turn.answer)
            except Exception as e:  # 예상하지 못한 오류: 내용은 로그에만 남기고 화면에는 짧게
                ops.log_event("error", level=logging.ERROR, session=st.session_state.session_id,
                              error=type(e).__name__, detail=str(e))
                st.error("일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
                st.session_state.messages.pop()
                st.stop()

            msg = {
                "role": "assistant", "content": turn.answer, "id": uuid.uuid4().hex,
                "meta": {
                    "question": question, "search_query": turn.search_query, "notice": notice,
                    "seconds": turn.seconds, "first_token": turn.first_token_seconds,
                    "model": model_label.split(" (")[0] if turn.model_id != "offline" else "오프라인",
                    "user_type": user_type, "prompt": prompt_version, "search": search_mode, "k": k,
                    "sources": turn.sources, "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
                    "prompt_chars": turn.prompt_chars, "cost": turn.cost,
                },
            }
            show_meta(msg)
    st.session_state.messages.append(msg)

    status_text = "오류→대체" if error else ("거절" if turn.refused else ("제한→오프라인" if notice else "정상"))
    tracker.add({"model": msg["meta"]["model"], "search": search_mode, "seconds": round(turn.seconds, 2),
                 "first_token": None if turn.first_token_seconds is None else round(turn.first_token_seconds, 2),
                 "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
                 "cost": turn.cost, "status": status_text})
    ops.log_event(
        "answer", level=logging.WARNING if error else logging.INFO,
        session=st.session_state.session_id, model=model_id, served_model=turn.served_model or None,
        search=search_mode, k=k, prompt=prompt_version, user_type=user_type,
        q_chars=len(question), followup=bool(history), rewritten=turn.search_query != question,
        pages=[d.metadata["page_no"] for d in turn.docs], seconds=round(turn.seconds, 2),
        first_token=turn.first_token_seconds and round(turn.first_token_seconds, 2),
        input_tokens=turn.input_tokens, output_tokens=turn.output_tokens, cost=round(turn.cost, 6),
        status=status_text, request_id=turn.request_id, error=type(error).__name__ if error else None,
    )

remaining = max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked)
remaining_slot.caption(f"이번 접속에서 AI 답변 {remaining}회 남음")

# 운영 현황은 질문 처리 뒤에 그려야 방금 질문까지 반영된다
with tab_ops:
    snap = get_tracker().snapshot()
    st.caption(f"서버가 켜진 뒤({snap['started_at']:%m-%d %H:%M} KST) 모든 접속자의 요청을 모은 값입니다. "
               "서버 메모리에만 있어 재시작하면 초기화되고, 요청별 상세 로그는 클라우드 로그에 남습니다. "
               "질문 내용은 저장하지 않습니다.")
    records = pd.DataFrame(snap["records"])
    if records.empty:
        st.info("아직 처리한 질문이 없습니다.")
    else:
        fb = snap["feedback"]
        with st.container(horizontal=True):
            st.metric("처리한 질문", f"{len(records)}건", border=True)
            st.metric("평균 응답 시간", f"{records['seconds'].mean():.1f}초", border=True)
            st.metric("95% 응답 시간", f"{records['seconds'].quantile(0.95):.1f}초", border=True,
                      help="요청 100건 중 95건은 이 시간 안에 끝났다는 뜻 (느린 요청 확인용)")
            st.metric("오늘 API 비용", f"${snap['cost_today']:.3f}", border=True,
                      help=f"하루 예산 ${DAILY_BUDGET_USD:.2f}를 넘으면 오프라인 모드로 전환")
            st.metric("오류·대체 응답", f"{(records['status'] == '오류→대체').sum()}건", border=True)
            st.metric("답변 평가", f"👍 {fb['좋아요']} · 👎 {fb['아쉬워요']}", border=True)
        st.markdown("**요청별 응답 시간 (초)**")
        st.line_chart(records["seconds"], alt="최근 요청의 응답 시간 추이")
        st.dataframe(
            records.rename(columns={"시각": "시각(KST)", "model": "모델", "search": "검색", "seconds": "응답(초)",
                                    "first_token": "첫 글자(초)", "input_tokens": "입력토큰",
                                    "output_tokens": "출력토큰", "cost": "비용(USD)", "status": "결과"}),
            width="stretch", hide_index=True, alt="최근 요청 목록",
        )
