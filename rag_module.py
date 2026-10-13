"""
부산과기대 학습지원 AI 조교 - RAG 핵심 로직 (Backend)

1주차 제안서의 시스템 구성을 그대로 코드로 옮겼습니다.
    학내 자료(PDF) → Vector Store(분할·임베딩) → AI 에이전트(Claude) → 유형별 응답

예제 코드의 8단계 구조는 그대로 유지하고, 아래를 바꿨습니다.
    - 임베딩: OpenAI → 무료 로컬 한국어 모델 (jhgan/ko-sroberta-multitask)
    - LLM:    GPT-4o → Claude (Sonnet 5.5 / Haiku 4.5)
    - 프롬프트: 기본(예제 원본) / 개선(역할·형식·Few-shot·제약조건) 2종 + 교수자/학습자 분리
    - 답변과 함께 참고한 자료(쪽수), 응답 시간, 토큰 수를 반환

3주차 개선
    - 하이브리드 검색: 키워드 검색(BM25) + 의미 검색(FAISS)을 RRF로 합침 → "사업 기간" 같은 키워드형 질문 보완
    - 대화 맥락: 후속 질문("그 기간은 언제부터야?")을 독립 질문으로 바꿔 검색하고, 답변에도 최근 대화를 넘김
    - 스트리밍: 답변을 생성되는 대로 내보내고, 첫 글자까지 걸린 시간을 따로 잼
    - 비용 계산: 모델 단가로 질문당 비용을 계산 (운영 현황 · 일일 예산에 사용)
"""
import os
import re
import time
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# .env 파일에 저장된 API 키 로드
load_dotenv()

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
REWRITE_MODEL = "claude-haiku-4-5"  # 검색어 재작성은 짧은 작업이라 저렴한 모델로

# 화면에 보이는 이름 → 실제 모델 ID
MODELS = {
    "Claude Sonnet 5.5 (데모용)": "claude-sonnet-5-5",
    "Claude Haiku 4.5 (실험용·저렴)": "claude-haiku-4-5",
    "오프라인 모드 (API 없이 검색 결과만)": "offline",
}

# USD / 1M tokens (입력, 출력)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# .env 에 LLM_MODE=offline 이면 화면 기본값을 오프라인 모드로
OFFLINE_DEFAULT = os.environ.get("LLM_MODE", "").lower() == "offline"

USER_TYPES = ["학습자", "교수자"]
PROMPT_VERSIONS = ["개선", "기본"]
SEARCH_MODES = ["하이브리드", "의미 검색"]
HISTORY_TURNS = 3  # 답변할 때 함께 넘기는 최근 대화 수 (질문·답변 한 쌍 = 1)

REFUSAL_MESSAGE = "죄송합니다. 이 질문에는 답변할 수 없습니다. 질문을 바꿔서 다시 시도해 주세요."

_embeddings = None
_client = None


def get_embeddings():
    """임베딩 모델은 처음 한 번만 불러와서 재사용 (첫 실행 시 약 0.5GB 다운로드)"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_client():
    global _client
    if _client is None:
        # ANTHROPIC_API_KEY 환경변수 사용.
        # 워크스페이스에 묶이지 않은 키는 ANTHROPIC_WORKSPACE_ID(wrkspc_...)도 함께 보내야 함
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        _client = anthropic.Anthropic(default_headers=headers)
    return _client


def estimate_cost(model_id, input_tokens, output_tokens):
    price_in, price_out = PRICES.get(model_id, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# ──────────────────────────────────────────────
# 하이브리드 검색 (5단계 보강)
# ──────────────────────────────────────────────

def tokenize_ko(text):
    """BM25용 토큰화. 한글은 2글자씩 잘라 조사·띄어쓰기가 달라도 겹치게 한다.
    ("사업 기간은" → 사업 / 기간 / 간은, "사업기간" → 사업 / 업기 / 기간)
    영문·숫자는 통째로 쓴다 (SFR-004 → sfr / 004)."""
    tokens = []
    for word in re.findall(r"[가-힣]+|[0-9a-z]+", text.lower()):
        if len(word) > 1 and "가" <= word[0] <= "힣":
            tokens += [word[i:i + 2] for i in range(len(word) - 1)]
        else:
            tokens.append(word)
    return tokens


def rrf_merge(rankings, k, k0=60):
    """Reciprocal Rank Fusion: 각 검색 결과에서 r위인 조각에 1/(60+r)점을 주고 합산해 다시 정렬.
    점수 단위가 다른 두 검색(BM25 점수, 코사인 유사도)을 순위만으로 공평하게 합칠 수 있다."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k0 + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


# ──────────────────────────────────────────────
# 프롬프트 (6단계)
# ──────────────────────────────────────────────

# [기본] 예제 코드의 프롬프트 원본 그대로 - 실험의 비교 기준
BASIC_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in Korean:"""

# [개선] 역할 지정 + 형식 지정(#명령문/#제약조건/#출력형식) + Few-shot + 제약조건
IMPROVED_SYSTEM = """#명령문
당신은 부산과학기술대학교 교수학습지원센터의 AI 조교 '부과기 AI 조교'입니다.
아래 #제약조건을 지켜, 사용자가 질문하면 <자료>에 담긴 학내 문서만 근거로 #출력형식에 맞게 답변하세요.

#제약조건
- 반드시 <자료>에 있는 내용만 근거로 답한다. 자료에 없는 내용은 추측하거나 지어내지 않는다.
- 자료에서 답을 찾을 수 없으면 "제공된 자료에서 해당 내용을 찾을 수 없습니다."라고 답하고, 학과 사무실이나 교수학습지원센터에 문의하도록 안내한다.
- 숫자, 날짜, 금액, 요구사항 코드는 자료에 적힌 그대로 옮긴다.
- 근거로 쓴 자료의 쪽수를 (p.쪽수) 형태로 반드시 표기한다.
- 앞선 대화가 있으면 이어지는 질문으로 이해하되, 근거는 이번 <자료>에서만 찾는다.
- 한국어로 답한다.

#사용자 유형
지금 질문하는 사람은 '{user_type}'입니다.

#출력형식
{output_format}

#예시
{examples}"""

OUTPUT_FORMATS = {
    "학습자": """- 처음 듣는 학생도 이해할 수 있게 쉬운 말로 3~5줄 이내로 설명한다.
- 전문 용어가 나오면 괄호 안에 짧게 풀어 쓴다.
- 마지막 줄에 "📌 출처: (p.쪽수)"를 적는다.
- 그 다음 줄에 "💡 이어서 물어볼 만한 질문: [질문 1개]"를 적는다.""",
    "교수자": """- 첫 줄에 한 문장 핵심 요약을 적는다.
- 이어서 핵심 내용을 마크다운 표(| 항목 | 내용 |)로 정리한다.
- 표 아래에 "🎓 수업·자료 활용 포인트:"로 시작하는 1~2줄을 적는다.
- 마지막 줄에 "📌 출처: (p.쪽수)"를 적는다.""",
}

# Few-shot 예시는 테스트 질문과 겹치지 않는 가상의 학사 자료로 작성 (정답 유출 방지)
EXAMPLES = {
    "학습자": """<예시 자료>
[자료 1 | 학사안내.pdf p.4] 중앙도서관 열람실은 학기 중 07:00~23:00 운영하며, 시험기간에는 24시간 개방한다.
</예시 자료>
질문: 도서관 열람실 몇 시까지 해?
답변:
학기 중에는 오전 7시부터 밤 11시까지 열고, 시험기간에는 24시간 열려 있어요.
📌 출처: (p.4)
💡 이어서 물어볼 만한 질문: 시험기간은 언제부터 언제까지인가요?

<예시 자료>
[자료 1 | 학사안내.pdf p.4] 중앙도서관 열람실은 학기 중 07:00~23:00 운영한다.
</예시 자료>
질문: 기숙사 식당 메뉴 알려줘
답변:
제공된 자료에서 해당 내용을 찾을 수 없습니다. 기숙사 행정실이나 교수학습지원센터에 문의해 주세요.""",
    "교수자": """<예시 자료>
[자료 1 | 학사안내.pdf p.7] 성적 입력 기간은 학기말 시험 종료 후 7일 이내이며, 성적 정정은 공시 후 3일 이내 신청할 수 있다.
</예시 자료>
질문: 성적 처리 일정 정리해줘
답변:
성적은 시험 종료 후 7일 안에 입력하고, 정정은 공시 후 3일 안에 신청받습니다.

| 항목 | 내용 |
|---|---|
| 성적 입력 | 학기말 시험 종료 후 7일 이내 |
| 성적 정정 신청 | 성적 공시 후 3일 이내 |

🎓 수업·자료 활용 포인트: 첫 수업 OT 자료에 정정 신청 기한을 미리 안내하면 문의를 줄일 수 있습니다.
📌 출처: (p.7)""",
}

IMPROVED_USER = """<자료>
{context}
</자료>

질문: {question}"""

# 후속 질문을 독립적인 검색 질문으로 바꾸는 프롬프트 (5-1단계)
REWRITE_SYSTEM = """당신은 검색어를 다듬는 도우미입니다.
<대화 기록>과 마지막 질문을 보고, 마지막 질문을 앞의 대화 없이도 뜻이 통하는 검색 질문 한 문장으로 바꿔 쓰세요.
- '그거', '그 기간', '거기'처럼 앞 대화를 가리키는 말은 실제 대상으로 바꿉니다.
- 이미 혼자서 뜻이 통하는 질문이면 그대로 씁니다.
- 질문에 답하지 말고, 바꾼 질문 한 문장만 출력합니다."""


def format_context(docs):
    """검색된 조각을 '[자료 n | 파일명 p.쪽]' 꼬리표와 함께 하나의 텍스트로 합침"""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[자료 {i} | {doc.metadata['source_name']} p.{doc.metadata['page_no']}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_prompt(docs, question, prompt_version, user_type, history=()):
    """(system, messages)를 만듦. 최근 대화는 앞에 그대로 두고, 마지막 질문에만 검색 자료를 붙인다"""
    context = format_context(docs)
    if prompt_version == "기본":
        system, user = None, BASIC_TEMPLATE.format(context=context, question=question)
    else:
        system = IMPROVED_SYSTEM.format(
            user_type=user_type,
            output_format=OUTPUT_FORMATS[user_type],
            examples=EXAMPLES[user_type],
        )
        user = IMPROVED_USER.format(context=context, question=question)
    return system, [*history, {"role": "user", "content": user}]


def rewrite_query(question, history, model_id):
    """[5-1단계] 대화 맥락 반영: 후속 질문을 독립적인 검색 질문으로 바꿈.
    반환: (검색어, 입력 토큰, 출력 토큰)"""
    if not history:
        return question, 0, 0
    last_question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if model_id == "offline":
        # API 없이: 직전 질문을 붙여서 검색 ("하자담보책임기간은? 그 기간은 언제부터야?")
        return f"{last_question} {question}".strip(), 0, 0

    transcript = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI 조교'}: {m['content'][:500]}" for m in history
    )
    try:
        response = get_client().messages.create(
            model=REWRITE_MODEL,
            max_tokens=300,  # 한 문장만 받으면 됨
            system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": f"<대화 기록>\n{transcript}\n</대화 기록>\n\n마지막 질문: {question}"}],
        )
    except anthropic.AnthropicError:
        # 재작성에 실패해도(키 오류 · 네트워크 · 과부하) 답변은 계속: 직전 질문을 붙이는 방식으로 대신함
        return f"{last_question} {question}".strip(), 0, 0
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or question, response.usage.input_tokens, response.usage.output_tokens


# ──────────────────────────────────────────────
# LLM 호출 (7단계)
# ──────────────────────────────────────────────

def offline_answer(docs, user_type):
    """API 없이 동작하는 대체 응답: LLM이 생성하지 않고, 검색된 자료 원문을 그대로 보여줌.
    (검색 단계까지는 실제로 동작하므로 RAG의 '찾기' 품질은 그대로 확인할 수 있음)"""
    lines = [f"🔌 **오프라인 모드** — Claude 대신, {user_type}의 질문과 관련도가 높은 순서로 찾은 자료 원문을 보여드립니다."]
    for i, d in enumerate(docs, start=1):
        text = " ".join(d.page_content.split())  # 줄바꿈·공백 정리
        if len(text) > 250:
            text = text[:250] + "…"
        lines += ["", f"**{i}위 · {d.metadata['source_name']} p.{d.metadata['page_no']}**", f"> {text}"]
    if not docs:
        lines.append("검색된 자료가 없습니다.")
    return "\n".join(lines)


@dataclass
class Turn:
    """질문 하나의 처리 과정과 결과. 화면 · 실험 · 운영 로그가 같은 값을 쓴다."""
    question: str
    search_query: str
    docs: list
    system: str | None
    messages: list
    model_id: str
    user_type: str
    started: float
    rewrite_input_tokens: int = 0
    rewrite_output_tokens: int = 0
    rewrite_seconds: float = 0.0  # 단계별 소요 시간: 느릴 때 어디가 원인인지 보려고 기록
    search_seconds: float = 0.0
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    first_token_seconds: float | None = None
    seconds: float = 0.0
    refused: bool = False
    served_model: str = ""
    request_id: str | None = None

    def stream(self):
        """[7~8단계] 답변 생성 → 출력. 글자를 생성되는 대로 내보낸다 (st.write_stream에 그대로 넘길 수 있음)"""
        chunks = iter([offline_answer(self.docs, self.user_type)]) if self.model_id == "offline" else self._stream_claude()
        parts = []
        for text in chunks:
            if self.first_token_seconds is None:
                self.first_token_seconds = time.time() - self.started
            parts.append(text)
            yield text
        self.seconds = time.time() - self.started
        self.answer = REFUSAL_MESSAGE if self.refused else "".join(parts)

    def _stream_claude(self):
        params = {"model": self.model_id, "max_tokens": 16000, "messages": self.messages}
        if self.system:
            params["system"] = self.system
        client = get_client()
        if self.model_id == "claude-sonnet-5-5":
            # 채팅 응답 속도(제안서 목표: 10초 이내)를 위해 effort는 low.
            # 드물게 안전 분류기가 거절하면 서버가 다른 모델로 자동 재시도하도록 fallbacks 설정.
            stream_ctx = client.beta.messages.stream(
                **params,
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        else:
            stream_ctx = client.messages.stream(**params)

        with stream_ctx as stream:
            self.request_id = stream.request_id  # 장애 문의 시 Anthropic에 전달하는 요청 번호 (로그에 남김)
            yield from stream.text_stream
            final = stream.get_final_message()
        self.input_tokens = final.usage.input_tokens
        self.output_tokens = final.usage.output_tokens
        self.refused = final.stop_reason == "refusal"
        self.served_model = final.model

    @property
    def cost(self):
        """질문 하나의 비용(USD): 검색어 재작성 + 답변 생성. 대체 모델이 답했으면 그 모델 단가로."""
        answer_model = self.served_model if self.served_model in PRICES else self.model_id
        return (estimate_cost(REWRITE_MODEL, self.rewrite_input_tokens, self.rewrite_output_tokens)
                + estimate_cost(answer_model, self.input_tokens, self.output_tokens))

    @property
    def sources(self):
        return [{"source": d.metadata["source_name"], "page": d.metadata["page_no"], "text": d.page_content}
                for d in self.docs]

    @property
    def prompt_chars(self):
        """LLM에 보내는 프롬프트 길이 (비용과 비례)"""
        return len(self.system or "") + sum(len(m["content"]) for m in self.messages)

    def to_dict(self):
        return {
            "answer": self.answer,
            "search_query": self.search_query,
            "prompt_chars": self.prompt_chars,
            "sources": self.sources,
            "seconds": self.seconds,
            "first_token_seconds": self.first_token_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "rewrite_seconds": self.rewrite_seconds,
            "search_seconds": self.search_seconds,
        }


# ──────────────────────────────────────────────
# RAG 파이프라인
# ──────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, pdf_paths, chunk_size=400, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # [1단계] 문서 로드 (Document Load)
        docs = []
        for path in pdf_paths:
            for doc in PyMuPDFLoader(path).load():
                doc.metadata["source_name"] = os.path.basename(path)
                doc.metadata["page_no"] = doc.metadata.get("page", 0) + 1  # 0부터 시작 → 1부터
                docs.append(doc)

        # [2단계] 문서 분할 (Text Split)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,        # 청크 사이즈 조절
            chunk_overlap=chunk_overlap,  # 오버랩 조절
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.chunks = [c for c in text_splitter.split_documents(docs) if c.page_content.strip()]
        if not self.chunks:
            raise ValueError("PDF에서 글자를 찾지 못했습니다. 스캔한 이미지 PDF는 지원하지 않습니다.")
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i  # 두 검색 결과를 합칠 때 같은 조각인지 알아보는 번호

        # [3~4단계] 임베딩 및 벡터 DB 저장 (Embedding & Vector DB)
        self.vectorstore = FAISS.from_documents(documents=self.chunks, embedding=get_embeddings())
        # [3~4단계 추가] 키워드 검색용 BM25 색인
        self.bm25 = BM25Okapi([tokenize_ko(c.page_content) for c in self.chunks])

    def retrieve(self, query, k=5, search_mode="하이브리드"):
        """[5단계] 검색: 의미 검색(FAISS) 또는 하이브리드(BM25 + FAISS → RRF)"""
        if search_mode == "의미 검색":
            return self.vectorstore.similarity_search(query, k=k)
        n = min(len(self.chunks), max(20, k * 4))  # 후보는 넉넉히 뽑아서 합친다
        vector_ids = [d.metadata["chunk_id"] for d in self.vectorstore.similarity_search(query, k=n)]
        scores = self.bm25.get_scores(tokenize_ko(query))
        keyword_ids = [i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n] if scores[i] > 0]
        return [self.chunks[i] for i in rrf_merge([vector_ids, keyword_ids], k)]

    def prepare(self, question, history=(), k=5, model_id="claude-haiku-4-5", prompt_version="개선",
                user_type="학습자", search_mode="하이브리드"):
        """[5~6단계] 검색어 정리 → 검색 → 프롬프트. 답변 생성(7단계)은 Turn.stream()에서."""
        started = time.time()
        history = list(history)[-HISTORY_TURNS * 2:]
        while history and history[0]["role"] != "user":  # 대화는 사용자 질문부터 시작해야 함
            history.pop(0)

        search_query, rw_in, rw_out = rewrite_query(question, history, model_id)
        rewritten_at = time.time()
        docs = self.retrieve(search_query, k=k, search_mode=search_mode)
        searched_at = time.time()
        system, messages = build_prompt(docs, question, prompt_version, user_type, history)
        return Turn(question=question, search_query=search_query, docs=docs, system=system, messages=messages,
                    model_id=model_id, user_type=user_type, started=started,
                    rewrite_input_tokens=rw_in, rewrite_output_tokens=rw_out,
                    rewrite_seconds=rewritten_at - started, search_seconds=searched_at - rewritten_at)

    def ask(self, question, history=(), **kwargs):
        """스트리밍 없이 한 번에 답을 받음 (실험 스크립트용)"""
        turn = self.prepare(question, history, **kwargs)
        for _ in turn.stream():
            pass
        return turn.to_dict()
