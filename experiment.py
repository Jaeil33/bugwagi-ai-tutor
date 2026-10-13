"""
파라미터 비교 실험 스크립트

기본값(chunk_size 400 / overlap 100 / k 3 / 기본 프롬프트 / 의미 검색)에서 한 번에 하나만 바꿔 가며
같은 테스트 질문 16개를 던지고, 결과를 results/ 폴더에 CSV로 저장합니다.
3주차: 검색 방식(하이브리드) 실험, 최종 추천 설정, 후속 질문(대화 맥락) 실험을 추가했습니다.

실행:  python experiment.py
       python experiment.py --only "H. 3주차 최종"   (일부 실험만 다시 돌려 기존 결과에서 그 행만 교체)
       python experiment.py --deploy-check            (배포 기본 모델 Sonnet 5.5 + 최종 설정으로 16문항 확인)
       (LLM_MODE=offline 이면 API 없이 검색 단계만 측정해 results/offline/ 에 저장)
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from rag_module import RAGPipeline

load_dotenv()

PDF_PATHS = ["data/RFP_교육.pdf"]
MODEL_ID = "claude-haiku-4-5"  # 실험은 저렴한 모델로
DEPLOY_MODEL_ID = "claude-sonnet-5-5"  # 배포한 앱의 기본 모델
OFFLINE = os.environ.get("LLM_MODE", "").lower() == "offline"  # API 없이 검색 단계만 측정
OUT_DIR = "results/offline" if OFFLINE else "results"

BASE = {"chunk_size": 400, "chunk_overlap": 100, "k": 3, "prompt": "기본", "search": "의미 검색"}
WEEK2 = {"chunk_size": 400, "chunk_overlap": 100, "k": 5, "prompt": "개선", "search": "의미 검색"}  # 2주차 추천 설정
CANDIDATE = {"chunk_size": 200, "chunk_overlap": 100, "k": 5, "prompt": "개선", "search": "하이브리드"}  # 3주차 1차 후보
FINAL = {"chunk_size": 200, "chunk_overlap": 100, "k": 7, "prompt": "개선", "search": "하이브리드"}  # 3주차 추천 설정

# (실험 이름, 바꾼 값)
EXPERIMENTS = [
    ("기준", {}),
    ("A. chunk_size", {"chunk_size": 200}),
    ("A. chunk_size", {"chunk_size": 800}),
    ("B. overlap", {"chunk_overlap": 0}),
    ("B. overlap", {"chunk_overlap": 200}),
    ("C. k", {"k": 1}),
    ("C. k", {"k": 5}),
    ("D. 프롬프트", {"prompt": "개선"}),
    ("E. 검색 방식", {"search": "하이브리드"}),  # 3주차
    ("F. 2주차 추천", {k: v for k, v in WEEK2.items() if BASE[k] != v}),
    ("G. 3주차 후보 (k=5)", {k: v for k, v in CANDIDATE.items() if BASE[k] != v}),  # Q3를 놓쳐서 k를 늘림
    ("H. 3주차 최종", {k: v for k, v in FINAL.items() if BASE[k] != v}),
]

# keywords: 그룹마다 하나 이상 포함되면 통과, 모든 그룹을 통과해야 정답
TEST_SET = [
    {"q": "이 사업 예산은 얼마야?", "keywords": [["103,792,000", "1억 379만"]], "pages": [2]},
    {"q": "사업 기간은 언제까지야?", "keywords": [["12월 31일", "12. 31", "12.31", "12/31"]], "pages": [2]},
    {"q": "요구사항은 총 몇 개고 어떻게 나뉘어?", "keywords": [["17"], ["기능"], ["보안"]], "pages": [5, 6]},
    {"q": "SFR-004는 무슨 요구사항이야?", "keywords": [["맞춤형 프롬프트", "맞춤 프롬프트", "맞춤형프롬프트"]], "pages": [3, 6]},
    {"q": "RAG는 어떤 데이터를 기반으로 답변해야 해?", "keywords": [["드라이브"], ["웹검색", "웹 검색"]], "pages": [2, 3]},
    {"q": "학생 입장에서 이 서비스의 기대효과는?", "keywords": [["즉시", "바로", "실시간"], ["맞춤"]], "pages": [3]},
    {"q": "기술평가와 가격평가 비율은?", "keywords": [["80"], ["20"]], "pages": [11]},
    {"q": "제안서 발표 시간은?", "keywords": [["10분"]], "pages": [10]},
    {"q": "이 시스템의 월 이용료는 얼마야?", "keywords": None, "pages": []},  # 문서에 없음 → 모른다고 해야 정답
    # 3주차 추가: 질문 9개로는 1문항이 11%p라 결과가 흔들려서, 키워드형 · 서술형 질문을 더함
    {"q": "하자담보책임기간은 얼마나 돼?", "keywords": [["1년"]], "pages": [15]},
    {"q": "제안서는 몇 쪽 이내로 써야 해?", "keywords": [["50쪽", "50페이지", "50 쪽"]], "pages": [7]},
    {"q": "제안서 제출 마감은 언제야?", "keywords": [["10월 2일", "10.2", "10. 2"], ["10:00", "10시", "오전 10"]], "pages": [10, 14]},
    {"q": "협상적격자는 어떤 기준으로 뽑아?", "keywords": [["70%", "70 %", "70퍼센트"]], "pages": [12]},
    {"q": "계약하고 나서 사업수행계획서는 언제까지 내야 해?", "keywords": [["10일"]], "pages": [15]},
    {"q": "제안 내용에 대해 문의하려면 어디로 연락해?", "keywords": [["7283", "LINC"]], "pages": [16]},
    {"q": "이 사업에 입찰한 업체는 몇 곳이야?", "keywords": None, "pages": []},  # 문서에 없음
]

# 후속 질문: 두 번째 질문만으로는 무엇을 묻는지 알 수 없어서, 앞 대화를 반영해야 맞힐 수 있다
FOLLOWUP_SET = [
    {"turns": ["하자담보책임기간은 얼마나 돼?", "그 기간은 언제부터 계산해?"],
     "keywords": [["종료한 날", "종료일", "종료된 날", "인도한 날", "인도일"]]},
    {"turns": ["제안서 제출 마감은 언제야?", "어디로 내면 돼?"],
     "keywords": [["본관 2층", "시랑로"]]},
    {"turns": ["기술평가와 가격평가 비율은?", "그중 정성평가는 몇 점이야?"],
     "keywords": [["70"]]},
]

NOT_FOUND_WORDS = ["찾을 수 없", "나와 있지 않", "포함되어 있지 않", "언급되어 있지 않", "명시되어 있지 않",
                   "알 수 없", "정보가 없", "확인할 수 없", "확인되지 않"]


def is_correct(answer, item):
    if item["keywords"] is None:
        return any(w in answer for w in NOT_FOUND_WORDS)
    return all(any(kw in answer for kw in group) for group in item["keywords"])


def label(change):
    if not change:
        return "기준 (400/100/3/기본/의미)"
    return ", ".join(f"{k}={v}" for k, v in change.items())


def run_followups(rag, model_id):
    """최종 설정에서 후속 질문을 '앞 대화 없이' / '앞 대화 반영'으로 각각 물어 비교"""
    rows = []
    for i, item in enumerate(FOLLOWUP_SET, start=1):
        first, second = item["turns"]
        r1 = rag.ask(first, k=FINAL["k"], model_id=model_id, prompt_version=FINAL["prompt"], search_mode=FINAL["search"])
        history = [{"role": "user", "content": first}, {"role": "assistant", "content": r1["answer"]}]
        for mode, hist in [("앞 대화 없이", []), ("앞 대화 반영", history)]:
            r = rag.ask(second, history=hist, k=FINAL["k"], model_id=model_id,
                        prompt_version=FINAL["prompt"], search_mode=FINAL["search"])
            context = "\n".join(s["text"] for s in r["sources"])
            row = {"문항": f"F{i}", "첫 질문": first, "후속 질문": second, "방식": mode, "검색어": r["search_query"],
                   "근거 포함": is_correct(context, item)}
            if not OFFLINE:
                row.update({"정답": is_correct(r["answer"], item), "비용(USD)": round(r["cost"], 5), "답변": r["answer"]})
            rows.append(row)
            print(f"  [F{i} {mode}] 검색어='{r['search_query']}' 근거포함={row['근거 포함']} 정답={row.get('정답')}")
    return pd.DataFrame(rows)


def deploy_check():
    """배포한 앱과 같은 조건(Sonnet 5.5 + 최종 설정)으로 16문항을 물어 정답률 · 응답 시간 · 비용 확인"""
    rag = RAGPipeline(PDF_PATHS, chunk_size=FINAL["chunk_size"], chunk_overlap=FINAL["chunk_overlap"])
    rows = []
    for i, item in enumerate(TEST_SET, start=1):
        r = rag.ask(item["q"], k=FINAL["k"], model_id=DEPLOY_MODEL_ID, prompt_version=FINAL["prompt"],
                    user_type="학습자", search_mode=FINAL["search"])
        rows.append({"문항": i, "질문": item["q"], "정답": is_correct(r["answer"], item),
                     "응답시간(초)": round(r["seconds"], 2), "첫 글자(초)": round(r["first_token_seconds"], 2),
                     "입력토큰": r["input_tokens"], "출력토큰": r["output_tokens"],
                     "비용(USD)": round(r["cost"], 5), "답변": r["answer"]})
        print(f"  Q{i} 정답={rows[-1]['정답']} {rows[-1]['응답시간(초)']}초")
    df = pd.DataFrame(rows)
    df.to_csv("results/deploy_check.csv", index=False, encoding="utf-8-sig")
    print(f"정답률 {df['정답'].mean():.0%} · 평균 {df['응답시간(초)'].mean():.2f}초 · 질문당 ${df['비용(USD)'].mean():.5f}")


def summarize(raw):
    agg = {
        "청크수": ("청크 수", "first"),
        "검색적중률": ("검색 적중", "mean"),
        "근거포함률": ("근거 포함", "mean"),
        "평균프롬프트글자수": ("프롬프트 글자수", "mean"),
        "평균응답시간": ("응답시간(초)", "mean"),
    }
    if not OFFLINE:
        agg.update({"정답률": ("정답", "mean"), "평균입력토큰": ("입력토큰", "mean"),
                    "평균출력토큰": ("출력토큰", "mean"), "질문당 비용(USD)": ("비용(USD)", "mean")})
    summary = raw.groupby(["실험", "설정"], sort=False).agg(**agg).reset_index()
    for col in ["검색적중률", "근거포함률", "정답률"]:
        if col in summary:
            summary[col] = (summary[col].astype(float) * 100).round(0)
    summary["평균프롬프트글자수"] = summary["평균프롬프트글자수"].round(0)
    summary["평균응답시간"] = summary["평균응답시간"].round(2)
    if not OFFLINE:
        summary["질문당 비용(USD)"] = summary["질문당 비용(USD)"].round(5)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="이 이름의 실험만 다시 실행 (예: \"H. 3주차 최종\")")
    parser.add_argument("--deploy-check", action="store_true", help="배포 설정(Sonnet 5.5)으로 16문항만 확인")
    args = parser.parse_args()
    if args.deploy_check:
        deploy_check()
        return

    model_id = "offline" if OFFLINE else MODEL_ID
    print(f"모드: {'오프라인 (검색 단계만 측정)' if OFFLINE else MODEL_ID}")
    os.makedirs(OUT_DIR, exist_ok=True)
    pipelines = {}  # (chunk_size, overlap) 별로 벡터DB를 한 번만 생성
    rows = []

    for exp_name, change in EXPERIMENTS:
        if args.only and exp_name not in args.only:
            continue
        cfg = {**BASE, **change}
        key = (cfg["chunk_size"], cfg["chunk_overlap"])
        if key not in pipelines:
            print(f"벡터DB 생성: chunk_size={key[0]}, overlap={key[1]}")
            pipelines[key] = RAGPipeline(PDF_PATHS, chunk_size=key[0], chunk_overlap=key[1])
        rag = pipelines[key]

        for i, item in enumerate(TEST_SET, start=1):
            r = rag.ask(item["q"], k=cfg["k"], model_id=model_id, prompt_version=cfg["prompt"],
                        user_type="학습자", search_mode=cfg["search"])
            pages = sorted({s["page"] for s in r["sources"]})
            context = "\n".join(s["text"] for s in r["sources"])
            row = {
                "실험": exp_name, "설정": label(change),
                "chunk_size": cfg["chunk_size"], "overlap": cfg["chunk_overlap"], "k": cfg["k"],
                "프롬프트": cfg["prompt"], "검색": cfg["search"],
                "청크 수": len(rag.chunks),
                "문항": i, "질문": item["q"],
                # 검색 적중: 정답이 있는 쪽을 찾아왔는가 / 근거 포함: 찾아온 조각에 정답 키워드가 실제로 들어 있는가
                "검색 적중": any(p in pages for p in item["pages"]) if item["pages"] else None,
                "근거 포함": is_correct(context, item) if item["keywords"] else None,
                "검색된 쪽": ",".join(map(str, pages)),
                "프롬프트 글자수": r["prompt_chars"],
                "응답시간(초)": round(r["seconds"], 2),
            }
            if not OFFLINE:
                row.update({"정답": is_correct(r["answer"], item), "입력토큰": r["input_tokens"],
                            "출력토큰": r["output_tokens"], "비용(USD)": r["cost"], "답변": r["answer"]})
            rows.append(row)
            print(f"  [{label(change)}] Q{i} 검색적중={row['검색 적중']} 근거포함={row['근거 포함']} 정답={row.get('정답')}")

    raw = pd.DataFrame(rows)
    raw_path = f"{OUT_DIR}/raw_results.csv"
    if args.only and os.path.exists(raw_path):
        # 기존 결과에서 다시 돌린 실험의 행만 바꾸고, 실험 목록 순서대로 정렬
        old = pd.read_csv(raw_path)
        raw = pd.concat([old[~old["실험"].isin(args.only)], raw], ignore_index=True)
        order = {name: n for n, (name, _) in enumerate(EXPERIMENTS)}
        raw = raw.sort_values(["실험", "문항"], key=lambda col: col.map(order) if col.name == "실험" else col,
                              kind="stable").reset_index(drop=True)
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = summarize(raw)
    summary.to_csv(f"{OUT_DIR}/summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))

    print("\n후속 질문 실험 (3주차 최종 설정)")
    key = (FINAL["chunk_size"], FINAL["chunk_overlap"])
    final_rag = pipelines.get(key) or RAGPipeline(PDF_PATHS, chunk_size=key[0], chunk_overlap=key[1])
    followup = run_followups(final_rag, model_id)
    followup.to_csv(f"{OUT_DIR}/followup.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
