# AWS App Runner / GCP Cloud Run / Azure Container Apps 같은 컨테이너 실행 서비스용 (코멘토 배포 튜토리얼 2절)
FROM python:3.13-slim
# 로그를 버퍼 없이 바로 내보내야 클라우드 로그(CloudWatch · Cloud Logging)에 즉시 보인다
ENV PYTHONUNBUFFERED=1

# 보안: root가 아닌 사용자로 실행
RUN useradd --create-home appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser
# 임베딩 모델(약 0.5GB)을 이미지에 미리 받아 둬서 첫 요청이 느리지 않게 한다
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"
COPY --chown=appuser . .

EXPOSE 8080
# 헬스 체크 경로: /_stcore/health (Streamlit 내장)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
