# Phase 5 (RAG) Lambda 컨테이너 이미지
#
# 빌드 전 rag_llm_3.py를 S3에서 이 디렉터리로 다운로드 필요:
#   aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py infra/lambda/
#
# 빌드 (프로젝트 루트에서 실행):
#   docker build --platform linux/amd64 -t rare-link-rag -f infra/lambda/Dockerfile .
#
# 또는 deploy.sh 사용:
#   bash infra/deploy.sh

FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# 1. 시스템 라이브러리 (PostgreSQL 클라이언트)
RUN yum install -y postgresql-libs && yum clean all

# 2. Python 의존성 (PyTorch 불필요 — Phase 5는 API + DB만 사용)
RUN pip install --no-cache-dir \
    aiohttp==3.9.5 \
    psycopg2-binary==2.9.9 \
    boto3>=1.34.0 \
    pandas>=2.0.0 \
    requests>=2.31.0

# 3. RAG 파이프라인 코드
#    (빌드 전 S3에서 다운로드: aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py infra/lambda/)
COPY infra/lambda/rag_llm_3.py ./

# 4. Lambda 핸들러
COPY infra/lambda/app.py ./

CMD ["app.lambda_handler"]
