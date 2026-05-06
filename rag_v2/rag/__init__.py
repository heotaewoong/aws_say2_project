# rag/__init__.py
# Rare-Link AI — RAG 컴포넌트 패키지
#
# 구성:
#   bedrock_extractor  — 임상 텍스트 → HPO (AWS Bedrock Claude)
#   lab_rules          — 혈액·폐기능 수치 → HPO (Rule-based)
#   lirical_scorer     — LIRICAL LR 스코어링 (Orphanet 기반)
#   knowledge_base     — Orphanet XML → CSV 파서
#   pubcasefinder      — HPO → 케이스리포트 (PubCaseFinder API)
#   pubmed_fetcher     — 질환명 → 최신 논문 (PubMed API + ChromaDB 캐시)
#   rag_engine         — HPO → 유사 질환 검색 (ChromaDB)
#   vector_store       — ChromaDB 공유 저장소
#   ragas_eval         — RAGAS 평가 + PMID 환각 체크

from .bedrock_extractor import BedrockHPOExtractor
from .lab_rules import lab_to_hpo
from .lirical_scorer import build_disease_database, rank_diseases, compute_lr_score
from .pubcasefinder import get_ranked_diseases, format_pcf_for_llm

__all__ = [
    "BedrockHPOExtractor",
    "lab_to_hpo",
    "build_disease_database",
    "rank_diseases",
    "compute_lr_score",
    "get_ranked_diseases",
    "format_pcf_for_llm",
]
