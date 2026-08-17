"""Shared application dependencies (singletons, lifespan)."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from psycopg_pool import AsyncConnectionPool
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor
from rag.integration import PDFParser, RAGIntegration
from rag.factory import VisionService
from app.store import init_store

logger = logging.getLogger(__name__)


async def _ensure_postgres_db():
    """Create the database if it doesn't exist."""
    uri = Config.POSTGRES_URI
    # Connect to default 'postgres' db to create target db
    idx = uri.rfind("/")
    db_name = uri[idx + 1:]
    base_uri = uri[:idx] + "/postgres"
    try:
        conn = await psycopg.AsyncConnection.connect(base_uri, autocommit=True)
        async with conn:
            cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not await cur.fetchone():
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Created database: {db_name}")
    except Exception as e:
        logger.warning(f"Could not auto-create database: {e}")


class RetrieverTool:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    def invoke(self, query: str, section_type_filter=None):
        return self._retriever.retrieve(
            query=query,
            k=Config.TOP_K,
            use_hyde=False,
            rerank=True,
            expand_parent=True,
            rrf_k=Config.RRF_K,
            fetch_k=Config.FETCH_K,
            section_type_filter=section_type_filter,
        )


_llm: ChatOpenAI | None = None
_retriever: Retriever | None = None
_retriever_tool: RetrieverTool | None = None
_pdf_parser: PDFParser | None = None
_rag_integration: RAGIntegration | None = None
_checkpointer: AsyncPostgresSaver | MemorySaver | None = None
_vision_service: VisionService | None = None


def get_llm() -> ChatOpenAI:
    return _llm  # type: ignore


def get_retriever() -> Retriever:
    return _retriever  # type: ignore


def get_retriever_tool() -> RetrieverTool:
    return _retriever_tool  # type: ignore


def get_pdf_parser() -> PDFParser:
    return _pdf_parser  # type: ignore


def get_rag_integration() -> RAGIntegration:
    return _rag_integration  # type: ignore


def get_checkpointer() -> AsyncPostgresSaver:
    return _checkpointer  # type: ignore


def get_vision_service() -> VisionService | None:
    return _vision_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _retriever, _retriever_tool, _pdf_parser, _rag_integration, _checkpointer, _vision_service

    logger.info("Starting up — loading models …")

    pool = None
    if Config.STORAGE_MODE == "postgres":
        await _ensure_postgres_db()
        pool = AsyncConnectionPool(Config.POSTGRES_URI, min_size=2, max_size=10)
        await pool.open()
        await init_store(pool, mode="postgres")
    elif Config.STORAGE_MODE == "sqlite":
        # SQLite 持久化文件列表和会话，不需要启动 PostgreSQL 服务。
        await init_store(None, sqlite_path=Config.SQLITE_PATH, mode="sqlite")
    else:
        # memory 仅适合临时调试：后端重启后文件列表会清空。
        await init_store(None, mode="memory")

    _llm = ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
        timeout=60,
        max_retries=3,
    )

    if Config.VLM_ENABLED:
        vlm_llm = ChatOpenAI(
            base_url=Config.VLM_BASE_URL,
            model=Config.VLM_MODEL,
            api_key=Config.VLM_API_KEY,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
        _vision_service = VisionService(vlm_llm)

    _retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        reranker_model=Config.RERANKER_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
        enable_cache=Config.ENABLE_CACHE,
    )
    _retriever_tool = RetrieverTool(_retriever)

    _pdf_parser = PDFParser(
        llm=_llm,
        # 本地 SQLite 模式跳过上传时的额外分类调用，避免每次上传都依赖 LLM。
        enable_section_classification=Config.STORAGE_MODE == "postgres",
    )
    _rag_integration = RAGIntegration(
        embedding_model=Config.EMBEDDING_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
    )

    upload_dir = Path(Config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    if Config.STORAGE_MODE in ("memory", "sqlite"):
        _checkpointer = MemorySaver()
        logger.info("Startup complete (local %s storage).", Config.STORAGE_MODE)
        yield
    else:
        async with AsyncPostgresSaver.from_conn_string(Config.POSTGRES_URI) as cp:
            await cp.setup()
            _checkpointer = cp
            logger.info("Startup complete.")
            yield

    _checkpointer = None
    if pool is not None:
        await pool.close()
    logger.info("Shutting down.")
