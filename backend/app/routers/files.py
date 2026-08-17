"""File upload and management APIs."""

import uuid
import hashlib
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles

from config import Config
from app.dependencies import get_pdf_parser, get_rag_integration, get_retriever
from app.store import add_file, list_files, delete_file_record

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    # 这是离线建库的入口：每个上传的 PDF 都会在本次请求中完成
    # “解析 → 节点化 → 切块 → 写入向量库”，而不只是保存到磁盘。
    upload_dir = Path(Config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    results = []

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "status": "error", "detail": "Only PDF files are supported"})
            continue

        content = await f.read()
        if len(content) > max_bytes:
            results.append({"filename": f.filename, "status": "error", "detail": f"File exceeds {Config.MAX_UPLOAD_SIZE_MB}MB limit"})
            continue

        # 用原始二进制内容算 hash，而不是用文件名；重命名后的同一论文也能去重。
        content_hash = hashlib.sha256(content).hexdigest()

        # 在子块集合的 metadata 中查重。若已入库，后续解析和 embedding 都可省掉。
        retriever = get_retriever()
        updater = retriever.get_updater()
        existing_paper = updater.has_content_hash(content_hash)
        if existing_paper:
            results.append({"filename": f.filename, "status": "duplicate", "detail": f"Same content as paper '{existing_paper}'"})
            continue

        file_id = str(uuid.uuid4())
        paper_id = Path(f.filename).stem
        save_path = upload_dir / f"{file_id}.pdf"

        async with aiofiles.open(save_path, "wb") as out:
            await out.write(content)

        try:
            parser = get_pdf_parser()
            # PDFParser 会保留段落、标题、表格、图片等论文结构，而非合并成纯文本。
            nodes = parser.parse(str(save_path), paper_id)

            integration = get_rag_integration()
            # 节点携带页码、章节、坐标等 metadata；切块后这些信息仍会随文档保存。
            docs = integration.nodes_to_documents(nodes, content_hash=content_hash)
            parents, children = integration.create_chunks(docs)

            # 子块用于精确召回，父块用于生成时提供完整上下文；二者分别存入 Milvus。
            integration.store_in_milvus(parents, children)

            record = await add_file(
                file_id=file_id,
                filename=f.filename,
                paper_id=paper_id,
                size_bytes=len(content),
                page_count=max((n.page_num for n in nodes), default=0),
                chunk_count=len(children),
            )
            results.append({"filename": f.filename, "status": "ok", **record})
        except Exception as e:
            logger.exception(f"Failed to process {f.filename}")
            save_path.unlink(missing_ok=True)
            results.append({"filename": f.filename, "status": "error", "detail": str(e)})

    return {"files": results}


@router.get("")
async def get_files():
    return await list_files()


@router.delete("/{file_id}")
async def remove_file(file_id: str):
    record = await delete_file_record(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    retriever = get_retriever()
    updater = retriever.get_updater()
    updater.delete_paper(record["paper_id"])

    save_path = Path(Config.UPLOAD_DIR) / f"{file_id}.pdf"
    save_path.unlink(missing_ok=True)

    figures_dir = Path("data/figures") / record["paper_id"]
    if figures_dir.exists():
        shutil.rmtree(figures_dir)

    return {"ok": True, "paper_id": record["paper_id"]}
