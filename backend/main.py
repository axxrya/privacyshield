"""
PrivacyShield FastAPI Server.

Serves both the REST API (/api/*) and the static Tactile Security Lab frontend
from a single process:
    cd backend && uvicorn main:app --reload
or from repository root:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import base64
import os
import sys
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from detector import PIIDetector  # noqa: E402
from redactor import render_annotated_overlay, render_redacted_image  # noqa: E402
from sample_generator import SampleKind, generate_sample_image  # noqa: E402

RedactionStyle = Literal["pixelate", "matte_bar", "synthetic_mask"]

MAX_CACHED_DOCS = 20
DOCUMENT_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

detector = PIIDetector()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    detector.warmup()
    yield
    DOCUMENT_CACHE.clear()


app = FastAPI(
    title="PrivacyShield Local PII Redaction Engine",
    version="0.1.0",
    description="100% On-Device OCR + Regex + NER PII detection and redaction API.",
    lifespan=lifespan,
)


class SampleRequest(BaseModel):
    kind: SampleKind = "indian_id"
    style: RedactionStyle = "pixelate"
    show_tags: bool = True


class RedactRequest(BaseModel):
    document_id: str
    enabled_entity_ids: List[str] = Field(default_factory=list)
    style: RedactionStyle = "pixelate"
    show_tags: bool = True


def _encode_png_data_url(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image file.")
    # Guard ultra-large images for crisp CPU latency while keeping high readability
    h, w = image_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 2200:
        scale = 2200.0 / max_dim
        image_bgr = cv2.resize(
            image_bgr,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image_bgr


def _cache_put(doc_id: str, payload: Dict[str, Any]) -> None:
    DOCUMENT_CACHE[doc_id] = payload
    DOCUMENT_CACHE.move_to_end(doc_id)
    while len(DOCUMENT_CACHE) > MAX_CACHED_DOCS:
        DOCUMENT_CACHE.popitem(last=False)


def _build_audit_log(
    entities: List[Dict[str, Any]],
    enabled_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    active_set = set(enabled_ids) if enabled_ids is not None else None
    logs: List[Dict[str, Any]] = []
    for ent in entities:
        is_redacted = ent.get("enabled", True) if active_set is None else ent["id"] in active_set
        logs.append(
            {
                "id": ent["id"],
                "type": ent["type"],
                "masked_value": ent["masked_preview"],
                "confidence": ent["confidence"],
                "source": ent["source"],
                "bbox": ent["bbox"],
                "action": "REDACTED" if is_redacted else "EXEMPTED",
            }
        )
    return logs


def _process_new_document(
    raw_bytes: bytes,
    filename: str,
    style: RedactionStyle,
    show_tags: bool,
) -> Dict[str, Any]:
    image_bgr = _decode_image_bytes(raw_bytes)
    detection = detector.detect(image_bgr)
    entities = detection["entities"]

    enabled_ids = [ent["id"] for ent in entities]
    redacted_bgr, redaction_ms = render_redacted_image(
        image_bgr,
        entities,
        style=style,
        enabled_ids=enabled_ids,
        show_tags=show_tags,
    )
    annotated_bgr = render_annotated_overlay(image_bgr, entities, enabled_ids=enabled_ids)

    original_url = _encode_png_data_url(image_bgr)
    redacted_url = _encode_png_data_url(redacted_bgr)
    annotated_url = _encode_png_data_url(annotated_bgr)

    doc_id = uuid.uuid4().hex[:12]
    base_metrics = detection["metrics"]
    total_latency_ms = round(base_metrics["detection_ms"] + redaction_ms, 2)

    _cache_put(
        doc_id,
        {
            "filename": filename,
            "image_bgr": image_bgr,
            "original_image": original_url,
            "entities": entities,
            "ocr_lines": detection["ocr_lines"],
            "base_metrics": base_metrics,
        },
    )

    metrics = {
        **base_metrics,
        "redaction_ms": redaction_ms,
        "inference_latency_ms": total_latency_ms,
        "active_redaction_count": len(enabled_ids),
    }

    return {
        "document_id": doc_id,
        "filename": filename,
        "style": style,
        "show_tags": show_tags,
        "original_image": original_url,
        "redacted_image": redacted_url,
        "annotated_image": annotated_url,
        "entities": entities,
        "audit_log": _build_audit_log(entities, enabled_ids),
        "ocr_lines": detection["ocr_lines"],
        "metrics": metrics,
        "privacy_guarantee": {
            "mode": "LOCAL_ON_DEVICE",
            "cloud_bytes_sent": 0,
            "external_requests": 0,
        },
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    engines = detector.warmup()
    rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    return {
        "status": "ok",
        "local_only": True,
        "cloud_bytes_sent": 0,
        "cpu_memory_mb": rss_mb,
        **engines,
    }


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    style: RedactionStyle = Form("pixelate"),
    show_tags: bool = Form(True),
) -> Dict[str, Any]:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return _process_new_document(
        raw_bytes=raw_bytes,
        filename=file.filename or "uploaded_document.png",
        style=style,
        show_tags=show_tags,
    )


@app.post("/api/sample")
def analyze_sample(req: SampleRequest) -> Dict[str, Any]:
    raw_bytes, filename = generate_sample_image(req.kind)
    return _process_new_document(
        raw_bytes=raw_bytes,
        filename=filename,
        style=req.style,
        show_tags=req.show_tags,
    )


@app.get("/api/sample/{kind}/raw")
def download_raw_sample(kind: SampleKind) -> Response:
    raw_bytes, filename = generate_sample_image(kind)
    return Response(
        content=raw_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/api/redact")
def re_redact_document(req: RedactRequest) -> Dict[str, Any]:
    cached = DOCUMENT_CACHE.get(req.document_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Document session not found in local memory cache. Please re-run analysis.",
        )

    DOCUMENT_CACHE.move_to_end(req.document_id)
    image_bgr: np.ndarray = cached["image_bgr"]
    entities: List[Dict[str, Any]] = cached["entities"]
    enabled_set = set(req.enabled_entity_ids)

    updated_entities = []
    for ent in entities:
        copy_ent = dict(ent)
        copy_ent["enabled"] = ent["id"] in enabled_set
        updated_entities.append(copy_ent)
    cached["entities"] = updated_entities

    redacted_bgr, redaction_ms = render_redacted_image(
        image_bgr,
        updated_entities,
        style=req.style,
        enabled_ids=enabled_set,
        show_tags=req.show_tags,
    )
    annotated_bgr = render_annotated_overlay(
        image_bgr, updated_entities, enabled_ids=enabled_set
    )

    rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    base_metrics = cached["base_metrics"]
    metrics = {
        **base_metrics,
        "cpu_memory_mb": rss_mb,
        "redaction_ms": redaction_ms,
        "inference_latency_ms": round(base_metrics["detection_ms"] + redaction_ms, 2),
        "active_redaction_count": sum(1 for e in updated_entities if e["enabled"]),
    }

    return {
        "document_id": req.document_id,
        "style": req.style,
        "show_tags": req.show_tags,
        "redacted_image": _encode_png_data_url(redacted_bgr),
        "annotated_image": _encode_png_data_url(annotated_bgr),
        "entities": updated_entities,
        "audit_log": _build_audit_log(updated_entities, list(enabled_set)),
        "metrics": metrics,
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(str(index_path))
