"""
PrivacyShield OCR + Regex + NER PII Detection Engine.

Runs 100% locally on CPU using RapidOCR (ONNX Runtime) with optional pytesseract
fallback, combined with high-precision Indian/global PII regex patterns and
spaCy (`en_core_web_sm`) named-entity recognition.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None  # type: ignore

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore


# Priority order when spans overlap (higher wins)
ENTITY_PRIORITY: Dict[str, int] = {
    "CARD": 100,
    "AADHAAR": 95,
    "PAN": 92,
    "EMAIL": 90,
    "UPI_ID": 88,
    "PHONE": 85,
    "ADDRESS": 70,
    "PERSON": 65,
}

# Known Indian UPI PSP handles (covers major banks & apps)
UPI_HANDLES = {
    "okaxis",
    "okhdfcbank",
    "okicici",
    "oksbi",
    "ybl",
    "ibl",
    "axl",
    "paytm",
    "ptyes",
    "ptaxis",
    "pthdfc",
    "ptsbi",
    "upi",
    "sbi",
    "icici",
    "hdfcbank",
    "kotak",
    "axisbank",
    "indus",
    "federal",
    "apl",
    "rapl",
    "waicici",
    "wahdfcbank",
    "wasbi",
    "waaxis",
    "barodampay",
    "fbl",
    "jupiteraxis",
    "ikwik",
    "pingpay",
    "freecharge",
    "airtel",
    "jio",
    "slice",
    "cred",
    "fam",
    "naviaxis",
    "yesbank",
    "idfcfirst",
    "rbl",
    "citi",
    "hsbc",
    "scb",
}

# Non-PII common document headings/words that NER can occasionally mistag
NER_STOPWORDS = {
    "aadhaar",
    "pan",
    "upi",
    "card",
    "government",
    "india",
    "republic",
    "income",
    "tax",
    "department",
    "verified",
    "payment",
    "receipt",
    "transaction",
    "successful",
    "completed",
    "merchant",
    "invoice",
    "privacyshield",
    "unique",
    "identification",
    "authority",
    "male",
    "female",
    "dob",
    "kyc",
    "status",
    "total",
    "amount",
    "visa",
    "mastercard",
    "rupay",
    "bhim",
    "google",
    "phonepe",
    "paytm",
    "bank",
    "ltd",
    "pvt",
}

PERSON_LABEL_RE = re.compile(
    r"^(?:name|full\s*name|holder|card\s*holder|account\s*holder|customer|paid\s*by|payer|sender|recipient|beneficiary|to|from)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

ADDRESS_LABEL_RE = re.compile(
    r"^(?:address|addr|billing\s*address|shipping\s*address|residence|location)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

PIN_CODE_RE = re.compile(r"\b[1-9]\d{5}\b")
ADDRESS_KEYWORD_RE = re.compile(
    r"\b(?:road|rd|street|st|lane|ln|avenue|ave|nagar|marg|colony|sector|phase|block|flat|apt|floor|cross|main|layout|vihar|enclave|extn|extension|plaza|chowk|bazar|bazaar|indiranagar|bandra|koramangala|bengaluru|bangalore|mumbai|delhi|hyderabad|chennai|pune|kolkata|ahmedabad|jaipur|gurugram|noida)\b",
    re.IGNORECASE,
)

# Regex rules
CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
UPI_RE = re.compile(r"\b[a-zA-Z0-9._\-]{2,64}@([a-zA-Z]{2,20})\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)")
PHONE_CONTEXT_RE = re.compile(r"\b(?:phone|mobile|mob|tel|contact|whatsapp|call)\b", re.IGNORECASE)
TXN_ID_CONTEXT_RE = re.compile(r"\b(?:ref\s*id|refid|txn|transaction|order\s*id|invoice|utr|rrn)\b", re.IGNORECASE)


@dataclass
class OCRItem:
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    polygon: List[List[float]]


@dataclass
class LineSpan:
    item: OCRItem
    start: int
    end: int


@dataclass
class ReconstructedLine:
    text: str
    spans: List[LineSpan]
    y_center: float
    height: float


@dataclass
class CandidateMatch:
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str
    line: ReconstructedLine


def _luhn_check(digits: str) -> bool:
    if len(digits) != 16 or not digits.isdigit():
        return False
    total = 0
    reverse = digits[::-1]
    for idx, ch in enumerate(reverse):
        n = int(ch)
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _mask_preview(entity_type: str, text: str) -> str:
    clean = text.strip()
    if entity_type in ("CARD", "AADHAAR"):
        digits = re.sub(r"\D", "", clean)
        if len(digits) >= 4:
            return f"{'*' * (len(digits) - 4)}{digits[-4:]}"
    if entity_type == "PAN" and len(clean) == 10:
        return f"{clean[:2]}*****{clean[-2:]}"
    if entity_type == "PHONE":
        digits = re.sub(r"\D", "", clean)
        if len(digits) >= 4:
            return f"******{digits[-4:]}"
    if entity_type in ("EMAIL", "UPI_ID") and "@" in clean:
        local, dom = clean.split("@", 1)
        head = local[:2] if len(local) >= 2 else local[:1]
        return f"{head}***@{dom}"
    if len(clean) <= 4:
        return "*" * len(clean)
    return f"{clean[:2]}{'*' * min(8, len(clean) - 3)}{clean[-1:]}"


class PIIDetector:
    """Thread-safe local OCR + Regex + spaCy NER PII detection pipeline."""

    def __init__(self) -> None:
        self._ocr_engine: Any = None
        self._ocr_name: str = "uninitialized"
        self._nlp: Any = None
        self._ner_name: str = "heuristic"

    def warmup(self) -> Dict[str, str]:
        """Pre-load OCR and spaCy models into memory."""
        self._ensure_ocr()
        self._ensure_nlp()
        return {"ocr_engine": self._ocr_name, "ner_engine": self._ner_name}

    def _ensure_ocr(self) -> None:
        if self._ocr_engine is not None:
            return
        if RapidOCR is not None:
            self._ocr_engine = RapidOCR()
            self._ocr_name = "rapidocr-onnxruntime"
        elif pytesseract is not None:
            self._ocr_engine = pytesseract
            self._ocr_name = "pytesseract"
        else:
            raise RuntimeError(
                "No OCR engine available. Install rapidocr-onnxruntime or pytesseract."
            )

    def _ensure_nlp(self) -> None:
        if self._nlp is not None or spacy is None:
            return
        try:
            self._nlp = spacy.load("en_core_web_sm")
            self._ner_name = "spacy:en_core_web_sm"
        except Exception:
            self._nlp = None
            self._ner_name = "regex+context-ner"

    def _run_ocr(self, image_bgr: np.ndarray) -> List[OCRItem]:
        self._ensure_ocr()
        h, w = image_bgr.shape[:2]
        items: List[OCRItem] = []

        if self._ocr_name == "rapidocr-onnxruntime":
            result, _ = self._ocr_engine(image_bgr)
            if not result:
                return items
            for entry in result:
                poly, text, conf = entry[0], str(entry[1]).strip(), float(entry[2])
                if not text:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x1 = max(0, int(round(min(xs))))
                y1 = max(0, int(round(min(ys))))
                x2 = min(w, int(round(max(xs))))
                y2 = min(h, int(round(max(ys))))
                if x2 > x1 and y2 > y1:
                    items.append(
                        OCRItem(
                            text=text,
                            confidence=conf,
                            bbox=[x1, y1, x2, y2],
                            polygon=[[float(p[0]), float(p[1])] for p in poly],
                        )
                    )
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                txt = str(data["text"][i]).strip()
                if not txt:
                    continue
                conf_val = float(data["conf"][i])
                conf = max(0.0, min(1.0, conf_val / 100.0)) if conf_val >= 0 else 0.75
                x, y, bw, bh = (
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                )
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                if x2 > x1 and y2 > y1:
                    items.append(
                        OCRItem(
                            text=txt,
                            confidence=conf,
                            bbox=[x1, y1, x2, y2],
                            polygon=[
                                [float(x1), float(y1)],
                                [float(x2), float(y1)],
                                [float(x2), float(y2)],
                                [float(x1), float(y2)],
                            ],
                        )
                    )

        return items

    @staticmethod
    def _group_into_lines(items: List[OCRItem]) -> List[ReconstructedLine]:
        if not items:
            return []

        sorted_items = sorted(items, key=lambda it: ((it.bbox[1] + it.bbox[3]) / 2.0, it.bbox[0]))
        groups: List[List[OCRItem]] = []

        for item in sorted_items:
            cy = (item.bbox[1] + item.bbox[3]) / 2.0
            ih = max(1.0, float(item.bbox[3] - item.bbox[1]))
            placed = False
            for group in groups:
                g_cy = sum((g.bbox[1] + g.bbox[3]) / 2.0 for g in group) / len(group)
                g_h = sum(max(1.0, float(g.bbox[3] - g.bbox[1])) for g in group) / len(group)
                tol = max(10.0, 0.55 * min(ih, g_h))
                if abs(cy - g_cy) <= tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])

        lines: List[ReconstructedLine] = []
        for group in groups:
            group.sort(key=lambda g: g.bbox[0])
            text_parts: List[str] = []
            spans: List[LineSpan] = []
            cursor = 0
            for idx, item in enumerate(group):
                if idx > 0:
                    text_parts.append(" ")
                    cursor += 1
                start = cursor
                text_parts.append(item.text)
                cursor += len(item.text)
                spans.append(LineSpan(item=item, start=start, end=cursor))
            y_center = sum((g.bbox[1] + g.bbox[3]) / 2.0 for g in group) / len(group)
            avg_h = sum(max(1.0, float(g.bbox[3] - g.bbox[1])) for g in group) / len(group)
            lines.append(
                ReconstructedLine(
                    text="".join(text_parts),
                    spans=spans,
                    y_center=y_center,
                    height=avg_h,
                )
            )

        lines.sort(key=lambda ln: ln.y_center)
        return lines

    @staticmethod
    def _span_to_bbox(
        line: ReconstructedLine,
        match_start: int,
        match_end: int,
        img_w: int,
        img_h: int,
    ) -> Tuple[List[int], float]:
        """Map a character span on a reconstructed line back to a tight pixel bbox."""
        x1s: List[int] = []
        y1s: List[int] = []
        x2s: List[int] = []
        y2s: List[int] = []
        confs: List[float] = []

        for span in line.spans:
            ov_start = max(match_start, span.start)
            ov_end = min(match_end, span.end)
            if ov_start >= ov_end:
                continue

            bx1, by1, bx2, by2 = span.item.bbox
            n_chars = max(1, len(span.item.text))
            rel_start = (ov_start - span.start) / n_chars
            rel_end = (ov_end - span.start) / n_chars

            width = bx2 - bx1
            sub_x1 = int(round(bx1 + rel_start * width))
            sub_x2 = int(round(bx1 + rel_end * width))

            x1s.append(sub_x1)
            y1s.append(by1)
            x2s.append(sub_x2)
            y2s.append(by2)
            confs.append(span.item.confidence)

        if not x1s:
            return [0, 0, 1, 1], 0.5

        pad_x = 4
        pad_y = 3
        bbox = [
            max(0, min(x1s) - pad_x),
            max(0, min(y1s) - pad_y),
            min(img_w, max(x2s) + pad_x),
            min(img_h, max(y2s) + pad_y),
        ]
        avg_conf = sum(confs) / len(confs)
        return bbox, avg_conf

    def _detect_regex_on_line(self, line: ReconstructedLine) -> List[CandidateMatch]:
        matches: List[CandidateMatch] = []
        text = line.text

        for m in CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            conf = 0.98 if _luhn_check(digits) else 0.94
            matches.append(
                CandidateMatch("CARD", m.group(0), m.start(), m.end(), conf, "regex", line)
            )

        for m in AADHAAR_RE.finditer(text):
            raw = m.group(0)
            # Exclude '+91...' or '91[6-9]...' Indian phone numbers
            if m.start() > 0 and text[m.start() - 1] == "+":
                continue
            digits = re.sub(r"\D", "", raw)
            is_444_grouped = bool(re.match(r"^\d{4}[\s\-]\d{4}[\s\-]\d{4}$", raw))
            if digits.startswith("91") and digits[2] in "6789" and (not is_444_grouped or PHONE_CONTEXT_RE.search(text)):
                continue
            matches.append(
                CandidateMatch("AADHAAR", raw, m.start(), m.end(), 0.96, "regex", line)
            )

        for m in PAN_RE.finditer(text):
            matches.append(
                CandidateMatch("PAN", m.group(0), m.start(), m.end(), 0.98, "regex", line)
            )

        for m in EMAIL_RE.finditer(text):
            matches.append(
                CandidateMatch("EMAIL", m.group(0), m.start(), m.end(), 0.98, "regex", line)
            )

        for m in UPI_RE.finditer(text):
            # Ensure we don't match email domains like '@gmail.com'
            after_pos = m.end()
            if after_pos < len(text) and text[after_pos] == ".":
                continue
            handle = m.group(1).lower()
            if handle in UPI_HANDLES or (len(handle) >= 3 and handle.isalpha()):
                conf = 0.96 if handle in UPI_HANDLES else 0.88
                matches.append(
                    CandidateMatch("UPI_ID", m.group(0), m.start(), m.end(), conf, "regex", line)
                )

        for m in PHONE_RE.finditer(text):
            raw = m.group(0).strip()
            # Exclude alphanumeric reference codes like TXN-9048127319
            if m.start() >= 2 and text[m.start() - 1] == "-" and text[m.start() - 2].isalnum():
                continue
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 10 or len(digits) > 13:
                continue
            has_country_prefix = raw.startswith("+") or (len(digits) == 12 and digits.startswith("91"))
            if not has_country_prefix and TXN_ID_CONTEXT_RE.search(text) and not PHONE_CONTEXT_RE.search(text):
                continue
            matches.append(
                CandidateMatch("PHONE", raw, m.start(), m.end(), 0.95, "regex", line)
            )

        return matches

    @staticmethod
    def _looks_like_person_name(value: str) -> bool:
        cleaned = value.strip(" .,-|")
        if not cleaned or len(cleaned) < 3 or len(cleaned) > 48:
            return False
        if any(ch.isdigit() for ch in cleaned) or "@" in cleaned:
            return False
        words = [w for w in re.split(r"\s+", cleaned) if w]
        if not (1 <= len(words) <= 4):
            return False
        # Require at least 2 words for label-free or mostly title-case words
        if any(w.lower() in NER_STOPWORDS for w in words):
            return False
        alpha_words = [w for w in words if re.match(r"^[A-Za-z][A-Za-z.'\-]*$", w)]
        if len(alpha_words) != len(words):
            return False
        return all(w[0].isupper() for w in words)

    def _detect_ner_on_lines(self, lines: List[ReconstructedLine]) -> List[CandidateMatch]:
        self._ensure_nlp()
        matches: List[CandidateMatch] = []
        address_continuation_active = False

        for idx, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                address_continuation_active = False
                continue

            # 1. Explicit label-assisted Person detection (high precision for Indian names)
            p_match = PERSON_LABEL_RE.match(line.text)
            if p_match:
                val = p_match.group(1).strip()
                # Trim trailing pipe/columns if another field is on the same line
                val_cut = re.split(r"\s{2,}|\s*\|\s*", val)[0].strip()
                if self._looks_like_person_name(val_cut):
                    v_start = line.text.find(val_cut, p_match.start(1))
                    v_end = v_start + len(val_cut)
                    conf = 0.92
                    src = "hybrid"
                    if self._nlp is not None:
                        doc = self._nlp(f"My name is {val_cut}.")
                        if any(e.label_ == "PERSON" for e in doc.ents):
                            conf = 0.95
                            src = "spacy_ner"
                    matches.append(
                        CandidateMatch("PERSON", val_cut, v_start, v_end, conf, src, line)
                    )
                    address_continuation_active = False
                    continue

            # 2. Explicit label-assisted Address detection
            a_match = ADDRESS_LABEL_RE.match(line.text)
            if a_match:
                val = a_match.group(1).strip()
                if len(val) >= 6:
                    v_start = line.text.find(val, a_match.start(1))
                    v_end = v_start + len(val)
                    matches.append(
                        CandidateMatch("ADDRESS", val, v_start, v_end, 0.93, "hybrid", line)
                    )
                    address_continuation_active = True
                    continue

            # Continuation line immediately following an Address: line
            if address_continuation_active:
                prev_line = lines[idx - 1]
                vertical_gap = line.y_center - prev_line.y_center
                if (
                    vertical_gap <= max(45.0, prev_line.height * 2.2)
                    and (PIN_CODE_RE.search(text) or ADDRESS_KEYWORD_RE.search(text))
                    and not PERSON_LABEL_RE.match(text)
                    and not AADHAAR_RE.search(text)
                    and not PAN_RE.search(text)
                    and not CARD_RE.search(text)
                ):
                    matches.append(
                        CandidateMatch("ADDRESS", text, 0, len(line.text), 0.90, "hybrid", line)
                    )
                    if PIN_CODE_RE.search(text):
                        address_continuation_active = False
                    continue
                address_continuation_active = False

            # 3. Unlabeled line with strong Indian address cues (PIN code + street/city cue)
            if (PIN_CODE_RE.search(text) and ADDRESS_KEYWORD_RE.search(text)) or (
                "," in text and ADDRESS_KEYWORD_RE.search(text) and len(text) >= 14
            ):
                # Strip any leading label if present before colon
                if ":" in line.text:
                    after = line.text.split(":", 1)[1].strip()
                    if len(after) >= 10:
                        s = line.text.find(after)
                        matches.append(
                            CandidateMatch("ADDRESS", after, s, s + len(after), 0.89, "hybrid", line)
                        )
                        continue
                else:
                    matches.append(
                        CandidateMatch("ADDRESS", text, 0, len(line.text), 0.89, "hybrid", line)
                    )
                    continue

            # 4. General spaCy NER fallback for unstructured text
            if self._nlp is not None:
                doc = self._nlp(line.text)
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    if len(ent_text) < 3:
                        continue
                    words_lower = {w.lower() for w in re.split(r"\W+", ent_text) if w}
                    if words_lower & NER_STOPWORDS:
                        continue
                    if ent.label_ == "PERSON" and self._looks_like_person_name(ent_text):
                        # Require 2+ words to minimize single-word false positives on labels
                        if len(ent_text.split()) >= 2:
                            matches.append(
                                CandidateMatch(
                                    "PERSON",
                                    ent_text,
                                    ent.start_char,
                                    ent.end_char,
                                    0.86,
                                    "spacy_ner",
                                    line,
                                )
                            )
                    elif ent.label_ in ("GPE", "LOC", "FAC") and (
                        PIN_CODE_RE.search(line.text) or ADDRESS_KEYWORD_RE.search(line.text)
                    ):
                        matches.append(
                            CandidateMatch(
                                "ADDRESS",
                                ent_text,
                                ent.start_char,
                                ent.end_char,
                                0.83,
                                "spacy_ner",
                                line,
                            )
                        )

        return matches

    @staticmethod
    def _resolve_line_overlaps(candidates: List[CandidateMatch]) -> List[CandidateMatch]:
        """Keep highest-priority non-overlapping matches on each line."""
        by_line: Dict[int, List[CandidateMatch]] = {}
        for c in candidates:
            by_line.setdefault(id(c.line), []).append(c)

        kept: List[CandidateMatch] = []
        for group in by_line.values():
            group.sort(
                key=lambda c: (
                    ENTITY_PRIORITY.get(c.entity_type, 0),
                    c.end - c.start,
                    c.confidence,
                ),
                reverse=True,
            )
            chosen: List[CandidateMatch] = []
            for cand in group:
                overlaps = any(
                    not (cand.end <= ex.start or cand.start >= ex.end) for ex in chosen
                )
                if not overlaps:
                    chosen.append(cand)
            chosen.sort(key=lambda c: c.start)
            kept.extend(chosen)

        kept.sort(key=lambda c: (c.line.y_center, c.start))
        return kept

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Run full OCR + Regex + NER PII detection and collect benchmark metrics."""
        img_h, img_w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        ocr_items = self._run_ocr(image_bgr)
        t_ocr = time.perf_counter()

        lines = self._group_into_lines(ocr_items)
        candidates: List[CandidateMatch] = []
        for line in lines:
            candidates.extend(self._detect_regex_on_line(line))
        candidates.extend(self._detect_ner_on_lines(lines))

        resolved = self._resolve_line_overlaps(candidates)
        t_nlp = time.perf_counter()

        entities: List[Dict[str, Any]] = []
        for idx, match in enumerate(resolved, start=1):
            bbox, ocr_conf = self._span_to_bbox(match.line, match.start, match.end, img_w, img_h)
            combined_conf = round(min(0.99, (match.confidence * 0.7 + ocr_conf * 0.3)), 3)
            entities.append(
                {
                    "id": f"ent_{idx:03d}",
                    "type": match.entity_type,
                    "text": match.text,
                    "masked_preview": _mask_preview(match.entity_type, match.text),
                    "confidence": combined_conf,
                    "ocr_confidence": round(ocr_conf, 3),
                    "source": match.source,
                    "bbox": bbox,
                    "enabled": True,
                }
            )

        proc = psutil.Process(os.getpid())
        rss_mb = round(proc.memory_info().rss / (1024 * 1024), 1)

        ocr_ms = round((t_ocr - t0) * 1000.0, 2)
        nlp_regex_ms = round((t_nlp - t_ocr) * 1000.0, 2)
        detection_ms = round((t_nlp - t0) * 1000.0, 2)

        return {
            "entities": entities,
            "ocr_lines": [ln.text for ln in lines],
            "ocr_box_count": len(ocr_items),
            "metrics": {
                "ocr_ms": ocr_ms,
                "nlp_regex_ms": nlp_regex_ms,
                "detection_ms": detection_ms,
                "cpu_memory_mb": rss_mb,
                "entity_count": len(entities),
                "ocr_engine": self._ocr_name,
                "ner_engine": self._ner_name,
                "image_width": img_w,
                "image_height": img_h,
            },
        }
