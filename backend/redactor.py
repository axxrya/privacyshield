"""
PrivacyShield Redaction & Detection Overlay Engine (OpenCV + Pillow).

Supports 3 selectable redaction styles on detected bounding boxes:
  1. pixelate       - Gaussian Pixelation / Mosaic privacy blur
  2. matte_bar      - Solid Matte Redaction Bar with optional [ENTITY] tag
  3. synthetic_mask - Smart format-preserving synthetic masking in-place
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category colors (BGR for OpenCV) — Safety Vermilion primary + distinct accents
CATEGORY_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "AADHAAR": (45, 77, 255),   # #FF4D2D vermilion
    "PAN": (36, 98, 235),       # deep amber-orange
    "CARD": (45, 77, 255),      # #FF4D2D vermilion
    "UPI_ID": (20, 120, 220),   # copper-orange
    "PHONE": (180, 90, 20),     # cobalt-slate
    "EMAIL": (160, 100, 30),    # steel blue
    "PERSON": (30, 30, 28),     # carbon ink
    "ADDRESS": (80, 110, 60),   # muted olive-slate
}


def _load_pil_font(size: int, mono: bool = True) -> ImageFont.ImageFont:
    candidates = (
        [
            "C:/Windows/Fonts/consolab.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ]
        if mono
        else [
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=max(10, size))
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=max(10, size))
    except TypeError:
        return ImageFont.load_default()


def _synthetic_replacement(entity_type: str, text: str) -> str:
    """Generate a format-preserving synthetic replacement string."""
    clean = text.strip()
    if entity_type == "AADHAAR":
        return "XXXX XXXX 0000"
    if entity_type == "PAN":
        return "XXXXX0000X"
    if entity_type == "CARD":
        return "4000 XXXX XXXX 0000"
    if entity_type == "UPI_ID" and "@" in clean:
        _, handle = clean.split("@", 1)
        return f"user.redacted@{handle}"
    if entity_type == "EMAIL" and "@" in clean:
        return "redacted@domain.tld"
    if entity_type == "PHONE":
        return "+91 9XXXX XXXXX"
    if entity_type == "PERSON":
        return "A. CITIZEN"
    if entity_type == "ADDRESS":
        return "REDACTED STREET, CITY 000000"
    return re.sub(r"[A-Za-z0-9]", "X", clean)[:18]


def _sample_region_palette(roi: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Sample dominant border background color and contrasting text color (RGB)."""
    h, w = roi.shape[:2]
    if h < 4 or w < 4:
        return (245, 245, 243), (20, 20, 19)

    top = roi[0, :, :]
    bottom = roi[-1, :, :]
    left = roi[:, 0, :]
    right = roi[:, -1, :]
    border_pixels = np.vstack([top, bottom, left, right])
    median_bgr = np.median(border_pixels, axis=0)
    bg_rgb = (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))

    luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    fg_rgb = (20, 20, 19) if luminance > 130 else (244, 243, 239)
    return bg_rgb, fg_rgb


def _apply_pixelate(img: np.ndarray, bbox: List[int]) -> None:
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return

    # Coarse mosaic block size proportional to ROI height
    block = max(6, min(14, rh // 3))
    small_w = max(2, rw // block)
    small_h = max(2, rh // block)

    # Downsample -> Gaussian soften -> Nearest-neighbor mosaic upscale
    down = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(down, (rw, rh), interpolation=cv2.INTER_NEAREST)

    k = max(3, (block // 2) | 1)
    blurred = cv2.GaussianBlur(mosaic, (k, k), sigmaX=1.4)
    img[y1:y2, x1:x2] = blurred


def _apply_matte_bar(
    pil_img: Image.Image,
    bbox: List[int],
    entity_type: str,
    show_tags: bool,
) -> None:
    draw = ImageDraw.Draw(pil_img)
    x1, y1, x2, y2 = bbox
    rh = max(1, y2 - y1)
    rw = max(1, x2 - x1)

    # Carbon matte block (#141413) with crisp 1px vermilion left accent bar
    draw.rectangle([x1, y1, x2, y2], fill=(20, 20, 19))
    accent_w = max(2, min(4, rw // 25))
    draw.rectangle([x1, y1, min(x2, x1 + accent_w), y2], fill=(255, 77, 45))

    if not show_tags or rw < 38 or rh < 14:
        return

    tag = f"[{entity_type}]"
    font_size = max(10, min(16, int(rh * 0.56)))
    font = _load_pil_font(font_size, mono=True)
    tb = draw.textbbox((0, 0), tag, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]

    # Shrink font if wider than box
    while tw > rw - 12 and font_size > 9:
        font_size -= 1
        font = _load_pil_font(font_size, mono=True)
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

    if tw <= rw - 6:
        tx = x1 + (rw - tw) // 2
        ty = y1 + (rh - th) // 2 - tb[1]
        draw.text((tx, ty), tag, fill=(244, 243, 239), font=font)


def _apply_synthetic_mask(
    pil_img: Image.Image,
    orig_bgr: np.ndarray,
    bbox: List[int],
    entity_type: str,
    orig_text: str,
) -> None:
    x1, y1, x2, y2 = bbox
    roi = orig_bgr[y1:y2, x1:x2]
    bg_rgb, fg_rgb = _sample_region_palette(roi)

    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], fill=bg_rgb, outline=(255, 77, 45), width=1)

    rw = max(1, x2 - x1)
    rh = max(1, y2 - y1)
    replacement = _synthetic_replacement(entity_type, orig_text)

    font_size = max(10, min(20, int(rh * 0.62)))
    font = _load_pil_font(font_size, mono=True)
    tb = draw.textbbox((0, 0), replacement, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]

    while tw > rw - 8 and font_size > 9:
        font_size -= 1
        font = _load_pil_font(font_size, mono=True)
        tb = draw.textbbox((0, 0), replacement, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

    if tw > rw - 6:
        # Truncate gracefully if box is very narrow
        chars = max(3, int(len(replacement) * ((rw - 8) / max(1, tw))))
        replacement = replacement[:chars]
        tb = draw.textbbox((0, 0), replacement, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

    tx = x1 + 4
    ty = y1 + (rh - th) // 2 - tb[1]
    draw.text((tx, ty), replacement, fill=fg_rgb, font=font)


def render_redacted_image(
    image_bgr: np.ndarray,
    entities: List[Dict[str, Any]],
    style: str = "pixelate",
    enabled_ids: Optional[Iterable[str]] = None,
    show_tags: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Apply the chosen redaction style to active entities on a copy of `image_bgr`.
    Returns `(redacted_bgr, redaction_ms)`.
    """
    t0 = time.perf_counter()
    active_set = set(enabled_ids) if enabled_ids is not None else None
    out = image_bgr.copy()
    h, w = out.shape[:2]

    active_entities = [
        ent
        for ent in entities
        if (ent.get("enabled", True) if active_set is None else ent["id"] in active_set)
    ]

    if style == "pixelate":
        for ent in active_entities:
            x1, y1, x2, y2 = ent["bbox"]
            clamped = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            _apply_pixelate(out, clamped)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        for ent in active_entities:
            x1, y1, x2, y2 = ent["bbox"]
            clamped = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            if style == "matte_bar":
                _apply_matte_bar(pil_img, clamped, ent["type"], show_tags=show_tags)
            elif style == "synthetic_mask":
                _apply_synthetic_mask(
                    pil_img, image_bgr, clamped, ent["type"], ent.get("text", "")
                )
            else:
                # Default fallback to matte_bar if unknown style string passed
                _apply_matte_bar(pil_img, clamped, ent["type"], show_tags=show_tags)
        out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    redaction_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return out, redaction_ms


def render_annotated_overlay(
    image_bgr: np.ndarray,
    entities: List[Dict[str, Any]],
    enabled_ids: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    Render an inspection overlay showing bounding boxes, confidence scores,
    and active/bypassed states for each detected PII entity.
    """
    active_set = set(enabled_ids) if enabled_ids is not None else None
    overlay = image_bgr.copy()
    tint_layer = image_bgr.copy()
    h, w = overlay.shape[:2]

    for ent in entities:
        x1, y1, x2, y2 = ent["bbox"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        is_active = ent.get("enabled", True) if active_set is None else ent["id"] in active_set
        color = CATEGORY_COLORS_BGR.get(ent["type"], (45, 77, 255)) if is_active else (140, 140, 136)

        cv2.rectangle(tint_layer, (x1, y1), (x2, y2), color, -1)

    alpha = 0.16
    overlay = cv2.addWeighted(tint_layer, alpha, overlay, 1.0 - alpha, 0)

    pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_pil_font(11, mono=True)

    for ent in entities:
        x1, y1, x2, y2 = ent["bbox"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        is_active = ent.get("enabled", True) if active_set is None else ent["id"] in active_set

        bgr = CATEGORY_COLORS_BGR.get(ent["type"], (45, 77, 255)) if is_active else (140, 140, 136)
        rgb = (bgr[2], bgr[1], bgr[0])

        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)

        conf_pct = int(round(float(ent.get("confidence", 0.9)) * 100))
        state_suffix = "" if is_active else " [OFF]"
        label = f"{ent['type']} {conf_pct}%{state_suffix}"
        tb = draw.textbbox((0, 0), label, font=font)
        lw, lh = tb[2] - tb[0] + 10, tb[3] - tb[1] + 6

        ly1 = y1 - lh - 2 if y1 - lh - 2 >= 0 else y2 + 2
        ly2 = ly1 + lh
        lx1 = x1
        lx2 = min(w, lx1 + lw)

        draw.rectangle([lx1, ly1, lx2, ly2], fill=rgb)
        draw.text((lx1 + 5, ly1 + 3 - tb[1]), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
