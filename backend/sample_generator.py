"""
Synthetic Indian ID Card & UPI Payment Receipt Generator (Pillow).

Generates high-DPI synthetic test images with realistic (strictly non-real)
Indian PII so judges and evaluators can test PrivacyShield in 1 click without
uploading real personal documents.
"""

from __future__ import annotations

import io
import os
import random
from typing import Literal, Tuple

from PIL import Image, ImageDraw, ImageFont


SampleKind = Literal["indian_id", "upi_receipt"]


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        paths = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ]
    else:
        paths = [
            "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_qr_matrix(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, seed: int) -> None:
    rng = random.Random(seed)
    cells = 15
    cell = size // cells
    draw.rectangle([x - 4, y - 4, x + cells * cell + 4, y + cells * cell + 4], fill=(255, 255, 255), outline=(210, 208, 200))
    for r in range(cells):
        for c in range(cells):
            in_corner = (
                (r < 4 and c < 4)
                or (r < 4 and c >= cells - 4)
                or (r >= cells - 4 and c < 4)
            )
            if in_corner or rng.random() > 0.48:
                px = x + c * cell
                py = y + r * cell
                draw.rectangle([px, py, px + cell - 1, py + cell - 1], fill=(28, 28, 30))


def generate_indian_id_sample() -> Tuple[bytes, str]:
    """Generate a realistic synthetic Indian KYC / Aadhaar + PAN specimen sheet."""
    w, h = 1040, 680
    img = Image.new("RGB", (w, h), (239, 238, 233))
    draw = ImageDraw.Draw(img)

    # Outer document sheet card
    draw.rounded_rectangle([32, 28, w - 32, h - 28], radius=16, fill=(255, 255, 255), outline=(30, 30, 32), width=2)

    # Tricolor-inspired clean top header strip
    draw.rectangle([34, 30, w - 34, 42], fill=(255, 118, 48))
    draw.rectangle([34, 42, w - 34, 52], fill=(255, 255, 255))
    draw.rectangle([34, 52, w - 34, 64], fill=(19, 136, 8))

    f_title = _font(25, bold=True)
    f_sub = _font(15, bold=True, mono=True)
    f_label = _font(17, bold=False)
    f_val = _font(22, bold=True)
    f_mono_lg = _font(28, bold=True, mono=True)
    f_mono_md = _font(21, bold=True, mono=True)

    draw.text((64, 78), "SYNTHETIC CITIZEN KYC SPECIMEN (TEST DOCUMENT)", fill=(20, 20, 19), font=f_title)
    draw.text((64, 110), "ON-DEVICE REDACTION BENCHMARK // NON-REAL DATA", fill=(115, 114, 108), font=f_sub)
    draw.line([64, 140, w - 64, 140], fill=(218, 216, 209), width=2)

    # Left avatar photo box
    draw.rounded_rectangle([64, 168, 244, 378], radius=10, fill=(235, 238, 242), outline=(180, 182, 186), width=2)
    draw.ellipse([114, 196, 194, 276], fill=(166, 174, 184))
    draw.pieslice([86, 282, 222, 412], start=180, end=360, fill=(142, 152, 164))
    draw.text((104, 348), "PHOTO ID", fill=(85, 90, 96), font=_font(16, bold=True, mono=True))

    # Core identity fields (clear spacing for accurate OCR)
    y = 170
    draw.text((278, y), "Name: Aarav Sharma", fill=(20, 20, 19), font=f_val)
    y += 46
    draw.text((278, y), "Phone: +91 98765 43210", fill=(20, 20, 19), font=f_val)
    y += 46
    draw.text((278, y), "Email: aarav.sharma94@gmail.com", fill=(20, 20, 19), font=f_val)
    y += 46
    draw.text((278, y), "PAN: ABCDE1234F", fill=(20, 20, 19), font=f_mono_md)
    y += 46
    draw.text((278, y), "UPI: aarav.sharma@okaxis", fill=(20, 20, 19), font=f_mono_md)

    # Address block
    draw.text((64, 412), "Address: 42 MG Road, Indiranagar,", fill=(28, 28, 30), font=f_val)
    draw.text((160, 444), "Bengaluru, Karnataka 560038", fill=(28, 28, 30), font=f_val)

    # QR code on right
    _draw_qr_matrix(draw, w - 216, 172, 138, seed=42)

    # Prominent Aadhaar number footer banner
    draw.rounded_rectangle([64, 504, w - 64, 610], radius=10, fill=(247, 246, 241), outline=(35, 35, 38), width=2)
    draw.text((92, 518), "AADHAAR NUMBER (UIDAI FORMAT)", fill=(108, 106, 100), font=f_sub)
    draw.text((92, 548), "4829 7310 5942", fill=(20, 20, 19), font=f_mono_lg)
    draw.text((540, 520), "LINKED DEBIT CARD", fill=(108, 106, 100), font=f_sub)
    draw.text((540, 550), "4532 8910 3476 9012", fill=(20, 20, 19), font=f_mono_md)

    # Subtle specimen stamp
    draw.text((64, 622), "SPECIMEN / SYNTHETIC TEST ARTIFACT", fill=(145, 142, 134), font=f_label)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "synthetic_indian_id.png"


def generate_upi_receipt_sample() -> Tuple[bytes, str]:
    """Generate a realistic synthetic UPI + Card payment receipt screenshot."""
    w, h = 860, 980
    img = Image.new("RGB", (w, h), (235, 234, 228))
    draw = ImageDraw.Draw(img)

    # Mobile receipt card
    draw.rounded_rectangle([40, 32, w - 40, h - 32], radius=20, fill=(255, 255, 255), outline=(28, 28, 30), width=2)

    # Top emerald status banner
    draw.rounded_rectangle([68, 60, w - 68, 186], radius=14, fill=(236, 253, 245), outline=(16, 185, 129), width=2)
    f_hdr = _font(28, bold=True)
    f_amt = _font(36, bold=True, mono=True)
    f_lbl = _font(18, bold=False)
    f_val = _font(22, bold=True)
    f_mono = _font(21, bold=True, mono=True)
    f_sm = _font(15, bold=True, mono=True)

    draw.text((96, 80), "PAYMENT SUCCESSFUL", fill=(6, 95, 70), font=f_hdr)
    draw.text((96, 124), "INR 2,499.00", fill=(17, 24, 39), font=f_amt)
    draw.text((w - 280, 92), "UPI / CARD RECEIPT", fill=(6, 95, 70), font=f_sm)

    # Section divider
    y = 222
    rows = [
        ("Customer: Priya Nair", f_val),
        ("Phone: +91 98201 55432", f_mono),
        ("Email: priya.nair@outlook.in", f_val),
        ("Paid By UPI: priya.nair@okicici", f_mono),
        ("Merchant UPI: freshmart.retail@ybl", f_mono),
        ("Card Used: 5412 7512 3412 8890", f_mono),
        ("Tax PAN: BNZPM9412K", f_mono),
    ]

    for text, font in rows:
        draw.rounded_rectangle([68, y - 10, w - 68, y + 48], radius=8, fill=(250, 250, 248), outline=(228, 226, 220), width=1)
        draw.text((92, y + 4), text, fill=(20, 20, 19), font=font)
        y += 74

    # Billing address card
    draw.rounded_rectangle([68, y, w - 68, y + 114], radius=10, fill=(250, 250, 248), outline=(228, 226, 220), width=1)
    draw.text((92, y + 14), "Billing Address: 18 Linking Road, Bandra West,", fill=(20, 20, 19), font=f_val)
    draw.text((92, y + 54), "Mumbai, Maharashtra 400050", fill=(20, 20, 19), font=f_val)

    # Footer metadata
    draw.text((68, h - 82), "REF ID: TXN-9048127319  //  SYNTHETIC RECEIPT FOR ON-DEVICE TESTING", fill=(120, 118, 112), font=f_lbl)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "synthetic_upi_receipt.png"


def generate_sample_image(kind: SampleKind = "indian_id") -> Tuple[bytes, str]:
    if kind == "upi_receipt":
        return generate_upi_receipt_sample()
    return generate_indian_id_sample()
