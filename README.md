# PrivacyShield — On-Device PII Redaction Lab

PrivacyShield is an on-device sensitive information (PII) detection and redaction prototype built for screenshots and document photos. All OCR, regex matching, named-entity recognition, and OpenCV/Pillow image masking run **100% locally on CPU** with **0 bytes sent to the cloud**.

---

## Architecture Overview

```text
privacyshield/
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI server (REST API + static frontend)
│   ├── detector.py           # RapidOCR + Regex + spaCy NER + latency/RSS benchmarks
│   ├── redactor.py           # OpenCV/Pillow pixelate, matte bar, synthetic mask + overlay
│   └── sample_generator.py   # Synthetic Indian ID Card & UPI Payment Receipt generator
├── frontend/
│   ├── index.html            # Tactile Security Lab Swiss-grid interface
│   ├── styles.css            # Braun / Teenage Engineering inspired hardware styling
│   └── app.js                # Interactive split-screen slider, entity toggles & audit log
├── requirements.txt          # Pinned CPU-only Python dependencies
└── README.md
```

### Pipeline Stages
1. **Local OCR (`backend/detector.py`):** Uses `rapidocr-onnxruntime` (with `pytesseract` fallback) to extract text tokens and polygon bounding boxes on CPU. Tokens are grouped into visual reading lines while maintaining character-to-pixel span mappings.
2. **Hybrid PII Detection:**
   - **High-Precision Regex:** Detects Indian Aadhaar numbers (`XXXX XXXX XXXX`), PAN cards (`[A-Z]{5}[0-9]{4}[A-Z]`), 16-digit debit/credit cards (with optional Luhn confidence boost), UPI IDs (`@okaxis`, `@ybl`, `@oksbi`, etc.), Indian phone numbers (`+91` / 10-digit), and email addresses.
   - **Contextual + spaCy NER (`en_core_web_sm`):** Extracts person names and multi-line Indian street/PIN addresses, mapping matched substrings back to sub-box pixel coordinates.
3. **Surgical Redaction (`backend/redactor.py`):**
   - **Gaussian Pixelation / Mosaic** (`pixelate`)
   - **Solid Matte Redaction Bar** (`matte_bar`) with optional `[ENTITY]` tags
   - **Smart Synthetic Masking** (`synthetic_mask`) with sampled background/foreground palette matching
   - Generates an **Annotated Detection Overlay** displaying color-coded bounding boxes and confidence percentages, plus a **JSON Audit Log** and real-time **Inference Latency (ms)**, **CPU Memory (MB)**, and **Entity Count** metrics.

---

## Setup & Running Locally

### 1. Create a virtual environment and install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Start the server (single command)
From the `backend/` directory:
```powershell
cd backend
uvicorn main:app --reload
```
Or from the repository root:
```powershell
uvicorn backend.main:app --reload
```

### 3. Open the application
Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- Click **GENERATE SAMPLE INDIAN ID / PAYMENT RECEIPT** to test immediately with synthetic Indian KYC ID or UPI Receipt fixtures.
- Drag the **Split Compare** slider handle across the document, toggle individual entities in the **Entity Inspector** drawer, or switch between **Gaussian Pixelation**, **Solid Matte Bar**, and **Smart Synthetic Mask**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Engine readiness, CPU RSS memory, and local-only verification |
| `POST` | `/api/analyze` | Upload an image (`file`, `style`, `show_tags`) for OCR + PII detection + redaction |
| `POST` | `/api/sample` | Generate and analyze a synthetic fixture (`kind`: `indian_id` \| `upi_receipt`) |
| `POST` | `/api/redact` | Re-render redactions instantaneously for toggled entity IDs or style changes |
| `GET` | `/api/sample/{kind}/raw` | Download the unredacted synthetic PNG fixture |
