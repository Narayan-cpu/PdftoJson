#!/usr/bin/env python3
"""
app.py

Usage:
    pip install -r requirements.txt
    python app.py input.pdf output.json

Description:
    Parses input.pdf and writes structured JSON to output.json.
    Extracts text (with headings), tables, images, and charts.
    Author: Narayan Naik
"""

import sys
import os
import json
import tempfile
from typing import List, Dict, Any, Optional

import fitz 
import pdfplumber


try:
    import camelot
    _HAS_CAMELot = True
except Exception:
    _HAS_CAMELot = False

try:
    import pytesseract
    from PIL import Image
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import pandas as pd

# ---------- Utility helpers ---------- Narayan naik----

def save_image_bytes_to_tempfile(img_bytes: bytes, suffix=".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

def page_image_from_fitz(page, zoom=2) -> Image.Image:
    """Render a fitz page to a PIL Image (for OCR or CV)."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img

# ---------- Text extraction & paragraph grouping ----------Narayan naik----

def extract_text_blocks_pdfplumber_page(p: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Use pdfplumber to get words and group into paragraphs.
    Returns list of dicts with keys: text, possible_heading (bool), font_size (avg)
    """
    words = p.extract_words(extra_attrs=["size", "fontname"])  # may include size
    if not words:
        return []


    lines = {}
    for w in words:
        top = round(w.get("top", 0))
        key = top
        lines.setdefault(key, []).append(w)

    sorted_lines_keys = sorted(lines.keys())
    line_texts = []
    for key in sorted_lines_keys:
        words_line = sorted(lines[key], key=lambda x: float(x.get("x0", 0)))
        text = " ".join(w["text"] for w in words_line)
        avg_size = sum(float(w.get("size", 0)) for w in words_line) / max(1, len(words_line))
        fontnames = [w.get("fontname", "") for w in words_line]
        line_texts.append({"text": text.strip(), "size": avg_size, "fontnames": fontnames})


    para_list = []
    current_para = {"lines": [], "avg_size": 0}
    for L in line_texts:
        # Heuristic: if line is empty -> new paragraph
        if not L["text"].strip():
            if current_para["lines"]:
                para_list.append(current_para)
                current_para = {"lines": [], "avg_size": 0}
            continue
        current_para["lines"].append(L["text"])
        current_para["avg_size"] += L["size"]
    if current_para["lines"]:
        para_list.append(current_para)

    # finalize paragraphs
    out = []
    for pblock in para_list:
        lines = pblock["lines"]
        avg_size = pblock["avg_size"] / max(1, len(lines))
        text = "\n".join(lines).strip()
        possible_heading = is_heading_by_text_and_size(lines[0], avg_size)
        out.append({"type": "paragraph", "text": text, "avg_size": avg_size, "possible_heading": possible_heading})
    return out

def is_heading_by_text_and_size(first_line_text: str, avg_size: float) -> bool:
    """
    Heuristic: If first_line is ALL CAPS or avg font size is large -> heading.
    """
    stripped = first_line_text.strip()
    if not stripped:
        return False
    # all caps heuristic (allow numbers and punctuation)
    letters = [c for c in stripped if c.isalpha()]
    all_caps = bool(letters) and all(c.isupper() for c in letters)
    large_font = avg_size >= 11.5  # heuristic threshold; many PDFs use 10-12 for body
    # also detect lines that end with ':' as a heading
    if all_caps or large_font or stripped.endswith(":"):
        return True
    return False

# ---------- Table extraction ----------Narayan naik----

def extract_tables_for_page_using_camelot(pdf_path: str, page_number: int) -> List[List[List[str]]]:
    """
    Use camelot to extract tables on a single page.
    Returns list of table data as lists of rows (cells).
    """
    if not _HAS_CAMELot:
        return []
    try:
        # Camelot page numbering is 1-based and accepts string ranges
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="lattice")  # lattice and stream heuristics possible
        results = []
        for t in tables:
            df = t.df.fillna("").astype(str)
            results.append(df.values.tolist())
        # If lattice returned nothing, fallback to stream
        if not results:
            tables_stream = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="stream")
            for t in tables_stream:
                df = t.df.fillna("").astype(str)
                results.append(df.values.tolist())
        return results
    except Exception:
        return []

def extract_tables_pdfplumber_page(p: pdfplumber.page.Page) -> List[List[List[str]]]:
    """
    Try pdfplumber table extraction for a page.
    Returns list of tables as list of rows.
    """
    tables = []
    try:
        raw_tables = p.extract_tables()
        for t in raw_tables:
            # t is list of rows
            rows = [[(cell if cell is not None else "") for cell in row] for row in t]
            if any(any(cell.strip() for cell in row if isinstance(cell, str)) for row in rows):
                tables.append(rows)
    except Exception:
        pass
    return tables

# ---------- Image & chart extraction ---------- Narayan naik----

def extract_images_with_fitz(doc: fitz.Document, page_index: int) -> List[Dict[str, Any]]:
    """
    Extract images embedded in the page and save bytes.
    Returns list of dicts: {'image_bytes': ..., 'ext': '.png', 'xrefs': xref}
    """
    out = []
    try:
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = "." + base_image.get("ext", "png")
            out.append({"image_bytes": img_bytes, "ext": ext, "xref": xref})
    except Exception:
        pass
    return out

def looks_like_chart(image_path: str) -> bool:
    """
    Basic heuristic using OpenCV:
      - High density of straight lines (axes, gridlines)
      - Presence of text (we won't do heavy text detection here)
    Returns True if likely a chart.
    """
    if not _HAS_CV2:
        return False
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False
        # Resize to speed up
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        # Edge detection
        edges = cv2.Canny(img, 50, 150)
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
        if lines is None:
            return False
        # count roughly vertical/horizontal lines
        cnt = 0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 or dy == 0:
                cnt += 1
            else:
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if angle < 10 or angle > 170 or (80 < angle < 100):
                    cnt += 1
        # Heuristic threshold: many straight lines -> chart
        return cnt >= 6
    except Exception:
        return False

# ---------- OCR fallback ----------Narayan naik----

def ocr_image_to_text(pil_image: Image.Image) -> str:
    if not _HAS_TESSERACT:
        return ""
    try:
        return pytesseract.image_to_string(pil_image)
    except Exception:
        return ""

# ---------- Main processing per page ----------Narayan naik----

def process_pdf(input_pdf_path: str) -> Dict[str, Any]:
    """
    Main orchestrator: opens pdf with both fitz and pdfplumber,
    processes each page, and builds the JSON structure.
    """
    result = {"pages": []}
    doc = fitz.open(input_pdf_path)
    with pdfplumber.open(input_pdf_path) as pdf:
        num_pages = len(doc)
        for i in range(num_pages):
            page_number = i + 1
            page_entry = {"page_number": page_number, "content": []}

            # pdfplumber page for text/tables
            try:
                p = pdf.pages[i]
            except Exception:
                p = None

            # 1) Text extraction -> paragraphs + headings detection
            text_blocks = []
            if p:
                try:
                    text_blocks = extract_text_blocks_pdfplumber_page(p)
                except Exception:
                    text_blocks = []

            # If text_blocks empty or nearly empty, use OCR on rendered page
            total_text_len = sum(len(b["text"]) for b in text_blocks) if text_blocks else 0
            if total_text_len < 30:
                # Render via fitz and OCR
                try:
                    fitz_page = doc[i]
                    pil_img = page_image_from_fitz(fitz_page, zoom=2)
                    ocr_text = ocr_image_to_text(pil_img) if _HAS_TESSERACT else ""
                    if ocr_text and ocr_text.strip():
                        # break into paragraphs by double newlines
                        paras = [p.strip() for p in ocr_text.split("\n\n") if p.strip()]
                        for para in paras:
                            # simple heading detection from OCR text line
                            first_line = para.splitlines()[0] if para.splitlines() else para
                            possible_heading = is_heading_by_text_and_size(first_line, 12)
                            page_entry["content"].append({
                                "type": "paragraph",
                                "section": None,
                                "sub_section": None,
                                "text": para,
                                "possible_heading": possible_heading,
                                "source": "ocr"
                            })
                except Exception:
                    pass
            else:
                # Convert text_blocks into the JSON structure, attaching headings as section markers
                current_section = None
                current_sub = None
                for block in text_blocks:
                    text = block["text"]
                    if block.get("possible_heading"):
                        # Decide if it's a section or subsection based on size
                        if block["avg_size"] >= 13:
                            current_section = text.strip()
                            current_sub = None
                            page_entry["content"].append({
                                "type": "paragraph",
                                "section": current_section,
                                "sub_section": None,
                                "text": text,
                                "possible_heading": True,
                                "source": "extracted"
                            })
                        else:
                            # treat as subsection
                            current_sub = text.strip()
                            page_entry["content"].append({
                                "type": "paragraph",
                                "section": current_section,
                                "sub_section": current_sub,
                                "text": text,
                                "possible_heading": True,
                                "source": "extracted"
                            })
                    else:
                        page_entry["content"].append({
                            "type": "paragraph",
                            "section": current_section,
                            "sub_section": current_sub,
                            "text": text,
                            "possible_heading": False,
                            "source": "extracted"
                        })

            # 2) Table extraction
            tables = []
            if _HAS_CAMELot:
                try:
                    tables = extract_tables_for_page_using_camelot(input_pdf_path, page_number)
                except Exception:
                    tables = []
            if not tables and p:
                try:
                    tables = extract_tables_pdfplumber_page(p)
                except Exception:
                    tables = []

            for t in tables:
                page_entry["content"].append({
                    "type": "table",
                    "section": None,
                    "description": None,
                    "table_data": t
                })

     
            images = extract_images_with_fitz(doc, i)
            for idx, imginfo in enumerate(images):
                img_bytes = imginfo["image_bytes"]
                ext = imginfo.get("ext", ".png")
                tmp_path = save_image_bytes_to_tempfile(img_bytes, suffix=ext)
                is_chart = False
                if _HAS_CV2:
                    try:
                        is_chart = looks_like_chart(tmp_path)
                    except Exception:
                        is_chart = False

                # encode image as base64 if you want or store path. Here we'll include placeholder and not inline large b64.
                # For now include a small metadata and save local temp path reference
                page_entry["content"].append({
                    "type": "chart" if is_chart else "image",
                    "section": None,
                    "description": None,
                    "image_path": tmp_path,
                    "detected_chart": is_chart
                })

            # Append page entry
            result["pages"].append(page_entry)
    return result

# ---------- CLI entry point ---------- written by Narayan naik----

def main():
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_structured_json.py input.pdf output.json")
        sys.exit(1)
    input_pdf = sys.argv[1]
    output_json = sys.argv[2]
    if not os.path.exists(input_pdf):
        print("Error: input PDF not found:", input_pdf)
        sys.exit(1)
    print("Processing:", input_pdf)
    data = process_pdf(input_pdf)
    # Clean some values (e.g., remove binary blobs). Already we used image_path.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Done. JSON saved to:", output_json)
    print("Note: Image files (if any) saved to system temp directory. Chart detection is heuristic.")

if __name__ == "__main__":
    main()
