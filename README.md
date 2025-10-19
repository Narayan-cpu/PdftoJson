# 📄 PDF to Structured JSON Extractor

This project provides a Python script (`app.py`) that extracts **structured content** from PDF files and converts it into **JSON** format. It can capture:

* ✅ Text (with heading/paragraph detection)
* ✅ Tables (using [camelot](https://camelot-py.readthedocs.io/en/master/))
* ✅ Images (embedded in the PDF)
* ✅ Charts/diagrams (if `opencv` is available for basic detection)
* ✅ OCR support for scanned PDFs (if `pytesseract` is installed)

---

## ⚙️ Requirements

Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```txt
fitz  # PyMuPDF
pdfplumber
camelot-py[cv]
pytesseract
Pillow
opencv-python
numpy
```

---

## 🚀 Usage

```bash
python app.py input.pdf output.json
```

### Arguments

* `input.pdf` → Path to the PDF file you want to parse.
* `output.json` → File where structured JSON output will be written.

---

## 📊 Example Output

```json
{
  "pages": [
    {
      "number": 1,
      "content": [
        {
          "type": "paragraph",
          "text": "This is a sample introduction paragraph."
        },
        {
          "type": "table",
          "data": [
            ["Header1", "Header2"],
            ["Row1Col1", "Row1Col2"]
          ]
        },
        {
          "type": "image",
          "path": "tmp/image_1.png"
        }
      ]
    }
  ]
}
```

---

## 🔧 Features & Optional Dependencies

* **Text extraction** → always works with `pdfplumber`
* **Tables** → needs `camelot-py`
* **OCR for scanned text** → needs `pytesseract` & Tesseract installed
* **Chart detection** → needs `opencv-python`

---

## 🛠 Development Notes

* Extracted images are temporarily saved in `/tmp` or system temp folder.
* Headings are approximated by font size and style detection.
* Charts are detected heuristically via contour detection (basic support).

---

## 👨‍💻 Author
**Narayan Naik**
---
