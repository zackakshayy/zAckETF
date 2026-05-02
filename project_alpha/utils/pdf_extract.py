# project_alpha/utils/pdf_extract.py
from __future__ import annotations

import os
import re
import io
import shutil
import tempfile
import concurrent.futures as cf
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# ---------- Text extraction (summary) ----------
try:
    from pdfminer.high_level import extract_text
except Exception:
    extract_text = None

RETURNS_KEYS = [
    r"YTD", r"1\s*Year", r"3\s*Year", r"5\s*Year", r"10\s*Year",
    r"Since\s*Inception", r"Since\-?Inception"
]
FUND_NAME_PAT = r"(Columbia\s+Research\s+Enhanced\s+Core\s+ETF|RECS)"
BENCH_PAT = r"(Russell\s*1000(?:\s*Index)?)"
INCEPTION_PAT = r"(Inception\s*Date|Fund\s*Inception)\s*[:\-]?\s*([A-Za-z]{3}\s*\d{1,2},\s*\d{4}|\d{2}/\d{2}/\d{4})"
EXPENSE_PAT = r"(Net\s*Expense\s*Ratio|Expense\s*Ratio)\s*[:\-]?\s*([0-9.]+\s*%)"
OBJECTIVE_PAT = r"(investment\s*objective|strategy|seeks)\s*[:\-]?\s*(.+)"
CUSIP_PAT = r"(CUSIP)\s*[:\-]?\s*([0-9A-Z]{9})"
TICKER_PAT = r"(Ticker)\s*[:\-]?\s*(RECS)"

def _extract_text(path: str) -> str:
    if extract_text is None:
        return ""
    try:
        return extract_text(path) or ""
    except Exception:
        return ""

def _grab(pattern: str, text: str, flags=re.IGNORECASE | re.MULTILINE) -> str | None:
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return (m.group(m.lastindex) if m.lastindex else m.group(0)).strip()

def _parse_returns_block(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in RETURNS_KEYS:
        pat = rf"{key}[^0-9\-+%]*([\-+]?\d{1,2}\.?\d*\s*%)"
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            pat2 = rf"{key}[^0-9\-+%]*([\-+]?\d{1,2}\.?\d*)"
            m = re.search(pat2, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            try:
                num = float(val.replace("%", "").strip())
                out[re.sub(r"\s+", "_", re.sub(r"[^A-Za-z0-9 ]", "", key)).lower()] = num / 100.0
            except Exception:
                pass
    return out

def _parse_pdf_facts(path: str) -> Dict[str, str | float]:
    txt = _extract_text(path)
    if not txt:
        return {"file": path, "parse_ok": False, "note": "No text layer / failed extraction"}
    info: Dict[str, str | float] = {"file": path, "parse_ok": True}
    info["fund_match"] = bool(re.search(FUND_NAME_PAT, txt, re.IGNORECASE))
    info["benchmark_match"] = bool(re.search(BENCH_PAT, txt, re.IGNORECASE))
    info["inception"] = _grab(INCEPTION_PAT, txt)
    info["expense_ratio"] = _grab(EXPENSE_PAT, txt)
    info["cusip"] = _grab(CUSIP_PAT, txt)
    info["ticker"] = _grab(TICKER_PAT, txt)
    obj = _grab(OBJECTIVE_PAT, txt)
    if obj and len(obj) > 500: obj = obj[:500] + " …"
    info["objective_or_method"] = obj
    for k, v in _parse_returns_block(txt).items():
        info[f"return_{k}"] = v
    return info

def summarize_recs_pdfs(pdf_dir: str, out_csv: str) -> pd.DataFrame:
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")
    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        raise FileNotFoundError(f"No PDFs found under {pdf_dir}")
    rows: List[Dict[str, str | float]] = []
    with cf.ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
        for res in tqdm(ex.map(_parse_pdf_facts, files), total=len(files), desc="Parse RECS PDFs", unit="pdf"):
            rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# ---------- Table extraction ----------
def _try_pdfplumber_tables(path: str) -> List[pd.DataFrame]:
    try:
        import pdfplumber
    except Exception:
        return []
    out: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tbl in tables or []:
                    df = pd.DataFrame(tbl)
                    # promote first non-empty row as header if looks like headers
                    if not df.empty:
                        header_row = df.iloc[0]
                        if header_row.isnull().sum() == 0:
                            df.columns = header_row
                            df = df.iloc[1:].reset_index(drop=True)
                    out.append(df)
    except Exception:
        return out
    return out

def _try_camelot_tables(path: str) -> List[pd.DataFrame]:
    try:
        import camelot
    except Exception:
        return []
    out: List[pd.DataFrame] = []
    try:
        # Try lattice first (explicit cell lines), then stream (white-space separated)
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(path, pages="all", flavor=flavor)
                for t in tables:
                    df = t.df
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out.append(df)
            except Exception:
                continue
    except Exception:
        return out
    return out

def _try_tabula_tables(path: str) -> List[pd.DataFrame]:
    try:
        import tabula
    except Exception:
        return []
    out: List[pd.DataFrame] = []
    try:
        dfs = tabula.read_pdf(path, pages="all", multiple_tables=True)
        for df in dfs or []:
            if isinstance(df, pd.DataFrame) and not df.empty:
                out.append(df)
    except Exception:
        return out
    return out

def _maybe_ocr_page_images(path: str, dpi: int = 300) -> List[pd.DataFrame]:
    """
    Optional OCR (disabled by default). Extract text-only CSV per page using pytesseract.
    This is a best-effort fallback for image-only PDFs, but will not perfectly reconstruct tables.
    """
    try:
        import pdf2image, pytesseract
    except Exception:
        return []
    out: List[pd.DataFrame] = []
    try:
        imgs = pdf2image.convert_from_path(path, dpi=dpi)
        for img in imgs:
            text = pytesseract.image_to_string(img)
            # crude table rows split on newlines and tabs
            rows = []
            for line in text.splitlines():
                parts = [p for p in re.split(r"\t+|\s{2,}", line.strip()) if p]
                if parts:
                    rows.append(parts)
            if rows:
                out.append(pd.DataFrame(rows))
    except Exception:
        return []
    return out

def extract_tables_from_pdfs(
    pdf_dir: str,
    out_dir: str,
    extractors: Optional[List[str]] = None,
    enable_ocr: bool = False,
    excel_name: str = "recs_pdf_tables.xlsx",
) -> Tuple[int, List[str]]:
    """
    Extract tables from all PDFs in `pdf_dir` using the configured extractor order.
    Saves each table to CSV and a master Excel with one sheet per table.
    Returns (num_tables, file_list).
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")
    os.makedirs(out_dir, exist_ok=True)
    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        raise FileNotFoundError(f"No PDFs found under {pdf_dir}")

    order = extractors or ["pdfplumber", "camelot", "tabula"]
    writer = pd.ExcelWriter(os.path.join(out_dir, excel_name), engine="xlsxwriter")
    saved_csvs: List[str] = []
    n_tables = 0

    def _extract_one(path: str) -> List[pd.DataFrame]:
        # Try extractors in order
        for name in order:
            if name == "pdfplumber":
                tables = _try_pdfplumber_tables(path)
            elif name == "camelot":
                tables = _try_camelot_tables(path)
            elif name == "tabula":
                tables = _try_tabula_tables(path)
            else:
                tables = []
            if tables:
                return tables
        # Optional OCR fallback for image PDFs
        if enable_ocr:
            return _maybe_ocr_page_images(path)
        return []

    for pdf_path in tqdm(files, desc="Extract tables from RECS PDFs", unit="pdf"):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        tables = _extract_one(pdf_path)
        for j, df in enumerate(tables):
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            # Clean minimal: drop all-empty cols/rows
            df = df.replace({"": None}).dropna(how="all", axis=0).dropna(how="all", axis=1)
            # Save CSV
            csv_path = os.path.join(out_dir, f"{base}_table{j+1}.csv")
            df.to_csv(csv_path, index=False)
            saved_csvs.append(csv_path)
            # Save Excel sheet
            sheet = f"{base[:25]}_{j+1}"
            try:
                df.to_excel(writer, sheet_name=sheet, index=False)
            except Exception:
                # fallback truncated columns to fit Excel limits
                df.iloc[:, :50].to_excel(writer, sheet_name=sheet, index=False)
            n_tables += 1

    try:
        writer.close()
    except Exception:
        pass

    return n_tables, saved_csvs
