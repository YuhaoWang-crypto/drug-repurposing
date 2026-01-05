\
from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def read_text_maybe_gz(path: Path, encoding: str = "utf-8", errors: str = "replace") -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding, errors=errors) as f:
            return f.read()
    return path.read_text(encoding=encoding, errors=errors)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def requests_get(url: str, timeout: int = 60, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code >= 500:
                raise RuntimeError(f"ServerError {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(backoff ** i)
    raise last  # type: ignore


def requests_post_json(url: str, payload: Dict[str, Any], timeout: int = 120, retries: int = 4, backoff: float = 2.0) -> Dict[str, Any]:
    headers = {"content-type": "application/json"}
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
            if r.status_code >= 500:
                raise RuntimeError(f"ServerError {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff ** i)
    raise last  # type: ignore


def download_file(url: str, out_path: Path, timeout: int = 120, retries: int = 4, backoff: float = 2.0, chunk: int = 1 << 20) -> Path:
    """
    Download URL to out_path. Skips download if file exists and non-empty.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    last = None
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code >= 500:
                    raise RuntimeError(f"ServerError {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for b in r.iter_content(chunk_size=chunk):
                        if b:
                            f.write(b)
            if out_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file: {out_path}")
            return out_path
        except Exception as e:
            last = e
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            time.sleep(backoff ** i)

    raise last  # type: ignore


def parse_href_list_from_html(html: str) -> List[str]:
    """
    NCBI ftp directory listings are simple HTML with <a href="...">.
    Return raw href values.
    """
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    # Remove parent links, dirs
    hrefs = [h for h in hrefs if h and not h.startswith("?") and h not in ("../", "./")]
    return hrefs
