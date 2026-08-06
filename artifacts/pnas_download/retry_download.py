#!/usr/bin/env python3
"""
Retry downloading DOI 10.1073/pnas.1718406115 via every paperscraper path
and several direct HTTP strategies. Writes attempt logs + any recovered files.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Ensure repo root is importable when run as a script
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DOI = "10.1073/pnas.1718406115"
URL = f"https://www.pnas.org/doi/{DOI}"
DOI_URL = f"https://doi.org/{DOI}"
MAIL = os.environ.get("PAPERSCRAPER_MAIL", "paperscraper-debug@example.com")
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "download_attempts.log"
SUMMARY_PATH = OUT_DIR / "download_attempts_summary.json"
PDF_DIR = OUT_DIR / "pdfs"

USER_AGENTS = [
    "paperscraper/1.0 (+https)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

DIRECT_CANDIDATE_URLS = [
    URL,
    DOI_URL,
    f"https://www.pnas.org/doi/pdf/{DOI}",
    f"https://www.pnas.org/doi/pdf/{DOI}?download=true",
    f"https://pnas.org/doi/pdf/{DOI}",
    f"https://www.pnas.org/doi/epdf/{DOI}",
    f"https://www.pnas.org/content/pnas/{DOI.replace('10.1073/', '')}.full.pdf",
]


class JsonListHandler(logging.Handler):
    def __init__(self, records: List[Dict[str, Any]]):
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
        )


attempts: List[Dict[str, Any]] = []
log_records: List[Dict[str, Any]] = []


def setup_logging() -> logging.Logger:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicate noise
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    jh = JsonListHandler(log_records)
    jh.setLevel(logging.DEBUG)
    jh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(jh)

    return logging.getLogger("pnas_retry")


def record_attempt(
    name: str,
    success: bool,
    detail: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    entry = {
        "attempt": len(attempts) + 1,
        "name": name,
        "success": success,
        "ts": datetime.now(timezone.utc).isoformat(),
        "detail": detail or {},
        "error": error,
        "path": path,
    }
    attempts.append(entry)
    logging.getLogger("pnas_retry").info(
        "ATTEMPT %-3d %-28s success=%s path=%s error=%s",
        entry["attempt"],
        name,
        success,
        path,
        (error or "")[:200],
    )


def is_pdf_bytes(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def save_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def try_http_get(
    name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    allow_redirects: bool = True,
    out_name: Optional[str] = None,
) -> bool:
    log = logging.getLogger("pnas_retry")
    try:
        resp = requests.get(
            url,
            headers=headers or {},
            timeout=90,
            allow_redirects=allow_redirects,
            stream=True,
        )
        content = resp.content
        detail = {
            "url": url,
            "final_url": str(resp.url),
            "status": resp.status_code,
            "content_type": resp.headers.get("Content-Type"),
            "bytes": len(content),
            "headers_req": headers or {},
        }
        if resp.status_code == 200 and is_pdf_bytes(content):
            out = PDF_DIR / (out_name or f"{name.replace('/', '_')}.pdf")
            save_bytes(out, content)
            record_attempt(name, True, detail=detail, path=str(out))
            return True
        # Save HTML/error body snippet for debugging
        snippet_path = PDF_DIR / f"{name.replace('/', '_')}.response.bin"
        save_bytes(snippet_path, content[:50_000])
        detail["snippet_path"] = str(snippet_path)
        detail["startswith"] = content[:40].decode("latin-1", errors="replace")
        record_attempt(
            name,
            False,
            detail=detail,
            error=f"HTTP {resp.status_code}, not a PDF (ctype={detail['content_type']})",
        )
        return False
    except Exception as e:
        log.debug("HTTP failure for %s: %s", name, traceback.format_exc())
        record_attempt(name, False, detail={"url": url}, error=str(e))
        return False


def try_citation_pdf_meta(ua: str) -> bool:
    """Follow doi.org / PNAS landing page and pull citation_pdf_url meta."""
    log = logging.getLogger("pnas_retry")
    name = f"citation_pdf_meta[{ua[:24]}]"
    try:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": ua}
        resp = requests.get(DOI_URL, headers=headers, timeout=90)
        soup = BeautifulSoup(resp.text, features="lxml")
        meta = soup.find("meta", {"name": "citation_pdf_url"})
        detail = {
            "landing_status": resp.status_code,
            "landing_url": str(resp.url),
            "meta_found": bool(meta and meta.get("content")),
            "meta_content": meta.get("content") if meta else None,
        }
        if not meta or not meta.get("content"):
            # also try PNAS page directly
            resp2 = requests.get(URL, headers=headers, timeout=90)
            soup2 = BeautifulSoup(resp2.text, features="lxml")
            meta = soup2.find("meta", {"name": "citation_pdf_url"})
            detail["pnas_status"] = resp2.status_code
            detail["pnas_url"] = str(resp2.url)
            detail["meta_found"] = bool(meta and meta.get("content"))
            detail["meta_content"] = meta.get("content") if meta else None
            # save HTML for inspection
            save_bytes(PDF_DIR / "landing_pnas.html", resp2.content[:200_000])
        if meta and meta.get("content"):
            pdf_url = meta["content"]
            ok = try_http_get(
                f"{name}->pdf",
                pdf_url,
                headers=headers,
                out_name=f"citation_meta_{abs(hash(ua)) % 10_000}.pdf",
            )
            if not ok:
                record_attempt(name, False, detail=detail, error="meta present but PDF fetch failed")
            return ok
        record_attempt(name, False, detail=detail, error="no citation_pdf_url meta")
        return False
    except Exception as e:
        log.debug(traceback.format_exc())
        record_attempt(name, False, error=str(e))
        return False


def try_unpaywall_probe() -> Optional[str]:
    """Query Unpaywall and return best OA PDF URL if any."""
    url = f"https://api.unpaywall.org/v2/{DOI}?email={MAIL}"
    try:
        r = requests.get(url, timeout=60)
        data = r.json()
        save_bytes(PDF_DIR / "unpaywall.json", json.dumps(data, indent=2).encode())
        best = (data.get("best_oa_location") or {}) if isinstance(data, dict) else {}
        pdf_url = best.get("url_for_pdf")
        record_attempt(
            "unpaywall_probe",
            bool(pdf_url),
            detail={
                "status": r.status_code,
                "is_oa": data.get("is_oa") if isinstance(data, dict) else None,
                "pdf_url": pdf_url,
                "oa_locations": len(data.get("oa_locations") or [])
                if isinstance(data, dict)
                else 0,
            },
            error=None if pdf_url else "no url_for_pdf",
        )
        return pdf_url
    except Exception as e:
        record_attempt("unpaywall_probe", False, error=str(e))
        return None


def try_paperscraper_save_pdf() -> bool:
    from paperscraper.pdf import load_api_keys, save_pdf

    api_keys = load_api_keys(None)
    out = PDF_DIR / "save_pdf.pdf"
    try:
        res = save_pdf(
            {"doi": DOI},
            filepath=out,
            save_metadata=True,
            api_keys=api_keys,
            mail=MAIL,
        )
        ok = bool(res.get("success"))
        path = None
        if out.with_suffix(".pdf").exists():
            path = str(out.with_suffix(".pdf"))
        elif out.with_suffix(".xml").exists():
            path = str(out.with_suffix(".xml"))
        record_attempt("save_pdf", ok, detail=res, path=path)
        return ok
    except Exception as e:
        record_attempt("save_pdf", False, error=f"{e}\n{traceback.format_exc()}")
        return False


def try_debug_save_pdf() -> bool:
    from paperscraper.pdf import load_api_keys
    from paperscraper.pdf.pdf import debug_save_pdf

    api_keys = load_api_keys(None)
    out = PDF_DIR / "debug_save_pdf.pdf"
    try:
        res = debug_save_pdf(
            {"doi": DOI},
            filepath=out,
            api_keys=api_keys,
            mail=MAIL,
            save_first_only=False,
        )
        ok = bool(res.get("successes"))
        record_attempt("debug_save_pdf_all", ok, detail=res)
        return ok
    except Exception as e:
        record_attempt(
            "debug_save_pdf_all", False, error=f"{e}\n{traceback.format_exc()}"
        )
        return False


def try_each_fallback() -> bool:
    from paperscraper.pdf import load_api_keys
    from paperscraper.pdf.fallbacks import FALLBACKS

    api_keys = load_api_keys(None)
    any_ok = False
    paper = {"doi": DOI}

    for name, fn in FALLBACKS.items():
        out = PDF_DIR / f"fallback_{name}"
        ok = False
        err = None
        detail: Dict[str, Any] = {"fallback": name}
        try:
            if name == "unpaywall":
                ok = bool(fn(DOI, out, MAIL, None))
            elif name == "bioc_pmc":
                ok = bool(fn(DOI, out, MAIL))
            elif name == "crossref":
                ok = bool(fn(DOI, out, MAIL))
            elif name in ("europepmc", "doaj", "openalex", "arxiv", "plos", "elife"):
                ok = bool(fn(DOI, out))
            elif name in ("s3", "medrxiv_s3"):
                if api_keys.get("AWS_ACCESS_KEY_ID") and api_keys.get(
                    "AWS_SECRET_ACCESS_KEY"
                ):
                    ok = bool(fn(DOI, out, api_keys))
                else:
                    err = "skipped: missing AWS credentials"
            elif name in ("wiley", "springer"):
                key = (
                    "WILEY_TDM_API_TOKEN"
                    if name == "wiley"
                    else "SPRINGER_API_KEY"
                )
                if not api_keys.get(key):
                    err = f"skipped: missing {key}"
                else:
                    ok = bool(fn(paper, out, api_keys))
            elif name == "elsevier":
                if not api_keys.get("ELSEVIER_TDM_API_KEY"):
                    err = "skipped: missing ELSEVIER_TDM_API_KEY"
                else:
                    ok = bool(fn(paper, out, api_keys, preferred_type="pdf"))
            else:
                # generic best-effort
                try:
                    ok = bool(fn(DOI, out))
                except TypeError:
                    ok = bool(fn(paper, out, api_keys))
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            ok = False

        path = None
        for suf in (".pdf", ".xml"):
            p = out.with_suffix(suf)
            if p.exists() and p.stat().st_size > 0:
                path = str(p)
                break
        record_attempt(f"fallback:{name}", ok, detail=detail, error=err, path=path)
        any_ok = any_ok or ok
        time.sleep(0.5)
    return any_ok


def try_europepmc_rest() -> bool:
    """Direct Europe PMC full-text / PDF endpoints."""
    urls = [
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{DOI}&format=json&resultType=core",
    ]
    try:
        r = requests.get(urls[0], timeout=60)
        data = r.json()
        save_bytes(PDF_DIR / "europepmc_search.json", json.dumps(data, indent=2).encode())
        results = (data.get("resultList") or {}).get("result") or []
        detail = {"n_results": len(results)}
        if not results:
            record_attempt("europepmc_rest_search", False, detail=detail, error="no hits")
            return False
        hit = results[0]
        detail["pmcid"] = hit.get("pmcid")
        detail["hasPDF"] = hit.get("hasPDF")
        detail["isOpenAccess"] = hit.get("isOpenAccess")
        pmcid = hit.get("pmcid")
        record_attempt("europepmc_rest_search", True, detail=detail)
        if pmcid:
            pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            return try_http_get(
                "europepmc_pdf_render",
                pdf_url,
                headers={"User-Agent": USER_AGENTS[1]},
                out_name="europepmc_render.pdf",
            )
        return False
    except Exception as e:
        record_attempt("europepmc_rest_search", False, error=str(e))
        return False


def try_openalex_probe() -> bool:
    url = f"https://api.openalex.org/works/https://doi.org/{DOI}"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": f"paperscraper ({MAIL})"})
        data = r.json()
        save_bytes(PDF_DIR / "openalex.json", json.dumps(data, indent=2).encode())
        loc = data.get("best_oa_location") or {}
        pdf_url = loc.get("pdf_url") or (data.get("primary_location") or {}).get("pdf_url")
        record_attempt(
            "openalex_probe",
            bool(pdf_url),
            detail={"status": r.status_code, "pdf_url": pdf_url, "oa": data.get("open_access")},
            error=None if pdf_url else "no pdf_url",
        )
        if pdf_url:
            return try_http_get(
                "openalex_pdf",
                pdf_url,
                headers={"User-Agent": USER_AGENTS[1]},
                out_name="openalex.pdf",
            )
        return False
    except Exception as e:
        record_attempt("openalex_probe", False, error=str(e))
        return False


def try_crossref_probe() -> bool:
    url = f"https://api.crossref.org/works/{DOI}"
    try:
        r = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": f"paperscraper/1.0 (mailto:{MAIL})"},
        )
        data = r.json()
        save_bytes(PDF_DIR / "crossref.json", json.dumps(data, indent=2).encode())
        message = data.get("message") or {}
        links = message.get("link") or []
        pdf_links = [
            L.get("URL")
            for L in links
            if L.get("content-type") == "application/pdf" or "pdf" in (L.get("URL") or "").lower()
        ]
        record_attempt(
            "crossref_probe",
            bool(pdf_links),
            detail={"status": r.status_code, "links": links, "pdf_links": pdf_links},
            error=None if pdf_links else "no pdf links",
        )
        any_ok = False
        for i, pdf_url in enumerate(pdf_links):
            any_ok = (
                try_http_get(
                    f"crossref_pdf_{i}",
                    pdf_url,
                    headers={"User-Agent": USER_AGENTS[1]},
                    out_name=f"crossref_{i}.pdf",
                )
                or any_ok
            )
        return any_ok
    except Exception as e:
        record_attempt("crossref_probe", False, error=str(e))
        return False


def try_semantic_scholar() -> bool:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}?fields=title,openAccessPdf,isOpenAccess,externalIds"
    headers = {"User-Agent": USER_AGENTS[1]}
    if os.environ.get("SS_API_KEY"):
        headers["x-api-key"] = os.environ["SS_API_KEY"]
    try:
        r = requests.get(url, headers=headers, timeout=60)
        data = r.json()
        save_bytes(PDF_DIR / "semanticscholar.json", json.dumps(data, indent=2).encode())
        oa = data.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
        record_attempt(
            "semantic_scholar_probe",
            bool(pdf_url),
            detail={"status": r.status_code, "pdf_url": pdf_url, "isOpenAccess": data.get("isOpenAccess")},
            error=None if pdf_url else "no openAccessPdf.url",
        )
        if pdf_url:
            return try_http_get(
                "semantic_scholar_pdf",
                pdf_url,
                headers={"User-Agent": USER_AGENTS[1]},
                out_name="semanticscholar.pdf",
            )
        return False
    except Exception as e:
        record_attempt("semantic_scholar_probe", False, error=str(e))
        return False


def try_pmc_oa() -> bool:
    """NCBI ID converter -> PMC OA PDF package / article PDF."""
    conv = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    try:
        r = requests.get(
            conv,
            params={
                "tool": "paperscraper",
                "email": MAIL,
                "ids": DOI,
                "idtype": "doi",
                "format": "json",
            },
            timeout=60,
        )
        data = r.json()
        save_bytes(PDF_DIR / "ncbi_idconv.json", json.dumps(data, indent=2).encode())
        records = data.get("records") or []
        pmcid = records[0].get("pmcid") if records else None
        record_attempt(
            "ncbi_idconv",
            bool(pmcid),
            detail={"status": r.status_code, "records": records},
            error=None if pmcid else "no pmcid",
        )
        if not pmcid:
            return False
        candidates = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
            f"https://europepmc.org/articles/{pmcid}?pdf=render",
        ]
        any_ok = False
        for i, u in enumerate(candidates):
            any_ok = (
                try_http_get(
                    f"pmc_pdf_{i}",
                    u,
                    headers={"User-Agent": USER_AGENTS[1]},
                    out_name=f"pmc_{i}.pdf",
                )
                or any_ok
            )
        return any_ok
    except Exception as e:
        record_attempt("ncbi_idconv", False, error=str(e))
        return False


def finalize() -> int:
    successes = [a for a in attempts if a["success"]]
    summary = {
        "doi": DOI,
        "url": URL,
        "started": attempts[0]["ts"] if attempts else None,
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_attempts": len(attempts),
        "n_successes": len(successes),
        "success_methods": [a["name"] for a in successes],
        "attempts": attempts,
        "recovered_files": sorted(
            str(p) for p in PDF_DIR.glob("*") if p.suffix.lower() in {".pdf", ".xml"} and p.stat().st_size > 100
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.getLogger("pnas_retry").info(
        "DONE: %d/%d successes: %s",
        len(successes),
        len(attempts),
        summary["success_methods"],
    )
    logging.getLogger("pnas_retry").info("Summary written to %s", SUMMARY_PATH)
    return 0 if successes else 1


def main() -> int:
    log = setup_logging()
    log.info("Starting exhaustive download retries for %s", DOI)
    log.info("Output dir: %s", OUT_DIR)

    # 1) Direct URL candidates with multiple UAs
    for ua in USER_AGENTS:
        for url in DIRECT_CANDIDATE_URLS:
            short = url.replace("https://", "").replace("/", "_")[:60]
            try_http_get(
                f"direct[{ua[:12]}]->{short}",
                url,
                headers={"User-Agent": ua, "Accept": "application/pdf,*/*"},
                out_name=f"direct_{abs(hash((ua, url))) % 100000}.pdf",
            )
            time.sleep(0.3)

    # 2) citation_pdf_url scraping
    for ua in USER_AGENTS:
        try_citation_pdf_meta(ua)
        time.sleep(0.5)

    # 3) Aggregator probes
    pdf = try_unpaywall_probe()
    if pdf:
        try_http_get(
            "unpaywall_best_oa_pdf",
            pdf,
            headers={"User-Agent": USER_AGENTS[1]},
            out_name="unpaywall_best.pdf",
        )
    try_openalex_probe()
    try_crossref_probe()
    try_semantic_scholar()
    try_europepmc_rest()
    try_pmc_oa()

    # 4) paperscraper APIs
    try_paperscraper_save_pdf()
    try_each_fallback()
    try_debug_save_pdf()

    return finalize()


if __name__ == "__main__":
    sys.exit(main())
