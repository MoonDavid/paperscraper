"""Functionalities to scrape PDF files of publications."""

import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import quote

import requests
import tldextract
from bs4 import BeautifulSoup
from tqdm import tqdm

from ..utils import load_papers_dump
from .fallbacks import FALLBACKS
from .utils import download_pdf_to_path, get_session, load_api_keys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ABSTRACT_ATTRIBUTE = {
    "biorxiv": ["DC.Description"],
    "arxiv": ["citation_abstract"],
    "chemrxiv": ["citation_abstract"],
}
DEFAULT_ATTRIBUTES = ["citation_abstract", "description"]
CHEMRXIV_API_BASE = "https://www.cambridge.org/engage/coe/public-api/v1/items/doi/"


def _get_chemrxiv_item(
    doi: str, user_agent: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Fetch ChemRxiv metadata from the Cambridge Open Engage API.

    Args:
        doi: The DOI to look up.
        user_agent: Headers to use for the request.

    Returns:
        Item metadata if available, otherwise None.
    """
    api_url = f"{CHEMRXIV_API_BASE}{doi}"
    try:
        response = get_session().get(api_url, headers=user_agent)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning(f"ChemRxiv API lookup failed for {doi}: {exc}")
        return None

    if isinstance(data, dict) and isinstance(data.get("item"), dict):
        return data["item"]
    return data if isinstance(data, dict) else None


def _chemrxiv_metadata_from_item(item: Dict[str, Any], doi: str) -> Dict[str, Any]:
    """Build metadata from a ChemRxiv API item payload.

    Args:
        item: API response payload for the item.
        doi: DOI for logging context.

    Returns:
        A metadata dict with title, authors, and abstract.
    """
    metadata: Dict[str, Any] = {
        "title": item.get("title") or "Title not found",
        "authors": [],
    }

    authors = []
    for author in item.get("authors", []) or []:
        first = (author or {}).get("firstName") or ""
        last = (author or {}).get("lastName") or ""
        name = " ".join(part for part in [first, last] if part).strip()
        if name:
            authors.append(name)
    metadata["authors"] = authors if authors else ["Author information not found"]

    abstract = item.get("abstract")
    if abstract:
        abstract_text = BeautifulSoup(abstract, "html.parser").get_text(separator="\n")
        abstract_text = abstract_text.strip()
        if abstract_text.startswith("Abstract"):
            abstract_text = abstract_text[8:].strip()
        metadata["abstract"] = abstract_text
    else:
        metadata["abstract"] = "Abstract not found"
        logger.warning(f"Could not find abstract for {doi}")

    return metadata


def _chemrxiv_pdf_url(item: Dict[str, Any]) -> Optional[str]:
    """Extract the PDF URL from a ChemRxiv API item payload."""
    asset = item.get("asset")
    if not isinstance(asset, dict):
        return None
    original = asset.get("original")
    if isinstance(original, dict) and original.get("url"):
        return original.get("url")
    return asset.get("url")


def _write_metadata(metadata: Dict[str, Any], output_path: Path) -> bool:
    """Write metadata to a JSON file next to the PDF."""
    try:
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True
    except Exception as exc:
        logger.error(f"Failed to save metadata to {str(output_path)}: {exc}")
        return False


# python
def _get_abstract_pubmed(pmid: str, timeout: int = 20) -> Optional[str]:
    """
    Query NCBI EFetch for PubMed and return the abstract text or None.
    Uses the XML retmode and extracts all AbstractText nodes.
    """
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        resp = get_session().get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        soup_xml = BeautifulSoup(resp.text, "xml")
        abstract_texts = soup_xml.find_all("abstracttext")
        if not abstract_texts:
            return None
        parts = []
        for node in abstract_texts:
            parts.append(node.get_text("\n").strip())
        return "\n".join([p for p in parts if p])
    except Exception as e:
        logger.warning(f"PubMed fetch failed for PMID={pmid}: {e}")
        return None


def _get_abstract_crossref(
    doi: str, timeout: int = 20, mail: Optional[str] = None
) -> Optional[str]:
    """
    Query Crossref works API and return the abstract (HTML cleaned) or None.
    Uses a polite mailto User-Agent when mail is provided.
    """
    try:
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        contact = mail or "paperscraper@example.com"
        headers = {
            "User-Agent": f"paperscraper/1.0 (mailto:{contact})",
            "Accept": "application/json",
        }
        resp = get_session().get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("message", {})
        raw = data.get("abstract")
        if not raw:
            return None
        return BeautifulSoup(raw, "html.parser").get_text("\n").strip()
    except Exception as e:
        logger.warning(f"Crossref fetch failed for DOI={doi}: {e}")
        return None


# python
def _get_abstract_europepmc(doi: str, timeout: int = 20) -> Optional[str]:
    """
    Query Europe PMC REST API for DOI and return the abstract (prefer `abstractText`) or None.
    Uses resultType=core&format=json as requested.
    """
    try:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {"query": f"DOI:{doi}", "resultType": "core", "format": "json"}
        resp = get_session().get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None
        first = results[0]
        # Try keys likely returned: 'abstractText', 'abstract' (case-insensitive fallback)
        abstract = first.get("abstractText") or first.get("abstract")
        if not abstract:
            # case-insensitive fallback
            for k, v in first.items():
                if k.lower() == "abstracttext" or k.lower() == "abstract":
                    abstract = v
                    break
        if not abstract:
            return None
        return BeautifulSoup(abstract, "html.parser").get_text("\n").strip()
    except Exception as e:
        logger.warning(f"EuropePMC fetch failed for DOI={doi}: {e}")
        return None


# --- Replace abstract retrieval section in save_file with the following block ---


def save_file(
    paper_metadata: Dict[str, Any],
    filepath: Union[str, Path],
    save_metadata: bool = False,
    api_keys: Optional[Union[str, Dict[str, str]]] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save a PDF file of a paper.

    Args:
        paper_metadata: A dictionary with the paper metadata. Must contain the `doi` key.
        filepath: Path to the PDF file to be saved (with or without suffix).
        save_metadata: A boolean indicating whether to save paper metadata as a separate json.
        api_keys: Either a dictionary containing API keys (if already loaded) or a string (path to API keys file).
                  If None, will try to load from `.env` file and if unsuccessful, skip API-based fallbacks.
        preferred_type: Preferred file type to download, 'pdf', 'xml', or 'both'.
            Defaults to 'pdf'. Use 'both' to download PDF and XML when available.
    Returns:
        A dict summary: {success: bool, method: str|None, filetype: 'pdf'|'xml'|None}
    """
    if preferred_type == "both":
        return save_file_and_xml(
            paper_metadata=paper_metadata,
            filepath=filepath,
            save_metadata=save_metadata,
            api_keys=api_keys,
            mail=mail,
        )
    if preferred_type not in ("pdf", "xml"):
        raise ValueError("preferred_type must be one of 'pdf', 'xml', or 'both'.")

    if not isinstance(paper_metadata, Dict):
        raise TypeError(f"paper_metadata must be a dict, not {type(paper_metadata)}.")
    if "doi" not in paper_metadata.keys():
        raise KeyError("paper_metadata must contain the key 'doi'.")
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"filepath must be a string or Path, not {type(filepath)}.")

    output_path = Path(filepath)

    if not output_path.parent.exists():
        raise ValueError(f"The folder: {output_path.parent} seems to not exist.")

    # load API keys from file if not already loaded via in save_file_from_dump (dict)
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)
    doi = paper_metadata["doi"]
    url = f"https://doi.org/{doi}"
    user_agent = {"User-Agent": "paperscraper/1.0 (+https)"}
    success = False
    used_method: Optional[str] = None
    used_filetype: Optional[str] = None
    soup = None
    final_url = None

    # ChemRxiv HTML pages are often Cloudflare-blocked; use the Open Engage API.
    if "chemrxiv" in doi.lower():
        item = _get_chemrxiv_item(doi, user_agent)
        if item:
            if save_metadata:
                _write_metadata(_chemrxiv_metadata_from_item(item, doi), output_path)
            pdf_url = _chemrxiv_pdf_url(item)
            if pdf_url:
                try:
                    if download_pdf_to_path(pdf_url, output_path, user_agent):
                        return {
                            "success": True,
                            "method": "chemrxiv",
                            "filetype": "pdf",
                        }
                    logger.warning(
                        f"ChemRxiv Open Engage PDF endpoint did not return a PDF: {pdf_url}"
                    )
                except Exception as e:
                    logger.warning(
                        f"ChemRxiv Open Engage PDF download failed for {doi}: {e}"
                    )
            else:
                logger.warning(f"ChemRxiv API response missing PDF URL for {doi}")

    # Fast path: OpenAlex-batched OA PDF URL (from prefetch_openalex_metadata).
    if preferred_type == "pdf":
        oa_meta = _OPENALEX_META_CACHE.get(_normalize_doi(doi)) or _OPENALEX_META_CACHE.get(
            _normalize_doi(doi).lower()
        )
        pdf_url = (oa_meta or {}).get("pdf_url")
        if pdf_url:
            try:
                if download_pdf_to_path(pdf_url, output_path, user_agent):
                    return {
                        "success": True,
                        "method": "openalex_cache",
                        "filetype": "pdf",
                    }
            except Exception as exc:
                logger.info(f"OpenAlex cached PDF URL failed for {doi}: {exc}")

    try:
        response = get_session().get(url)
        soup = BeautifulSoup(response.text, features="lxml")
        response.raise_for_status()
        final_url = response.url
        soup = BeautifulSoup(response.text, features="lxml")
        meta_pdf = soup.find("meta", {"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            pdf_url = meta_pdf.get("content")
            pdf_response = get_session().get(pdf_url)
            pdf_response.raise_for_status()

            if pdf_response.content[:4] == b"%PDF":
                with open(output_path.with_suffix(".pdf"), "wb+") as f:
                    f.write(pdf_response.content)
                success = True
                used_method = "direct"
                used_filetype = "pdf"
            else:
                logger.warning(
                    f"The file from {pdf_url} does not appear to be a valid PDF."
                )

    except Exception as e:
        logger.warning(f"Could not download from: {final_url} - {e}. Trying fallbacks.")

    if success:
        if not save_metadata:
            return {"success": True, "method": used_method, "filetype": used_filetype}

        metadata = {}
        # Extract title
        title_tag = soup.find("meta", {"name": "citation_title"})
        metadata["title"] = title_tag.get("content") if title_tag else "Title not found"

        # Extract authors
        authors = []
        for author_tag in soup.find_all("meta", {"name": "citation_author"}):
            if author_tag.get("content"):
                authors.append(author_tag["content"])
        metadata["authors"] = authors if authors else ["Author information not found"]

        # Extract abstract
        domain = tldextract.extract(url).domain
        abstract_keys = ABSTRACT_ATTRIBUTE.get(domain, DEFAULT_ATTRIBUTES)

        for key in abstract_keys:
            abstract_tag = soup.find("meta", {"name": key})
            if abstract_tag:
                raw_abstract = BeautifulSoup(
                    abstract_tag.get("content", "None"), "html.parser"
                ).get_text("\n")
                if raw_abstract.strip().startswith("Abstract"):
                    raw_abstract = raw_abstract.strip()[8:]
                metadata["abstract"] = raw_abstract.strip()
                break

        if "abstract" not in metadata.keys():
            metadata["abstract"] = "Abstract not found"
            logger.warning(f"Could not find abstract for {url}")
        elif metadata["abstract"].endswith("..."):
            logger.warning(f"Abstract truncated from {url}")

        # Save metadata to JSON
        try:
            with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata to {str(output_path)}: {e}")
        return {"success": True, "method": used_method, "filetype": used_filetype}

    # If primary download failed, try fallbacks
    logger.info(f"Primary download failed for {doi}. Attempting fallbacks.")

    # Order of fallbacks tries to maximize OA coverage first
    if mail and FALLBACKS["unpaywall"](doi, output_path, mail, final_url):
        return {"success": True, "method": "unpaywall", "filetype": "pdf"}

    if FALLBACKS["europepmc"](doi, output_path, preferred_type=preferred_type):
        filetype = "pdf" if output_path.with_suffix(".pdf").exists() else "xml"
        return {"success": True, "method": "europepmc", "filetype": filetype}

    if FALLBACKS["bioc_pmc"](doi, output_path, mail or "your_email@example.com"):
        return {"success": True, "method": "bioc_pmc", "filetype": "xml"}

    # bioRxiv / medRxiv share the 10.1101 DOI prefix. Prefer explicit name/URL matches.
    doi_l = doi.lower()
    final_l = (final_url or "").lower()
    has_aws = bool(
        api_keys.get("AWS_ACCESS_KEY_ID") and api_keys.get("AWS_SECRET_ACCESS_KEY")
    )
    is_medrxiv = "medrxiv" in doi_l or "medrxiv" in final_l
    is_biorxiv = "biorxiv" in doi_l or "biorxiv" in final_l
    is_1101 = doi_l.startswith("10.1101/")

    if has_aws and is_medrxiv and "medrxiv_s3" in FALLBACKS:
        if FALLBACKS["medrxiv_s3"](doi, output_path, api_keys):
            return {"success": True, "method": "medrxiv_s3", "filetype": "pdf"}

    if has_aws and (is_biorxiv or (is_1101 and not is_medrxiv)):
        if FALLBACKS["s3"](doi, output_path, api_keys):
            return {"success": True, "method": "biorxiv_s3", "filetype": "pdf"}
        # Ambiguous 10.1101 (no explicit bioRxiv signal): also try medRxiv S3.
        if (
            is_1101
            and not is_biorxiv
            and "medrxiv_s3" in FALLBACKS
            and FALLBACKS["medrxiv_s3"](doi, output_path, api_keys)
        ):
            return {"success": True, "method": "medrxiv_s3", "filetype": "pdf"}

    if "plos" in doi_l:
        if FALLBACKS["plos"](doi, output_path):
            return {"success": True, "method": "plos", "filetype": "pdf"}

    if "elife" in doi.lower():
        if FALLBACKS["elife"](doi, output_path):
            return {"success": True, "method": "elife", "filetype": "xml"}

    # Non-publisher OA aggregators
    if "openalex" in FALLBACKS and FALLBACKS["openalex"](
        doi, output_path, api_keys=api_keys, mail=mail
    ):
        return {"success": True, "method": "openalex", "filetype": "pdf"}

    if "crossref" in FALLBACKS and FALLBACKS["crossref"](
        doi, output_path, mail or "your_email@example.com"
    ):
        return {"success": True, "method": "crossref", "filetype": "pdf"}

    if "doaj" in FALLBACKS and FALLBACKS["doaj"](doi, output_path):
        return {"success": True, "method": "doaj", "filetype": "pdf"}

    if "arxiv" in FALLBACKS and FALLBACKS["arxiv"](doi, output_path):
        return {"success": True, "method": "arxiv", "filetype": "pdf"}

    # Publisher TDM / Open Access APIs — only when Crossref/domain says that publisher owns the DOI
    if api_keys:
        has_springer_key = bool(
            api_keys.get("SPRINGER_OPEN_ACCESS_API") or api_keys.get("SPRINGER_API_KEY")
        )
        if (
            has_springer_key
            and FALLBACKS.get("springer")
            and _publisher_api_allowed(
                doi, "springer", mail=mail, final_url=final_url, api_keys=api_keys
            )
        ):
            if FALLBACKS["springer"](paper_metadata, output_path, api_keys):
                return {"success": True, "method": "springer", "filetype": "pdf"}
        if api_keys.get("WILEY_TDM_API_TOKEN") and _publisher_api_allowed(
            doi, "wiley", mail=mail, final_url=final_url, api_keys=api_keys
        ):
            if FALLBACKS["wiley"](paper_metadata, output_path, api_keys):
                return {"success": True, "method": "wiley", "filetype": "pdf"}
        if api_keys.get("ELSEVIER_TDM_API_KEY") and _publisher_api_allowed(
            doi, "elsevier", mail=mail, final_url=final_url, api_keys=api_keys
        ):
            if FALLBACKS["elsevier"](
                paper_metadata, output_path, api_keys, preferred_type=preferred_type
            ):
                return {
                    "success": True,
                    "method": "elsevier",
                    "filetype": preferred_type,
                }

    logger.warning(f"All download attempts failed for {doi}.")
    # --- Replace the previous "save abstract as .txt when all attempts failed" block with this ---
    abstract_text = None

    # 1) Try Europe PMC first (prefer AbstractText)
    try:
        abstract_text = _get_abstract_europepmc(doi)
    except Exception:
        abstract_text = None

    # 2) If no abstract yet and pmid present, try PubMed Entrez
    if (
        not abstract_text
        and isinstance(paper_metadata, dict)
        and paper_metadata.get("pubmed_id")
    ):
        pmid = str(paper_metadata.get("pubmed_id"))
        abstract_text = _get_abstract_pubmed(pmid)

    # 3) If still no abstract, try Crossref for the DOI
    if not abstract_text:
        try:
            abstract_text = _get_abstract_crossref(doi, mail=mail)
        except Exception:
            abstract_text = None

    if not abstract_text:
        logger.warning(f"Could not retrieve abstract for {doi}.")
        return {"success": False, "method": None, "filetype": None}
    else:
        try:
            with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(abstract_text)
            logger.info(f"Saved abstract to {str(output_path.with_suffix('.txt'))}.")
        except Exception as e:
            logger.error(f"Failed to save abstract to {str(output_path)}: {e}")
        # Abstract saved, but not a full text
        return {"success": False, "method": "abstract", "filetype": "txt"}


def _try_download_xml(
    paper_metadata: Dict[str, Any],
    output_path: Union[str, Path],
    api_keys: Optional[Dict[str, str]] = None,
    mail: Optional[str] = None,
) -> Dict[str, Any]:
    """Try XML-only sources for a paper. Skips work if `.xml` already exists."""
    output_path = Path(output_path)
    xml_path = output_path.with_suffix(".xml")
    if xml_path.exists():
        return {"success": True, "method": "existing", "filetype": "xml"}

    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    doi = paper_metadata["doi"]
    contact = mail or "your_email@example.com"

    if FALLBACKS["europepmc"](doi, output_path, preferred_type="xml"):
        return {"success": True, "method": "europepmc", "filetype": "xml"}
    if FALLBACKS["bioc_pmc"](doi, output_path, contact):
        return {"success": True, "method": "bioc_pmc", "filetype": "xml"}
    if "elife" in doi.lower() and FALLBACKS["elife"](doi, output_path):
        return {"success": True, "method": "elife", "filetype": "xml"}
    if api_keys.get("ELSEVIER_TDM_API_KEY") and _publisher_api_allowed(
        doi, "elsevier", mail=mail, api_keys=api_keys
    ):
        if FALLBACKS["elsevier"](
            paper_metadata, output_path, api_keys, preferred_type="xml"
        ):
            return {"success": True, "method": "elsevier", "filetype": "xml"}

    return {"success": False, "method": None, "filetype": None}


def save_file_and_xml(
    paper_metadata: Dict[str, Any],
    filepath: Union[str, Path],
    save_metadata: bool = False,
    api_keys: Optional[Union[str, Dict[str, str]]] = None,
    mail: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download both PDF and XML when available.

    Runs PDF retrieval first, then XML-only sources for any missing XML. If PDF is
    still missing after a successful XML hit (e.g. BioC-PMC), retries Elsevier PDF.
    """
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    output_path = Path(filepath)
    pdf_path = output_path.with_suffix(".pdf")
    xml_path = output_path.with_suffix(".xml")

    pdf_result: Dict[str, Any] = {
        "success": False,
        "method": None,
        "filetype": None,
    }
    xml_result: Dict[str, Any] = {
        "success": False,
        "method": None,
        "filetype": None,
    }

    if pdf_path.exists():
        pdf_result = {"success": True, "method": "existing", "filetype": "pdf"}
    else:
        # Prefer PDF sources; may still land an XML via PMC fallbacks.
        pdf_result = save_file(
            paper_metadata,
            filepath=output_path,
            save_metadata=save_metadata,
            api_keys=api_keys,
            preferred_type="pdf",
            mail=mail,
        )
        # If an XML-only source succeeded, treat that as the XML result.
        if pdf_result.get("filetype") == "xml" and xml_path.exists():
            xml_result = {
                "success": True,
                "method": pdf_result.get("method"),
                "filetype": "xml",
            }
            pdf_result = {"success": False, "method": None, "filetype": None}

    if xml_path.exists() and not xml_result.get("success"):
        xml_result = {"success": True, "method": "existing", "filetype": "xml"}
    elif not xml_path.exists():
        xml_result = _try_download_xml(
            paper_metadata, output_path, api_keys=api_keys, mail=mail
        )

    # If we only got XML so far, try Elsevier PDF explicitly (Elsevier DOIs only).
    if (
        not pdf_path.exists()
        and api_keys.get("ELSEVIER_TDM_API_KEY")
        and _publisher_api_allowed(
            paper_metadata["doi"], "elsevier", mail=mail, api_keys=api_keys
        )
    ):
        if FALLBACKS["elsevier"](
            paper_metadata, output_path, api_keys, preferred_type="pdf"
        ):
            pdf_result = {"success": True, "method": "elsevier", "filetype": "pdf"}

    has_pdf = pdf_path.exists()
    has_xml = xml_path.exists()
    parts = []
    methods = []
    if has_pdf:
        parts.append("pdf")
        methods.append(f"pdf:{pdf_result.get('method') or 'unknown'}")
    if has_xml:
        parts.append("xml")
        methods.append(f"xml:{xml_result.get('method') or 'unknown'}")

    filetype = "+".join(parts) if parts else (
        "txt" if pdf_result.get("filetype") == "txt" else None
    )
    return {
        "success": has_pdf or has_xml,
        "method": ";".join(methods) if methods else pdf_result.get("method"),
        "filetype": filetype,
        "pdf": pdf_result if has_pdf else {"success": False, "method": None, "filetype": None},
        "xml": xml_result if has_xml else {"success": False, "method": None, "filetype": None},
    }


def save_file_from_dump(
    dump_path: str,
    pdf_path: str,
    key_to_save: str = "doi",
    save_metadata: bool = False,
    api_keys: Optional[str] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
    max_workers: int = 16,
) -> Dict[str, Any]:
    """
    Receives a path to a paper metadata dump and saves the PDF/XML files of
    each paper.

    Supported dump formats:
        - ``.jsonl`` paperscraper dumps (one JSON object per line)
        - Web of Science tab-delimited UTF-8 exports (typically ``savedrecs.txt``)

    Args:
        dump_path: Path to a ``.jsonl`` dump or a WoS TBA ``.txt``/``.tsv`` export.
        pdf_path: Path to a folder where the files will be stored.
        key_to_save: Key in the paper metadata to use as filename.
            Has to be `doi`, `title`, or `date`. Defaults to `doi`.
        save_metadata: A boolean indicating whether to save paper metadata as a separate json.
        api_keys: Path to a file with API keys. If None, API-based fallbacks will be skipped.
        preferred_type: Preferred file type to download, 'pdf', 'xml', or 'both'.
            Defaults to 'pdf'. Use 'both' to save PDF and XML when both are available.
        mail: Optional email address to use for Unpaywall API requests.
        max_workers: Concurrent paper downloads (rate-limited per API). Defaults to 16.
    Returns:
        A dict containing per-DOI results and counts. Also writes fallback_stats.json to pdf_path.
    """

    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path must be a string, not {type(dump_path)}.")
    lower = dump_path.lower()
    if not (
        lower.endswith(".jsonl")
        or lower.endswith(".txt")
        or lower.endswith(".tsv")
        or lower.endswith(".csv")
    ):
        raise ValueError(
            "Please provide a dump_path with .jsonl or Web of Science "
            "tab-delimited (.txt/.tsv) extension."
        )

    if not isinstance(pdf_path, str):
        raise TypeError(f"pdf_path must be a string, not {type(pdf_path)}.")

    if not isinstance(key_to_save, str):
        raise TypeError(f"key_to_save must be a string, not {type(key_to_save)}.")
    if key_to_save not in ("doi", "title", "date"):
        raise ValueError(
            f"key_to_save must be one of 'doi', 'title', or 'date', not {key_to_save!r}."
        )
    if preferred_type not in ["pdf", "xml", "both"]:
        raise ValueError("preferred_type must be one of 'pdf', 'xml', or 'both'.")
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError(f"max_workers must be a positive int, got {max_workers!r}.")

    papers = load_papers_dump(dump_path)

    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    # One OpenAlex batch up front for publisher/OA gating (no Crossref).
    dois = [p.get("doi") for p in papers if p.get("doi")]
    if dois:
        prefetch_openalex_metadata(dois, api_keys=api_keys, mail=mail)

    os.makedirs(pdf_path, exist_ok=True)

    results_by_doi: Dict[str, Dict[str, Any]] = {}
    counts_by_method: Dict[str, int] = {}
    lock = threading.Lock()

    def _record(doi: str, result: Dict[str, Any]) -> None:
        with lock:
            results_by_doi[doi] = result
            if result and result.get("method"):
                if result.get("success"):
                    counts_by_method[result["method"]] = (
                        counts_by_method.get(result["method"], 0) + 1
                    )
                elif result.get("method") == "abstract":
                    counts_by_method["abstract_only"] = (
                        counts_by_method.get("abstract_only", 0) + 1
                    )
                else:
                    counts_by_method["failed"] = counts_by_method.get("failed", 0) + 1
            else:
                counts_by_method["failed"] = counts_by_method.get("failed", 0) + 1

    def _process_one(paper: Dict[str, Any]) -> None:
        if "doi" not in paper.keys() or paper["doi"] is None:
            logger.warning("Skipping paper since no DOI available.")
            return
        if key_to_save not in paper.keys() or paper[key_to_save] is None:
            logger.warning(
                f"Skipping paper {paper.get('doi')} since key {key_to_save!r} is missing."
            )
            return
        filename = str(paper[key_to_save]).replace("/", "_")
        # Soft-sanitize Windows/POSIX-hostile characters from titles etc.
        for bad in (":", "*", "?", '"', "<", ">", "|", "\\"):
            filename = filename.replace(bad, "_")
        # Avoid overly long filenames from long titles.
        if len(filename) > 180:
            filename = filename[:180].rstrip(" ._")
        pdf_file = Path(os.path.join(pdf_path, f"{filename}.pdf"))
        xml_file = pdf_file.with_suffix(".xml")
        want_pdf = preferred_type in ("pdf", "both")
        want_xml = preferred_type in ("xml", "both")
        doi = paper["doi"]

        if preferred_type == "both":
            if pdf_file.exists() and xml_file.exists():
                _record(
                    doi,
                    {
                        "success": True,
                        "method": "existing",
                        "filetype": "pdf+xml",
                        "pdf": {
                            "success": True,
                            "method": "existing",
                            "filetype": "pdf",
                        },
                        "xml": {
                            "success": True,
                            "method": "existing",
                            "filetype": "xml",
                        },
                    },
                )
                return
        else:
            if want_pdf and pdf_file.exists():
                _record(
                    doi,
                    {"success": True, "method": "existing", "filetype": "pdf"},
                )
                return
            if want_xml and xml_file.exists() and preferred_type == "xml":
                _record(
                    doi,
                    {"success": True, "method": "existing", "filetype": "xml"},
                )
                return

        result = save_file(
            paper,
            str(pdf_file),
            save_metadata=save_metadata,
            api_keys=api_keys,
            preferred_type=preferred_type,
            mail=mail,
        )
        _record(doi, result)

    workers = min(max_workers, max(len(papers), 1))
    logger.info(
        f"Downloading {len(papers)} papers with max_workers={workers} "
        f"(publisher APIs rate-limited)."
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one, paper) for paper in papers]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Processing papers"
        ):
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"Worker failed: {exc}")

    # Save stats to file in the target directory
    try:
        stats = {
            "total": len(papers),
            "counts": counts_by_method,
            "by_doi": results_by_doi,
        }
        stats_path = Path(pdf_path) / "fallback_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved fallback stats to {stats_path}")

        import csv

        report_path = Path(pdf_path) / "paper_download_report.csv"
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "doi",
                    "title",
                    "journal",
                    "year",
                    "success",
                    "filetype",
                    "source",
                    "wos_oa",
                    "openalex_publisher",
                    "openalex_is_oa",
                ],
            )
            writer.writeheader()
            for paper in papers:
                doi = paper.get("doi")
                if not doi:
                    continue
                result = results_by_doi.get(doi, {})
                oa_meta = _OPENALEX_META_CACHE.get(_normalize_doi(doi)) or {}
                writer.writerow(
                    {
                        "doi": doi,
                        "title": paper.get("title") or paper.get("Title") or "",
                        "journal": paper.get("journal") or "",
                        "year": paper.get("date") or "",
                        "success": bool(result.get("success")),
                        "filetype": result.get("filetype") or "",
                        "source": result.get("method") or "",
                        "wos_oa": paper.get("oa") or "",
                        "openalex_publisher": oa_meta.get("publisher") or "",
                        "openalex_is_oa": oa_meta.get("is_oa"),
                    }
                )
        logger.info(f"Saved per-paper report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to write fallback stats: {e}")

    return {"counts": counts_by_method, "by_doi": results_by_doi}


# Publisher gating via OpenAlex (batched), not Crossref.
_OPENALEX_META_CACHE: Dict[str, Dict[str, Any]] = {}

_PUBLISHER_API_RULES: Dict[str, Dict[str, Any]] = {
    "wiley": {
        "publisher_substrings": ("wiley", "blackwell", "john wiley"),
    },
    "springer": {
        "publisher_substrings": (
            "springer",
            "springer nature",
            "nature publishing",
            "nature portfolio",
            "biomed central",
            "bmc",
        ),
    },
    "elsevier": {
        "publisher_substrings": ("elsevier", "cell press", "the lancet", "lancet"),
    },
}


def _normalize_doi(doi: str) -> str:
    d = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.lower().startswith(prefix):
            d = d[len(prefix) :]
            break
    return d.strip()


def _openalex_headers(mail: Optional[str] = None) -> Dict[str, str]:
    contact = mail or "paperscraper@example.com"
    return {
        "User-Agent": f"paperscraper/1.0 (mailto:{contact})",
        "Accept": "application/json",
    }


def prefetch_openalex_metadata(
    dois: list,
    api_keys: Optional[Dict[str, str]] = None,
    mail: Optional[str] = None,
    chunk_size: int = 50,
    timeout: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch-fetch OpenAlex work metadata (publisher + OA) for many DOIs.

    Uses filter=doi:https://doi.org/A|https://doi.org/B with OPENALEX_API_KEY when set.
    Results are stored in the module cache and returned as {doi: meta}.
    """
    api_keys = api_keys or {}
    api_key = api_keys.get("OPENALEX_API_KEY")
    unique = []
    seen = set()
    for doi in dois:
        norm = _normalize_doi(doi)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)

    headers = _openalex_headers(mail)
    for i in range(0, len(unique), chunk_size):
        chunk = unique[i : i + chunk_size]
        filter_dois = "|".join(f"https://doi.org/{d}" for d in chunk)
        params: Dict[str, Any] = {
            "filter": f"doi:{filter_dois}",
            "per_page": max(len(chunk), 1),
            "select": "doi,primary_location,open_access,best_oa_location",
        }
        if api_key:
            params["api_key"] = api_key
        elif mail:
            params["mailto"] = mail

        try:
            resp = get_session().get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except Exception as exc:
            logger.error(f"OpenAlex batch publisher lookup failed: {exc}")
            continue

        for work in results:
            raw_doi = _normalize_doi(work.get("doi") or "")
            if not raw_doi:
                continue
            source = ((work.get("primary_location") or {}).get("source")) or {}
            publisher = (
                source.get("host_organization_name")
                or source.get("display_name")
                or source.get("host_organization")
            )
            oa = (work.get("open_access") or {})
            best = work.get("best_oa_location") or {}
            meta = {
                "publisher": str(publisher) if publisher else None,
                "is_oa": bool(oa.get("is_oa")) if oa else None,
                "oa_status": oa.get("oa_status"),
                "pdf_url": best.get("pdf_url") or (work.get("primary_location") or {}).get("pdf_url"),
            }
            _OPENALEX_META_CACHE[raw_doi] = meta
            _OPENALEX_META_CACHE[raw_doi.lower()] = meta

        # Mark misses in this chunk so we don't refetch endlessly.
        for d in chunk:
            key = d.lower()
            if key not in _OPENALEX_META_CACHE and d not in _OPENALEX_META_CACHE:
                _OPENALEX_META_CACHE[d] = {
                    "publisher": None,
                    "is_oa": None,
                    "oa_status": "not_in_openalex",
                    "pdf_url": None,
                }
                _OPENALEX_META_CACHE[key] = _OPENALEX_META_CACHE[d]

        logger.info(
            f"OpenAlex batch: fetched publishers for DOIs {i + 1}-{min(i + chunk_size, len(unique))} / {len(unique)}"
        )

    return {
        d: _OPENALEX_META_CACHE.get(d) or _OPENALEX_META_CACHE.get(d.lower()) or {}
        for d in unique
    }


def _get_openalex_publisher(
    doi: str,
    api_keys: Optional[Dict[str, str]] = None,
    mail: Optional[str] = None,
) -> Optional[str]:
    """Return cached OpenAlex host_organization_name for a DOI, prefetching if needed."""
    norm = _normalize_doi(doi)
    meta = _OPENALEX_META_CACHE.get(norm) or _OPENALEX_META_CACHE.get(norm.lower())
    if meta is None:
        prefetch_openalex_metadata([norm], api_keys=api_keys, mail=mail)
        meta = _OPENALEX_META_CACHE.get(norm) or _OPENALEX_META_CACHE.get(norm.lower()) or {}
    return meta.get("publisher")


def _get_redirect_domain(doi: str, timeout: int = 10) -> Optional[str]:
    """
    Resolve https://doi.org/{doi} and return the extracted domain (e.g. 'wiley') or None on failure.
    """
    try:
        resp = get_session().get(
            f"https://doi.org/{doi}", timeout=timeout, allow_redirects=True
        )
        resp.raise_for_status()
        return tldextract.extract(resp.url).domain or None
    except Exception:
        return None


def _publisher_api_allowed(
    doi: str,
    publisher_key: str,
    mail: Optional[str] = None,
    final_url: Optional[str] = None,
    timeout: int = 15,
    api_keys: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Return True only if OpenAlex says this DOI belongs to the given publisher API.

    Publisher comes from a batched OpenAlex lookup (host_organization_name).
    """
    del final_url, timeout  # kept for call-site compatibility
    rules = _PUBLISHER_API_RULES.get(publisher_key)
    if not rules:
        return False

    publisher = (_get_openalex_publisher(doi, api_keys=api_keys, mail=mail) or "").lower()
    if not publisher:
        return False
    return any(s in publisher for s in rules.get("publisher_substrings", ()))


def _crossref_publisher_is_wiley(
    doi: str, timeout: int = 10, mail: Optional[str] = None
) -> bool:
    """Backward-compatible wrapper; now uses OpenAlex publisher matching."""
    del timeout
    return _publisher_api_allowed(doi, "wiley", mail=mail)


def _wiley_allowed(
    doi: str,
    final_url: Optional[str] = None,
    timeout: int = 10,
    mail: Optional[str] = None,
) -> bool:
    """Return True if OpenAlex publisher indicates Wiley."""
    return _publisher_api_allowed(
        doi, "wiley", mail=mail, final_url=final_url, timeout=timeout
    )


def debug_save_file(
    paper_metadata: Dict[str, Any],
    filepath: Union[str, Path],
    api_keys: Optional[Union[str, Dict[str, str]]] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
    save_first_only: bool = True,
) -> Dict[str, Any]:
    """
    Debug version that attempts the direct method and all fallbacks independently.
    Writes results per-fallback to distinct files with ".{fallback}" suffix to avoid clobbering.

    Returns a dict with keys: direct, successes (list), results (per-fallback bool), first_saved (fallback name or None).
    """
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    doi = paper_metadata["doi"]
    base_output = Path(filepath)
    successes = []
    per = {}

    # Use a unique path for the initial direct check so save_file doesn't
    # already save a fallback to the main output and interfere with later attempts.
    direct_check_path = Path(str(base_output) + ".direct_check")
    direct_res = save_file(
        paper_metadata,
        direct_check_path,
        save_metadata=False,
        api_keys=api_keys,
        preferred_type=preferred_type,
        mail=mail,
    )

    # Only treat "direct" as successful if the returned method is actually "direct"
    is_direct = bool(direct_res.get("success") and direct_res.get("method") == "direct")
    if is_direct:
        successes.append("direct")
    per["direct"] = is_direct
    first_saved = "direct" if per["direct"] else None

    # Build a deterministic list of fallbacks
    order = [
        "unpaywall",
        "europepmc",
        "bioc_pmc",
        "plos",
        "elife",
        "openalex",
        "crossref",
        "doaj",
        "arxiv",
        "springer",
        "wiley",
        "elsevier",
    ]

    def _attempt(name: str) -> bool:
        # derive unique output stem for debug
        out = Path(str(base_output) + f".{name}")
        try:
            if name == "unpaywall" and mail:
                return FALLBACKS[name](doi, out, mail, None)
            if name == "europepmc":
                return FALLBACKS[name](doi, out, preferred_type=preferred_type)
            if name in ("doaj", "arxiv"):
                return FALLBACKS[name](doi, out)
            if name == "openalex":
                return FALLBACKS[name](doi, out, api_keys=api_keys, mail=mail)
            if name == "crossref":
                return FALLBACKS[name](doi, out, mail or "your_email@example.com")
            if name in ("s3", "medrxiv_s3"):
                if api_keys.get("AWS_ACCESS_KEY_ID") and api_keys.get(
                    "AWS_SECRET_ACCESS_KEY"
                ):
                    return FALLBACKS[name](doi, out, api_keys)
                return False
            if name in ("plos", "elife"):
                return FALLBACKS[name](doi, out)
            if name in ("wiley", "springer"):
                if name == "wiley":
                    if not api_keys.get("WILEY_TDM_API_TOKEN"):
                        return False
                    try:
                        if not _publisher_api_allowed(
                            doi, "wiley", mail=mail, api_keys=api_keys
                        ):
                            return False
                    except Exception:
                        return False
                if name == "springer":
                    if not (
                        api_keys.get("SPRINGER_OPEN_ACCESS_API")
                        or api_keys.get("SPRINGER_API_KEY")
                    ):
                        return False
                    try:
                        if not _publisher_api_allowed(
                            doi, "springer", mail=mail, api_keys=api_keys
                        ):
                            return False
                    except Exception:
                        return False
                return FALLBACKS[name](paper_metadata, out, api_keys)
            if name == "elsevier":
                if not api_keys.get("ELSEVIER_TDM_API_KEY"):
                    return False
                try:
                    if not _publisher_api_allowed(
                        doi, "elsevier", mail=mail, api_keys=api_keys
                    ):
                        return False
                except Exception:
                    return False
                return FALLBACKS[name](
                    paper_metadata, out, api_keys, preferred_type=preferred_type
                )
        except Exception:
            return False
        return False

    for fb in order:
        if fb not in FALLBACKS:
            per[fb] = False
            continue
        ok = _attempt(fb)
        per[fb] = ok
        if ok:
            successes.append(fb)
            if not first_saved:
                first_saved = fb
            if save_first_only:
                # stop after the first saved to limit writes
                break

    return {
        "direct": per.get("direct", False),
        "results": per,
        "successes": successes,
        "first_saved": first_saved,
    }


# python
def debug_save_file_from_dump(
    dump_path: str,
    pdf_path: str,
    api_keys: Optional[str] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
    save_first_only: bool = True,
    save_interval: int = 10,
) -> Dict[str, Any]:
    """
    Debug variant for batch processing that tests all fallbacks per paper and records which work.
    Writes a debug_fallback_stats.json with detailed per-DOI outcomes.
    Saves intermediate stats every `save_interval` papers so partial results are available.
    """
    papers = load_papers_dump(dump_path)
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    by_doi = {}
    counts: Dict[str, int] = {}

    pbar = tqdm(papers, total=len(papers), desc="Debug processing")

    def _write_debug_stats(
        target_dir: str, by_doi_obj: Dict[str, Any], counts_obj: Dict[str, int]
    ):
        try:
            stats_path = Path(target_dir) / "debug_fallback_stats.json"
            tmp_path = stats_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"by_doi": by_doi_obj, "counts": counts_obj},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            tmp_path.replace(stats_path)
            logger.info(f"Saved debug fallback stats to {stats_path}")
        except Exception as e:
            logger.error(f"Failed to write debug fallback stats: {e}")

    for i, paper in enumerate(pbar):
        if "doi" not in paper or not paper["doi"]:
            continue
        filename = paper["doi"].replace("/", "_")
        out = str(Path(os.path.join(pdf_path, f"{filename}.pdf")))
        res = debug_save_file(
            paper,
            out,
            api_keys=api_keys,
            preferred_type=preferred_type,
            mail=mail,
            save_first_only=save_first_only,
        )
        by_doi[paper["doi"]] = res
        # count successes per fallback
        for fb, ok in res.get("results", {}).items():
            if ok:
                counts[fb] = counts.get(fb, 0) + 1

        # periodically save partial stats so you can inspect mid-run
        if save_interval > 0 and ((i + 1) % save_interval == 0):
            _write_debug_stats(pdf_path, by_doi, counts)

    # write final debug stats
    try:
        _write_debug_stats(pdf_path, by_doi, counts)
    except Exception as e:
        logger.error(f"Failed to write final debug fallback stats: {e}")
    return {"by_doi": by_doi, "counts": counts}

# Backward-compatible aliases
save_pdf = save_file
save_pdf_and_xml = save_file_and_xml
save_pdf_from_dump = save_file_from_dump
debug_save_pdf = debug_save_file
debug_save_pdf_from_dump = debug_save_file_from_dump

