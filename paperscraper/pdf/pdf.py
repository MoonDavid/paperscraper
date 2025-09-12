"""Functionalities to scrape PDF files of publications."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
import tldextract
from bs4 import BeautifulSoup
from tqdm import tqdm

from ..utils import load_jsonl
from .fallbacks import FALLBACKS
from .utils import load_api_keys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ABSTRACT_ATTRIBUTE = {
    "biorxiv": ["DC.Description"],
    "arxiv": ["citation_abstract"],
    "chemrxiv": ["citation_abstract"],
}
DEFAULT_ATTRIBUTES = ["citation_abstract", "description"]

# python
def _get_abstract_pubmed(pmid: str, timeout: int = 20) -> Optional[str]:
    """
    Query NCBI EFetch for PubMed and return the abstract text or None.
    Uses the XML retmode and extracts all AbstractText nodes.
    """
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        resp = requests.get(url, params=params, timeout=timeout)
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

def _get_abstract_crossref(doi: str, timeout: int = 20) -> Optional[str]:
    """
    Query Crossref works API and return the abstract (HTML cleaned) or None.
    """
    try:
        url = f"https://api.crossref.org/works/{doi}"
        resp = requests.get(url, timeout=timeout)
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
        resp = requests.get(url, params=params, timeout=timeout)
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

# --- Replace abstract retrieval section in save_pdf with the following block ---



def save_pdf(
    paper_metadata: Dict[str, Any],
    filepath: Union[str, Path],
    save_metadata: bool = False,
    api_keys: Optional[Union[str, Dict[str, str]]] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save a PDF file of a paper.

    Args:
        paper_metadata: A dictionary with the paper metadata. Must contain the `doi` key.
        filepath: Path to the PDF file to be saved (with or without suffix).
        save_metadata: A boolean indicating whether to save paper metadata as a separate json.
        api_keys: Either a dictionary containing API keys (if already loaded) or a string (path to API keys file).
                  If None, will try to load from `.env` file and if unsuccessful, skip API-based fallbacks.
        preferred_type: Preferred file type to download, 'pdf' or 'xml'. Defaults to 'pdf'.
    Returns:
        A dict summary: {success: bool, method: str|None, filetype: 'pdf'|'xml'|None}
    """
    if not isinstance(paper_metadata, Dict):
        raise TypeError(f"paper_metadata must be a dict, not {type(paper_metadata)}.")
    if "doi" not in paper_metadata.keys():
        raise KeyError("paper_metadata must contain the key 'doi'.")
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"filepath must be a string or Path, not {type(filepath)}.")

    output_path = Path(filepath)

    if not output_path.parent.exists():
        raise ValueError(f"The folder: {output_path.parent} seems to not exist.")

    # load API keys from file if not already loaded via in save_pdf_from_dump (dict)
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    doi = paper_metadata["doi"]
    url = f"https://doi.org/{doi}"
    success = False
    used_method: Optional[str] = None
    used_filetype: Optional[str] = None
    soup = None
    final_url = None

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        final_url = response.url
        soup = BeautifulSoup(response.text, features="lxml")
        meta_pdf = soup.find("meta", {"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            pdf_url = meta_pdf.get("content")
            pdf_response = requests.get(pdf_url, timeout=60)
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

    if FALLBACKS["europepmc"](doi, output_path):
        return {"success": True, "method": "europepmc", "filetype": "xml"}

    if FALLBACKS["bioc_pmc"](doi, output_path, mail):
        return {"success": True, "method": "bioc_pmc", "filetype": "xml"}

    if (
        "biorxiv" in doi.lower()
        and api_keys.get("AWS_ACCESS_KEY_ID")
        and api_keys.get("AWS_SECRET_ACCESS_KEY")
    ):
        if FALLBACKS["s3"](doi, output_path, api_keys):
            return {"success": True, "method": "biorxiv_s3", "filetype": "pdf"}

    if (
        "medrxiv" in doi.lower()
        and api_keys.get("AWS_ACCESS_KEY_ID")
        and api_keys.get("AWS_SECRET_ACCESS_KEY")
        and "medrxiv_s3" in FALLBACKS
    ):
        if FALLBACKS["medrxiv_s3"](doi, output_path, api_keys):
            return {"success": True, "method": "medrxiv_s3", "filetype": "pdf"}

    if "plos" in doi.lower():
        if FALLBACKS["plos"](doi, output_path):
            return {"success": True, "method": "plos", "filetype": "pdf"}

    if "elife" in doi.lower():
        if FALLBACKS["elife"](doi, output_path):
            return {"success": True, "method": "elife", "filetype": "xml"}

    # Non-publisher OA aggregators
    if "openalex" in FALLBACKS and FALLBACKS["openalex"](doi, output_path):
        return {"success": True, "method": "openalex", "filetype": "pdf"}

    if "crossref" in FALLBACKS and FALLBACKS["crossref"](doi, output_path, mail or "your_email@example.com"):
        return {"success": True, "method": "crossref", "filetype": "pdf"}

    if "doaj" in FALLBACKS and FALLBACKS["doaj"](doi, output_path):
        return {"success": True, "method": "doaj", "filetype": "pdf"}

    if "arxiv" in FALLBACKS and FALLBACKS["arxiv"](doi, output_path):
        return {"success": True, "method": "arxiv", "filetype": "pdf"}

    # Publisher TDM APIs
    if api_keys:
        if api_keys.get("SPRINGER_API_KEY") and FALLBACKS.get("springer"):
            if FALLBACKS["springer"](paper_metadata, output_path, api_keys):
                return {"success": True, "method": "springer", "filetype": "pdf"}
        if api_keys.get("WILEY_TDM_API_TOKEN"):
            if FALLBACKS["wiley"](
                paper_metadata, output_path, api_keys
            ):
                return {"success": True, "method": "wiley", "filetype": "pdf"}
        if api_keys.get("ELSEVIER_TDM_API_KEY"):
            if FALLBACKS["elsevier"](
                paper_metadata, output_path, api_keys, preferred_type=preferred_type
            ):
                return {"success": True, "method": "elsevier", "filetype": preferred_type}


    logger.warning(f"All download attempts failed for {doi}.")
    # --- Replace the previous "save abstract as .txt when all attempts failed" block with this ---
    abstract_text = None

    # 1) Try Europe PMC first (prefer AbstractText)
    try:
        abstract_text = _get_abstract_europepmc(doi)
    except Exception:
        abstract_text = None

    # 2) If no abstract yet and pmid present, try PubMed Entrez
    if not abstract_text and isinstance(paper_metadata, dict) and paper_metadata.get("pubmed_id"):
        pmid = str(paper_metadata.get("pubmed_id"))
        abstract_text = _get_abstract_pubmed(pmid)

    # 3) If still no abstract, try Crossref for the DOI
    if not abstract_text:
        try:
            abstract_text = _get_abstract_crossref(doi)
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


def save_pdf_from_dump(
    dump_path: str,
    pdf_path: str,
    key_to_save: str = "doi",
    save_metadata: bool = False,
    api_keys: Optional[str] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None
) -> Dict[str, Any]:
    """
    Receives a path to a `.jsonl` dump with paper metadata and saves the PDF files of
    each paper.

    Args:
        dump_path: Path to a `.jsonl` file with paper metadata, one paper per line.
        pdf_path: Path to a folder where the files will be stored.
        key_to_save: Key in the paper metadata to use as filename.
            Has to be `doi` or `title`. Defaults to `doi`.
        save_metadata: A boolean indicating whether to save paper metadata as a separate json.
        api_keys: Path to a file with API keys. If None, API-based fallbacks will be skipped.
        preferred_type: Preferred file type to download, 'pdf' or 'xml'. Defaults to 'pdf'.
        mail: Optional email address to use for Unpaywall API requests.
    Returns:
        A dict containing per-DOI results and counts. Also writes fallback_stats.json to pdf_path.
    """

    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path must be a string, not {type(dump_path)}.")
    if not dump_path.endswith(".jsonl"):
        raise ValueError("Please provide a dump_path with .jsonl extension.")

    if not isinstance(pdf_path, str):
        raise TypeError(f"pdf_path must be a string, not {type(pdf_path)}.")

    if not isinstance(key_to_save, str):
        raise TypeError(f"key_to_save must be a string, not {type(key_to_save)}.")
    if preferred_type not in ["pdf", "xml"]:
        raise ValueError("preferred_type must be one of 'pdf' or 'xml'.")

    papers = load_jsonl(dump_path)

    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    results_by_doi: Dict[str, Dict[str, Any]] = {}
    counts_by_method: Dict[str, int] = {}

    pbar = tqdm(papers, total=len(papers), desc="Processing")
    for i, paper in enumerate(pbar):
        pbar.set_description(f"Processing paper {i + 1}/{len(papers)}")

        if "doi" not in paper.keys() or paper["doi"] is None:
            logger.warning(f"Skipping paper since no DOI available.")
            continue
        filename = paper[key_to_save].replace("/", "_")
        pdf_file = Path(os.path.join(pdf_path, f"{filename}.pdf"))
        xml_file = pdf_file.with_suffix(".xml")
        if pdf_file.exists():
            logger.info(f"File {pdf_file} already exists. Skipping download.")
            results_by_doi[paper["doi"]] = {"success": True, "method": "existing", "filetype": "pdf"}
            counts_by_method["existing"] = counts_by_method.get("existing", 0) + 1
            continue
        if xml_file.exists():
            logger.info(f"File {xml_file} already exists. Skipping download.")
            results_by_doi[paper["doi"]] = {"success": True, "method": "existing", "filetype": "xml"}
            counts_by_method["existing"] = counts_by_method.get("existing", 0) + 1
            continue
        output_path = str(pdf_file)
        result = save_pdf(
            paper,
            output_path,
            save_metadata=save_metadata,
            api_keys=api_keys,
            preferred_type=preferred_type,
            mail=mail
        )
        doi = paper["doi"]
        results_by_doi[doi] = result
        if result and result.get("method"):
            if result.get("success"):
                counts_by_method[result["method"]] = counts_by_method.get(result["method"], 0) + 1
            else:
                # track abstract-only separately
                if result.get("method") == "abstract":
                    counts_by_method["abstract_only"] = counts_by_method.get("abstract_only", 0) + 1
        else:
            counts_by_method["failed"] = counts_by_method.get("failed", 0) + 1

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
    except Exception as e:
        logger.error(f"Failed to write fallback stats: {e}")

    return {"counts": counts_by_method, "by_doi": results_by_doi}


# Debug variants: try all fallbacks independently of order and record which work

def debug_save_pdf(
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

    # Try direct
    direct_path = str(base_output)
    direct_res = save_pdf(
        paper_metadata,
        direct_path,
        save_metadata=False,
        api_keys=api_keys,
        preferred_type=preferred_type,
        mail=mail,
    )
    if direct_res.get("success"):
        successes.append("direct")
    per["direct"] = bool(direct_res.get("success"))
    first_saved = "direct" if per["direct"] else None

    # Build a deterministic list of fallbacks
    order = [
        "unpaywall",
        "europepmc",
        "bioc_pmc",
        "s3",
        "medrxiv_s3",
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
            if name in ("europepmc", "doaj", "openalex", "arxiv"):
                return FALLBACKS[name](doi, out)
            if name == "crossref":
                return FALLBACKS[name](doi, out, mail or "your_email@example.com")
            if name in ("s3", "medrxiv_s3"):
                if api_keys.get("AWS_ACCESS_KEY_ID") and api_keys.get("AWS_SECRET_ACCESS_KEY"):
                    return FALLBACKS[name](doi, out, api_keys)
                return False
            if name in ("plos", "elife"):
                return FALLBACKS[name](doi, out)
            if name in ("wiley", "springer"):
                if name == "wiley" and not api_keys.get("WILEY_TDM_API_TOKEN"):
                    return False
                if name == "springer" and not api_keys.get("SPRINGER_API_KEY"):
                    return False
                return FALLBACKS[name](paper_metadata, out, api_keys)
            if name == "elsevier":
                if not api_keys.get("ELSEVIER_TDM_API_KEY"):
                    return False
                return FALLBACKS[name](paper_metadata, out, api_keys, preferred_type=preferred_type)
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

    return {"direct": per.get("direct", False), "results": per, "successes": successes, "first_saved": first_saved}


def debug_save_pdf_from_dump(
    dump_path: str,
    pdf_path: str,
    api_keys: Optional[str] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
    save_first_only: bool = True,
) -> Dict[str, Any]:
    """
    Debug variant for batch processing that tests all fallbacks per paper and records which work.
    Writes a debug_fallback_stats.json with detailed per-DOI outcomes.
    """
    papers = load_jsonl(dump_path)
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    by_doi = {}
    counts: Dict[str, int] = {}

    pbar = tqdm(papers, total=len(papers), desc="Debug processing")
    for i, paper in enumerate(pbar):
        if "doi" not in paper or not paper["doi"]:
            continue
        filename = paper["doi"].replace("/", "_")
        out = str(Path(os.path.join(pdf_path, f"{filename}.pdf")))
        res = debug_save_pdf(
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

    # write debug stats
    try:
        stats_path = Path(pdf_path) / "debug_fallback_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"by_doi": by_doi, "counts": counts}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved debug fallback stats to {stats_path}")
    except Exception as e:
        logger.error(f"Failed to write debug fallback stats: {e}")
    return {"by_doi": by_doi, "counts": counts}
