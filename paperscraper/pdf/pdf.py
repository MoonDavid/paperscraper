"""Functionalities to scrape PDF files of publications."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import time

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


def save_pdf(
    paper_metadata: Dict[str, Any],
    filepath: Union[str, Path],
    save_metadata: bool = False,
    api_keys: Optional[Union[str, Dict[str, str]]] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None,
) -> None:
    """
    Save a PDF (or XML) file of a paper, given its metadata.

    Args:
        paper_metadata: A dictionary with the paper metadata. Must contain the `doi` key.
        filepath: Path to the PDF file to be saved (with or without suffix).
        save_metadata: If True, also saves extracted metadata (title, authors, abstract) as JSON.
        api_keys: Either a dict containing API keys, a path to an API key file, or None.
        preferred_type: Preferred file type to download, 'pdf' or 'xml'. Defaults to 'pdf'.
        mail: Email address (recommended for Unpaywall API).
    """
    if not isinstance(paper_metadata, Dict):
        raise TypeError(f"paper_metadata must be a dict, not {type(paper_metadata)}.")
    if "doi" not in paper_metadata:
        raise KeyError("paper_metadata must contain the key 'doi'.")
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"filepath must be a string or Path, not {type(filepath)}.")

    output_path = Path(filepath)
    if not output_path.parent.exists():
        raise ValueError(f"The folder: {output_path.parent} does not exist.")

    # Load API keys if not already loaded
    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    doi = paper_metadata["doi"]
    url = f"https://doi.org/{doi}"
    success = False
    soup = None
    final_url = None

    # === Primary attempt: resolve DOI and fetch citation_pdf_url ===
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

            if pdf_response.content.startswith(b"%PDF"):
                with open(output_path.with_suffix(".pdf"), "wb+") as f:
                    f.write(pdf_response.content)
                success = True
            else:
                logger.warning(f"The file from {pdf_url} is not a valid PDF.")

    except Exception as e:
        logger.warning(f"Could not download from {final_url or url}: {e}. Trying fallbacks.")

    # === If successful, optionally save metadata ===
    if success:
        if save_metadata and soup:
            metadata = {}
            # Title
            title_tag = soup.find("meta", {"name": "citation_title"})
            metadata["title"] = title_tag.get("content") if title_tag else "Title not found"

            # Authors
            authors = [a["content"] for a in soup.find_all("meta", {"name": "citation_author"}) if a.get("content")]
            metadata["authors"] = authors if authors else ["Author information not found"]

            # Abstract
            domain = tldextract.extract(final_url or url).domain
            abstract_keys = ABSTRACT_ATTRIBUTE.get(domain, DEFAULT_ATTRIBUTES)
            metadata["abstract"] = "Abstract not found"

            for key in abstract_keys:
                abstract_tag = soup.find("meta", {"name": key})
                if abstract_tag:
                    raw_abstract = BeautifulSoup(
                        abstract_tag.get("content", "None"), "html.parser"
                    ).get_text(separator="\n")
                    if raw_abstract.strip().startswith("Abstract"):
                        raw_abstract = raw_abstract.strip()[8:]
                    metadata["abstract"] = raw_abstract.strip()
                    break

            if metadata["abstract"].endswith("..."):
                logger.warning(f"Abstract truncated from {url}")

            # Save metadata JSON
            try:
                with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"Failed to save metadata JSON for {doi}: {e}")
        return

    # === Fallbacks: Publisher-aware routing ===
    logger.info(f"Primary download failed for {doi}. Attempting fallbacks.")
    domain = tldextract.extract(final_url or url).domain.lower() if (final_url or url) else ""
    logger.info(f"Resolved domain for DOI {doi}: '{domain}' (from {final_url or url})")

    # Publisher-specific fallbacks first
    if "plos" in domain:
        if FALLBACKS["plos"](doi, output_path): return
    elif "elife" in domain:
        if FALLBACKS["elife"](doi, output_path): return
    elif "biorxiv" in domain:
        if api_keys.get("AWS_ACCESS_KEY_ID") and api_keys.get("AWS_SECRET_ACCESS_KEY"):
            if FALLBACKS["s3"](doi, output_path, api_keys): return
    elif "wiley" in domain and api_keys.get("WILEY_TDM_API_TOKEN"):
        if FALLBACKS["wiley"](paper_metadata, output_path, api_keys): return
    elif "elsevier" in domain and api_keys.get("ELSEVIER_TDM_API_KEY"):
        if FALLBACKS["elsevier"](paper_metadata, output_path, api_keys,
                                preferred_type=preferred_type): return

    # General-purpose open access fallbacks
    if mail and FALLBACKS["unpaywall"](doi, output_path, mail, final_url): return
    if FALLBACKS["europepmc"](doi, output_path): return
    if FALLBACKS["bioc_pmc"](doi, output_path, mail): return

    logger.warning(f"All download attempts failed for {doi}.")



def save_pdf_from_dump(
    dump_path: str,
    pdf_path: str,
    key_to_save: str = "doi",
    save_metadata: bool = False,
    api_keys: Optional[str] = None,
    preferred_type: str = "pdf",
    mail: Optional[str] = None
) -> None:
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
    """

    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path must be a string, not {type(dump_path)}.")
    if not dump_path.endswith(".jsonl"):
        raise ValueError("Please provide a dump_path with .jsonl extension.")

    if not isinstance(pdf_path, str):
        raise TypeError(f"pdf_path must be a string, not {type(pdf_path)}.")

    if not isinstance(key_to_save, str):
        raise TypeError(f"key_to_save must be a string, not {type(key_to_save)}.")
    if key_to_save not in ["doi", "title", "date"]:
        raise ValueError("key_to_save must be one of 'doi' or 'title'.")

    if preferred_type not in ["pdf", "xml"]:
        raise ValueError("preferred_type must be one of 'pdf' or 'xml'.")

    papers = load_jsonl(dump_path)

    if not isinstance(api_keys, dict):
        api_keys = load_api_keys(api_keys)

    pbar = tqdm(papers, total=len(papers), desc="Processing")
    for i, paper in enumerate(pbar):
        pbar.set_description(f"Processing paper {i + 1}/{len(papers)}")

        if "doi" not in paper.keys() or paper["doi"] is None:
            logger.warning(f"Skipping {paper['title']} since no DOI available.")
            continue
        filename = paper[key_to_save].replace("/", "_")
        pdf_file = Path(os.path.join(pdf_path, f"{filename}.pdf"))
        xml_file = pdf_file.with_suffix(".xml")
        if pdf_file.exists():
            logger.info(f"File {pdf_file} already exists. Skipping download.")
            continue
        if xml_file.exists():
            logger.info(f"File {xml_file} already exists. Skipping download.")
            continue
        output_path = str(pdf_file)
        save_pdf(
            paper,
            output_path,
            save_metadata=save_metadata,
            api_keys=api_keys,
            preferred_type=preferred_type,
            mail=mail
        )
