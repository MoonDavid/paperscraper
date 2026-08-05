import csv
import json
import logging
import sys
from importlib import resources
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Web of Science tab-delimited (UTF-8) field tags → paperscraper keys.
# See: https://images.webofknowledge.com/WOKRS535R111/help/WOK/hs_wos_fieldtags.html
WOS_TBA_FIELD_MAP = {
    "DI": "doi",
    "TI": "title",
    "AF": "authors",
    "AU": "authors_short",
    "AB": "abstract",
    "PY": "date",
    "SO": "journal",
    "PM": "pubmed_id",
    "UT": "wos_id",
    "DL": "doi_url",
    "DT": "document_type",
}


def get_server_dumps_dir() -> str:
    """Return the filesystem path to the bundled server_dumps directory."""
    return str(resources.files("paperscraper").joinpath("server_dumps"))


def dump_papers(papers: pd.DataFrame, filepath: str) -> None:
    """
    Receives a pd.DataFrame, one paper per row and dumps it into a .jsonl
    file with one paper per line.

    Args:
        papers (pd.DataFrame): A dataframe of paper metadata, one paper per row.
        filepath (str): Path to dump the papers, has to end with `.jsonl`.
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")
    if not filepath.endswith(".jsonl"):
        raise ValueError("Please provide a filepath with .jsonl extension")

    if isinstance(papers, List) and all([isinstance(p, Dict) for p in papers]):
        papers = pd.DataFrame(papers)
        logger.warning(
            "Preferably pass a pd.DataFrame, not a list of dictionaries. "
            "Passing a list is a legacy functionality that might become deprecated."
        )

    if not isinstance(papers, pd.DataFrame):
        raise TypeError(f"papers must be a pd.DataFrame, not {type(papers)}")

    paper_list = list(papers.T.to_dict().values())

    with open(filepath, "w") as f:
        for paper in paper_list:
            f.write(json.dumps(paper) + "\n")


def get_filename_from_query(query: List[str]) -> str:
    """Convert a keyword query into filenames to dump the paper.

    Args:
        query (list): List of string with keywords.

    Returns:
        str: Filename.
    """
    filename = "_".join([k if isinstance(k, str) else k[0] for k in query]) + ".jsonl"
    filename = filename.replace(" ", "").lower()
    return filename


def load_jsonl(filepath: str) -> List[Dict[str, str]]:
    """
    Load data from a `.jsonl` file, i.e., a file with one dictionary per line.

    Args:
        filepath (str): Path to `.jsonl` file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, one per paper.
    """

    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def _split_wos_authors(value: str) -> List[str]:
    """Split a Web of Science author field into a list of names."""
    if not value or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def is_wos_tba_file(filepath: str) -> bool:
    """
    Return True if ``filepath`` looks like a Web of Science tab-delimited export.

    Detection is based on a UTF-8 (optional BOM) header line containing the
    classic WoS tags ``PT``, ``TI``, and ``DI``, separated by tabs.
    """
    path = Path(filepath)
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().strip("\n\r")
    except (OSError, UnicodeDecodeError):
        return False
    if "\t" not in header:
        return False
    fields = {part.strip() for part in header.split("\t") if part.strip()}
    return {"PT", "TI", "DI"}.issubset(fields)


def load_wos_tba(
    filepath: str,
    *,
    require_doi: bool = False,
    keep_empty: bool = False,
) -> List[Dict[str, object]]:
    """
    Load a Web of Science tab-delimited (UTF-8) export into paperscraper records.

    WoS "Tab-delimited (Win, UTF-8)" / "Tab delimited" downloads are typically
    named ``savedrecs.txt``. Each row is mapped to a dictionary with at least
    the keys used by PDF download helpers (``doi``, ``title``, ``authors``,
    ``abstract``, ``date``, ``journal``, ``pubmed_id``).

    Args:
        filepath: Path to the WoS TBA ``.txt`` / ``.tsv`` export.
        require_doi: If True, skip rows without a DOI.
        keep_empty: If True, keep empty string fields; otherwise omit them.

    Returns:
        List of paper metadata dictionaries.
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"WoS TBA file not found: {filepath}")
    if not is_wos_tba_file(filepath):
        raise ValueError(
            f"{filepath} does not look like a Web of Science tab-delimited export "
            "(expected a header with PT/TI/DI fields)."
        )

    papers: List[Dict[str, object]] = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            paper: Dict[str, object] = {}
            for wos_key, target_key in WOS_TBA_FIELD_MAP.items():
                raw = (row.get(wos_key) or "").strip()
                if not raw and not keep_empty:
                    continue
                if target_key in ("authors", "authors_short"):
                    paper[target_key] = _split_wos_authors(raw)
                else:
                    paper[target_key] = raw

            # Prefer full author names (AF) over short initials (AU).
            authors = paper.pop("authors", None)
            authors_short = paper.pop("authors_short", None)
            if authors:
                paper["authors"] = authors
            elif authors_short:
                paper["authors"] = authors_short

            doi = paper.get("doi")
            if require_doi and not doi:
                continue
            if doi:
                paper["doi"] = str(doi).strip()
            papers.append(paper)

    logger.info(f"Loaded {len(papers)} records from WoS TBA file {filepath}")
    return papers


def load_papers_dump(filepath: str) -> List[Dict[str, object]]:
    """
    Load a paper metadata dump from ``.jsonl`` or Web of Science TBA ``.txt``.

    Args:
        filepath: Path to a ``.jsonl`` dump or a WoS tab-delimited export.

    Returns:
        List of paper metadata dictionaries.
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")

    lower = filepath.lower()
    if lower.endswith(".jsonl"):
        return load_jsonl(filepath)
    if lower.endswith((".txt", ".tsv", ".csv")) or is_wos_tba_file(filepath):
        return load_wos_tba(filepath)
    raise ValueError(
        "Unsupported dump format. Provide a .jsonl file or a Web of Science "
        "tab-delimited (.txt/.tsv) export."
    )


def wos_tba_to_jsonl(
    tba_path: str,
    jsonl_path: str,
    *,
    require_doi: bool = False,
) -> str:
    """
    Convert a Web of Science TBA export to a paperscraper ``.jsonl`` dump.

    Args:
        tba_path: Path to the WoS tab-delimited export.
        jsonl_path: Destination ``.jsonl`` path.
        require_doi: If True, skip rows without a DOI.

    Returns:
        The ``jsonl_path`` written.
    """
    papers = load_wos_tba(tba_path, require_doi=require_doi)
    dump_papers(pd.DataFrame(papers), jsonl_path)
    return jsonl_path
