"""Optional PDF/XML → Markdown conversion via Firecrawl anydoc."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Markdown conversion requires the optional 'markdown' extra. "
    "Install with: pip install 'paperscraper[markdown]' "
    "(provides firecrawl-anydoc, imported as `anydoc`). "
    "Requires Python >= 3.10."
)


def _import_anydoc():
    try:
        import anydoc  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(_INSTALL_HINT) from exc
    return anydoc


def _xml_to_markdown(xml_path: Path) -> str:
    """Best-effort Markdown from full-text XML when anydoc cannot parse it."""
    from bs4 import BeautifulSoup

    raw = xml_path.read_bytes()
    soup = BeautifulSoup(raw, features="lxml-xml")
    # Prefer article body / abstract-ish regions when present
    chunks = []
    title = soup.find(["article-title", "title"])
    if title and title.get_text(strip=True):
        chunks.append(f"# {title.get_text(' ', strip=True)}")
    for tag_name in ("abstract", "body", "sec"):
        for node in soup.find_all(tag_name):
            text = node.get_text("\n", strip=True)
            if text:
                chunks.append(text)
    if not chunks:
        text = soup.get_text("\n", strip=True)
        chunks.append(text)
    md = "\n\n".join(chunks)
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def convert_file_to_markdown(
    source_path: Union[str, Path],
    markdown_path: Optional[Union[str, Path]] = None,
    *,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Convert a downloaded PDF (or XML full text) to Markdown.

    PDFs are converted with Firecrawl's ``anydoc`` (PyPI package
    ``firecrawl-anydoc``). XML is converted with a lightweight text extract
    because anydoc does not support JATS/PMC XML.

    Args:
        source_path: Path to a ``.pdf`` or ``.xml`` file.
        markdown_path: Destination ``.md`` path. Defaults to the same stem
            next to ``source_path``.
        overwrite: If False and the ``.md`` already exists, skip conversion.

    Returns:
        Path to the written Markdown file, or None if conversion failed.
    """
    source = Path(source_path)
    if not source.exists():
        logger.warning(f"Cannot convert missing file to Markdown: {source}")
        return None

    out = Path(markdown_path) if markdown_path else source.with_suffix(".md")
    if out.exists() and not overwrite:
        logger.info(f"Markdown already exists, skipping: {out}")
        return out

    suffix = source.suffix.lower()
    try:
        if suffix == ".pdf":
            anydoc = _import_anydoc()
            markdown = anydoc.to_markdown(str(source))
        elif suffix == ".xml":
            # Try anydoc first (unsupported today), then XML text fallback.
            try:
                anydoc = _import_anydoc()
                markdown = anydoc.to_markdown(str(source))
            except ImportError:
                raise
            except Exception:
                markdown = _xml_to_markdown(source)
        else:
            logger.warning(
                f"Unsupported source for Markdown conversion: {source} "
                f"(expected .pdf or .xml)"
            )
            return None
    except ImportError:
        raise
    except Exception as exc:
        logger.warning(f"Markdown conversion failed for {source}: {exc}")
        return None

    if not isinstance(markdown, str) or not markdown.strip():
        logger.warning(f"Markdown conversion produced empty output for {source}")
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    logger.info(f"Wrote Markdown to {out}")
    return out


def maybe_convert_download(
    output_stem: Union[str, Path],
    filetype: Optional[str],
    *,
    to_markdown: bool,
) -> Optional[Path]:
    """
    If ``to_markdown`` is set and a PDF/XML was saved under ``output_stem``,
    convert it to Markdown beside the binary.
    """
    if not to_markdown or not filetype:
        return None
    stem = Path(output_stem)
    if filetype == "pdf":
        source = stem.with_suffix(".pdf")
    elif filetype == "xml":
        source = stem.with_suffix(".xml")
    else:
        return None
    return convert_file_to_markdown(source)
