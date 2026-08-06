import os
import threading
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import find_dotenv, load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = (10, 30)
_thread_local = threading.local()


class _PooledSession(requests.Session):
    """Session with connection pooling and a default timeout on every request."""

    def request(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        return super().request(*args, **kwargs)


def get_session() -> requests.Session:
    """
    Return a per-thread pooled requests session.

    Sessions are thread-local because requests.Session is not safe to share across
    threads, while pooling still removes the TCP/TLS handshake from repeated calls
    to the same host within a worker.
    """
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _PooledSession()
        # Reused keep-alive connections can be closed server-side between calls, so
        # retry connection drops and transient 5xx. raise_on_status stays False so
        # callers keep seeing the usual HTTPError from raise_for_status().
        retry = Retry(
            total=2,
            connect=2,
            read=1,
            status=2,
            backoff_factor=0.3,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


def load_api_keys(filepath: Optional[str] = None) -> Dict[str, str]:
    """
    Reads API keys from a file and returns them as a dictionary.
    The file should have each API key on a separate line in the format:
        KEY_NAME=API_KEY_VALUE

    Example:
        WILEY_TDM_API_TOKEN=your_wiley_token_here
        ELSEVIER_TDM_API_KEY=your_elsevier_key_here
        SPRINGER_OPEN_ACCESS_API=your_springer_open_access_key_here
        SPRINGER_API_KEY=your_springer_metadata_key_here
        AWS_ACCESS_KEY_ID=your_aws_access_key_here
        AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

    Args:
        filepath: Optional path to the file containing API keys.

    Returns:
        Dict[str, str]: A dictionary where keys are API key names and values are their respective API keys.
    """
    candidates: list[Path] = []
    if filepath:
        candidates.append(Path(filepath))
    else:
        # Repo root: .../paperscraper/pdf/utils.py -> parents[2]
        repo_root = Path(__file__).resolve().parents[2]
        home = Path.home()
        candidates.extend(
            [
                Path.cwd() / ".env",
                Path.cwd() / ".env.txt",
                repo_root / ".env",
                repo_root / ".env.txt",
                home / ".env",
                home / ".env.txt",
            ]
        )
        found = find_dotenv(usecwd=True)
        if found:
            candidates.append(Path(found))

    loaded_any = False
    for env_path in candidates:
        try:
            if env_path.is_file():
                load_dotenv(dotenv_path=env_path, override=False)
                loaded_any = True
        except OSError:
            continue

    if not loaded_any and not filepath:
        # Last resort: default dotenv discovery (may be a no-op).
        load_dotenv(find_dotenv(usecwd=True))

    # Accept common typo for the Springer Open Access key name.
    springer_oa = os.getenv("SPRINGER_OPEN_ACCESS_API") or os.getenv(
        "SPRINGER_OPEN_ACESS_API"
    )

    return {
        "WILEY_TDM_API_TOKEN": os.getenv("WILEY_TDM_API_TOKEN"),
        "ELSEVIER_TDM_API_KEY": os.getenv("ELSEVIER_TDM_API_KEY"),
        "SPRINGER_OPEN_ACCESS_API": springer_oa,
        "SPRINGER_API_KEY": os.getenv("SPRINGER_API_KEY"),
        "OPENALEX_API_KEY": os.getenv("OPENALEX_API_KEY"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }


def download_pdf_to_path(pdf_url: str, out_path: Path, headers: dict) -> bool:
    # Stream to avoid loading the entire PDF in memory
    with get_session().get(
        pdf_url, stream=True, headers=headers, allow_redirects=True
    ) as r:
        r.raise_for_status()
        # Read first chunk to validate it’s a PDF
        it = r.iter_content(chunk_size=64 * 1024)
        first = next(it, b"")
        if not first.startswith(b"%PDF"):
            return False
        with open(out_path.with_suffix(".pdf"), "wb") as f:
            f.write(first)
            for chunk in it:
                if chunk:
                    f.write(chunk)
    return True
