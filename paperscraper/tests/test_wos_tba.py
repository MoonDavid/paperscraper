import os
from pathlib import Path

import pytest

from paperscraper.pdf import save_pdf_from_dump
from paperscraper.utils import (
    is_wos_tba_file,
    load_papers_dump,
    load_wos_tba,
    wos_tba_to_jsonl,
)

TEST_WOS_PATH = str(Path(__file__).parent / "test_wos_tba.tsv")
SAVE_PATH = "tmp_wos_pdf_storage"


class TestWosTba:
    def test_detect_wos_tba(self):
        assert is_wos_tba_file(TEST_WOS_PATH)

    def test_load_wos_tba(self):
        papers = load_wos_tba(TEST_WOS_PATH)
        assert len(papers) >= 1
        first = papers[0]
        assert "doi" in first and first["doi"].startswith("10.")
        assert "title" in first and first["title"]
        assert isinstance(first.get("authors"), list)
        assert first["authors"]

    def test_load_papers_dump_auto(self):
        papers = load_papers_dump(TEST_WOS_PATH)
        assert papers[0]["doi"]

    def test_wos_tba_to_jsonl(self, tmp_path):
        out = tmp_path / "wos.jsonl"
        wos_tba_to_jsonl(TEST_WOS_PATH, str(out))
        assert out.exists()
        lines = [line for line in out.read_text().splitlines() if line.strip()]
        assert len(lines) >= 1

    def test_load_wos_tba_bad_file(self, tmp_path):
        bad = tmp_path / "not_wos.txt"
        bad.write_text("hello\tworld\n1\t2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_wos_tba(str(bad))

    def test_save_pdf_from_wos_tba(self):
        os.makedirs(SAVE_PATH, exist_ok=True)
        # Only download the first record to keep the test fast: write a 1-row TBA.
        import csv

        with open(TEST_WOS_PATH, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames
            first = next(reader)
        one_row = Path(SAVE_PATH) / "one.txt"
        with one_row.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(first)

        stats = save_pdf_from_dump(
            str(one_row),
            pdf_path=SAVE_PATH,
            key_to_save="doi",
            mail="dev@example.com",
        )
        assert first["DI"] in stats["by_doi"]
        result = stats["by_doi"][first["DI"]]
        # Full text may be PDF or XML depending on OA path.
        assert result.get("success") in (True, False)
        if result.get("success"):
            doi_name = first["DI"].replace("/", "_")
            assert (
                Path(SAVE_PATH, f"{doi_name}.pdf").exists()
                or Path(SAVE_PATH, f"{doi_name}.xml").exists()
            )
        import shutil

        shutil.rmtree(SAVE_PATH)
