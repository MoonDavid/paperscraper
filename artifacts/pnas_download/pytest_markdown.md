============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0 -- /workspace/.venv/bin/python
cachedir: .pytest_cache
rootdir: /workspace
configfile: pyproject.toml
plugins: anyio-4.12.1, cov-7.0.0
collecting ... collected 5 items

paperscraper/tests/test_pdf.py::TestPDF::test_convert_file_to_markdown_pdf PASSED [ 20%]
paperscraper/tests/test_pdf.py::TestPDF::test_save_pdf_to_markdown_option PASSED [ 40%]
paperscraper/tests/test_pdf.py::TestPDF::test_save_pdf_from_dump_to_markdown PASSED [ 60%]
paperscraper/tests/test_pdf.py::TestPDF::test_to_markdown_requires_bool PASSED [ 80%]
paperscraper/tests/test_pdf.py::TestPDF::test_fallback_europepmc_pdf_when_xml_missing PASSED [100%]

============================== 5 passed in 7.68s ===============================
