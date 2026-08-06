# PNAS download attempts log

DOI: 10.1073/pnas.1718406115
URL: https://www.pnas.org/doi/10.1073/pnas.1718406115
Attempts: 53
Reported successes (incl. probes): 7

## Final verdict
- SUCCESS via Europe PMC PDF render (PMC5924899)
- Saved: `artifacts/pnas_download/pnas.1718406115.pdf` (1452260 bytes, 9 pages)

## Attempt table

| # | Method | Success | Error |
|---|--------|---------|-------|
| 1 | `direct[paperscraper]->www.pnas.org_doi_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 2 | `direct[paperscraper]->doi.org_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 3 | `direct[paperscraper]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 4 | `direct[paperscraper]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 5 | `direct[paperscraper]->pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 6 | `direct[paperscraper]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 7 | `direct[paperscraper]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 8 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 9 | `direct[Mozilla/5.0 ]->doi.org_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 10 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 11 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 12 | `direct[Mozilla/5.0 ]->pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 13 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 14 | `direct[Mozilla/5.0 ]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 15 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 16 | `direct[Mozilla/5.0 ]->doi.org_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 17 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 18 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 19 | `direct[Mozilla/5.0 ]->pnas.org_doi_pdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 20 | `direct[Mozilla/5.0 ]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 21 | `direct[Mozilla/5.0 ]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 22 | `citation_pdf_meta[paperscraper/1.0 (+https]` | False | no citation_pdf_url meta |
| 23 | `citation_pdf_meta[Mozilla/5.0 (X11; Linux ]` | False | no citation_pdf_url meta |
| 24 | `citation_pdf_meta[Mozilla/5.0 (Windows NT ]` | False | no citation_pdf_url meta |
| 25 | `unpaywall_probe` | False | no url_for_pdf |
| 26 | `openalex_probe` | True |  |
| 27 | `openalex_pdf` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 28 | `crossref_probe` | True |  |
| 29 | `crossref_pdf_0` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 30 | `semantic_scholar_probe` | True |  |
| 31 | `semantic_scholar_pdf` | False | HTTP 403, not a PDF (ctype=text/html; charset=UTF-8) |
| 32 | `europepmc_rest_search` | True |  |
| 33 | `europepmc_pdf_render` | True |  |
| 34 | `ncbi_idconv` | True |  |
| 35 | `pmc_pdf_0` | False | HTTP 200, not a PDF (ctype=text/html; charset=utf-8) |
| 36 | `pmc_pdf_1` | False | HTTP 200, not a PDF (ctype=text/html; charset=utf-8) |
| 37 | `pmc_pdf_2` | True |  |
| 38 | `save_pdf` | False |  |
| 39 | `fallback:bioc_pmc` | False |  |
| 40 | `fallback:elife` | False |  |
| 41 | `fallback:elsevier` | False | skipped: missing ELSEVIER_TDM_API_KEY |
| 42 | `fallback:europepmc` | False |  |
| 43 | `fallback:s3` | False | skipped: missing AWS credentials |
| 44 | `fallback:wiley` | False | skipped: missing WILEY_TDM_API_TOKEN |
| 45 | `fallback:unpaywall` | False |  |
| 46 | `fallback:springer` | False | skipped: missing SPRINGER_API_KEY |
| 47 | `fallback:plos` | False |  |
| 48 | `fallback:openalex` | False |  |
| 49 | `fallback:crossref` | False |  |
| 50 | `fallback:arxiv` | False |  |
| 51 | `fallback:medrxiv_s3` | False | skipped: missing AWS credentials |
| 52 | `fallback:doaj` | False |  |
| 53 | `debug_save_pdf_all` | False |  |

## Working methods

- europepmc.org/articles/PMC5924899?pdf=render
- europepmc.org/api/getPdf?pmcid=PMC5924899
- wget same URL
- curl -L same URL

## Notable failures

- pnas.org direct (HTTP 403 Cloudflare/bot block)
- doi.org redirect to pnas (403)
- paperscraper save_pdf (all built-in fallbacks failed: XML 404, PNAS PDF 403)
- Unpaywall (422 with example.com email / no url_for_pdf)
- BioC-PMC XML missing
- PMC OA package (idIsNotOpenAccess)
- NCBI PMC /pdf/ HTML interstitial

## Library fix (follow-up)

`fallback_europepmc` previously only fetched `fullTextXML` (404 for this DOI).
It now falls back to `https://europepmc.org/articles/{pmcid}?pdf=render`.

After the fix, `save_pdf({"doi": "10.1073/pnas.1718406115"}, ...)` returns:
`{'success': True, 'method': 'europepmc', 'filetype': 'pdf'}`.
