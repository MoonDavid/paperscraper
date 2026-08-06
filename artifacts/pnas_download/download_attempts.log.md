2026-08-06 11:42:47,965 | INFO    | pnas_retry | Starting exhaustive download retries for 10.1073/pnas.1718406115
2026-08-06 11:42:47,965 | INFO    | pnas_retry | Output dir: /workspace/artifacts/pnas_download
2026-08-06 11:42:47,967 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:48,027 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:48,028 | INFO    | pnas_retry | ATTEMPT 1   direct[paperscraper]->www.pnas.org_doi_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:48,329 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:48,371 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:48,372 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:48,428 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:48,429 | INFO    | pnas_retry | ATTEMPT 2   direct[paperscraper]->doi.org_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:48,731 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:48,762 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:48,763 | INFO    | pnas_retry | ATTEMPT 3   direct[paperscraper]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:49,064 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:49,103 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115?download=true HTTP/1.1" 403 None
2026-08-06 11:42:49,103 | INFO    | pnas_retry | ATTEMPT 4   direct[paperscraper]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:49,405 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:49,439 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:49,439 | INFO    | pnas_retry | ATTEMPT 5   direct[paperscraper]->pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:49,741 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:49,772 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/epdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:49,772 | INFO    | pnas_retry | ATTEMPT 6   direct[paperscraper]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:50,074 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:50,103 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /content/pnas/pnas.1718406115.full.pdf HTTP/1.1" 403 None
2026-08-06 11:42:50,103 | INFO    | pnas_retry | ATTEMPT 7   direct[paperscraper]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:50,405 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:50,435 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:50,435 | INFO    | pnas_retry | ATTEMPT 8   direct[Mozilla/5.0 ]->www.pnas.org_doi_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:50,737 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:50,775 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:50,776 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:50,810 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:50,810 | INFO    | pnas_retry | ATTEMPT 9   direct[Mozilla/5.0 ]->doi.org_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:51,112 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:51,142 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:51,143 | INFO    | pnas_retry | ATTEMPT 10  direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:51,445 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:51,477 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115?download=true HTTP/1.1" 403 None
2026-08-06 11:42:51,477 | INFO    | pnas_retry | ATTEMPT 11  direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:51,778 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:51,808 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:51,809 | INFO    | pnas_retry | ATTEMPT 12  direct[Mozilla/5.0 ]->pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:52,110 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:52,138 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/epdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:52,138 | INFO    | pnas_retry | ATTEMPT 13  direct[Mozilla/5.0 ]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:52,439 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:52,470 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /content/pnas/pnas.1718406115.full.pdf HTTP/1.1" 403 None
2026-08-06 11:42:52,470 | INFO    | pnas_retry | ATTEMPT 14  direct[Mozilla/5.0 ]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:52,772 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:52,805 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:52,805 | INFO    | pnas_retry | ATTEMPT 15  direct[Mozilla/5.0 ]->www.pnas.org_doi_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:53,106 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:53,138 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:53,139 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:53,173 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:53,173 | INFO    | pnas_retry | ATTEMPT 16  direct[Mozilla/5.0 ]->doi.org_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:53,476 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:53,506 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:53,507 | INFO    | pnas_retry | ATTEMPT 17  direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:53,808 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:53,840 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115?download=true HTTP/1.1" 403 None
2026-08-06 11:42:53,841 | INFO    | pnas_retry | ATTEMPT 18  direct[Mozilla/5.0 ]->www.pnas.org_doi_pdf_10.1073_pnas.1718406115?download=true success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:54,142 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:54,174 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:54,174 | INFO    | pnas_retry | ATTEMPT 19  direct[Mozilla/5.0 ]->pnas.org_doi_pdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:54,476 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:54,512 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/epdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:54,513 | INFO    | pnas_retry | ATTEMPT 20  direct[Mozilla/5.0 ]->www.pnas.org_doi_epdf_10.1073_pnas.1718406115 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:54,814 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:54,845 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /content/pnas/pnas.1718406115.full.pdf HTTP/1.1" 403 None
2026-08-06 11:42:54,846 | INFO    | pnas_retry | ATTEMPT 21  direct[Mozilla/5.0 ]->www.pnas.org_content_pnas_pnas.1718406115.full.pdf success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:55,249 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:55,288 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:55,288 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:55,320 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:55,321 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:55,349 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:55,350 | INFO    | pnas_retry | ATTEMPT 22  citation_pdf_meta[paperscraper/1.0 (+https] success=False path=None error=no citation_pdf_url meta
2026-08-06 11:42:55,853 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:55,889 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:55,890 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:55,919 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:55,920 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:55,951 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:55,952 | INFO    | pnas_retry | ATTEMPT 23  citation_pdf_meta[Mozilla/5.0 (X11; Linux ] success=False path=None error=no citation_pdf_url meta
2026-08-06 11:42:56,454 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): doi.org:443
2026-08-06 11:42:56,489 | DEBUG   | urllib3.connectionpool | https://doi.org:443 "GET /10.1073/pnas.1718406115 HTTP/1.1" 302 None
2026-08-06 11:42:56,490 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:56,520 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/full/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:56,521 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:56,551 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /doi/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:56,552 | INFO    | pnas_retry | ATTEMPT 24  citation_pdf_meta[Mozilla/5.0 (Windows NT ] success=False path=None error=no citation_pdf_url meta
2026-08-06 11:42:57,055 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): api.unpaywall.org:443
2026-08-06 11:42:57,102 | DEBUG   | urllib3.connectionpool | https://api.unpaywall.org:443 "GET /v2/10.1073/pnas.1718406115?email=paperscraper-debug@example.com HTTP/1.1" 422 154
2026-08-06 11:42:57,103 | INFO    | pnas_retry | ATTEMPT 25  unpaywall_probe              success=False path=None error=no url_for_pdf
2026-08-06 11:42:57,104 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): api.openalex.org:443
2026-08-06 11:42:57,311 | DEBUG   | urllib3.connectionpool | https://api.openalex.org:443 "GET /works/https://doi.org/10.1073/pnas.1718406115 HTTP/1.1" 200 None
2026-08-06 11:42:57,314 | INFO    | pnas_retry | ATTEMPT 26  openalex_probe               success=True path=None error=
2026-08-06 11:42:57,315 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:57,345 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /content/pnas/115/17/E3879.full.pdf HTTP/1.1" 403 None
2026-08-06 11:42:57,346 | INFO    | pnas_retry | ATTEMPT 27  openalex_pdf                 success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:57,347 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): api.crossref.org:443
2026-08-06 11:42:57,408 | DEBUG   | urllib3.connectionpool | https://api.crossref.org:443 "GET /works/10.1073/pnas.1718406115 HTTP/1.1" 200 3486
2026-08-06 11:42:57,409 | INFO    | pnas_retry | ATTEMPT 28  crossref_probe               success=True path=None error=
2026-08-06 11:42:57,409 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pnas.org:443
2026-08-06 11:42:57,438 | DEBUG   | urllib3.connectionpool | https://pnas.org:443 "GET /doi/pdf/10.1073/pnas.1718406115 HTTP/1.1" 403 None
2026-08-06 11:42:57,438 | INFO    | pnas_retry | ATTEMPT 29  crossref_pdf_0               success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:57,439 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): api.semanticscholar.org:443
2026-08-06 11:42:57,598 | DEBUG   | urllib3.connectionpool | https://api.semanticscholar.org:443 "GET /graph/v1/paper/DOI:10.1073/pnas.1718406115?fields=title,openAccessPdf,isOpenAccess,externalIds HTTP/1.1" 200 400
2026-08-06 11:42:57,598 | INFO    | pnas_retry | ATTEMPT 30  semantic_scholar_probe       success=True path=None error=
2026-08-06 11:42:57,599 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.pnas.org:443
2026-08-06 11:42:57,629 | DEBUG   | urllib3.connectionpool | https://www.pnas.org:443 "GET /content/pnas/115/17/E3879.full.pdf HTTP/1.1" 403 None
2026-08-06 11:42:57,630 | INFO    | pnas_retry | ATTEMPT 31  semantic_scholar_pdf         success=False path=None error=HTTP 403, not a PDF (ctype=text/html; charset=UTF-8)
2026-08-06 11:42:57,631 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.ebi.ac.uk:443
2026-08-06 11:42:58,072 | DEBUG   | urllib3.connectionpool | https://www.ebi.ac.uk:443 "GET /europepmc/webservices/rest/search?query=DOI:10.1073/pnas.1718406115&format=json&resultType=core HTTP/1.1" 200 None
2026-08-06 11:42:58,078 | INFO    | pnas_retry | ATTEMPT 32  europepmc_rest_search        success=True path=None error=
2026-08-06 11:42:58,079 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): europepmc.org:443
2026-08-06 11:42:58,466 | DEBUG   | urllib3.connectionpool | https://europepmc.org:443 "GET /articles/PMC5924899?pdf=render HTTP/1.1" 302 232
2026-08-06 11:42:58,749 | DEBUG   | urllib3.connectionpool | https://europepmc.org:443 "GET /api/getPdf?pmcid=PMC5924899 HTTP/1.1" 200 1452260
2026-08-06 11:43:00,053 | INFO    | pnas_retry | ATTEMPT 33  europepmc_pdf_render         success=True path=/workspace/artifacts/pnas_download/pdfs/europepmc_render.pdf error=
2026-08-06 11:43:00,055 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,088 | DEBUG   | urllib3.connectionpool | https://www.ncbi.nlm.nih.gov:443 "GET /pmc/utils/idconv/v1.0/?tool=paperscraper&email=paperscraper-debug%40example.com&ids=10.1073%2Fpnas.1718406115&idtype=doi&format=json HTTP/1.1" 301 432
2026-08-06 11:43:00,089 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pmc.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,141 | DEBUG   | urllib3.connectionpool | https://pmc.ncbi.nlm.nih.gov:443 "GET /tools/idconv/api/v1/articles/?tool=paperscraper&email=paperscraper-debug%40example.com&ids=10.1073%2Fpnas.1718406115&idtype=doi&format=json HTTP/1.1" 200 None
2026-08-06 11:43:00,142 | INFO    | pnas_retry | ATTEMPT 34  ncbi_idconv                  success=True path=None error=
2026-08-06 11:43:00,142 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,173 | DEBUG   | urllib3.connectionpool | https://www.ncbi.nlm.nih.gov:443 "GET /pmc/articles/PMC5924899/pdf/ HTTP/1.1" 301 244
2026-08-06 11:43:00,174 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pmc.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,227 | DEBUG   | urllib3.connectionpool | https://pmc.ncbi.nlm.nih.gov:443 "GET /articles/PMC5924899/pdf/ HTTP/1.1" 200 None
2026-08-06 11:43:00,228 | INFO    | pnas_retry | ATTEMPT 35  pmc_pdf_0                    success=False path=None error=HTTP 200, not a PDF (ctype=text/html; charset=utf-8)
2026-08-06 11:43:00,229 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): www.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,260 | DEBUG   | urllib3.connectionpool | https://www.ncbi.nlm.nih.gov:443 "GET /pmc/articles/PMC5924899/pdf/PMC5924899.pdf HTTP/1.1" 301 315
2026-08-06 11:43:00,261 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): pmc.ncbi.nlm.nih.gov:443
2026-08-06 11:43:00,312 | DEBUG   | urllib3.connectionpool | https://pmc.ncbi.nlm.nih.gov:443 "GET /articles/PMC5924899/pdf/PMC5924899.pdf HTTP/1.1" 200 None
2026-08-06 11:43:00,312 | INFO    | pnas_retry | ATTEMPT 36  pmc_pdf_1                    success=False path=None error=HTTP 200, not a PDF (ctype=text/html; charset=utf-8)
2026-08-06 11:43:00,313 | DEBUG   | urllib3.connectionpool | Starting new HTTPS connection (1): europepmc.org:443
2026-08-06 11:43:00,680 | DEBUG   | urllib3.connectionpool | https://europepmc.org:443 "GET /articles/PMC5924899?pdf=render HTTP/1.1" 302 232
2026-08-06 11:43:00,843 | DEBUG   | urllib3.connectionpool | https://europepmc.org:443 "GET /api/getPdf?pmcid=PMC5924899 HTTP/1.1" 200 1452260
2026-08-06 11:43:02,192 | INFO    | pnas_retry | ATTEMPT 37  pmc_pdf_2                    success=True path=/workspace/artifacts/pnas_download/pdfs/pmc_2.pdf error=
2026-08-06 11:43:03,300 | WARNING | paperscraper.load_dumps |  No dump found for biorxiv. Skipping entry.
2026-08-06 11:43:03,301 | WARNING | paperscraper.load_dumps |  No dump found for chemrxiv. Skipping entry.
2026-08-06 11:43:03,301 | WARNING | paperscraper.load_dumps |  No dump found for medrxiv. Skipping entry.
2026-08-06 11:43:03,301 | WARNING | paperscraper.load_dumps |  No dumps found for either biorxiv, medrxiv and chemrxiv. Consider using paperscraper.get_dumps.* to fetch the dumps.
2026-08-06 11:43:03,647 | WARNING | paperscraper.pdf.pdf | Could not download from: None - 403 Client Error: Forbidden for url: https://pnas.org/doi/full/10.1073/pnas.1718406115. Trying fallbacks.
2026-08-06 11:43:03,647 | INFO    | paperscraper.pdf.pdf | Primary download failed for 10.1073/pnas.1718406115. Attempting fallbacks.
2026-08-06 11:43:03,684 | WARNING | paperscraper.pdf.fallbacks | Error during Unpaywall fallback for 10.1073/pnas.1718406115: 422 Client Error: Unprocessable Entity for url: https://api.unpaywall.org/v2/10.1073/pnas.1718406115?email=paperscraper-debug@example.com
2026-08-06 11:43:04,045 | INFO    | paperscraper.pdf.fallbacks | Found PMCID PMC5924899 for DOI 10.1073/pnas.1718406115 in Europe PMC (result 1 of 1).
2026-08-06 11:43:04,439 | ERROR   | paperscraper.pdf.fallbacks | Failed to download XML from Europe PMC for DOI 10.1073/pnas.1718406115: 404 Client Error: Not Found for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5924899/fullTextXML
2026-08-06 11:43:04,520 | INFO    | paperscraper.pdf.fallbacks | Converted DOI 10.1073/pnas.1718406115 to PMCID PMC5924899.
2026-08-06 11:43:04,520 | INFO    | paperscraper.pdf.fallbacks | Attempting to download XML from BioC-PMC URL: https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode
2026-08-06 11:43:04,891 | WARNING | paperscraper.pdf.fallbacks | No XML found for DOI 10.1073/pnas.1718406115 at BioC-PMC URL https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode.
2026-08-06 11:43:05,151 | ERROR   | paperscraper.pdf.fallbacks | OpenAlex fallback failed for 10.1073/pnas.1718406115: 403 Client Error: Forbidden for url: https://www.pnas.org/content/pnas/115/17/E3879.full.pdf
2026-08-06 11:43:05,188 | INFO    | paperscraper.pdf.fallbacks | Crossref: no usable PDF links for 10.1073/pnas.1718406115.
2026-08-06 11:43:05,359 | INFO    | paperscraper.pdf.fallbacks | DOAJ: no usable fulltext PDF for 10.1073/pnas.1718406115.
2026-08-06 11:43:05,407 | INFO    | paperscraper.pdf.fallbacks | arXiv: no entry for DOI 10.1073/pnas.1718406115
2026-08-06 11:43:05,407 | WARNING | paperscraper.pdf.pdf | All download attempts failed for 10.1073/pnas.1718406115.
2026-08-06 11:43:05,764 | INFO    | paperscraper.pdf.pdf | Saved abstract to /workspace/artifacts/pnas_download/pdfs/save_pdf.txt.
2026-08-06 11:43:05,766 | INFO    | pnas_retry | ATTEMPT 38  save_pdf                     success=False path=None error=
2026-08-06 11:43:05,849 | INFO    | paperscraper.pdf.fallbacks | Converted DOI 10.1073/pnas.1718406115 to PMCID PMC5924899.
2026-08-06 11:43:05,849 | INFO    | paperscraper.pdf.fallbacks | Attempting to download XML from BioC-PMC URL: https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode
2026-08-06 11:43:06,284 | WARNING | paperscraper.pdf.fallbacks | No XML found for DOI 10.1073/pnas.1718406115 at BioC-PMC URL https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode.
2026-08-06 11:43:06,286 | INFO    | pnas_retry | ATTEMPT 39  fallback:bioc_pmc            success=False path=None error=
2026-08-06 11:43:06,787 | ERROR   | paperscraper.pdf.fallbacks | Unable to parse eLife DOI: 10.1073/pnas.1718406115
2026-08-06 11:43:06,787 | INFO    | pnas_retry | ATTEMPT 40  fallback:elife               success=False path=None error=
2026-08-06 11:43:07,287 | INFO    | pnas_retry | ATTEMPT 41  fallback:elsevier            success=False path=None error=skipped: missing ELSEVIER_TDM_API_KEY
2026-08-06 11:43:08,163 | INFO    | paperscraper.pdf.fallbacks | Found PMCID PMC5924899 for DOI 10.1073/pnas.1718406115 in Europe PMC (result 1 of 1).
2026-08-06 11:43:08,546 | ERROR   | paperscraper.pdf.fallbacks | Failed to download XML from Europe PMC for DOI 10.1073/pnas.1718406115: 404 Client Error: Not Found for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5924899/fullTextXML
2026-08-06 11:43:08,548 | INFO    | pnas_retry | ATTEMPT 42  fallback:europepmc           success=False path=None error=
2026-08-06 11:43:09,049 | INFO    | pnas_retry | ATTEMPT 43  fallback:s3                  success=False path=None error=skipped: missing AWS credentials
2026-08-06 11:43:09,549 | INFO    | pnas_retry | ATTEMPT 44  fallback:wiley               success=False path=None error=skipped: missing WILEY_TDM_API_TOKEN
2026-08-06 11:43:10,087 | WARNING | paperscraper.pdf.fallbacks | Error during Unpaywall fallback for 10.1073/pnas.1718406115: 422 Client Error: Unprocessable Entity for url: https://api.unpaywall.org/v2/10.1073/pnas.1718406115?email=paperscraper-debug@example.com
2026-08-06 11:43:10,088 | INFO    | pnas_retry | ATTEMPT 45  fallback:unpaywall           success=False path=None error=
2026-08-06 11:43:10,588 | INFO    | pnas_retry | ATTEMPT 46  fallback:springer            success=False path=None error=skipped: missing SPRINGER_API_KEY
2026-08-06 11:43:11,089 | INFO    | pnas_retry | ATTEMPT 47  fallback:plos                success=False path=None error=
2026-08-06 11:43:11,807 | ERROR   | paperscraper.pdf.fallbacks | OpenAlex fallback failed for 10.1073/pnas.1718406115: 403 Client Error: Forbidden for url: https://www.pnas.org/content/pnas/115/17/E3879.full.pdf
2026-08-06 11:43:11,809 | INFO    | pnas_retry | ATTEMPT 48  fallback:openalex            success=False path=None error=
2026-08-06 11:43:12,345 | INFO    | paperscraper.pdf.fallbacks | Crossref: no usable PDF links for 10.1073/pnas.1718406115.
2026-08-06 11:43:12,346 | INFO    | pnas_retry | ATTEMPT 49  fallback:crossref            success=False path=None error=
2026-08-06 11:43:12,889 | INFO    | paperscraper.pdf.fallbacks | arXiv: no entry for DOI 10.1073/pnas.1718406115
2026-08-06 11:43:12,890 | INFO    | pnas_retry | ATTEMPT 50  fallback:arxiv               success=False path=None error=
2026-08-06 11:43:13,391 | INFO    | pnas_retry | ATTEMPT 51  fallback:medrxiv_s3          success=False path=None error=skipped: missing AWS credentials
2026-08-06 11:43:14,030 | INFO    | paperscraper.pdf.fallbacks | DOAJ: no usable fulltext PDF for 10.1073/pnas.1718406115.
2026-08-06 11:43:14,032 | INFO    | pnas_retry | ATTEMPT 52  fallback:doaj                success=False path=None error=
2026-08-06 11:43:14,611 | WARNING | paperscraper.pdf.pdf | Could not download from: None - 403 Client Error: Forbidden for url: https://pnas.org/doi/full/10.1073/pnas.1718406115. Trying fallbacks.
2026-08-06 11:43:14,611 | INFO    | paperscraper.pdf.pdf | Primary download failed for 10.1073/pnas.1718406115. Attempting fallbacks.
2026-08-06 11:43:14,644 | WARNING | paperscraper.pdf.fallbacks | Error during Unpaywall fallback for 10.1073/pnas.1718406115: 422 Client Error: Unprocessable Entity for url: https://api.unpaywall.org/v2/10.1073/pnas.1718406115?email=paperscraper-debug@example.com
2026-08-06 11:43:15,130 | INFO    | paperscraper.pdf.fallbacks | Found PMCID PMC5924899 for DOI 10.1073/pnas.1718406115 in Europe PMC (result 1 of 1).
2026-08-06 11:43:15,549 | ERROR   | paperscraper.pdf.fallbacks | Failed to download XML from Europe PMC for DOI 10.1073/pnas.1718406115: 404 Client Error: Not Found for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5924899/fullTextXML
2026-08-06 11:43:15,636 | INFO    | paperscraper.pdf.fallbacks | Converted DOI 10.1073/pnas.1718406115 to PMCID PMC5924899.
2026-08-06 11:43:15,636 | INFO    | paperscraper.pdf.fallbacks | Attempting to download XML from BioC-PMC URL: https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode
2026-08-06 11:43:15,792 | WARNING | paperscraper.pdf.fallbacks | No XML found for DOI 10.1073/pnas.1718406115 at BioC-PMC URL https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5924899/unicode.
2026-08-06 11:43:15,916 | ERROR   | paperscraper.pdf.fallbacks | OpenAlex fallback failed for 10.1073/pnas.1718406115: 403 Client Error: Forbidden for url: https://www.pnas.org/content/pnas/115/17/E3879.full.pdf
2026-08-06 11:43:15,950 | INFO    | paperscraper.pdf.fallbacks | Crossref: no usable PDF links for 10.1073/pnas.1718406115.
2026-08-06 11:43:16,086 | INFO    | paperscraper.pdf.fallbacks | DOAJ: no usable fulltext PDF for 10.1073/pnas.1718406115.
2026-08-06 11:43:16,119 | INFO    | paperscraper.pdf.fallbacks | arXiv: no entry for DOI 10.1073/pnas.1718406115
2026-08-06 11:43:16,120 | WARNING | paperscraper.pdf.pdf | All download attempts failed for 10.1073/pnas.1718406115.
2026-08-06 11:43:16,526 | INFO    | paperscraper.pdf.pdf | Saved abstract to /workspace/artifacts/pnas_download/pdfs/debug_save_pdf.pdf.txt.
2026-08-06 11:43:16,558 | WARNING | paperscraper.pdf.fallbacks | Error during Unpaywall fallback for 10.1073/pnas.1718406115: 422 Client Error: Unprocessable Entity for url: https://api.unpaywall.org/v2/10.1073/pnas.1718406115?email=paperscraper-debug@example.com
2026-08-06 11:43:16,915 | INFO    | paperscraper.pdf.fallbacks | Found PMCID PMC5924899 for DOI 10.1073/pnas.1718406115 in Europe PMC (result 1 of 1).
2026-08-06 11:43:17,285 | ERROR   | paperscraper.pdf.fallbacks | Failed to download XML from Europe PMC for DOI 10.1073/pnas.1718406115: 404 Client Error: Not Found for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5924899/fullTextXML
2026-08-06 11:43:17,286 | ERROR   | paperscraper.pdf.fallbacks | Unable to parse eLife DOI: 10.1073/pnas.1718406115
2026-08-06 11:43:17,419 | ERROR   | paperscraper.pdf.fallbacks | OpenAlex fallback failed for 10.1073/pnas.1718406115: 403 Client Error: Forbidden for url: https://www.pnas.org/content/pnas/115/17/E3879.full.pdf
2026-08-06 11:43:17,456 | INFO    | paperscraper.pdf.fallbacks | Crossref: no usable PDF links for 10.1073/pnas.1718406115.
2026-08-06 11:43:17,593 | INFO    | paperscraper.pdf.fallbacks | DOAJ: no usable fulltext PDF for 10.1073/pnas.1718406115.
2026-08-06 11:43:17,627 | INFO    | paperscraper.pdf.fallbacks | arXiv: no entry for DOI 10.1073/pnas.1718406115
2026-08-06 11:43:17,627 | INFO    | pnas_retry | ATTEMPT 53  debug_save_pdf_all           success=False path=None error=
2026-08-06 11:43:17,628 | INFO    | pnas_retry | DONE: 7/53 successes: ['openalex_probe', 'crossref_probe', 'semantic_scholar_probe', 'europepmc_rest_search', 'europepmc_pdf_render', 'ncbi_idconv', 'pmc_pdf_2']
2026-08-06 11:43:17,628 | INFO    | pnas_retry | Summary written to /workspace/artifacts/pnas_download/download_attempts_summary.json
=== EXTRA ATTEMPTS 2026-08-06T11:43:25Z ===
--- curl GET https://europepmc.org/articles/PMC5924899?pdf=render
result: 200 application/pdf 1452260 https://europepmc.org/api/getPdf?pmcid=PMC5924899
--- curl GET https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5924899/pdf/
result: 200 text/html; charset=utf-8 21255 https://pmc.ncbi.nlm.nih.gov/articles/PMC5924899/pdf/
--- curl GET https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/../../
result: 200 text/html;charset=UTF-8 14456 https://ftp.ncbi.nlm.nih.gov/pub/
--- curl GET https://www.pnas.org/doi/pdf/10.1073/pnas.1718406115
result: 403 text/html; charset=UTF-8 5517 https://www.pnas.org/doi/pdf/10.1073/pnas.1718406115
--- curl GET https://cdn.ncbi.nlm.nih.gov/pmc/blobs/pdf/PMC5924899.pdf
result: 404 text/html 117805 https://cdn.ncbi.nlm.nih.gov/pmc/blobs/pdf/PMC5924899.pdf
--- wget europepmc
pdfs/extra/wget_epmc.pdf: PDF document, version 1.4, 9 page(s)
--- PMC OA service
<OA><responseDate>2026-08-06 07:43:30</responseDate><request>https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC5924899</request><error code="idIsNotOpenAccess">identifier 'PMC5924899' is not Open Access</error></OA>

