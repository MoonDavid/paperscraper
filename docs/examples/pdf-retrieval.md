# PDF Retrieval

`paperscraper` downloads full text from DOI metadata. Preprint servers are the
most reliable targets; journal papers can depend on open-access status,
publisher pages, and institutional/API access.

## Single DOI

Download a single paper by DOI:

```pycon
>>> from paperscraper.pdf import save_pdf
>>> paper = {"doi": "10.48550/arXiv.2207.03928"}
>>> save_pdf(paper, filepath="gt4sd_paper.pdf")
True
```

`filepath` can be provided with or without `.pdf`. XML fallbacks write an `.xml`
file next to the requested path when XML full text is the available format.

Pass `save_metadata=True` to store paper metadata next to the downloaded file:

```pycon
>>> save_pdf(paper, filepath="gt4sd_paper.pdf", save_metadata=True)
True
```

## Batch Downloads

Download PDFs or XMLs from a metadata dump:

```py
from paperscraper.pdf import save_pdf_from_dump

save_pdf_from_dump(
    "ai_quantum_chemistry.jsonl",
    pdf_path="papers",
    key_to_save="doi",
)
```

`key_to_save` can be `"doi"`, `"title"`, or `"date"`.

### Web of Science tab-delimited exports

`save_pdf_from_dump` also accepts Web of Science **Tab-delimited (Win, UTF-8)**
exports (usually named `savedrecs.txt`):

```py
from paperscraper.pdf import save_pdf_from_dump

save_pdf_from_dump(
    "savedrecs.txt",
    pdf_path="wos_pdfs",
    key_to_save="doi",
    mail="you@example.com",  # helps Unpaywall / NCBI polite use
)
```

In Web of Science: Export → Tab delimited / Tab-delimited (Win, UTF-8). The
loader maps WoS tags such as `DI`→`doi`, `TI`→`title`, `AF`→`authors`,
`AB`→`abstract`, `PY`→`date`, `SO`→`journal`, and `PM`→`pubmed_id`.

To convert a WoS export to a paperscraper `.jsonl` dump first:

```py
from paperscraper.utils import wos_tba_to_jsonl

wos_tba_to_jsonl("savedrecs.txt", "savedrecs.jsonl")
```

## Fallbacks

When direct PDF retrieval fails, `paperscraper` tries supported fallbacks:

- BioC-PMC XML for open-access papers in PubMed Central.
- eLife XML from the eLife article XML repository.
- Publisher APIs when matching credentials are available.
- bioRxiv S3 access when AWS requester-pays credentials are provided.

These improve retrieval success, but they cannot bypass publisher restrictions
or paywalls.

## Publisher API Keys

Publisher API keys can be supplied via a file or loaded from `.env`:

```txt
WILEY_TDM_API_TOKEN=your_wiley_token_here
ELSEVIER_TDM_API_KEY=your_elsevier_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
```

Then pass the path when downloading from a dump:

```py
save_pdf_from_dump(
    "pubmed_query_results.jsonl",
    pdf_path="papers",
    key_to_save="doi",
    api_keys="api_keys.txt",
)
```

Or load the keys once and reuse them across calls:

```py
from paperscraper.pdf import load_api_keys, save_pdf

api_keys = load_api_keys("api_keys.txt")
save_pdf(
    {"doi": "10.1101/786871"},
    filepath="taskload.pdf",
    api_keys=api_keys,
)
```

Wiley and Elsevier TDM APIs are generally free for eligible academic users with
institutional access. For bioRxiv S3 access, use an AWS IAM key with
`AmazonS3ReadOnlyAccess`.

## Downstream Analysis

Retrieved PDFs can be passed to document conversion and analysis tools. For
example, [Docling](https://github.com/docling-project/docling) can convert PDFs
into structured text/Markdown for downstream extraction, indexing, or RAG
pipelines. See the [Docling technical report](https://arxiv.org/abs/2408.09869)
for details.
