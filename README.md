# PakLawRAG

A RAG (Retrieval-Augmented Generation) system for the **Pakistan Penal Code (PPC)**. Given a natural language query it retrieves the most relevant PPC sections and generates a concise answer using a local LLM.

---

## Project structure

```
PakLawRAG/
├── scripts/
│   ├── scrape_ppc.py                 # Fetch and parse all PPC sections from the web
│   ├── normalise_sections.py         # Replace Urdu/Arabic legal terms with English equivalents
│   ├── build_vectorstore_sections.py # Embed sections and build FAISS index
│   ├── query_vectorstore_sections.py # Interactive RAG CLI (retrieval + LLM answer)
│   └── eval_retrieval.py             # Retrieval evaluation (Hit@k, MRR)
├── output/
│   └── ppc_sections.json             # 636 parsed + normalised sections
├── vectorstore_sections/             # FAISS index (commit this — no need to rebuild)
│   ├── index.faiss                   # Embedding vectors (636 sections × 1024 dims)
│   └── index.pkl                     # Document metadata (section IDs, original text)
└── requirements.txt
```

---

## Pipeline

### Build phase (run once)

```
pakistani.org/pakistan/legislation/1860/actXLVof1860.html
      │
      ▼
scrape_ppc.py
      │   ├─ Fetches raw HTML, extracts plain text via BeautifulSoup
      │   ├─ Trims to PPC body (between "Pakistan Penal Code" and last "Schedule")
      │   ├─ Detects section headings with a single compiled regex
      │   │    handles: "375."  "375A."  "337-A."  "120-A"  "Section 375"
      │   ├─ Deduplicates — keeps longest text per section_id
      │   └─ Saves {"section_id", "text"} for 636 sections
      │
      ▼
normalise_sections.py
      │   ├─ Expands Islamized Urdu/Arabic terms inline using single-pass regex
      │   │    e.g. "qatl-i-amd" → "qatl-i-amd (intentional murder, murder, ...)"
      │   ├─ Extracts keywords found per section
      │   └─ Saves {"section_id", "text", "normalized_text", "keywords"}
      │
      ▼
build_vectorstore_sections.py
      │   ├─ Loads sections from ppc_sections.json
      │   ├─ Embeds normalized_text using BAAI/bge-large-en-v1.5
      │   ├─ Stores original text in document metadata for display
      │   └─ Builds and saves FAISS flat index
      │
      ▼
vectorstore_sections/
```

### Query phase

```
User question (natural language)
      │
      ▼
query_vectorstore_sections.py
      │   ├─ Loads FAISS index + BGE embedding model (singleton — loads once)
      │   ├─ Embeds the question with the same model
      │   ├─ similarity_search(query, k=3) → top-3 sections
      │   └─ Passes retrieved sections as context to gemma3:4b via Ollama
      │
      ▼
LLM answer citing section numbers + source list
```

### What is embedded vs what is displayed

| Field | Content | Used for |
|---|---|---|
| `normalized_text` | Urdu terms + English meanings in `()` | Embedding / retrieval |
| `text` (original) | Verbatim legal text as scraped | Passed to LLM as context |

The embedding model sees English equivalents of Urdu terms. The LLM receives the actual legal text unchanged.

---

## Setup

```bash
pip install -r requirements.txt
pip install langchain-huggingface langchain-ollama
```

Install and start Ollama, then pull the LLM:

```bash
ollama pull gemma3:4b
```

### Run the pipeline

```bash
cd scripts

# 1. Scrape sections from the web
python scrape_ppc.py

# 2. Normalise Urdu/Arabic terms
python normalise_sections.py

# 3. Build FAISS index
python build_vectorstore_sections.py
```

> The `vectorstore_sections/` directory is committed to the repo — skip step 3 if you haven't changed the corpus.

### Query

```bash
python query_vectorstore_sections.py
```

### Evaluate

```bash
python eval_retrieval.py
```

> On macOS you may need to prefix with `KMP_DUPLICATE_LIB_OK=TRUE` due to a known OpenMP conflict between PyTorch and FAISS. Add `export KMP_DUPLICATE_LIB_OK=TRUE` to your `~/.zshrc` to set it permanently.

---

## Design decisions

### Web scraping instead of PDF parsing

The original pipeline loaded the PPC from a PDF using `PyPDFLoader`. This required:
- A local copy of the PDF
- A two-pass anchor-based section extractor (300+ lines)
- A separate cleaning step to strip page markers (`<<<PAGE_N>>>`), footnote refs (`3[Pakistan]`), and omission placeholders (`* * *`)
- `load_data.py`, `parser_data.py`, `split_data.py`, `inspect_data.py`, `clean_and_rebuild.py` — five scripts for one job

The web source at `pakistani.org` provides clean, structured HTML. A single `scrape_ppc.py` replaces all five scripts, produces cleaner text, and has no local file dependency. The only downside is a network request at build time.

### Section-level granularity instead of fixed-size chunks

`RecursiveCharacterTextSplitter` (chunk_size=800, overlap=80) was tried first and discarded. A PPC section is the natural retrieval unit. When a user asks "what is the punishment for robbery?" they want Section 392, not an 800-character fragment that may cut off the penalty clause mid-sentence. Section-level chunking also makes `section_id` metadata clean and unambiguous.

### Terminology normalisation

The PPC was Islamized between 1979–1990. Key offence titles were replaced with Arabic/Urdu terms:

| English query term | PPC section title | Sections |
|---|---|---|
| Murder | Qatl-e-Amd | 300, 302 |
| Culpable homicide | Qatl-i-khata | 318, 319 |
| Accidental killing | Qatl-bis-sabab | 321, 322 |
| Blood money | Diyat | 323, 330 |
| Retaliation | Qisas | 304, 307 |
| Head/face wound | Shajjah | 337, 337A |
| Body wound | Jurh | 337B |
| Penetrating wound | Jaifah | 337C, 337D |
| Dismemberment | Itlaf-i-udw | 333, 334 |
| Organ impairment | Itlaf-i-salahiyyat-i-udw | 335, 336 |
| Abortion (early) | Isqat-i-hamal | 338, 338A |
| Abortion (late) | Isqat-i-janin | 338B, 338C |
| Pardon | Afw | 309 |
| Compounding | Sulh | 310 |
| Giving woman as settlement | Badal-i-sulh / Wanni / Swara | 310A |
| Legal heir | Wali | 305 |
| Coercion | Ikrah | 303 |
| Bone-exposing wound | Mudihah | 337, 337A |
| Bone-breaking wound | Hashimah | 337, 337A |
| Bone-displacing wound | Munaqqilah | 337, 337A |
| Brain membrane wound | Damighah | 337, 337A |

A query for "murder" contains no token that appears in Section 300 ("Qatl-e-Amd"). `normalise_sections.py` addresses this by annotating Urdu terms inline:

```
"commits qatl-i-amd"  →  "commits qatl-i-amd (intentional murder, murder, ...)"
```

The original Urdu term is preserved (legal accuracy), and the English equivalents are appended so the embedding model has both vocabularies to match against.

**Important limitation:** `normalise_sections.py` is purely dictionary-based. Any Urdu/Arabic term not explicitly listed in `term_map` passes through unchanged.

#### Why single-pass regex

The initial implementation used a loop of `re.sub` calls, one per term, longest-first. This caused cascading replacements: after `qatl-i-amd` was expanded, the shorter `qatl` pattern fired again inside the replacement. The fix is a single combined `re.compile("term1|term2|...", re.IGNORECASE)` with a callback — each position in the string is visited exactly once.

### Embedding model

`BAAI/bge-large-en-v1.5` — 1024-dimensional embeddings, retrieval-optimised, free and local.

Chosen over `all-MiniLM-L6-v2` (previous model) for significantly better retrieval quality — Hit@1 improved from 47.1% to 64.7%, MRR from 0.667 to 0.794. No API or credentials required.

### LLM

`gemma3:4b` via Ollama — runs entirely locally, no API key needed.

The LLM receives the top-3 retrieved sections as context and is instructed to answer using only those sections and cite section numbers. It does not hallucinate law outside the retrieved context.

### BM25 was tried and rejected

A hybrid BM25 + FAISS approach using Reciprocal Rank Fusion was tested. It made scores worse. Root cause: BM25 is a keyword frequency model. Queries like "Defamation under Pakistan Penal Code" cause BM25 to score sections containing "Pakistan Penal Code" literally higher than the actual defamation sections. Dense-only retrieval is better for this corpus.

---

## Retrieval performance

Evaluated on 17 hand-labelled queries split into terminology-gap (5) and standard (12) groups. Metrics: **Hit@k** (did a correct section appear in the top-k results?) and **MRR** (mean reciprocal rank).

```
  Query                                            Expected        Retrieved@5                        RR  Note
-------------------------------------------------------------------------------------------------------------------
✓  What is the punishment for qatl-e-amd?         300,302         324,302,316,312,308              0.50  terminology gap
✓  Define culpable homicide                       318,319         38,318,319,301,300               0.50  terminology gap
✓  What is accidental killing?                    318,321         318,300,301,315,80               1.00  terminology gap
✓  Blood money compensation for killing           323,330         331,330,323,337X,337Z            0.50  terminology gap
✓  Hurt and grievous hurt                         332,337L        332,337,333,367,87               1.00  terminology gap
✓  What constitutes theft?                        378,379         410,378,411,379,380              0.50
✓  Punishment for robbery                         390,392         392,393,397,394,398              1.00
✓  Definition of criminal breach of trust         405,406         406,409,405,408,407              1.00
✓  What is the punishment for rape?               375,376         376,375A,377B,354A,354           1.00
✓  Kidnapping or abduction                        359,360,362     364A,362,365,365A,359            0.50
✓  Cheating and dishonestly inducing delivery     415,420         420,206,415,462H,462I            1.00
✓  What is forgery?                               463,465         463,470,468,465,467              1.00
✓  Defamation                                     499,500         499,500,232,209,74               1.00
✗  Public servant taking bribe                    161             162,165,171B,50,163              0.00
✓  Sedition against the state                     124A            124A,123A,505,152,298            1.00
✓  What is the punishment for dacoity?            391,395         395,399,400,396,402              1.00
✓  Trespass criminal trespass                     441,447         441,447,442,451,452              1.00

--- Summary (all queries) ---
  Hit@1:  64.7%
  Hit@2:  94.1%
  Hit@3:  94.1%
  Hit@5:  94.1%
  MRR:    0.794

--- Terminology-gap queries (5) ---
  Hit@1:  40.0%
  Hit@2:  100.0%
  Hit@3:  100.0%
  Hit@5:  100.0%
  MRR:    0.700

--- Standard queries (12) ---
  Hit@1:  75.0%
  Hit@2:  91.7%
  Hit@3:  91.7%
  Hit@5:  91.7%
  MRR:    0.833
```

### Model comparison

| Metric | MiniLM-L6-v2 | BGE-large-en-v1.5 |
|---|---|---|
| Hit@1 | 47.1% | **64.7%** |
| Hit@3 | 82.4% | **94.1%** |
| Hit@5 | 94.1% | **94.1%** |
| MRR | 0.667 | **0.794** |
| Terminology-gap MRR | 0.400 | **0.700** |

---

## What is working

- **636/636 sections scraped** — correctly handles all section ID formats including hyphenated variants (`337-A`, `52-A`, `120-A`)
- **Normalisation is active** — `term_map` covers 35+ Urdu/Arabic terms including all major Islamized offence titles, wound sub-categories, and tribal settlement terms
- **Original text preserved** — the LLM receives verbatim legal text; the normalised version is internal to the embedding step only
- **Singleton vectorstore** — embedding model and FAISS index load once per session, not on every query
- **All paths use `Path(__file__).parent`** — scripts run correctly from any directory
- **Standard English queries**: 91.7% Hit@3, MRR 0.833
- **Terminology-gap queries**: 100% Hit@3 with BGE-large

---

## What is not working

### 1. "Public servant taking bribe" misses §161

BGE-large ranks §162, §165, §171B ahead of §161. The definitional section ranks just outside top-5. Not a terminology issue.

### 2. English queries for Urdu-titled sections rank low

Plain English queries like "What is punishment for murder?" retrieve §396 (dacoity with murder) instead of §302. Using the Urdu term directly ("qatl-e-amd") retrieves §302 at rank 1. The normalisation helps but doesn't fully close the gap for very short queries.

### 3. English queries only

Urdu script queries are not supported — `bge-large-en-v1.5` is English-only.

---

## Known limitations

1. **Dictionary-based normalisation** — any Urdu/Arabic term not in `term_map` is invisible to the embedding model; new terms require manual entries
2. **Static index** — adding new legal documents requires re-running the full build pipeline
3. **Single source** — only the Pakistan Penal Code is indexed; other Pakistani statutes (CrPC, Evidence Act, etc.) are not included
4. **LLM quality bounded by retrieval** — if the wrong sections are retrieved, the LLM answer will be wrong or incomplete regardless of model quality
