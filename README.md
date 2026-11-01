# QueryLens 🧠

[![CI](https://github.com/SnehaNprog/queryLens/actions/workflows/ci.yml/badge.svg)](https://github.com/SnehaNprog/queryLens/actions/workflows/ci.yml)

**Ask questions about webpages and PDFs in any language and get cited answers in that same language.**

QueryLens is a retrieval-augmented generation (RAG) app. You add a webpage or a PDF, ask a question in your own language (even if the source is written in another), and it answers only from the source, showing the exact passages it used. There is no translation API: a multilingual embedding model finds the right passages across languages, and the prompt tells the LLM to reply in the language of the question.

## Features

- 🌐 **Webpages and PDFs**: paste a URL or upload a file; add as many sources as you like
- 🌍 **Cross-lingual retrieval**: a question in Hindi can find an English passage (`BAAI/bge-m3`, which supports 100+ languages)
- 🗣 **Answers in the question's language**: the model is instructed to reply in the language of the question, whatever the source language
- 📌 **Cited answers**: every answer shows the passages it came from (with page numbers for PDFs); the cited ones come first
- 🎯 **Reranking**: a cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the shortlist, which measurably improves the top result (see [Evaluation](#evaluation))
- ⚡ **Streaming answers** from an open-weight LLM on Groq
- 💬 **Follow-up questions**: a follow-up like "and what colour is it?" is rewritten into a standalone question before searching
- 🛡 **Grounded**: if the sources don't contain the answer, the model says so instead of guessing
- 🧯 **Friendly errors**: bad URLs, 404s, unreadable PDFs and a missing or wrong API key produce a clear message, not a traceback

## How it works

```
 URL / PDF ─► loader ─► chunker ─► bge-m3 embeddings ─► FAISS index
 (BeautifulSoup4 / pypdf)  (500 chars, 100 overlap)

 Question ─► (follow-up? rewrite using chat history)
          ─► embed ─► top 20 chunks from FAISS
          ─► bge-reranker-v2-m3 ─► top 5 chunks
          ─► prompt (context + question) ─► LLM on Groq (streamed)
          ─► answer in the question's language, with [1] [2] citations
```

1. **Load**: `webloader` fetches a page and strips menus, scripts and footers; `pdfloader` reads a PDF page by page and keeps page numbers.
2. **Chunk and embed**: text is split into overlapping chunks and embedded with `BAAI/bge-m3`, which maps all languages into one vector space.
3. **Retrieve and rerank**: the question is embedded, FAISS returns the 20 closest chunks, and the reranker reads the question and each chunk together to pick the best 5.
4. **Answer**: a LangChain chain sends those chunks to the LLM with rules to answer only from them, cite them as `[1]`, `[2]`, and reply in the question's language.

## Quick start

### With Docker (one command)

```bash
git clone https://github.com/SnehaNprog/queryLens.git
cd queryLens
echo "GROQ_API_KEY=your_key_here" > .env
docker compose up --build
```

Open <http://localhost:8501>. Get a key at [console.groq.com](https://console.groq.com/). Write the line exactly as shown, with no spaces around `=` and no quotes.

The first question downloads about 4 GB of models (embedding and reranker) into a Docker volume. They are kept for later runs.

### Without Docker

Requires Python 3.12.

```bash
git clone https://github.com/SnehaNprog/queryLens.git
cd queryLens
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

### Using it

1. In the sidebar, paste a URL and click **Add webpage**, or upload a PDF and click **Add PDF**.
2. Ask a question in the chat, in any language.
3. Open **Sources** under the answer to see the passages behind it.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none (required) | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Which Groq model answers. Groq adds and retires models, so change this if the default disappears. |

## Evaluation

`python -m eval.evaluate` (run from the project root) measures retrieval on a small multilingual benchmark in [eval/](eval/): 12 short fictional documents (English, Hindi, Spanish, French, German) and 37 questions in those five languages. **32 of the 37 questions are cross-lingual** (the question is in a different language than the document). A question counts as a hit when the retrieved chunk comes from the right document and contains the answer text.

| System | hit@1 | hit@3 | MRR | Median retrieval time |
|---|---|---|---|---|
| Vector search only | 0.89 | 1.00 | 0.94 | 26 ms |
| **Vector search + reranker** (what the app uses) | **0.97** | 1.00 | **0.99** | 880 ms |

On the 32 cross-lingual questions alone: hit@1 rises from 0.88 to 0.97 with the reranker. The reranker costs roughly 0.85 s per question on a laptop CPU, with no GPU. Raw results for each question are saved in [eval/results/latest.json](eval/results/latest.json).

**Read these numbers with care.** The benchmark is small (19 chunks in total), synthetic, and was written for this project, so it shows the approach works but is not a general accuracy figure. With so few chunks, hit@3 is 1.00 for both systems, so hit@1 and MRR are the informative columns. The language of the generated answers was checked by hand on Hindi and Spanish questions; it is not part of the benchmark.

**Why there is no keyword search.** An earlier version also combined the vector search with BM25 keyword search (hybrid search). On this benchmark it scored hit@1 of about 0.41 against 0.89 for vector search alone, and added nothing once the reranker was on, so it was removed (it is in the git history). Keyword search cannot match words across languages, which is most of this benchmark. It might still help on same-language documents full of exact codes or names, which this benchmark does not test well.

## Project structure

```
queryLens/
├── app.py                         # Streamlit UI (entry point)
├── querylens/                     # the application package
│   ├── ingestion/
│   │   ├── loaders.py             # webloader, pdfloader
│   │   └── chunker.py             # split text into overlapping chunks
│   ├── retrieval/
│   │   ├── embedder.py            # bge-m3 embeddings
│   │   ├── vector_store.py        # FAISS index (save/load supported)
│   │   ├── reranker.py            # bge-reranker-v2-m3
│   │   └── retriever.py           # add sources, search + rerank
│   ├── generation/
│   │   ├── llm.py                 # Groq client
│   │   └── qa_chain.py            # prompts, citations, streaming, follow-up rewriting
│   └── errors.py                  # friendly error messages
├── eval/
│   ├── evaluate.py                # retrieval benchmark runner
│   ├── docs/, questions.json      # benchmark documents and questions
│   └── results/latest.json        # saved results
├── tests/                         # pytest suite (96 tests)
├── data/                          # sample files
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Tests

```bash
python -m pytest -q
```

The suite has 96 tests. It runs the real embedding and reranker models, so the first run downloads them. Tests that touch the web, Groq, or the Streamlit UI use fakes, so they need no network or API key. GitHub Actions runs the suite on every push.

## Limitations

- **The index lives in memory.** Sources are lost when the app restarts (saving and loading an index is implemented in `Retriever`, but the UI does not use it yet).
- **One shared collection.** The app keeps a single index for the whole server, so everyone using the same instance shares the same sources. It is meant for local use, not as a multi-user service.
- **No OCR.** Scanned PDFs (images of text) have no text to read and are rejected with a message.
- **Pages that need JavaScript** to show their content may load as empty.
- **Simple chunking.** Text is cut at a fixed length, so a passage can start mid-sentence.
- **Latency.** Reranking on a CPU adds under a second; model loading on the first question takes longer.
- **The LLM is whatever Groq currently serves.** Model availability on Groq changes over time.

## License

MIT, see [LICENSE](LICENSE).
