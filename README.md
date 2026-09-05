# jobsearch

Rank job vacancies against your CV, using an LLM. See `task.md` for the
original brief and `docs/superpowers/specs/` (or ask for the plan file) for
the full design rationale.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"       # add ".[cloud]" for Anthropic
```

For local rating and embeddings, either install [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

or point at a [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`
instance (its own OpenAI-compatible API, not Ollama's) with `JOBSEARCH_LLM=llamacpp`
and `JOBSEARCH_LLAMACPP_HOST`. Start it with `--embedding` if you also want
`JOBSEARCH_RETRIEVER=embeddings` to work.

Config is environment variables (see `.env.example`). For local CLI use, copy
it to `.env` and load it into your shell with `set -a && source .env && set +a`
— a bare `source .env` sets shell variables but does not export them, so the
`jobsearch` process would not see them.

## Docker

```bash
cp .env.example .env      # fill in JOBSEARCH_LLM and the host that serves it
docker compose build
docker compose run --rm jobsearch ingest
docker compose run --rm jobsearch doctor
```

The container is one-shot per command, not a long-running service. `./state/`
holds the SQLite database and `profile.json`, persisted across runs via a
volume mount; `./data/` (company list, golden sets) is mounted read-write too;
`./profile_storage/` holds every archived CV + extracted profile version (see
`jobsearch profiles` below).

Your LLM server runs outside this compose file. `docker-compose.yml` documents
two ways to reach it: `network_mode: host` (Linux, when llama.cpp/Ollama run on
the same machine as Docker — simplest, `localhost` just works) or the default
`host.docker.internal` + `extra_hosts` (works across a LAN, requires the server
to listen on `0.0.0.0` rather than only `127.0.0.1`).

## Usage

```bash
jobsearch ingest                    # fetch vacancies into jobsearch.db
jobsearch doctor                    # check sources, list dead company slugs
jobsearch profile ./cv.pdf          # extract a candidate profile (archived automatically)
jobsearch profiles                  # list every archived profile version
jobsearch use-profile <id>          # reactivate one (copies it onto profile.json)
jobsearch match                     # filter, shortlist, rate
jobsearch show                      # ranked results
jobsearch show --full               # with evidence for each match
jobsearch mark <vacancy-id> dismissed
jobsearch eval --retrievers bm25,embeddings   # retrieval quality vs. data/golden/
```

Sending your CV to `JOBSEARCH_LLM=anthropic` sends personal data to a third
party; `ollama` and `llamacpp` keep it local.

## Sources

`data/companies.yaml` lists Greenhouse/Lever/Ashby company slugs verified live
during design (Europe/DACH-leaning). Slugs rot — `jobsearch doctor` reports
dead ones to prune. Arbeitnow needs no list and covers ~2,000 active EU jobs
on its own.

## Tests

```bash
pytest              # 53 tests, all offline (recorded fixtures, no network, no LLM)
```

`jobsearch eval` is not part of the suite — it measures retrieval quality
against `data/golden/`, not correctness.
