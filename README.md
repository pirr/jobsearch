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
jobsearch ingest --source jobicy    # or just one: arbeitnow|remoteok|jobicy|himalayas|remotive|
                                     #   workingnomads|remotejobsorg|hackernews|adzuna|greenhouse|
                                     #   lever|ashby|smartrecruiters|workable|recruitee|teamtailor|
                                     #   personio
jobsearch doctor                    # check sources, list dead company slugs
jobsearch discover-ats "Some Company"           # guess & probe ATS slugs for a company name
jobsearch discover-ats "Some Company" --write   # ...and append live hits to companies.yaml
jobsearch profile ./cv.pdf          # extract a candidate profile (archived automatically)
jobsearch profiles                  # list every archived profile version
jobsearch use-profile <id>          # reactivate one (copies it onto profile.json)
jobsearch match                     # filter, shortlist, rate
jobsearch show                      # ranked results
jobsearch show --full               # with evidence for each match
jobsearch mark <vacancy-id> dismissed
jobsearch mark <vacancy-id> rejected          # also: seen|applied|interview|accepted|closed
                                     #   (attach a free-text reason from the TUI's `n` key)
jobsearch tui                       # interactive terminal UI: scrollable list,
                                     #   no display limit, filter/sort/status/
                                     #   duplicate/description actions in place
jobsearch eval --retrievers bm25,embeddings   # retrieval quality vs. data/golden/
```

Sending your CV to `JOBSEARCH_LLM=anthropic` sends personal data to a third
party; `ollama` and `llamacpp` keep it local.

### Interactive TUI

`jobsearch tui` replaces `show`'s hard display limit with a keyset-paginated,
scrollable list -- keep scrolling (or reaching the bottom loads more
automatically) instead of re-running `show --limit N`. Inside it:

- `/` search by title/company/location; the sort dropdown and remote-only
  switch in the filter bar re-query as you change them
- `Enter` opens a vacancy's full detail (matched/missing skills, evidence,
  blockers) -- the URL there is a clickable terminal hyperlink
- `s` / `d` / `a` mark seen / dismissed / applied in place
- `i` / `o` / `x` mark interview / accepted / rejected -- the application
  outcome, once there is one
- `c` cycles the view: active -> dismissed -> closed -> duplicates
- `m` opens the duplicate picker (search any vacancy, confirm, hide); `u` in
  the duplicates view undoes a mark
- `e` opens $EDITOR on the description (same flow as `update-description`),
  then re-rates in the background without blocking the UI
- `n` opens $EDITOR on a free-text reason/note for the vacancy (e.g. why it
  was rejected) -- mark a status first, there's nowhere to attach it to yet
  otherwise
- `r` refreshes, `q`/`Esc` goes back/quits

Under Docker: `docker compose run --rm jobsearch tui` -- no compose file
changes needed, a pseudo-TTY is allocated the same way any interactive
`docker compose run` already gets one.

## Sources

`data/companies.yaml` lists Greenhouse/Lever/Ashby/SmartRecruiters company
slugs verified live during design (Europe/DACH-leaning, plus a US/UK batch).
Slugs rot — `jobsearch doctor` reports dead ones to prune, and
`jobsearch discover-ats "Company Name" --write` finds new ones: it guesses
slug candidates from a company name and probes each ATS platform for a live
board, with no auth and no LLM tokens involved.

Workable, Recruitee, Teamtailor and Personio are wired up as ATS platforms
too; `data/companies.yaml` has a few verified Workable/Teamtailor/Personio
slugs so far and none for Recruitee — `jobsearch discover-ats --write` finds
more.

Personio is the only XML feed (`{tenant}.jobs.personio.de/xml`). A tenant
that has left Personio 307-redirects to personio.com instead of 404ing; that
redirect is treated as a dead slug, not followed.

SmartRecruiters is the odd one out among the eight ATS platforms: its listing
endpoint never inlines a job's description, so each posting needs a second,
per-job request. To keep one large company from turning into hundreds of
requests every ingest run, postings are capped (`max_postings`, default 150)
rather than fetched exhaustively.

Eight sources need no company list and cover many companies on their own:
Arbeitnow (~2,000 active EU jobs), RemoteOK (global remote tech jobs),
Jobicy (queried with `geo=europe&industry=dev`, so only European engineering
roles), Himalayas (global remote jobs, structured per-job country
restrictions, bounded cursor pagination), Remotive (global remote jobs;
its free API currently exposes only ~18 jobs, so it adds little volume),
Working Nomads (global remote jobs, one bounded request, no pagination),
RemoteJobs.org (global remote jobs; only the first page of its own paginated
feed is fetched, same bounded-slice tradeoff as Himalayas/Adzuna) and Hacker
News "Who is hiring?" (the newest monthly thread only, via the public Algolia
HN API; posts have no schema, so company/role/location are parsed best-effort
from the conventional `Company | Role | Location` header line, and posts
without such a header are skipped).

A ninth, Adzuna, is optional: unlike the others it's a general job board (not
remote-only) and needs a free App ID/Key from
[developer.adzuna.com](https://developer.adzuna.com/signup) — set
`JOBSEARCH_ADZUNA_APP_ID`/`JOBSEARCH_ADZUNA_APP_KEY` in `.env` to enable it
(see `.env.example`); left unset, it's silently skipped. Queries
`gb`/`us`/`de`/`fr`/`nl`/`pl` only — `ru` is never queried even if granted on
the account, and Belarus isn't an Adzuna market at all.

## Tests

```bash
pytest              # 198 tests, all offline (recorded fixtures, no network, no LLM)
```

`jobsearch eval` is not part of the suite — it measures retrieval quality
against `data/golden/`, not correctness.
