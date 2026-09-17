# 🎵 HarmonyRanker — Music Recommender Simulation

**Original project:** *Music Recommender Simulation*
The original goal was to represent songs and a listener's "taste profile" as structured data, design a scoring rule that turns that data into ranked recommendations, and evaluate where the rule succeeds and fails. Its capabilities were: load a small song catalog from CSV, score each song against a user profile using genre, mood, and energy, and return the top-k matches. This final version completes that implementation, adds human-readable explanations for every recommendation, adds a multi-profile demo harness, and adds a test suite.

---

## Summary

HarmonyRanker is a **retrieval-augmented music recommender with content-based reranking**. It does not listen to audio and it does not use collaborative filtering. It answers a natural-language request in two stages: embeddings retrieve a shortlist of candidate songs and the background prose that explains them, then a hand-tuned weighted scorer reranks that shortlist against genre, mood, and a numeric energy target. A language model writes the final explanation — but only about the songs the scorer already chose.

Why it matters: it is a working, fully legible model of how commercial recommenders actually behave. Every recommendation is still traceable to a specific weight in a specific line of code, which makes it a good vehicle for seeing how a single weighting decision can quietly dominate an entire system's output — the exact failure mode documented in [Testing Summary](#testing-summary) and [model_card.md](model_card.md).

**The model never picks the songs.** That division is the point of the design: retrieval and the scorer make the decision, and generation describes it. Every factual sentence the model writes must cite the retrieved passage it came from, and anything it cannot support is dropped or withheld — see [Grounding](#what-is-and-isnt-deterministic-now).

It runs with no API key at all: retrieval falls back to a deterministic offline embedder and generation to the original template explainer.

---

## Architecture Overview

The system diagram lives at [diagrams/architecture.mmd](diagrams/architecture.mmd) (Mermaid). The pipeline runs in two phases.

**Build time — once, via `python3 -m src.ingest`:**

- **Corpus** ([src/corpus.py](src/corpus.py)) — two sources become one uniform list of chunks. Each of the 203 catalog rows is rendered as a *song card*: a couple of sentences of natural language that keep every raw number verbatim, because embeddings match "high energy" far better than they match `0.82`, while the grounding check needs the digits. Alongside them, ~7,200 words of hand-written prose in [docs/kb/](docs/kb/) covering all 19 genres, all 13 moods, seven eras, and ten listening contexts.
- **Chunking** ([src/chunking.py](src/chunking.py)) — prose is split on `##` headings, packed into ~900-character windows with 150 characters of word-aligned overlap. Overlap never crosses a heading boundary; a runt final chunk is merged back rather than orphaned. Song cards are never split.
- **Embedding and storage** ([src/embeddings.py](src/embeddings.py), [src/vector_store.py](src/vector_store.py)) — 277 chunks embedded with `gemini-embedding-2` at 768 dimensions and written to `data/index/gemini_index.jsonl`, which is committed. A corpus fingerprint in the index header is how a stale index gets caught.

**Query time — per question:**

- **Input** — a natural-language query enters through the CLI runner, [src/main.py](src/main.py). `--mode classic` still accepts the original structured taste profile.
- **Retrieval** ([src/retriever.py](src/retriever.py)) — the query is embedded and matched by cosine similarity, *stratified* so song cards and prose are retrieved to separate depths (a single top-k would let 203 cards crowd the prose out entirely). This yields a 20-song shortlist plus three prose passages.
- **Reranking** — `extract_prefs()` derives a structured profile from the query text deterministically, with no model call: longest-match against the catalog's own genre and mood vocabulary, a small synonym map, an energy-hint table, and negation handling so *"nothing too energetic"* does not request energetic music. The original `evaluate_song()` then scores the shortlist exactly as it always did, and the results are ordered by a blend of similarity and weighted score.
- **Retrieved records** — each result is a `RetrievedSong` carrying the song, its score, a `SignalMatch` per signal (hit *or* miss), confidence, band, and now the retrieval similarity and chunk id it came from.
- **Generation** ([src/answerer.py](src/answerer.py), [src/llm_client.py](src/llm_client.py)) — the retrieved passages, the query, and the already-chosen song list are assembled into a prompt and sent to `gemini-flash-lite-latest`. The prompt fences the list explicitly: *do not add, drop, or substitute*.
- **Guardrails** ([src/guardrails.py](src/guardrails.py)) — now five layers. The original three (malformed rows skipped, invalid profiles rejected, template claims checked against the record) are unchanged. Two more guard model output: a sentence citing a passage that was not retrieved is dropped, and an answer naming a song the ranker did not select is withheld *in full*. A withheld answer falls back to the deterministic template explainer.
- **Logging** ([src/logging_setup.py](src/logging_setup.py)) — every stage logs to `logs/run.log` (DEBUG) with warnings surfaced on the console. `--log-level INFO` shows the pipeline narrating itself.
- **Evaluation** — 252 automated tests across twelve files, plus a per-run reliability report (average confidence, chunk similarity, citation density, fabricated citations, substituted songs, grounding rate). See [Testing Summary](#testing-summary) and [Reproducible Execution Evidence](#reproducible-execution-evidence).

The class structure is in [diagrams/uml.mmd](diagrams/uml.mmd).

### What is and isn't deterministic now

Earlier versions of this README claimed the whole system was deterministic, with "no model call, no network, no API key". That was true and is no longer. Precisely what changed:

| Stage | Deterministic? |
|---|---|
| Chunking | **Yes** — pure function of the text; pinned by tests |
| Embedding | **Yes**, given a fixed index. The committed index is a fixed artifact |
| Song *selection* | **Yes** — retrieval order plus `evaluate_song()`; no model involvement |
| Confidence and bands | **Yes** — the offline embedder deliberately does not move them |
| Explanation *wording* | **No** — this is the model's contribution, and the only non-deterministic part |
| The entire offline path | **Yes** — hashing embedder plus template explainer, end to end |

Two things keep this honest rather than aspirational. The offline path is what the **test suite exercises**: `conftest.py` installs autouse fixtures that block every socket and strip `GEMINI_API_KEY`, so a passing run is proof that nothing was sent anywhere and nothing was billed. And the run report always names which backend actually served the request, so a silent downgrade to offline can never be mistaken for a working Gemini run.

---

## Setup Instructions

**Requirements:** Python 3.10+ (the project targets 3.12). The `google-genai` SDK requires 3.10 or newer, so the older 3.8 floor no longer applies.

1. Clone and enter the repo:

```bash
git clone https://github.com/jeff-sp/applied-ai-system-final.git && cd applied-ai-system-final
```

2. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the demo from the repo root:

```bash
python3 -m src.main
```

> Run it from the repo root — the catalog is resolved at the relative path `data/songs.csv`. (`python3 src/main.py` also works.)

5. Run the test suite:

```bash
python3 -m pytest -q
```

Expected: `252 passed`. Run it with the virtualenv's interpreter — `tests/test_embeddings.py` imports `google.genai`, so a bare system Python without the dependencies installed fails 8 of them for that reason alone. The suite never touches the network — see [What is and isn't deterministic now](#what-is-and-isnt-deterministic-now).

### Running offline (no API key)

**This step is optional.** With no `GEMINI_API_KEY` set, everything above already works: retrieval uses a deterministic offline embedder built in memory in ~50 ms, and generation uses the original template explainer. The run prints a banner saying so, and the run report names both backends. Nothing is silently degraded.

### Enabling Gemini

One key covers both embeddings and generation.

```bash
cp .env.example .env      # then put your key in it: GEMINI_API_KEY=...
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `.env` is gitignored.

Then build the vector index once. This is the **only step that costs anything** — 277 embedding calls, run once, not per query:

```bash
python3 -m src.ingest --dry-run    # chunk and report sizes; zero API calls
python3 -m src.ingest              # embed 277 chunks and write the index
python3 -m src.ingest --check      # exit 0 when the index matches the corpus
```

After that, `python3 -m src.main` uses Gemini for both retrieval and generation, and each query costs one embedding call plus one generation call. Re-run ingest only when `data/songs.csv` or `docs/kb/` changes — `--check` tells you when that has happened.

> **Note on `data/index/gemini_index.jsonl`:** the index is committed, so anyone cloning the repo gets working Gemini retrieval without rebuilding it or spending anything. The three ingest commands above are only needed if you change `data/songs.csv` or `docs/kb/`. With no key set, the committed index cannot be used — a query has to be embedded by the same model that built it — so the run warns and falls back to the offline embedder rather than comparing incompatible vectors.

### CLI options

```bash
python3 -m src.main --query "something moody for a late night drive" --show-chunks
```

| Flag | Default | What it does |
|---|---|---|
| `--mode` | `rag` | `rag` is the full pipeline. `classic` runs the original pre-RAG scoring path unchanged |
| `--query` | — | Ask one natural-language question instead of running a demo profile |
| `--profile` | `all` | One of `pop`, `lofi`, `rock`, `metal`, `city pop`, `jazz`, `funk`, or `all`. Quote the ones with a space: `--profile "city pop"` |
| `-k` | `5` | How many songs to recommend |
| `--context-k` | `3` | How many prose passages are injected into the prompt |
| `--embedder` | `auto` | `auto`, `gemini`, or `offline`. `auto` uses Gemini when a key and a fresh index are both present |
| `--generator` | `auto` | `auto`, `gemini`, `template`, or `off`. `off` retrieves but generates nothing |
| `--index` | `data/index/gemini_index.jsonl` | Point at a different vector index |
| `--catalog` | `data/songs.csv` | Point at a different catalog CSV |
| `--kb-dir` | `docs/kb` | Point at a different prose knowledge base |
| `--show-chunks` | off | Print the retrieved passage ids and their similarities |
| `--log-level` | `WARNING` | Console verbosity; `logs/run.log` always gets the full DEBUG trail |

### Asking your own question

Just ask:

```bash
python3 -m src.main --query "something confident to walk into a room to" -k 3
```

The system prints what it understood the request to mean (`understood as: genre=funk · mood=confident · energy=0.62`) before the recommendations, so a misreading is visible rather than hidden.

To add a permanent demo profile, add an entry to `PROFILES` in [src/main.py](src/main.py). Each carries both representations — the natural-language `query` used by rag mode and the structured `prefs` used by classic mode:

```python
"my profile": {
    "label": "Relaxed jazz",
    "query": "relaxed jazz for a quiet evening",
    "prefs": {"favorite_genre": "jazz", "favorite_mood": "relaxed", "target_energy": 0.35},
},
```

A test asserts that `extract_prefs()` recovers the `prefs` from the `query`, so the two cannot drift apart. Anything malformed is rejected by the profile guardrail with a message naming the offending field, rather than silently producing confident nonsense.

### Interactive mode

`src.main` answers one question per process, which means every follow-up pays the startup cost again — building the corpus, loading 277 vectors, and, when the committed index cannot be used, embedding every chunk from scratch. `src.repl` builds that stack once and then loops:

```bash
python3 -m src.repl
```

```
ask> mellow lofi beats to study to, nothing too energetic
  ... recommendations ...

ask> why 1
====================================================================
  WHY #1: Staying There — L'Indécis
  lofi · chill · energy 0.35 · 72 bpm
====================================================================

  signal    points        song value vs target
  ----------------------------------------------------------------
  genre     1.50 / 1.50        lofi vs lofi        hit
  mood      0.50 / 0.50       chill vs chill       hit
  energy    1.80 / 2.00        0.35 vs 0.25        hit
  ----------------------------------------------------------------
  score     3.80 / 4.00  3 of 3 signals matched

  confidence : 0.97 (high)
               0.6 × score + 0.4 × coverage
  similarity : +0.015 — recorded, NOT folded into confidence
               this embedder's scores are not calibrated enough to trust
  retrieved  : [song:4]
```

`why <n>` computes nothing. Every recommendation is already scored through `evaluate_song()`, which records a `SignalMatch` per signal — hits *and* misses — and the normal output shows almost none of it. This is the rest of the record: per-signal points out of maximum, what the song actually held against what the profile wanted, how far the confidence sits from the next band, and the chunk the song was retrieved from. It also states plainly when retrieval similarity was recorded but *not* counted toward confidence, which is what happens on the offline path because `HashingEmbedder.contributes_confidence` is `False`.

The session accepts the same backend and catalog flags as the one-shot CLI (`-k`, `--embedder`, `--generator`, `--catalog`, `--index`, `--context-k`, `--show-chunks`, `--log-level`), but not `--mode`, `--profile`, or `--query`. A malformed question ends that turn, not the session. `quit`, `exit`, Ctrl-D, and Ctrl-C all leave with exit code 0.

One thing it deliberately does not offer is a command to change the log level mid-session: `configure_logging` is idempotent by design, so the level is fixed when the session starts and such a command would silently do nothing.

---

## Sample Interactions

Three of the seven demo profiles are shown below — real output against the 203-song catalog. Every sentence after the score line is generated by the Explanation Agent from the retrieved evidence — nothing is hard-coded per song. Each block is verbatim except for the leading `Loaded 203 songs from data/songs.csv` line and the trailing run report, which are trimmed here and shown in full under [Reproducible Execution Evidence](#reproducible-execution-evidence).

These three run in **`--mode classic`** on purpose: it is the pre-RAG scoring path with no retrieval or generation in front of it, which is what makes the weighting behaviour below readable in isolation. The default `rag` mode prints more — a `query:` line, an `understood as:` line, a per-song `similarity`, a generated `ANSWER` block — and it can change the picks, because the scorer only ever sees the shortlist retrieval hands it. On the Gemini path the pop profile returns these same three songs, with confidence shifted slightly by the similarity term (rank 3: 0.75 rather than 0.78). On the offline embedder it does not: rank 3 becomes `Sunday Morning` at 0.58 (medium), because the hashing fallback never puts `Run the World (Girls)` in the shortlist. That gap between backends is measured in [E12](#e12--retrieval-quality); the RAG path is shown end to end in [E9](#e9--the-rag-pipeline-offline).

### Example 1 — High-energy pop

**Input:** `{"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}`

```
$ python3 -m src.main --mode classic --profile pop -k 3

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.92
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Levitating  —  Dua Lipa
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.78) sits close to your target of 0.80.

  3. Run the World (Girls)  —  Beyoncé
     score 3.40 · confidence 0.78 (high)
     Strong match (confidence 0.78): recommended because it's pop, your favorite genre and its energy (0.85) sits close to your target of 0.80. Caveat: the mood is confident, not happy.
```

Rank 3 is where the genre-over-mood weighting shows itself: a `pop`/`confident` track (1.5 + 0 + 1.90 = 3.40) outranks every `indie pop`/`happy` candidate, which can only reach 0.5 + energy. Under the earlier flat 1.0/1.0 weighting those two were interchangeable. The agent still names the mood miss as a caveat rather than letting it read as a full match.

### Example 2 — Chill lofi

**Input:** `{"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3}`

```
$ python3 -m src.main --mode classic --profile lofi -k 3

====================================================================
  CHILL LOFI
  profile: genre=lofi · mood=chill · energy=0.3
  average confidence: 0.97
====================================================================

  1. Staying There  —  L'Indécis
     score 3.90 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.35) sits close to your target of 0.30.

  2. Soulful  —  L'Indécis
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.38) sits close to your target of 0.30.

  3. Aruarian Dance  —  Nujabes
     score 3.76 · confidence 0.96 (high)
     Strong match (confidence 0.96): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.42) sits close to your target of 0.30.
```

All three are full three-signal matches, and the top two are the same artist which is the no-diversity-control limitation listed below showing up in practice.

### Example 3 — Intense rock

**Input:** `{"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.7}`

```
$ python3 -m src.main --mode classic --profile rock -k 3

====================================================================
  INTENSE ROCK
  profile: genre=rock · mood=intense · energy=0.7
  average confidence: 0.84
====================================================================

  1. Search and Destroy  —  The Stooges
     score 3.62 · confidence 0.94 (high)
     Strong match (confidence 0.94): recommended because it's rock, your favorite genre, the mood is intense, exactly what you wanted and its energy (0.89) sits close to your target of 0.70.

  2. Mr. Brightside  —  The Killers
     score 3.58 · confidence 0.80 (high)
     Strong match (confidence 0.80): recommended because it's rock, your favorite genre and the mood is intense, exactly what you wanted. Caveat: its energy (0.91) is off your target of 0.70.

  3. Sweet Child O' Mine  —  Guns N' Roses
     score 3.48 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's rock, your favorite genre and its energy (0.71) sits close to your target of 0.70. Caveat: the mood is happy, not intense.
```

Rank 2 is the interesting one: it matches genre *and* mood but misses energy by 0.21 — just past the 0.20 threshold that counts energy as "close" — so the agent flags energy as the caveat while still calling it a strong match. Compare rank 3, which trades the mood match for a near-perfect energy fit and lands lower. This profile is the hardest of the seven for the catalog and still averages 0.84; at 53 songs it averaged 0.79, and at 18 songs its rank 3 was a **low-confidence 0.43** stretch pick. That is not straightforwardly an improvement — see [Testing Summary](#testing-summary).

The low band never fires for any of the seven demo profiles — not for `metal`/`angry` (0.90 average, 5 of 5 high) and not for the thinnest ones. It takes a *double* miss to reach it: `polka` — a genre with **zero** songs in the catalog — still returns 5 of 5 *medium* at 0.63 when the requested mood exists, and only drops to 5 of 5 *low* at 0.43 when the mood is absent too. Both cases are captured under [Uncovered genre](#e6--uncovered-genre-polka).

---

## Reproducible Execution Evidence

Every block in this section is verbatim captured output from this repo — nothing is retyped or tidied. Every command here runs on a deterministic path: `--mode classic`, the offline embedder, or the template explainer, none of which makes a model call or touches the network. So re-running any command reproduces its block exactly, apart from the timestamps in `logs/run.log` and pytest's own timing line. The one thing this section cannot show reproducibly is Gemini's *wording*, which is why no block records it — see [What is and isn't deterministic now](#what-is-and-isnt-deterministic-now).

Reading the blocks:

- A line starting with `$` is the command; everything under it is that command's output.
- `[exit N]` is `echo $?` immediately after the command. `0` = ran, `1` = a fatal condition handled cleanly.
- Log output goes to **stderr**, program output to **stdout**. Blocks were captured with `2>&1`, so log lines may appear grouped ahead of the stdout they relate to; in a live terminal they interleave.
- Captured on Python 3.12.13 (`python3 --version`), macOS, from the repo root, inside the virtualenv from [Setup](#setup-instructions).

### E1 — Test suite

```bash
python3 -m pytest -q
```

```
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 85%]
....................................                                     [100%]
252 passed in 21.55s
[exit 0]
```

252 tests across 12 files. The timing figure is the only part of this block that varies between runs.

A passing run also proves the suite is hermetic. `conftest.py` installs two autouse fixtures: one replaces `socket.connect`, `socket.connect_ex`, and `socket.create_connection` with a raiser, and one deletes `GEMINI_API_KEY` from the environment. So these tests cannot have reached the Gemini API, cannot have been billed, and produce identical results on a machine with a key and one without.

### E2 — Full demo run (all seven profiles)

```bash
python3 -m src.main --mode classic -k 5
```

This is the run behind every headline number in the [Testing Summary](#testing-summary). Full transcript is 213 lines; the first profile, one middle profile and the run report are shown, with the cuts marked.

```
$ python3 -m src.main --mode classic -k 5

Loaded 203 songs from data/songs.csv

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.86
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Levitating  —  Dua Lipa
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.78) sits close to your target of 0.80.

  3. Run the World (Girls)  —  Beyoncé
     score 3.40 · confidence 0.78 (high)
     Strong match (confidence 0.78): recommended because it's pop, your favorite genre and its energy (0.85) sits close to your target of 0.80. Caveat: the mood is confident, not happy.

  4. Higher Love  —  Kygo, Whitney Houston
     score 3.34 · confidence 0.77 (high)
     Strong match (confidence 0.77): recommended because it's pop, your favorite genre and its energy (0.72) sits close to your target of 0.80. Caveat: the mood is hopeful, not happy.

  5. Don't Stop the Music  —  Rihanna
     score 3.30 · confidence 0.76 (high)
     Strong match (confidence 0.76): recommended because it's pop, your favorite genre and its energy (0.90) sits close to your target of 0.80. Caveat: the mood is energetic, not happy.

====================================================================

… (cut: CHILL LOFI, INTENSE ROCK — 57 lines)

====================================================================
  ANGRY METAL
  profile: genre=metal · mood=angry · energy=0.9
  average confidence: 0.90
====================================================================

  1. Angel of Death  —  Slayer
     score 3.88 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.96) sits close to your target of 0.90.

  2. Master of Puppets  —  Metallica
     score 3.86 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.97) sits close to your target of 0.90.

  3. Raining Blood  —  Slayer
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.98) sits close to your target of 0.90.

  4. Rise Today  —  Alter Bridge
     score 3.50 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's metal, your favorite genre and its energy (0.90) sits close to your target of 0.90. Caveat: the mood is hopeful, not angry.

  5. Hallowed Be Thy Name  —  Iron Maiden
     score 3.46 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's metal, your favorite genre and its energy (0.88) sits close to your target of 0.90. Caveat: the mood is moody, not angry.

====================================================================

… (cut: DRIVEABLE CITY POP, HIGH-ENERGY JAZZ, CONFIDENT FUNK — 85 lines)

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  mode                  : classic
  queries served        : 7
  recommendations made  : 35
  average confidence    : 0.88
  low-confidence picks  : 0/35
  claims made / dropped : 105 / 0
  explanations withheld : 0
  grounding rate        : 1.00
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

Note ranks 3–5 of the pop block: three genre-only picks whose mood misses, every one of them still labelled "Strong match". That is the confidence-threshold failure described in [What didn't work](#what-didnt-work), visible in the output rather than asserted.

### E3 — Pipeline narrating itself

```bash
python3 -m src.main --mode classic --profile lofi -k 3 --log-level INFO
```

```
$ python3 -m src.main --mode classic --profile lofi -k 3 --log-level INFO
[INFO] music_rec.recommender: Loading catalog from data/songs.csv
[INFO] music_rec.recommender: Loaded 203 song(s), skipped 0
[INFO] music_rec.retriever: Retriever ready: 203 songs from data/songs.csv across 19 genre(s)
[INFO] music_rec.retriever: Query: genre=lofi mood=chill target_energy=0.30 k=3
[INFO] music_rec.main: Run complete: 3 recommendation(s), avg confidence 0.975, grounding rate 1.000

Loaded 203 songs from data/songs.csv

====================================================================
  CHILL LOFI
  profile: genre=lofi · mood=chill · energy=0.3
  average confidence: 0.97
====================================================================

  1. Staying There  —  L'Indécis
     score 3.90 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.35) sits close to your target of 0.30.

  2. Soulful  —  L'Indécis
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.38) sits close to your target of 0.30.

  3. Aruarian Dance  —  Nujabes
     score 3.76 · confidence 0.96 (high)
     Strong match (confidence 0.96): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.42) sits close to your target of 0.30.

====================================================================

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  mode                  : classic
  queries served        : 1
  recommendations made  : 3
  average confidence    : 0.97
  low-confidence picks  : 0/3
  claims made / dropped : 9 / 0
  explanations withheld : 0
  grounding rate        : 1.00
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

### E4 — The DEBUG trail in `logs/run.log`

The console shows warnings only by default; the file always gets everything, including the per-rank scoring trace.

```bash
rm -f logs/run.log && python3 -m src.main --mode classic --profile metal -k 2 > /dev/null && cat logs/run.log
```

```
2026-08-03 19:50:44,947 INFO     music_rec.recommender | Loading catalog from data/songs.csv
2026-08-03 19:50:44,949 INFO     music_rec.recommender | Loaded 203 song(s), skipped 0
2026-08-03 19:50:44,949 INFO     music_rec.retriever | Retriever ready: 203 songs from data/songs.csv across 19 genre(s)
2026-08-03 19:50:44,949 INFO     music_rec.retriever | Query: genre=metal mood=angry target_energy=0.90 k=2
2026-08-03 19:50:44,951 DEBUG    music_rec.retriever |   #1 Angel of Death (metal) score=3.88 confidence=0.98 band=high
2026-08-03 19:50:44,951 DEBUG    music_rec.retriever |   #2 Master of Puppets (metal) score=3.86 confidence=0.98 band=high
2026-08-03 19:50:44,951 INFO     music_rec.main | Run complete: 2 recommendation(s), avg confidence 0.980, grounding rate 1.000
```

Timestamps are the only non-reproducible values anywhere in this section.

### E5 — Reliability sweep across all seven profiles

Where the "15 of 35 genre-only, 0 flagged" figure comes from. The sweep runs the same `Retriever` and `ExplanationAgent` the CLI uses, and counts bands and signal hits per profile.

```bash
python3 - <<'PY'
from collections import Counter
from src.explainer import ExplanationAgent
from src.logging_setup import configure_logging
from src.main import PROFILES
from src.retriever import Retriever
configure_logging(console_level="ERROR")

retriever, agent = Retriever.from_csv("data/songs.csv"), ExplanationAgent()
print(f"{'profile':<10} {'avg':>5} {'high':>5} {'med':>4} {'low':>4} {'genre-only':>11}")
totals, genre_only_total = [], 0
for name, profile in PROFILES.items():
    results = retriever.retrieve(profile["prefs"], k=5)
    for r in results:
        agent.explain(profile["prefs"], r)
    bands = Counter(r.band for r in results)
    matched = [{s.name for s in r.breakdown.signals if s.matched} for r in results]
    genre_only = sum(1 for m in matched if m in ({"genre", "energy"}, {"genre"}))
    genre_only_total += genre_only
    totals.extend(r.confidence for r in results)
    print(f"{name:<10} {retriever.average_confidence(results):>5.2f} {bands['high']:>5} "
          f"{bands['medium']:>4} {bands['low']:>4} {genre_only:>11}")
print(f"{'ALL':<10} {sum(totals)/len(totals):>5.2f} {'':>5} {'':>4} "
      f"{sum(1 for c in totals if c < 0.45):>4} {genre_only_total:>11}")
print()
print("recommendations:", len(totals), "| claims:", agent.stats["claims_made"],
      "| dropped:", agent.stats["claims_dropped"], "| grounding:", agent.grounding_rate(),
      "| withheld:", agent.stats["withheld"])
PY
```

```
profile      avg  high  med  low  genre-only
pop         0.86     5    0    0           3
lofi        0.90     5    0    0           2
rock        0.82     5    0    0           3
metal       0.90     5    0    0           2
city pop    0.90     5    0    0           2
jazz        0.95     5    0    0           1
funk        0.85     5    0    0           2
ALL         0.88               0          15

recommendations: 35 | claims: 105 | dropped: 0 | grounding: 1.0 | withheld: 0
[exit 0]
```

35 of 35 land in the `high` band. 15 of those 35 matched genre and energy but **not** the requested mood, and none was flagged — the detector's recall on its own failure mode is zero. `rock` (0.82) is the weakest profile, `jazz` (0.95) the strongest.

### E6 — Uncovered genre (`polka`)

`polka` has zero songs in the catalog. The retriever says so; the confidence layer's reaction depends on whether the *mood* is covered too.

```bash
python3 - <<'PY'
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="WARNING")

prefs = {"favorite_genre": "polka", "favorite_mood": "cheerful", "target_energy": 0.6}
retriever = Retriever.from_csv("data/songs.csv")
print("input:", prefs)
for rank, record in enumerate(retriever.retrieve(prefs, k=5), 1):
    print(f"  {rank}. {record.song['title']} ({record.song['genre']}) "
          f"score {record.score:.2f} confidence {record.confidence:.2f} ({record.band})")
PY
```

```
[WARNING] music_rec.retriever: No song in data/songs.csv has genre 'polka' - results will lean on energy alone
[WARNING] music_rec.retriever: Every result for genre=polka mood=cheerful is low confidence - the catalog does not really serve this profile
input: {'favorite_genre': 'polka', 'favorite_mood': 'cheerful', 'target_energy': 0.6}
  1. Four (jazz) score 2.00 confidence 0.43 (low)
  2. Yes Sir, That's My Baby (jazz) score 2.00 confidence 0.43 (low)
  3. Midnight Pretenders (city pop) score 2.00 confidence 0.43 (low)
  4. Wait for the Moment (funk) score 2.00 confidence 0.43 (low)
  5. Cigarette Daydreams (indie rock) score 2.00 confidence 0.43 (low)
[exit 0]
```

Both guardrail warnings fire and all five picks are correctly banded `low`. But swap the mood for one the catalog *does* contain, or just move the requested energy, and the bands move with it — same catalog, same weights, same missing genre:

```bash
python3 - <<'PY'
from collections import Counter
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="CRITICAL")

retriever = Retriever.from_csv("data/songs.csv")
probes = [
    ("polka", "cheerful", 0.6), ("polka", "happy", 0.6), ("polka", "energetic", 0.7),
    ("ska", "energetic", 0.3), ("ska", "energetic", 0.5), ("ska", "energetic", 0.7),
    ("classical", "relaxed", 0.3), ("classical", "relaxed", 0.8),
]
for genre, mood, energy in probes:
    results = retriever.retrieve(
        {"favorite_genre": genre, "favorite_mood": mood, "target_energy": energy}, k=5)
    bands = Counter(r.band for r in results)
    print(f"{genre:<9} {mood:<9} energy={energy}  avg={retriever.average_confidence(results):.2f}  "
          f"{dict(bands)}")
PY
```

```
polka     cheerful  energy=0.6  avg=0.43  {'low': 5}
polka     happy     energy=0.6  avg=0.63  {'medium': 5}
polka     energetic energy=0.7  avg=0.64  {'medium': 5}
ska       energetic energy=0.3  avg=0.60  {'medium': 5}
ska       energetic energy=0.5  avg=0.69  {'high': 2, 'medium': 3}
ska       energetic energy=0.7  avg=0.85  {'high': 5}
classical relaxed   energy=0.3  avg=0.83  {'high': 5}
classical relaxed   energy=0.8  avg=0.60  {'medium': 5}
[exit 0]
```

Two readings of the same run. First, a single mood match plus a decent energy fit is enough to lift a *completely off-genre* result out of the `low` band: `polka` is unavailable at any price, and the system says `medium`. The retriever knows the genre is missing and logs it; the confidence score never sees that fact. Second, `ska` and `classical` move across two bands on `target_energy` alone — the songs, the weights and the genre coverage are identical in every row. That is the "confidence is relative to the catalog, not to the user" limitation, measured.

### E7 — Guardrail results

| # | Guardrail | Trigger | Observed | Exit |
|---|---|---|---|---|
| G1 | Malformed catalog rows | `--catalog data/broken.csv` | 3 rows skipped with line + reason, run continues on the 3 valid ones | 0 |
| G2 | Missing catalog | `--catalog data/nope.csv` | one-line error, no traceback | 1 |
| G3 | No valid rows | catalog where every row is corrupt | rows skipped, then a fatal `contained no valid songs` | 1 |
| G4 | Invalid `k` | `-k 0` | rejected before any work is done | 1 |
| G5 | Invalid profile | missing key / non-numeric energy / empty genre | `ProfileError` naming the offending field | n/a |
| G6 | Out-of-range energy | `target_energy=5.0` | clamped to 1.0, warning logged, run continues | n/a |
| G7 | Ungrounded claim | evidence desynchronised from the song | claim dropped; all claims corrupt ⇒ explanation withheld | n/a |

**G1 — malformed rows are skipped, not fatal.** `data/broken.csv` holds 6 rows: an out-of-range energy, a row missing artist and mood, an unparseable energy value, and 3 good rows.

```bash
python3 -m src.main --mode classic --catalog data/broken.csv --profile pop -k 3
```

```
$ python3 -m src.main --mode classic --catalog data/broken.csv --profile pop -k 3
[WARNING] music_rec.guardrails: data/broken.csv: skipped 3 malformed row(s); first: line 5: energy=9.9 is outside the 0.0-1.0 range

Loaded 3 songs from data/broken.csv

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.53
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Mr. Brightside  —  The Killers
     score 1.78 · confidence 0.40 (low)
     Weak match (confidence 0.40): recommended because its energy (0.91) sits close to your target of 0.80. Caveat: it's rock, not the pop you asked for; the mood is intense, not happy. Treat this as a stretch pick rather than a real match.

  3. Aruarian Dance  —  Nujabes
     score 1.24 · confidence 0.19 (low)
     Weak match (confidence 0.19): nothing in your profile lines up with this song. Caveat: it's lofi, not the pop you asked for; the mood is chill, not happy; its energy (0.42) is off your target of 0.80. Treat this as a stretch pick rather than a real match.

====================================================================

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  mode                  : classic
  queries served        : 1
  recommendations made  : 3
  average confidence    : 0.53
  low-confidence picks  : 2/3
  claims made / dropped : 9 / 0
  explanations withheld : 0
  grounding rate        : 1.00
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

Two things at once: the catalog guardrail degrades instead of collapsing, **and** the confidence layer works exactly as designed when the catalog is genuinely thin — 2 of 3 picks flagged `low`, both worded as stretch picks with the mismatch spelled out. The band is not broken; the 203-song catalog simply never puts it under strain.

**G2 — missing catalog.**

```bash
python3 -m src.main --catalog data/nope.csv
```

```
$ python3 -m src.main --catalog data/nope.csv
[ERROR] music_rec.main: Catalog load failed: Catalog not found at data/nope.csv - run from the repo root
Error: Catalog not found at data/nope.csv - run from the repo root
[exit 1]
```

**G3 — a catalog where nothing survives validation.** Build a two-row catalog, both rows broken (row 1 has no artist, row 2 has `energy=7.7`):

```bash
printf 'id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n1,Ghost Track,,pop,happy,0.5,120,0.5,0.5,0.5\n2,Bad Energy,Someone,pop,happy,7.7,120,0.5,0.5,0.5\n' > data/all_invalid.csv && python3 -m src.main --catalog data/all_invalid.csv; echo "[exit $?]"; rm data/all_invalid.csv
```

```
[WARNING] music_rec.guardrails: data/all_invalid.csv: skipped 2 malformed row(s); first: line 2: missing value(s) for artist
[ERROR] music_rec.main: Catalog load failed: data/all_invalid.csv contained no valid songs (2 row(s) rejected)
Error: data/all_invalid.csv contained no valid songs (2 row(s) rejected)
[exit 1]
```

**G4 — invalid `k`.**

```bash
python3 -m src.main -k 0
```

```
$ python3 -m src.main -k 0
Error: -k must be a positive integer.
[exit 1]
```

**G5 / G6 — profile validation and energy clamping.** Four inputs through `validate_user_prefs()`: a missing key, a non-numeric energy, an empty genre, and a valid-but-out-of-range profile.

```bash
python3 - <<'PY'
from src.guardrails import validate_user_prefs, ProfileError
from src.logging_setup import configure_logging
configure_logging(console_level="WARNING")

cases = [
    {"favorite_mood": "happy", "target_energy": 0.8},
    {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": "loud"},
    {"favorite_genre": "", "favorite_mood": "happy", "target_energy": 0.8},
    {"favorite_genre": "POP ", "favorite_mood": "Happy", "target_energy": 5.0},
]
for prefs in cases:
    print("input :", prefs)
    try:
        print("output:", validate_user_prefs(prefs))
    except ProfileError as exc:
        print("output: ProfileError:", exc)
    print()
PY
```

```
[WARNING] music_rec.guardrails: target_energy=5.0 is outside 0.0-1.0; clamping to 1.0
input : {'favorite_mood': 'happy', 'target_energy': 0.8}
output: ProfileError: profile is missing required key(s): favorite_genre

input : {'favorite_genre': 'pop', 'favorite_mood': 'happy', 'target_energy': 'loud'}
output: ProfileError: profile.target_energy must be numeric, got 'loud'

input : {'favorite_genre': '', 'favorite_mood': 'happy', 'target_energy': 0.8}
output: ProfileError: profile.favorite_genre must be a non-empty string, got ''

input : {'favorite_genre': 'POP ', 'favorite_mood': 'Happy', 'target_energy': 5.0}
output: {'favorite_genre': 'pop', 'favorite_mood': 'happy', 'target_energy': 1.0}

[exit 0]
```

Three fatal cases name the exact offending field; the fourth is recoverable, so `target_energy` is clamped to 1.0 and the case of the strings normalised — with the clamp logged rather than applied silently.

**G7 — fault injection: ungrounded claims.** This is the grounding guardrail under deliberate attack. A real retrieved record is fetched, then its evidence is desynchronised from the song it describes — first one signal, then all of them.

```bash
python3 - <<'PY'
from src.explainer import ExplanationAgent
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="ERROR")

prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
record = Retriever.from_csv("data/songs.csv").retrieve(prefs, k=1)[0]
agent = ExplanationAgent()

print("song   :", record.song["title"], "| genre:", record.song["genre"], "| mood:", record.song["mood"])
print("normal :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print()

# Fault injection 1: desynchronise one signal from the song it describes.
record.breakdown.signals[0].song_value = "polka"
print("after mutating the genre signal to 'polka' (song is still pop):")
print("output :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print()

# Fault injection 2: desynchronise every signal.
for signal in record.breakdown.signals:
    signal.song_value = "nonsense" if isinstance(signal.song_value, str) else -1.0
print("after corrupting every signal:")
print("output :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print("grounding rate:", agent.grounding_rate())
PY
```

```
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': genre='polka' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': genre='nonsense' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': mood='nonsense' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': energy=-1.0 not supported by the record
[ERROR] music_rec.explainer: Withholding explanation for 'Can't Stop the Feeling!': no claim survived grounding
song   : Can't Stop the Feeling! | genre: pop | mood: happy
normal : Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.
stats  : {'explanations': 1, 'claims_made': 3, 'claims_dropped': 0, 'withheld': 0}

after mutating the genre signal to 'polka' (song is still pop):
output : Strong match (confidence 0.99): recommended because the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.
stats  : {'explanations': 2, 'claims_made': 6, 'claims_dropped': 1, 'withheld': 0}

after corrupting every signal:
output : No verifiable evidence for this pick, so no explanation is offered.
stats  : {'explanations': 3, 'claims_made': 9, 'claims_dropped': 4, 'withheld': 1}
grounding rate: 0.556
[exit 0]
```

The system never says "polka". The unsupported claim is dropped and the sentence is rebuilt from what remains; when nothing survives, the explanation is withheld rather than invented, and the run report's grounding rate falls from 1.00 to 0.556 so the damage is visible in the metrics.

### E8 — Exit codes

| Command | Exit |
|---|---|
| `python3 -m src.main -k 5` | 0 |
| `python3 -m src.main --catalog data/broken.csv --profile pop -k 3` | 0 |
| `python3 -m src.main --catalog data/nope.csv` | 1 |
| `python3 -m src.main --catalog data/all_invalid.csv` | 1 |
| `python3 -m src.main -k 0` | 1 |
| `python3 -m pytest -q` | 0 |

No failure path produces a traceback; every one prints a single line naming what went wrong.

### E9 — The RAG pipeline, offline

Retrieval is the in-memory hashing embedder; generation is the template explainer. The flags force that path so the block is reproducible with or without a key — with no key set, plain `--profile "city pop" -k 3 --show-chunks` reaches the same place on its own, after a warning that the committed Gemini index cannot serve an offline query.

```bash
python3 -m src.main --profile "city pop" -k 3 --show-chunks --embedder offline --generator template
```

```
====================================================================
  DRIVEABLE CITY POP
  query: "nostalgic city pop for a late night drive"
  understood as: genre=city pop · mood=nostalgic · energy=0.45
  average confidence: 0.79
====================================================================

  retrieved context:
    +0.129  [listening_contexts:late-night-driving:0] Late-night driving
    +0.124  [genres:city-pop:0] City pop
    +0.123  [genres:folk:1] Folk

  1. Mayonaka no Door~stay with me  —  Miki Matsubara
     score 3.50 · confidence 0.79 (high) · similarity 0.117

  2. Plastic Love  —  Mariya Takeuchi
     score 3.42 · confidence 0.78 (high) · similarity 0.124

  3. Fly-Day Chinatown  —  Nanako Sato
     score 3.46 · confidence 0.79 (high) · similarity 0.117

… (cut: the template ANSWER block and the run report — 37 lines)
```

Three things are worth reading off this. The free-text query was resolved to a structured profile with no model call. Retrieval surfaced the *late-night driving* context passage, not just the genre one — the prose knowledge base is doing real work. And the top passage similarities (`0.129`, `0.124`) exceed the song-card ones (`0.117`) because prose chunks share more vocabulary with a conversational query than a templated card does; that asymmetry is exactly why retrieval is stratified by kind rather than run as a single top-k.

The third passage, `genres:folk:1`, is a genuine miss — the lexical fallback matching on shared words like "unhurried" and "drives". Gemini embeddings do not make that mistake: the same command with a key and the committed index retrieves `genres:city-pop:1`, `listening_contexts:late-night-driving:0` and `genres:city-pop:0` at similarities of 0.806 / 0.771 / 0.755, against 0.117–0.129 offline. The songs are the same three; the *evidence* behind them is not. That block is not recorded here because its generated prose is not reproducible.

### E10 — The grounding guardrail catching a misbehaving model

`BrokenGenerator` ([src/llm_client.py](src/llm_client.py)) deliberately produces what a real model can produce on a bad day. Both cases are run against a real retrieved record, on the offline embedder so the record is reproducible.

```bash
python3 - <<'PY'
from src.answerer import AnswerAgent
from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import HashingEmbedder
from src.llm_client import BrokenGenerator
from src.logging_setup import configure_logging
from src.retriever import Retriever, SemanticRetriever
from src.vector_store import VectorStore
configure_logging(console_level="ERROR")

QUERY = "nostalgic city pop for a late night drive"
retriever = Retriever.from_csv("data/songs.csv")
embedder = HashingEmbedder()
chunks = build_corpus("data/songs.csv", "docs/kb")
store = VectorStore.build(chunks, embedder, corpus_fingerprint(chunks))
records, prose, prefs = SemanticRetriever(retriever, store, embedder).retrieve(QUERY, k=1)
titles = [s["title"] for s in retriever.songs]

for label, gen in [("fabricated citation", BrokenGenerator()),
                   ("substituted song", BrokenGenerator("Master of Puppets"))]:
    agent = AnswerAgent(generator=gen)
    answer = agent.answer(QUERY, prose, records, prefs, catalog_titles=titles)
    stats = agent.stats
    print(f"-- {label} --")
    print("  model wrote :", gen.generate(""))
    print(f"  verdict     : withheld={stats['withheld']} "
          f"fabricated={stats['fabricated_citations']} "
          f"substituted={stats['substituted_titles']} fell_back={answer.fell_back}")
    print("  shown       :", answer.text[:75] + "...")
PY
```

```
[ERROR] music_rec.guardrails: Dropping sentence citing unretrieved chunk(s) song:9999: 'This song is a great pick for you [song:9999].'
[ERROR] music_rec.guardrails: Withholding generated answer: only 0/2 sentences were cited (minimum ratio 0.6)
[ERROR] music_rec.guardrails: Withholding generated answer: answer named song(s) the ranker did not select: Master of Puppets
-- fabricated citation --
  model wrote : This song is a great pick for you [song:9999]. It has excellent vibes.
  verdict     : withheld=1 fabricated=1 substituted=0 fell_back=True
  shown       : Mayonaka no Door~stay with me — Strong match (confidence 0.79): recommended...

-- substituted song --
  model wrote : Strong match: you should listen to "Master of Puppets", which is a perfect fit [song:9999].
  verdict     : withheld=1 fabricated=0 substituted=1 fell_back=True
  shown       : Mayonaka no Door~stay with me — Strong match (confidence 0.79): recommended...
[exit 0]
```

The three log lines are the decisions themselves: the fabricated citation costs that sentence, which then leaves too few cited sentences to keep the answer at all, and the substitution is refused outright.

The two failures are treated differently on purpose. A fabricated citation drops *that sentence* — one bad provenance marker does not necessarily poison the rest. A substituted song withholds the *entire answer*, because the deterministic ranker choosing the recommendations is the system's central claim, and a model quietly swapping one in is precisely the thing that must not be salvageable by editing a sentence. In both cases the user still gets a correct, grounded explanation from the template fallback rather than an error.

### E11 — Cache invalidation

One row appended to `data/songs.csv`, nothing else changed:

```bash
python3 -m src.ingest --check
```

```
STALE: data/index/gemini_index.jsonl was built from a different corpus.
  index fingerprint : 120cd06939e8b095... (277 vectors)
  corpus fingerprint: 9c9b4a435a02bbf8... (278 chunks)
  Rebuild with: python3 -m src.ingest
[exit 1]
```

`python3 -m src.main` in the same state prints a warning naming the drift and **falls back to the offline embedder** rather than querying a stale index. It deliberately does not rebuild automatically: silently triggering paid API calls as a side effect of editing a CSV is a worse failure than a loud warning.

The fingerprint covers chunk ids, chunk text, *and* the prefix-scheme version. That last part matters — changing `QUERY_PREFIX` alters what gets embedded without altering any chunk's text, and would otherwise leave a subtly mismatched index looking fresh.

Against the committed index, `--check` reports the healthy case:

```bash
python3 -m src.ingest --check
```

```
[INFO] music_rec.recommender: Loading catalog from data/songs.csv
[INFO] music_rec.recommender: Loaded 203 song(s), skipped 0
[INFO] music_rec.corpus: Corpus: 203 song card(s) + 74 prose chunk(s) = 277 total
OK: data/index/gemini_index.jsonl matches the corpus (277 vectors, gemini-embedding-2@768)
[exit 0]
```

### E12 — Retrieval quality

The two-stage design depends on exactly one property: **the requested genre has to appear in the 20-song shortlist**, because the reranker can only reorder what retrieval hands it. This is the measurement of that property, on the offline embedder — the weaker of the two backends, so it is a floor rather than a best case.

```bash
python3 - <<'PY'
from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import HashingEmbedder
from src.logging_setup import configure_logging
from src.main import PROFILES
from src.recommender import load_songs
from src.vector_store import VectorStore
configure_logging(console_level="CRITICAL")

songs = load_songs("data/songs.csv")
chunks = build_corpus("data/songs.csv", "docs/kb")
embedder = HashingEmbedder()
store = VectorStore.build(chunks, embedder, corpus_fingerprint(chunks))
by_id = {s["id"]: s for s in songs}

def shortlist(query):
    hits, _ = store.search_stratified(embedder.embed_query(query), k_songs=20, k_prose=3)
    return [by_id[h.chunk.metadata["song_id"]] for h in hits
            if h.chunk.metadata.get("song_id") in by_id]

genres = sorted({s["genre"] for s in songs})
recall = sum(1 for g in genres if any(s["genre"] == g for s in shortlist(f"{g} music")))
top1 = sum(1 for g in genres if shortlist(f"{g} music")[0]["genre"] == g)
demo = sum(1 for p in PROFILES.values()
           if any(s["genre"] == p["prefs"]["favorite_genre"] for s in shortlist(p["query"])))

print(f"recall@20, one query per genre : {recall}/{len(genres)}")
print(f"recall@20, seven demo queries  : {demo}/{len(PROFILES)}")
print(f"top-1 by genre                 : {top1}/{len(genres)}")
PY
```

```
recall@20, one query per genre : 19/19
recall@20, seven demo queries  : 7/7
top-1 by genre                 : 14/19
[exit 0]
```

Recall is perfect and top-1 is not, which is the intended shape. Top-1 is *deliberately* not optimised: the reranker decides the final order, so retrieval only has to get the right songs into the room, and 5 lexical confusions in the lead position cost nothing downstream.

Swapping `HashingEmbedder()` for `GeminiEmbedder()` and `VectorStore.build(...)` for `VectorStore.load("data/index/gemini_index.jsonl")` runs the same measurement against the committed index. That scores **19/19 recall and 19/19 top-1** — it costs 19 embedding calls and its exact figures depend on a model alias that moves, which is why the reproducible block above is the offline one. The gap between 14/19 and 19/19 is the concrete value of setting a key.

---

## Cost, latency, and what's committed

The vector index at `data/index/gemini_index.jsonl` **is committed**, which is an unusual choice for a generated artifact and a deliberate one.

| | Cost |
|---|---|
| Fresh clone, offline | Zero. Index built in memory in ~50 ms |
| Fresh clone, with a key and a committed index | Zero to set up |
| Building the index | 277 embedding calls, **once** |
| One query, thereafter | **1 embedding call + 1 generation call** |
| Re-ingest | Only when `data/songs.csv` or `docs/kb/` changes |

Separating ingest from query is what keeps the per-question cost at one embedding call. Without it, every query would re-embed the whole corpus — 277 calls per question instead of one. Committing the result extends that saving to anyone who clones the repo.

The trade is a 2.3 MB generated file in git and a large diff whenever ingest re-runs. For a project whose point is that a grader can clone it and see real retrieval work without setting up billing, that is the right side of the trade.

**Why there is no numpy.** Retrieval is 277 chunks × 768 dimensions ≈ 210,000 multiply-adds per query, measured at 20–30 ms in pure Python — invisible next to a ~300 ms network round trip. numpy would add a 15 MB wheel and a platform-specific install failure mode to save time that is not on the critical path, and would replace a legible loop in [src/vector_store.py](src/vector_store.py) with an opaque one. The measurement is the reason, not the aesthetic.

---

## Design Decisions

**Content-based, not collaborative.** There is one user and no interaction history that survives a process, so there is nothing to collaboratively filter. `src.repl` keeps the last result set in memory so `why` can index into it, but nothing is written down and nothing carries across sessions. Content-based scoring is also fully inspectable, which is a requirement for the explanation feature — and it is what makes `why` a formatter over evidence rather than a second opinion about the pick.

**Categorical vs. continuous fields are scored differently.** `genre` and `mood` can only match or miss, so they earn a flat bonus. `energy` lives on a 0–1 scale where "close" is meaningfully different from "far," so it earns partial credit via linear closeness: `1 - abs(song.energy - target_energy)`. Because energy is already normalized to 0–1, this needs no scaling and is guaranteed bounded, so it composes cleanly with the flat bonuses.

**The weights, and the ones I changed.** The Phase 2 design (see [docs/algorithm_recipe.md](docs/algorithm_recipe.md)) was *genre-first*: genre +2.0, mood +1.0, energy ×1.0. I first shipped an *energy-first* variant — genre +1.0, mood +1.0, energy ×2.0 — deliberately, as an experiment, and kept it because what it revealed about the system was more interesting than the safer configuration. That experiment flattened genre and mood into a single interchangeable tie-breaker, which is wrong: a listener who asks for rock would rather hear a *confident* rock track than an *intense* funk one. The current weights restore the ranking between the two categoricals while keeping energy's granularity: **genre +1.5, mood +0.5, energy ×2.0**. A perfect song caps at 4.0 under all three configurations, so the pinned scoring tests still assert exact numbers.

**Trade-offs I accepted:**

| Decision | Gained | Gave up |
|---|---|---|
| Energy weighted ×2.0 | Smooth, granular ranking; no large blocks of tied scores | Genre/mood demoted to tie-breakers; off-genre songs surface as soon as a genre has fewer songs than `k` |
| Genre +1.5 vs. mood +0.5 | Right-genre/wrong-mood (3.5) always beats wrong-genre/right-mood (2.5) | A mood match alone is now nearly invisible next to energy; mood barely moves the order |
| Score only 3 of 10 fields in v1 | Readable math, explanations a non-programmer can follow | `tempo_bpm`, `valence`, `danceability`, `acousticness` are parsed but unused |
| Exact string match on genre | Trivially simple and predictable | `indie pop` ≠ `pop`; no notion of related genres |
| Reasons built during scoring, not after | Explanations can't drift from the score that produced them | Reason wording is coupled to the scoring function |
| Dict-based functional pipeline for the CLI | Direct CSV→dict flow, easy to test | Duplicates the `Song`/`UserProfile`/`Recommender` dataclass API, which remains as scaffolding |

**Why `tempo_bpm` is excluded.** It ranges 54–200 across the catalog, not 0–1. Dropping it into the same closeness formula without min-max scaling would let it swamp every other signal. Excluding it was cheaper than scaling it correctly for v1.

---

## Testing Summary

**The short version:** 252 of 252 automated tests pass. Across the seven demo profiles the system made 35 recommendations with an average confidence of **0.88**, flagging **0 of 35** as low-confidence. All **105 generated claims were grounded** in retrieved data (grounding rate 1.00, 0 explanations withheld). Fault injection confirmed the guardrails: a catalog with 3 corrupt rows loaded the 3 good ones and skipped the rest, and an artificially desynchronised record had its unsupported claim dropped rather than printed. The biggest problem is still the confidence layer: **15 of those 35 picks were genre-only matches that missed the requested mood, and not one was flagged** — and it takes a genre *and* a mood the catalog does not contain before the low band fires at all.

Every figure in this section is reproduced by a command in [Reproducible Execution Evidence](#reproducible-execution-evidence): the test count in [E1](#e1--test-suite), the 35/0.88/105/1.00 run figures in [E2](#e2--full-demo-run-all-seven-profiles) and [E5](#e5--reliability-sweep-across-all-seven-profiles), the band behaviour in [E6](#e6--uncovered-genre-polka), and the fault injections in [E7](#e7--guardrail-results).

### How it is tested

| Layer | File | Tests | What it pins down |
|---|---|---|---|
| Scoring & ranking | [tests/test_recommender.py](tests/test_recommender.py) | 10 | The weight invariants (genre > mood; weights sum to `MAX_SCORE`), exact score values, descending order, `k`, the empty-reasons fallback, CSV typing |
| Guardrails | [tests/test_guardrails.py](tests/test_guardrails.py) | 14 | Bad rows rejected, bad profiles rejected, energy clamping, grounding check |
| Retrieval & confidence | [tests/test_retriever.py](tests/test_retriever.py) | 40 | Catalog failures, ranking, confidence bounds/ordering/bands, `extract_prefs`, the semantic path, blended ordering |
| Explanation grounding | [tests/test_explainer.py](tests/test_explainer.py) | 9 | Wording tracks evidence, ungrounded claims dropped, explanations withheld, no genre leaks across every demo profile |
| Chunking | [tests/test_chunking.py](tests/test_chunking.py) | 17 | Heading boundaries, overlap that never crosses one, the hard-max invariant, runt merging, determinism |
| Corpus & fingerprint | [tests/test_corpus.py](tests/test_corpus.py) | 32 | Song-card rendering keeps every number verbatim, KB assembly, fingerprint invalidation |
| Embedding backends | [tests/test_embeddings.py](tests/test_embeddings.py) | 29 | One vector per chunk, the `types.Content` footgun, prefix application, rate-limit retry, IDF hashing |
| Vector store | [tests/test_vector_store.py](tests/test_vector_store.py) | 27 | Cosine, top-k ordering, stratified retrieval, JSONL round-trip, staleness detection |
| Prompting & grounding | [tests/test_answerer.py](tests/test_answerer.py) | 23 | Prompt fences the song list, fabricated citations dropped, substituted songs withheld, template fallback |
| Ingest CLI | [tests/test_ingest.py](tests/test_ingest.py) | 9 | `--check` and `--dry-run` contracts, exit codes, that ingest is the only index writer |
| End-to-end CLI | [tests/test_main.py](tests/test_main.py) | 22 | Full pipeline runs in both modes, every demo profile runs alone and is covered by the catalog, reliability report prints, clean failure + exit code 1 |

Three independent reliability mechanisms, not one:

1. **Automated tests** — including two *fault-injection* tests that deliberately break the system to prove the guardrail fires: a catalog whose rows are corrupt, and a retrieved record whose evidence has been desynchronised from the song it describes.
2. **Confidence scoring** — every recommendation is rated 0–1 from two ingredients (share of points earned, and how many distinct signals matched), then banded high/medium/low. The band controls the *wording*: only a high-confidence pick is allowed to say "Strong match", and a low-confidence pick must carry a "stretch pick" warning.
3. **Logging and error handling** — every stage writes to `logs/run.log`; malformed rows are skipped with the line number and reason; unrecoverable problems (missing catalog, no valid rows, invalid profile) exit with a one-line message and code 1, never a traceback.
4. **Human evaluation** — reading the output of contrasting runs and judging it by eye, which is the only mechanism that catches "plausible but wrong."

```json
{
  "human_evaluation": [
    {
      "test_input": "-k 3",
      "evaluation_criteria": "Songs match by genre then mood",
      "result": "Pass"
    },
    {
      "test_input": "Empty input",
      "evaluation_criteria": "Handles gracefully",
      "result": "Pass"
    }
  ]
}
```

| Test input | Criteria | Result | Evidence |
|---|---|---|---|
| `-k 3` | Songs match by genre first, then mood | Pass | [Sample Interactions](#sample-interactions) — every rank matches the requested genre; mood is the signal that slips |
| Empty / unusable input | Handles gracefully, no traceback | Pass | [E7 G2–G5](#e7--guardrail-results) — missing catalog, all-invalid catalog, `-k 0`, malformed profiles |

### What worked

- **Scoring math is exact and pinned.** A perfect match returns precisely 4.0; a genre/mood miss at energy distance 0.4 returns precisely 1.2. Any weight change breaks these loudly, which is the point.
- **The explanation cannot outrun the evidence.** Claims are built from the same `SignalMatch` objects that produced the score and checked against the retrieved record before printing. Injecting a mismatch drops the claim; corrupting every signal makes the agent withhold the explanation instead of inventing one.
- **The caveats are always right, even when the confidence number isn't.** Whatever the band says, the agent names every mismatch it found: `city pop`, `indie pop`, `funk` and `jazz` results all state what they actually are instead of passing as the requested genre. Swept across all seven profiles and the full 203-song catalog — 1,421 explanations — not one names a genre the song does not have, and no claim was dropped. The grounded-claim mechanism held up under both catalog changes; the threshold policy did not.
- **Genre now reliably outranks mood in the output, not just in the constants.** The pop profile's rank 3 is a `pop`/`confident` track at 3.40, ahead of every `indie pop`/`happy` candidate — the ordering the 1.5/0.5 split was meant to produce. Two tests pin the rule itself (`W_GENRE > W_MOOD`, and the weights summing to `MAX_SCORE`) so a future retune fails as a violated rule rather than as an unexplained arithmetic mismatch.
- **Guardrails degrade instead of collapsing.** 3 corrupt rows out of 6 → the run continued on the 3 valid songs and logged the reason for each skip.
- **Two problems from the last phase are fixed.** The `Song`/`UserProfile`/`Recommender` dataclasses no longer duplicate the pipeline — they delegate to it. And `python3 -m src.main` now works (both invocations do).

### What didn't work

- **Growing the catalog broke the confidence signal, and the second expansion finished the job.** This is still the biggest finding. At 18 songs the demo profiles produced 6 low-confidence picks and averaged 0.66; at 53 songs, **zero** and 0.76; at 203 songs, zero and **0.88**. Nothing improved — a deeper catalog just guarantees that *some* song has the right genre and close energy, which now clears the `high` threshold outright. The scoreboard across 35 picks: **20 full genre+mood matches, 15 genre-only picks, 0 flagged.** A detector that flags nothing has perfect recall and no value. The thresholds (0.75 / 0.45) were hand-picked against an 18-song catalog and have not survived either expansion.
- **The low band now takes a double miss to reach.** Worse than the last phase, where it still fired on thin genres. With ten songs in every genre but jazz, `metal`/`angry` averages 0.90 with 5 of 5 high, and `classical`, `country`, `ska` and `r&b` land 5 of 5 high whenever the requested energy sits inside the range that genre actually spans. The decisive case: `polka` — **zero** songs in the catalog — returns 5 of 5 *medium* at 0.63 when the requested mood exists. The retriever logs "no song has genre 'polka'" and the confidence layer still declines to call any of it a weak match; coverage is measured by the retriever and ignored by the score. Only when the *mood* misses as well does the band drop to 5 of 5 low at 0.43 ([E6](#e6--uncovered-genre-polka)). So the mechanism is not dead — it is just calibrated so that one matching categorical, or a lucky energy target, is enough to suppress it. The same run also shows the band working properly on a genuinely thin catalog: against `data/broken.csv` it flags 2 of 3 picks low and words them as stretch picks ([E7 G1](#e7--guardrail-results)).
- **Confidence is relative to the catalog, not to the user.** Because both ingredients are catalog-relative, the same song can score 0.43 in a small catalog and 0.70 in a large one without changing at all. Any threshold expressed as an absolute number inherits that instability. A rank-relative measure (how much better is this than the median candidate?) would not.
- **Confidence cannot audit the policy it comes from.** It is computed from the same weights it reports on, so it measures *agreement with the scoring rule*, not correctness. If the energy-first weighting is the wrong policy — and the previous phase argued it is — a confidently wrong recommendation still scores high. Self-reported confidence is a consistency check, not a truth check.
- **Energy still dominates the ranking.** With `W_ENERGY = 2.0` energy contributes to every song and is always positive, so genre and mood act as tie-breakers on an energy-driven order. Splitting the categorical weight 1.5/0.5 in favor of genre made the tie-break break the *right way* — an off-genre song can no longer draw level with an on-genre one on a mood match — but energy's 2.0 share is untouched, so the underlying dominance remains. The confidence layer makes it visible rather than fixing it: rock is still the weakest profile (0.82 against a 0.88 mean), which is the one the catalog serves worst. The cost lands on mood, not genre: at 0.5 a mood match moves a score less than an energy difference of 0.25 does, which is why 15 of 35 picks miss the requested mood.
- **Exact string matching is unchanged by the new data.** `city pop` and `indie pop` both score zero for a `pop` listener, and `indie rock` scores zero for `rock` — three near-misses the catalog now contains and the matcher still cannot see.

**What I learned:** the useful question is not "did the tests pass" but "what would the system say when it shouldn't be sure?" All 7 tests passed in the previous phase while the system confidently recommended pop to a rock listener. What changed that wasn't more tests of the same kind — it was making the system *rate its own evidence* and forcing the wording to obey the rating, then testing that mechanism by breaking it on purpose.

Then adding songs silently disabled it — twice. Every test still passed, the average confidence went *up* both times (0.66 → 0.76 → 0.88), and the run report looked healthier at every step, while the system's ability to tell a real match from a coincidence went to zero. That is the same lesson as the energy-weight finding, one level up: the metric moved in the flattering direction for a reason that had nothing to do with quality. A number that only ever gets better is not measuring anything.

The profile expansion taught a narrower version of the same thing. Four new profiles went in and the whole suite stayed green — because the end-to-end test asserted on three profile names it already knew about. A test that names its expectations instead of deriving them will keep passing over exactly the code you just added, so the CLI tests now build their assertions from `PROFILES` itself.

---

## Limitations and Risks

- **Genre coverage is no longer the binding constraint (203 songs)** — this limitation is largely resolved. The catalog now spans 19 genres with 10 songs each, plus 23 jazz; there are no singleton genres left, and every demo profile has both its genre and its mood represented. What it exposed is that coverage was never the real problem: with the thin-catalog excuse removed, the confidence layer stopped flagging anything across the seven demo profiles, and it will not flag a genre the catalog lacks entirely unless the requested mood is missing too. Mood coverage is thinner than genre coverage — 13 moods across 203 songs — so a mood miss remains the common failure.
- **No audio understanding** — it compares human-assigned labels, never the music itself. Every bias in the labeling is inherited wholesale.
- **Energy over-weighting** — the dominant signal by construction; documented above and in the model card.
- **No related-genre knowledge** — exact string equality only.
- **No diversity control** — nothing prevents the top-5 from being the same artist repeatedly.
- **Single-preference profiles** — a listener with genuinely mixed taste cannot be expressed.
- **Self-reported confidence** — the confidence score is derived from the same weights it rates, so it measures internal consistency, not correctness. It cannot flag a recommendation that the scoring policy itself gets wrong.

### Risks introduced by adding RAG

- **Non-deterministic wording** — explanations are no longer byte-reproducible when Gemini is enabled. Song selection still is. See [What is and isn't deterministic now](#what-is-and-isnt-deterministic-now).
- **Hallucination surface** — a language model can now fabricate. The guardrails catch citation fabrication and song substitution ([E10](#e10--the-grounding-guardrail-catching-a-misbehaving-model)), but they check *provenance*, not truth: a sentence that cites a real passage and misdescribes it would pass. Constraining the model to two sentences per song limits the room for this, but does not eliminate it.
- **Prose knowledge base authorship bias** — ~7,200 words of one person's characterisations of 19 genres and 13 moods now steer retrieval. Writing that reggae suits "sunshine and cooking" is an opinion, and it is now load-bearing infrastructure. This is a genuinely new fairness surface and the most under-examined part of the system.
- **Embedding model bias** — `gemini-embedding-2` was trained on internet text, which is English-centric and popularity-weighted. A genre that is well documented online embeds more usefully than one that is not, so the retrieval quality of city pop and lofi is unlikely to match that of, say, a regional tradition with a thin web footprint.
- **API cost and availability** — the system now has an external dependency that can be rate-limited, deprecated, or billed. The offline path exists so that none of those are fatal.
- **Moving model ids** — `gemini-flash-lite-latest` is an alias that tracks upstream. Recorded outputs in this README are dated, and a future run may differ for reasons outside this repository.
- **`extract_prefs` is heuristic** — vocabulary matching with a synonym map and negation handling. It handles the demo queries and much conversational phrasing, but it is not a parser. Its failures are bounded: it only affects reranking, never retrieval, so the worst case is a slightly worse ordering within an already-relevant shortlist.

Full analysis, including responsible-AI reflection and where bias enters: [model_card.md](model_card.md).

---

## Project Structure

```
├── src/
│   ├── main.py            # CLI runner: argparse, demo profiles, run report
│   ├── repl.py            # interactive session over a warm stack + the why command
│   ├── ingest.py          # build-time CLI: the only writer of the vector index
│   ├── config.py          # every tunable: model ids, dims, paths, chunk sizes, weights
│   ├── corpus.py          # song-card rendering, KB assembly, corpus fingerprint
│   ├── chunking.py        # chunk_text: heading-aware splitting with overlap
│   ├── embeddings.py      # Embedder ABC, GeminiEmbedder, offline HashingEmbedder
│   ├── vector_store.py    # cosine top-k, stratified retrieval, JSONL persistence
│   ├── retriever.py       # Retriever + SemanticRetriever + extract_prefs
│   ├── llm_client.py      # GeminiGenerator, TemplateGenerator, test doubles
│   ├── answerer.py        # AnswerAgent: prompt building, grounding, fallback
│   ├── explainer.py       # ExplanationAgent: deterministic template backend
│   ├── recommender.py     # scoring/ranking core + load_songs + OOP facade
│   ├── guardrails.py      # row/profile/query validation, grounding checks, errors
│   ├── models.py          # Chunk, RetrievedSong, SignalMatch, confidence
│   └── logging_setup.py   # console + logs/run.log configuration
├── data/
│   ├── songs.csv          # 203-song catalog, 19 genres, 10 fields per song
│   └── index/gemini_index.jsonl   # committed vectors, 277 chunks (see Cost)
├── docs/
│   ├── kb/                # the prose knowledge base, ~7,200 words
│   │   ├── genres.md      # all 19 catalog genres
│   │   ├── moods.md       # all 13 catalog moods
│   │   ├── eras.md        # seven periods and scenes
│   │   └── listening_contexts.md   # ten situations people actually ask about
│   ├── algorithm_recipe.md    # Phase 2 scoring design (code-free, superseded)
│   └── retrieval_recipe.md    # chunking, prefixes, blending (code-free)
├── tests/
│   ├── test_recommender.py    # scoring + ranking math
│   ├── test_guardrails.py     # validation and grounding
│   ├── test_retriever.py      # catalog failures, ranking, confidence, semantic path
│   ├── test_explainer.py      # grounded generation, dropped claims
│   ├── test_chunking.py       # boundaries, overlap, determinism
│   ├── test_embeddings.py     # one-vector-per-chunk, the types.Content footgun
│   ├── test_vector_store.py   # cosine, top-k ordering, persistence, freshness
│   ├── test_corpus.py         # card rendering, fingerprint invalidation
│   ├── test_answerer.py       # prompt building and the grounding guardrail
│   ├── test_ingest.py         # --check / --dry-run contracts
│   ├── test_main.py           # end-to-end CLI, both modes
│   └── test_repl.py           # interactive session, the why command, warm stack
├── diagrams/
│   ├── uml.mmd            # class diagram
│   └── architecture.mmd   # system flow diagram
├── conftest.py            # sys.path + autouse fixtures blocking network and key
├── .env.example           # GEMINI_API_KEY placeholder (.env is gitignored)
└── model_card.md          # intended use, models used, data, limitations, reflection
```

Data flows one way. At build time: `ingest -> corpus -> chunking -> embeddings -> vector_store`. At query time: `main -> retriever (embeddings + vector_store + recommender core) -> answerer -> llm_client -> output`, with `guardrails` and `logging_setup` used at every stage. `config.py` imports no project code and `models.py` imports only `config`, so the dependency graph stays acyclic.

---

## Reflection

Building this taught me that a recommender doesn't understand anything. It compares labels and numbers and adds them up, and everything a user experiences as "taste" or "insight" is really a set of weights somebody picked. The system's opinion about music is entirely an artifact of the three constants at the top of `score_song()`.

The sharpest lesson was how invisible that is from the inside. I raised the energy weight and the system kept producing recommendations that read as perfectly reasonable — plausible songs, confident explanations, no errors, all tests green. Only when I laid several different user profiles next to each other did the pattern show: the same reason text on almost every result, genre demoted to a tiebreaker. Nothing *failed*; the system just quietly became a different product than the one I designed. Splitting the categorical weight 1.5/0.5 fixed the ordering between genre and mood, but it took writing the rule down as a test — `W_GENRE > W_MOOD` — before the diagrams stopped advertising three different weightings the code had never used. That reframed problem-solving for me — evaluating an AI system isn't checking whether the output looks right, it's checking whether the output is right *for reasons you can name*. Now when a streaming app recommends something, my first thought is about whoever decided which factors count and how much.

The graded responsible-AI reflection — how I collaborated with AI tools, one helpful and one flawed AI suggestion, and the system's limitations — is in [model_card.md](model_card.md).
