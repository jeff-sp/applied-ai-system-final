# 🎵 HarmonyRanker — Music Recommender Simulation

**Original project:** *Music Recommender Simulation* (CodePath Applied AI, Modules 1–3).
The original goal was to represent songs and a listener's "taste profile" as structured data, design a transparent scoring rule that turns that data into ranked recommendations, and evaluate where the rule succeeds and fails. Its capabilities were: load a small song catalog from CSV, score each song against a user profile using genre, mood, and energy, and return the top-k matches. This final version completes that implementation, adds human-readable explanations for every recommendation, adds a multi-profile demo harness, and adds a test suite.

---

## Summary

HarmonyRanker is a **content-based** music recommender. It does not listen to audio and it does not use collaborative filtering — it compares each song's labeled attributes (genre, mood, energy) against a user's stated preferences, produces a numeric score, and ranks the catalog.

Why it matters: it is a working, fully legible model of how commercial recommenders actually behave. Every number in the ranking is traceable to a specific weight in a specific line of code, which makes it a good vehicle for seeing how a single weighting decision can quietly dominate an entire system's output — the exact failure mode documented in [Testing Summary](#testing-summary) and [model_card.md](model_card.md).

---

## Architecture Overview

The system diagram lives at [diagrams/architecture.mmd](diagrams/architecture.mmd) (Mermaid). It maps the recommender onto a retrieval-augmented pattern:

- **Input** — a user taste profile (`favorite_genre`, `favorite_mood`, `target_energy`) enters through the CLI runner, [src/main.py](src/main.py).
- **Retriever** ([src/retriever.py](src/retriever.py)) — owns the knowledge base. `load_songs()` parses, type-casts and validates `data/songs.csv`; `evaluate_song()` scores each song against the profile *and records the evidence for that score*; `rank_songs()` sorts descending and cuts to top-k. Scoring (a per-song number) stays separate from ranking (the sort-and-cut policy) so either can change without touching the other.
- **Retrieved records** — the retriever does not hand back bare songs. Each result is a `RetrievedSong` carrying the song, its score, a `SignalMatch` per signal (hit *or* miss), a confidence value, and a confidence band.
- **Explanation Agent** ([src/explainer.py](src/explainer.py)) — the generation half. It never touches the catalog; it sees only the retrieved record and builds its sentences from that evidence. Retrieval doesn't decorate the answer, it *is* the answer: the wording, the hedging, and the caveats are all derived from which signals matched. Generation is template-based and deterministic — no model call, no network, no API key — which is what makes the output testable.
- **Guardrails** ([src/guardrails.py](src/guardrails.py)) — three layers: malformed catalog rows are skipped and logged instead of crashing the run or poisoning the ranking; invalid profiles are rejected before scoring (an out-of-range `target_energy` is clamped and logged); and every claim the agent wants to make is checked against the retrieved record before it can be printed. An unsupported claim is dropped; if no claim survives, the explanation is withheld rather than invented.
- **Logging** ([src/logging_setup.py](src/logging_setup.py)) — every stage logs to `logs/run.log` (DEBUG) with warnings surfaced on the console. `--log-level INFO` shows the pipeline narrating itself.
- **Evaluation** — 63 automated tests across five files, plus a per-run reliability report (average confidence, low-confidence share, grounding rate) and human review of seven contrasting profiles. See [Testing Summary](#testing-summary).

The class structure is in [diagrams/uml.mmd](diagrams/uml.mmd): `Song`, `UserProfile`, and `Recommender`.

---

## Setup Instructions

**Requirements:** Python 3.8+

1. Clone and enter the repo:

```bash
git clone https://github.com/jeff-sp/applied-ai-system-final.git && cd applied-ai-system-final
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate`.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> The recommender itself uses only the Python standard library — `pip install` is needed for the test runner, not for the app.

4. Run the demo from the repo root:

```bash
python3 -m src.main
```

> Run it from the repo root — the catalog is resolved at the relative path `data/songs.csv`. (`python3 src/main.py` also works.)

5. Run the test suite:

```bash
python3 -m pytest -q
```

Expected: `63 passed`.

### CLI options

```bash
python3 -m src.main --profile lofi -k 3 --log-level INFO
```

| Flag | Default | What it does |
|---|---|---|
| `--profile` | `all` | One of `pop`, `lofi`, `rock`, `metal`, `city pop`, `jazz`, `funk`, or `all`. Quote the ones with a space: `--profile "city pop"` |
| `-k` | `5` | How many songs to recommend |
| `--catalog` | `data/songs.csv` | Point at a different catalog CSV |
| `--log-level` | `WARNING` | Console verbosity; `logs/run.log` always gets the full DEBUG trail |

### Trying your own profile

Add an entry to `PROFILES` in [src/main.py](src/main.py). A profile needs exactly three keys:

```python
my_profile = {
    "favorite_genre": "jazz",     # must match a genre string in data/songs.csv
    "favorite_mood": "relaxed",
    "target_energy": 0.35,        # 0.0 (calm) to 1.0 (intense)
}
```

Anything else is rejected by the profile guardrail with a message naming the offending field, rather than silently producing confident nonsense.

---

## Sample Interactions

Three of the seven demo profiles are shown below — real output from `python3 -m src.main -k 3` against the 203-song catalog. Every sentence after the score line is generated by the Explanation Agent from the retrieved evidence — nothing is hard-coded per song.

### Example 1 — High-energy pop

**Input:** `{"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}`

```
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

All three are full three-signal matches, and the top two are the same artist — which is exactly the no-diversity-control limitation listed below showing up in practice.

### Example 3 — Intense rock

**Input:** `{"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.7}`

```
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

The low band no longer fires anywhere. Not for `metal`/`angry` (0.90 average, 5 of 5 high), not for the thinnest profiles, and not even for `polka` — a genre with **zero** songs in the catalog, which returns 5 of 5 *medium* at 0.63 despite an explicit coverage warning in the log.

### Guardrails in action

Pointing the runner at a catalog with three corrupt rows (out-of-range energy, missing fields, an unparseable number):

```
[WARNING] music_rec.guardrails: broken.csv: skipped 3 malformed row(s); first: line 5: energy=9.9 is outside the 0.0-1.0 range

Loaded 3 songs from broken.csv
```

The run continues on the valid rows instead of crashing, and the skip is on the console and in `logs/run.log`. A catalog where *nothing* is valid, or the file is missing, fails fast with a one-line error and exit code 1 — no traceback.

---

## Design Decisions

**Content-based, not collaborative.** There is one user and no interaction history, so there is nothing to collaboratively filter. Content-based scoring is also fully inspectable — a requirement for the explanation feature.

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

**Why `tempo_bpm` is excluded.** It ranges 60–168, not 0–1. Dropping it into the same closeness formula without min-max scaling would let it swamp every other signal. Excluding it was cheaper than scaling it correctly for v1.

---

## Testing Summary

**The short version:** 63 of 63 automated tests pass. Across the seven demo profiles the system made 35 recommendations with an average confidence of **0.88**, flagging **0 of 35** as low-confidence. All **105 generated claims were grounded** in retrieved data (grounding rate 1.00, 0 explanations withheld). Fault injection confirmed the guardrails: a catalog with 3 corrupt rows loaded the 3 good ones and skipped the rest, and an artificially desynchronised record had its unsupported claim dropped rather than printed. The headline problem is still the confidence layer: **15 of those 35 picks were genre-only matches that missed the requested mood, and not one was flagged** — and the low band has now stopped firing even for a genre the catalog does not contain at all.

### How it is tested

| Layer | File | Tests | What it pins down |
|---|---|---|---|
| Scoring & ranking | [tests/test_recommender.py](tests/test_recommender.py) | 10 | The weight invariants (genre > mood; weights sum to `MAX_SCORE`), exact score values, descending order, `k`, the empty-reasons fallback, CSV typing |
| Guardrails | [tests/test_guardrails.py](tests/test_guardrails.py) | 14 | Bad rows rejected, bad profiles rejected, energy clamping, grounding check |
| Retrieval & confidence | [tests/test_retriever.py](tests/test_retriever.py) | 16 | Catalog failures, ranking, confidence bounds/ordering/bands |
| Explanation grounding | [tests/test_explainer.py](tests/test_explainer.py) | 9 | Wording tracks evidence, ungrounded claims dropped, explanations withheld, no genre leaks across every demo profile |
| End-to-end CLI | [tests/test_main.py](tests/test_main.py) | 14 | Full pipeline runs, every demo profile runs alone and is covered by the catalog, reliability report prints, clean failure + exit code 1 |

Three independent reliability mechanisms, not one:

1. **Automated tests** — including two *fault-injection* tests that deliberately break the system to prove the guardrail fires: a catalog whose rows are corrupt, and a retrieved record whose evidence has been desynchronised from the song it describes.
2. **Confidence scoring** — every recommendation is rated 0–1 from two ingredients (share of points earned, and how many distinct signals matched), then banded high/medium/low. The band controls the *wording*: only a high-confidence pick is allowed to say "Strong match", and a low-confidence pick must carry a "stretch pick" warning.
3. **Logging and error handling** — every stage writes to `logs/run.log`; malformed rows are skipped with the line number and reason; unrecoverable problems (missing catalog, no valid rows, invalid profile) exit with a one-line message and code 1, never a traceback.

### What worked

- **Scoring math is exact and pinned.** A perfect match returns precisely 4.0; a genre/mood miss at energy distance 0.4 returns precisely 1.2. Any weight change breaks these loudly, which is the point.
- **The explanation cannot outrun the evidence.** Claims are built from the same `SignalMatch` objects that produced the score and checked against the retrieved record before printing. Injecting a mismatch drops the claim; corrupting every signal makes the agent withhold the explanation instead of inventing one.
- **The caveats are always right, even when the confidence number isn't.** Whatever the band says, the agent names every mismatch it found: `city pop`, `indie pop`, `funk` and `jazz` results all state what they actually are instead of passing as the requested genre. Swept across all seven profiles and the full 203-song catalog — 1,421 explanations — not one names a genre the song does not have, and no claim was dropped. The grounded-claim mechanism held up under both catalog changes; the threshold policy did not.
- **Genre now reliably outranks mood in the output, not just in the constants.** The pop profile's rank 3 is a `pop`/`confident` track at 3.40, ahead of every `indie pop`/`happy` candidate — the ordering the 1.5/0.5 split was meant to produce. Two tests pin the rule itself (`W_GENRE > W_MOOD`, and the weights summing to `MAX_SCORE`) so a future retune fails as a violated rule rather than as an unexplained arithmetic mismatch.
- **Guardrails degrade instead of collapsing.** 3 corrupt rows out of 6 → the run continued on the 3 valid songs and logged the reason for each skip.
- **Two problems from the last phase are fixed.** The `Song`/`UserProfile`/`Recommender` dataclasses no longer duplicate the pipeline — they delegate to it. And `python3 -m src.main` now works (both invocations do).

### What didn't work

- **Growing the catalog broke the confidence signal, and the second expansion finished the job.** This is still the biggest finding. At 18 songs the demo profiles produced 6 low-confidence picks and averaged 0.66; at 53 songs, **zero** and 0.76; at 203 songs, zero and **0.88**. Nothing improved — a deeper catalog just guarantees that *some* song has the right genre and close energy, which now clears the `high` threshold outright. The scoreboard across 35 picks: **20 full genre+mood matches, 15 genre-only picks, 0 flagged.** A detector that flags nothing has perfect recall and no value. The thresholds (0.75 / 0.45) were hand-picked against an 18-song catalog and have not survived either expansion.
- **The low band is now unreachable in practice.** Worse than the last phase, where it at least still fired on thin genres. With ten songs in every genre but jazz, `metal`/`angry` averages 0.90 with 5 of 5 high, and `classical`, `country`, `ska` and `r&b` all land 5 of 5 high. The decisive case: `polka` — **zero** songs in the catalog — returns 5 of 5 *medium* at 0.63. The retriever logs "no song has genre 'polka'" and the confidence layer still declines to call any of it a weak match. Coverage is measured by the retriever and ignored by the score.
- **Confidence is relative to the catalog, not to the user.** Because both ingredients are catalog-relative, the same song can score 0.43 in a small catalog and 0.70 in a large one without changing at all. Any threshold expressed as an absolute number inherits that instability. A rank-relative measure (how much better is this than the median candidate?) would not.
- **Confidence cannot audit the policy it comes from.** It is computed from the same weights it reports on, so it measures *agreement with the scoring rule*, not correctness. If the energy-first weighting is the wrong policy — and the previous phase argued it is — a confidently wrong recommendation still scores high. Self-reported confidence is a consistency check, not a truth check.
- **Energy still dominates the ranking.** With `W_ENERGY = 2.0` energy contributes to every song and is always positive, so genre and mood act as tie-breakers on an energy-driven order. Splitting the categorical weight 1.5/0.5 in favor of genre made the tie-break break the *right way* — an off-genre song can no longer draw level with an on-genre one on a mood match — but energy's 2.0 share is untouched, so the underlying dominance remains. The confidence layer makes it visible rather than fixing it: rock is still the weakest profile (0.82 against a 0.88 mean), which is the one the catalog serves worst. The cost lands on mood, not genre: at 0.5 a mood match moves a score less than an energy difference of 0.25 does, which is why 15 of 35 picks miss the requested mood.
- **Exact string matching is unchanged by the new data.** `city pop` and `indie pop` both score zero for a `pop` listener, and `indie rock` scores zero for `rock` — three near-misses the catalog now contains and the matcher still cannot see.

**What I learned:** the useful question is not "did the tests pass" but "what would the system say when it shouldn't be sure?" All 7 tests passed in the previous phase while the system confidently recommended pop to a rock listener. What changed that wasn't more tests of the same kind — it was making the system *rate its own evidence* and forcing the wording to obey the rating, then testing that mechanism by breaking it on purpose.

Then adding songs silently disabled it — twice. Every test still passed, the average confidence went *up* both times (0.66 → 0.76 → 0.88), and the run report looked healthier at every step, while the system's ability to tell a real match from a coincidence went to zero. That is the same lesson as the energy-weight finding, one level up: the metric moved in the flattering direction for a reason that had nothing to do with quality. A number that only ever gets better is not measuring anything.

The profile expansion taught a narrower version of the same thing. Four new profiles went in and the whole suite stayed green — because the end-to-end test asserted on three profile names it already knew about. A test that names its expectations instead of deriving them will keep passing over exactly the code you just added, so the CLI tests now build their assertions from `PROFILES` itself.

---

## Limitations and Risks

- **Genre coverage is no longer the binding constraint (203 songs)** — this limitation is largely resolved. The catalog now spans 19 genres with 10 songs each, plus 23 jazz; there are no singleton genres left, and every demo profile has both its genre and its mood represented. What it exposed is that coverage was never the real problem: with the thin-catalog excuse removed, the confidence layer stopped flagging anything at all, including genres the catalog does not contain. Mood coverage is thinner than genre coverage — 13 moods across 203 songs — so a mood miss remains the common failure.
- **No audio understanding** — it compares human-assigned labels, never the music itself. Every bias in the labeling is inherited wholesale.
- **Energy over-weighting** — the dominant signal by construction; documented above and in the model card.
- **No related-genre knowledge** — exact string equality only.
- **No diversity control** — nothing prevents the top-5 from being the same artist repeatedly.
- **Single-preference profiles** — a listener with genuinely mixed taste cannot be expressed.
- **Self-reported confidence** — the confidence score is derived from the same weights it rates, so it measures internal consistency, not correctness. It cannot flag a recommendation that the scoring policy itself gets wrong.
- **Template-based explanations** — deterministic and fully auditable, but the phrasing is fixed. It explains *why the score came out that way*, which is not the same as musical insight.

Full analysis, including responsible-AI reflection and where bias enters: [model_card.md](model_card.md).

---

## Project Structure

```
├── src/
│   ├── main.py            # CLI runner: argparse, demo profiles, run report
│   ├── retriever.py       # Retriever: knowledge base + top-k retrieval with evidence
│   ├── explainer.py       # ExplanationAgent: grounded generation from retrieved records
│   ├── recommender.py     # scoring/ranking core + load_songs + OOP facade
│   ├── guardrails.py      # row/profile validation, grounding checks, error types
│   ├── models.py          # Song, UserProfile, SignalMatch, RetrievedSong, confidence
│   └── logging_setup.py   # console + logs/run.log configuration
├── data/songs.csv         # 203-song catalog, 19 genres, 10 fields per song
├── tests/
│   ├── test_recommender.py    # scoring + ranking math
│   ├── test_guardrails.py     # validation and grounding
│   ├── test_retriever.py      # catalog failures, ranking, confidence
│   ├── test_explainer.py      # grounded generation, dropped claims
│   └── test_main.py           # end-to-end CLI
├── docs/algorithm_recipe.md   # Phase 2 scoring design (code-free, superseded)
├── diagrams/
│   ├── uml.mmd            # class diagram
│   └── architecture.mmd   # system flow diagram
├── conftest.py            # puts the repo root on sys.path for pytest
└── model_card.md          # intended use, data, limitations, reflection
```

Data flows one way: `main -> retriever -> (recommender core) -> explainer -> output`, with `guardrails` and `logging_setup` used at every stage. `models.py` imports no project code, so the dependency graph stays acyclic.

---

## Reflection

Building this taught me that a recommender doesn't *understand* anything. It compares labels and numbers and adds them up, and everything a user experiences as "taste" or "insight" is really a set of weights somebody picked. The system's opinion about music is entirely an artifact of the three constants at the top of `score_song()`.

The sharpest lesson was how invisible that is from the inside. I raised the energy weight and the system kept producing recommendations that read as perfectly reasonable — plausible songs, confident explanations, no errors, all tests green. Only when I laid several different user profiles next to each other did the pattern show: the same reason text on almost every result, genre demoted to a tiebreaker. Nothing *failed*; the system just quietly became a different product than the one I designed. Splitting the categorical weight 1.5/0.5 fixed the ordering between genre and mood, but it took writing the rule down as a test — `W_GENRE > W_MOOD` — before the diagrams stopped advertising three different weightings the code had never used. That reframed problem-solving for me — evaluating an AI system isn't checking whether the output looks right, it's checking whether the output is right *for reasons you can name*. Now when a streaming app recommends something, my first thought is about whoever decided which factors count and how much.

The graded responsible-AI reflection — how I collaborated with AI tools, one helpful and one flawed AI suggestion, and the system's limitations — is in [model_card.md](model_card.md).
